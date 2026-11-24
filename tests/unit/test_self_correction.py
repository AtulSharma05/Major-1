"""Comprehensive unit tests for Stage 5 Bounded Self-Correction Loop.

Covers all 13 core self-correction requirements:
1. Successful correction after an execution failure (syntax/column error).
2. Successful correction after a deterministic semantic failure (metric mismatch).
3. Successful correction after an actionable numerical mismatch.
4. No correction for INCONCLUSIVE validation (ambiguous data / missing intent).
5. No correction for ambiguous intent.
6. Strict retry limit enforcement: exactly two retries (3 total attempts).
7. Successful correction on the second retry (attempt 3 succeeds).
8. Both retries fail: clean termination with validation_failed.
9. Full validation rerun across all layers after every correction.
10. Preservation of original structured intent across all attempts.
11. Immutable attempt history with complete diagnostic traceability.
12. Safety check enforcement: corrector cannot bypass SQL safety checks.
13. No correction after an initial successful execution.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.agent.codegen.generator import CodeGenerator
from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import profile_table
from src.agent.planning.intent import AggregationType, MetricSpec, QuerySpecification
from src.agent.planning.planner import AnalyticalPlan, PlanStep, QueryPlanner
from src.agent.validation.contracts import (
    ValidationStatus,
    WorkflowValidationReport,
)
from src.agent.validation.execution import ExecutionValidator
from src.agent.validation.numerical import NumericalValidator
from src.agent.validation.semantic import SemanticValidator
from src.agent.workflow.graph import build_analytical_graph
from src.agent.workflow.state import AgentState


class SequenceCodeGenerator(CodeGenerator):
    """Yields a sequence of SQL queries across calls to simulate iterative corrections."""

    def __init__(self, sql_sequence: list[str]):
        self.sql_sequence = list(sql_sequence)
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    def generate_sql(
        self,
        question: str,
        profile=None,
        intent=None,
        error_feedback=None,
        prior_step_context=None,
        **kwargs,
    ) -> str:
        idx = min(self.call_count, len(self.sql_sequence) - 1)
        sql = self.sql_sequence[idx]
        self.calls.append(
            {
                "question": question,
                "profile": profile,
                "intent": intent,
                "error_feedback": error_feedback,
                "prior_step_context": prior_step_context,
                "returned_sql": sql,
            }
        )
        self.call_count += 1
        return sql


class FixedIntentPlanner(QueryPlanner):
    """Planner that returns a pre-configured AnalyticalPlan with explicit structured intent."""

    def __init__(self, intent: QuerySpecification):
        self.intent = intent

    def plan(self, question: str, active_table: str = "", profile=None, **kwargs) -> AnalyticalPlan:
        step = PlanStep(
            step_id=1,
            question=question,
            target_table=active_table or self.intent.table_name,
            depends_on=[],
            structured_intent=self.intent,
        )
        return AnalyticalPlan(
            original_question=question,
            steps=[step],
            execution_order=[1],
            is_multi_step=False,
        )


@pytest.fixture
def workflow_env(data_dir):
    """Fixture providing initialized database, profile, and standard validators."""
    db = SessionDatabase()
    db.register_csv(data_dir / "sample_superstore.csv", "superstore")
    profile = profile_table(db, "superstore")
    exec_val = ExecutionValidator()
    num_val = NumericalValidator()
    sem_val = SemanticValidator()
    yield db, profile, exec_val, num_val, sem_val
    db.close()


# ---------------------------------------------------------------------------
# 1. Execution Failure Correction
# ---------------------------------------------------------------------------
def test_correction_execution_failure_resolved_on_attempt_2(workflow_env):
    """Verify syntax/column error in attempt 1 is corrected and verified on attempt 2."""
    db, profile, exec_val, num_val, sem_val = workflow_env

    intent = QuerySpecification(
        raw_question="What is total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )

    # Attempt 1: Syntax / non-existent column error
    # Attempt 2: Correct SQL
    generator = SequenceCodeGenerator(
        [
            "SELECT SUM(invalid_col) AS total_sales FROM superstore",
            "SELECT SUM(sales) AS total_sales FROM superstore",
        ]
    )

    graph = build_analytical_graph(
        db=db,
        generator=generator,
        planner=FixedIntentPlanner(intent),
        validator=exec_val,
        numerical_validator=num_val,
        semantic_validator=sem_val,
    )

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_exec_correction",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "validated"
    assert final_state["retry_count"] == 1
    assert generator.call_count == 2

    # Verify attempt history
    val_report: WorkflowValidationReport = final_state["validation_report"]
    history = val_report.step_history[1]
    assert len(history) == 2

    # Attempt 1 failed at Layer 1
    assert history[0].attempt_number == 1
    assert history[0].status == ValidationStatus.FAILED
    assert history[0].execution_result.status == ValidationStatus.FAILED
    assert history[0].diagnostic is not None
    assert history[0].diagnostic.is_correctable is True

    # Attempt 2 succeeded across all 3 layers
    assert history[1].attempt_number == 2
    assert history[1].status == ValidationStatus.VERIFIED
    assert history[1].execution_result.status == ValidationStatus.VERIFIED
    assert history[1].numerical_result.status == ValidationStatus.VERIFIED
    assert history[1].semantic_result.status == ValidationStatus.VERIFIED

    # Check that Corrector received structured error feedback
    assert generator.calls[1]["error_feedback"] is not None
    assert "invalid_col" in generator.calls[1]["error_feedback"]
    assert "CORRECTION DIAGNOSTIC" in generator.calls[1]["error_feedback"]


# ---------------------------------------------------------------------------
# 2. Deterministic Semantic Failure Correction
# ---------------------------------------------------------------------------
def test_correction_semantic_failure_resolved_on_attempt_2(workflow_env):
    """Verify semantic failure (wrong metric) is corrected and verified on attempt 2."""
    db, profile, exec_val, num_val, sem_val = workflow_env

    intent = QuerySpecification(
        raw_question="What is total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )

    # Attempt 1 executes successfully but calculates profit instead of sales
    # Attempt 2 calculates sales
    generator = SequenceCodeGenerator(
        [
            "SELECT SUM(profit) AS total_sales FROM superstore",
            "SELECT SUM(sales) AS total_sales FROM superstore",
        ]
    )

    graph = build_analytical_graph(
        db=db,
        generator=generator,
        planner=FixedIntentPlanner(intent),
        validator=exec_val,
        numerical_validator=num_val,
        semantic_validator=sem_val,
    )

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_sem_correction",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "validated"
    assert final_state["retry_count"] == 1
    assert generator.call_count == 2

    val_report: WorkflowValidationReport = final_state["validation_report"]
    history = val_report.step_history[1]
    assert len(history) == 2

    # Attempt 1: Semantic check failed
    assert history[0].status == ValidationStatus.FAILED
    assert (
        history[0].semantic_result.status == ValidationStatus.FAILED
        or history[0].numerical_result.status == ValidationStatus.FAILED
    )
    assert history[0].diagnostic is not None
    assert history[0].diagnostic.is_correctable is True

    # Attempt 2: All layers verified
    assert history[1].status == ValidationStatus.VERIFIED
    assert history[1].semantic_result.status == ValidationStatus.VERIFIED


# ---------------------------------------------------------------------------
# 3. Actionable Numerical Mismatch Correction
# ---------------------------------------------------------------------------
def test_correction_numerical_failure_resolved_on_attempt_2(workflow_env):
    """Verify numerical mismatch (wrong formula) is corrected on attempt 2."""
    db, profile, exec_val, num_val, sem_val = workflow_env

    intent = QuerySpecification(
        raw_question="What is total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )

    # Attempt 1: Multiplies by 2 -> numerical mismatch in Layer 2
    # Attempt 2: Correct formula
    generator = SequenceCodeGenerator(
        [
            "SELECT SUM(sales) * 2 AS total_sales FROM superstore",
            "SELECT SUM(sales) AS total_sales FROM superstore",
        ]
    )

    graph = build_analytical_graph(
        db=db,
        generator=generator,
        planner=FixedIntentPlanner(intent),
        validator=exec_val,
        numerical_validator=num_val,
        semantic_validator=sem_val,
    )

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_num_correction",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "validated"
    assert final_state["retry_count"] == 1

    val_report: WorkflowValidationReport = final_state["validation_report"]
    history = val_report.step_history[1]
    assert len(history) == 2
    assert history[0].numerical_result.status == ValidationStatus.FAILED
    assert history[1].numerical_result.status == ValidationStatus.VERIFIED


# ---------------------------------------------------------------------------
# 4 & 5. No Correction for INCONCLUSIVE Validation or Ambiguous Intent
# ---------------------------------------------------------------------------
def test_no_correction_on_inconclusive_validation(workflow_env):
    """Verify INCONCLUSIVE validation halts immediately without attempting correction."""
    db, profile, exec_val, num_val, sem_val = workflow_env

    # Single-step planner with NO structured intent -> Layer 2 & 3 return INCONCLUSIVE
    class NoIntentPlanner(QueryPlanner):
        def plan(
            self, question: str, active_table: str = "", profile=None, **kwargs
        ) -> AnalyticalPlan:
            step = PlanStep(
                step_id=1,
                question=question,
                target_table=active_table or "superstore",
                depends_on=[],
                structured_intent=None,
            )
            return AnalyticalPlan(
                original_question=question,
                steps=[step],
                execution_order=[1],
                is_multi_step=False,
            )

    generator = SequenceCodeGenerator(
        [
            "SELECT SUM(sales) AS total_sales FROM superstore",
            "SELECT SUM(sales) AS total_sales FROM superstore",
        ]
    )

    graph = build_analytical_graph(
        db=db,
        generator=generator,
        planner=NoIntentPlanner(),
        validator=exec_val,
        numerical_validator=num_val,
        semantic_validator=sem_val,
    )

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_inconclusive_no_retry",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    # Inconclusive must NOT retry!
    assert final_state["status"] == "inconclusive"
    assert final_state.get("retry_count", 0) == 0
    assert generator.call_count == 1

    val_report: WorkflowValidationReport = final_state["validation_report"]
    history = val_report.step_history[1]
    assert len(history) == 1
    assert history[0].status == ValidationStatus.INCONCLUSIVE


# ---------------------------------------------------------------------------
# 6. Strict Retry Limit: Exactly 2 Retries (3 Total Attempts)
# ---------------------------------------------------------------------------
def test_strict_retry_limit_exhaustion(workflow_env):
    """Verify persistent failure halts after exactly 2 retries (3 total attempts)."""
    db, profile, exec_val, num_val, sem_val = workflow_env

    intent = QuerySpecification(
        raw_question="What is total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )

    # All attempts return invalid SQL
    generator = SequenceCodeGenerator(
        [
            "SELECT fail_1 FROM superstore",
            "SELECT fail_2 FROM superstore",
            "SELECT fail_3 FROM superstore",
            "SELECT fail_4 FROM superstore",  # Must never be called
        ]
    )

    graph = build_analytical_graph(
        db=db,
        generator=generator,
        planner=FixedIntentPlanner(intent),
        validator=exec_val,
        numerical_validator=num_val,
        semantic_validator=sem_val,
    )

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_retry_limit",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    # Strict bounds: 1 initial + 2 retries = 3 total attempts
    assert final_state["status"] == "validation_failed"
    assert final_state["retry_count"] == 2
    assert generator.call_count == 3

    val_report: WorkflowValidationReport = final_state["validation_report"]
    history = val_report.step_history[1]
    assert len(history) == 3
    assert all(att.status == ValidationStatus.FAILED for att in history)


# ---------------------------------------------------------------------------
# 7. Second Retry Success (Attempt 3 Succeeds)
# ---------------------------------------------------------------------------
def test_second_retry_succeeds(workflow_env):
    """Verify attempt 1 and attempt 2 fail, but attempt 3 succeeds and marks VERIFIED."""
    db, profile, exec_val, num_val, sem_val = workflow_env

    intent = QuerySpecification(
        raw_question="What is total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )

    # Attempt 1: Syntax error
    # Attempt 2: Numerical error
    # Attempt 3: Valid query
    generator = SequenceCodeGenerator(
        [
            "SELECT bad_syntax FROM superstore",
            "SELECT SUM(sales) * 5 AS total_sales FROM superstore",
            "SELECT SUM(sales) AS total_sales FROM superstore",
        ]
    )

    graph = build_analytical_graph(
        db=db,
        generator=generator,
        planner=FixedIntentPlanner(intent),
        validator=exec_val,
        numerical_validator=num_val,
        semantic_validator=sem_val,
    )

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_second_retry_success",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "validated"
    assert final_state["retry_count"] == 2
    assert generator.call_count == 3

    val_report: WorkflowValidationReport = final_state["validation_report"]
    history = val_report.step_history[1]
    assert len(history) == 3
    assert history[0].status == ValidationStatus.FAILED
    assert history[1].status == ValidationStatus.FAILED
    assert history[2].status == ValidationStatus.VERIFIED


# ---------------------------------------------------------------------------
# 8. Both Retries Fail
# ---------------------------------------------------------------------------
def test_both_retries_fail_records_all_outcomes(workflow_env):
    """Verify all 3 failed attempts are recorded with diagnostic outcomes."""
    db, profile, exec_val, num_val, sem_val = workflow_env

    intent = QuerySpecification(
        raw_question="What is total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )

    generator = SequenceCodeGenerator(
        [
            "SELECT err1 FROM superstore",
            "SELECT err2 FROM superstore",
            "SELECT err3 FROM superstore",
        ]
    )

    graph = build_analytical_graph(
        db=db,
        generator=generator,
        planner=FixedIntentPlanner(intent),
        validator=exec_val,
        numerical_validator=num_val,
        semantic_validator=sem_val,
    )

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_both_retries_fail",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "validation_failed"
    assert "budget exhausted" in final_state["error_feedback"].lower()

    val_report: WorkflowValidationReport = final_state["validation_report"]
    history = val_report.step_history[1]
    assert len(history) == 3
    for i, att in enumerate(history, 1):
        assert att.attempt_number == i
        assert att.status == ValidationStatus.FAILED
        assert att.diagnostic is not None


# ---------------------------------------------------------------------------
# 9. Full Validation Rerun on Every Attempt
# ---------------------------------------------------------------------------
def test_full_validation_rerun_on_every_attempt(workflow_env):
    """Verify Layer 1, 2, and 3 are re-executed independently on each attempt."""
    db, profile, exec_val, num_val, sem_val = workflow_env

    intent = QuerySpecification(
        raw_question="What is total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )

    generator = SequenceCodeGenerator(
        [
            "SELECT SUM(sales) * 3 AS total_sales FROM superstore",
            "SELECT SUM(sales) AS total_sales FROM superstore",
        ]
    )

    graph = build_analytical_graph(
        db=db,
        generator=generator,
        planner=FixedIntentPlanner(intent),
        validator=exec_val,
        numerical_validator=num_val,
        semantic_validator=sem_val,
    )

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_full_rerun",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    val_report: WorkflowValidationReport = final_state["validation_report"]
    history = val_report.step_history[1]
    assert len(history) == 2

    # Attempt 1: L1 passed, L2 failed, L3 passed
    assert history[0].execution_result.status == ValidationStatus.VERIFIED
    assert history[0].numerical_result.status == ValidationStatus.FAILED
    assert history[0].semantic_result.status == ValidationStatus.VERIFIED

    # Attempt 2: L1 passed, L2 passed, L3 passed
    assert history[1].execution_result.status == ValidationStatus.VERIFIED
    assert history[1].numerical_result.status == ValidationStatus.VERIFIED
    assert history[1].semantic_result.status == ValidationStatus.VERIFIED


# ---------------------------------------------------------------------------
# 10. Preservation of Original Intent
# ---------------------------------------------------------------------------
def test_preservation_of_original_intent(workflow_env):
    """Verify structured intent is passed completely unchanged to the corrector."""
    db, profile, exec_val, num_val, sem_val = workflow_env

    intent = QuerySpecification(
        raw_question="What is total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )

    generator = SequenceCodeGenerator(
        [
            "SELECT bad_col FROM superstore",
            "SELECT SUM(sales) AS total_sales FROM superstore",
        ]
    )

    graph = build_analytical_graph(
        db=db,
        generator=generator,
        planner=FixedIntentPlanner(intent),
        validator=exec_val,
        numerical_validator=num_val,
        semantic_validator=sem_val,
    )

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_intent_preservation",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    graph.invoke(initial_state)

    # Verify both calls received the exact same intent object without mutation
    assert generator.calls[0]["intent"] == intent
    assert generator.calls[1]["intent"] == intent


# ---------------------------------------------------------------------------
# 11. Immutable Attempt History
# ---------------------------------------------------------------------------
def test_immutable_attempt_history(workflow_env):
    """Verify attempt history retains distinct queries, results, and diagnostics."""
    db, profile, exec_val, num_val, sem_val = workflow_env

    intent = QuerySpecification(
        raw_question="What is total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )

    q1 = "SELECT bad_col FROM superstore"
    q2 = "SELECT SUM(sales) AS total_sales FROM superstore"
    generator = SequenceCodeGenerator([q1, q2])

    graph = build_analytical_graph(
        db=db,
        generator=generator,
        planner=FixedIntentPlanner(intent),
        validator=exec_val,
        numerical_validator=num_val,
        semantic_validator=sem_val,
    )

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_history_immutability",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    val_report: WorkflowValidationReport = final_state["validation_report"]
    history = val_report.step_history[1]

    assert history[0].executed_sql == q1
    assert history[1].executed_sql == q2
    assert history[0].raw_execution_result.success is False
    assert history[1].raw_execution_result.success is True


# ---------------------------------------------------------------------------
# 12. Safety Check Enforcement on Corrector Output
# ---------------------------------------------------------------------------
def test_safety_check_blocks_malicious_corrector_output(workflow_env):
    """Verify the Corrector cannot bypass SQL security checks (e.g. DROP TABLE)."""
    db, profile, exec_val, num_val, sem_val = workflow_env

    intent = QuerySpecification(
        raw_question="What is total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )

    # Attempt 1: Syntax error
    # Attempt 2: Malicious DROP TABLE
    generator = SequenceCodeGenerator(
        [
            "SELECT bad_syntax FROM superstore",
            "DROP TABLE superstore",
            "SELECT SUM(sales) FROM superstore",
        ]
    )

    graph = build_analytical_graph(
        db=db,
        generator=generator,
        planner=FixedIntentPlanner(intent),
        validator=exec_val,
        numerical_validator=num_val,
        semantic_validator=sem_val,
    )

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_security_check",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    # The table must NOT be dropped!
    assert db.is_table_allowed("superstore") is True
    assert "superstore" in db.get_allowed_tables()

    # Attempt 2 must be flagged as an execution/security failure
    val_report: WorkflowValidationReport = final_state["validation_report"]
    history = val_report.step_history[1]
    assert history[1].execution_result.status == ValidationStatus.FAILED


# ---------------------------------------------------------------------------
# 13. No Correction on Initial Success
# ---------------------------------------------------------------------------
def test_no_correction_on_initial_success(workflow_env):
    """Verify that a successful initial attempt routes straight to assembler/validator."""
    db, profile, exec_val, num_val, sem_val = workflow_env

    intent = QuerySpecification(
        raw_question="What is total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )

    generator = SequenceCodeGenerator(
        [
            "SELECT SUM(sales) AS total_sales FROM superstore",
            "SELECT bad FROM superstore",
        ]
    )

    graph = build_analytical_graph(
        db=db,
        generator=generator,
        planner=FixedIntentPlanner(intent),
        validator=exec_val,
        numerical_validator=num_val,
        semantic_validator=sem_val,
    )

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_initial_success",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "validated"
    assert final_state.get("retry_count", 0) == 0
    assert generator.call_count == 1

    val_report: WorkflowValidationReport = final_state["validation_report"]
    history = val_report.step_history[1]
    assert len(history) == 1
    assert history[0].status == ValidationStatus.VERIFIED


# ---------------------------------------------------------------------------
# 14. Multi-Step Workflow Correction and Step Advance
# ---------------------------------------------------------------------------
def test_multi_step_workflow_correction_resets_retry_budget(workflow_env):
    """Verify Step 1 self-corrects on attempt 2, advances to Step 2 with reset budget."""
    db, profile, exec_val, num_val, sem_val = workflow_env

    intent1 = QuerySpecification(
        raw_question="What is total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    intent2 = QuerySpecification(
        raw_question="What is total profit?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_profit", column="profit", aggregation=AggregationType.SUM)],
    )

    class MultiStepPlanner(QueryPlanner):
        def plan(
            self, question: str, active_table: str = "", profile=None, **kwargs
        ) -> AnalyticalPlan:
            step1 = PlanStep(
                step_id=1,
                question="Sales step",
                target_table="superstore",
                depends_on=[],
                structured_intent=intent1,
            )
            step2 = PlanStep(
                step_id=2,
                question="Profit step",
                target_table="superstore",
                depends_on=[1],
                structured_intent=intent2,
            )
            return AnalyticalPlan(
                original_question=question,
                steps=[step1, step2],
                execution_order=[1, 2],
                is_multi_step=True,
            )

    # Step 1 Attempt 1 fails, Step 1 Attempt 2 succeeds.
    # Step 2 Attempt 1 succeeds.
    generator = SequenceCodeGenerator(
        [
            "SELECT bad_col FROM superstore",
            "SELECT SUM(sales) AS total_sales FROM superstore",
            "SELECT SUM(profit) AS total_profit FROM superstore",
        ]
    )

    graph = build_analytical_graph(
        db=db,
        generator=generator,
        planner=MultiStepPlanner(),
        validator=exec_val,
        numerical_validator=num_val,
        semantic_validator=sem_val,
    )

    initial_state: AgentState = {
        "question": "Sales and Profit",
        "session_id": "test_multi_step_correction",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "validated"
    assert generator.call_count == 3

    val_report: WorkflowValidationReport = final_state["validation_report"]
    assert len(val_report.step_history[1]) == 2
    assert len(val_report.step_history[2]) == 1
    assert val_report.step_history[1][1].status == ValidationStatus.VERIFIED
    assert val_report.step_history[2][0].status == ValidationStatus.VERIFIED
    assert val_report.all_steps_verified is True
