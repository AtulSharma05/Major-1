"""Unit tests for LangGraph state machine, 7-node workflow, and execution routing."""

import pytest

from src.agent.codegen.generator import CodeGenerator
from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import profile_table
from src.agent.execution.engine import ExecutionResult
from src.agent.planning.assembler import AssembledResult
from src.agent.planning.intent import AggregationType, MetricSpec, QuerySpecification
from src.agent.planning.planner import AnalyticalPlan, PlanStep, QueryPlanner
from src.agent.validation.contracts import (
    ExecutionValidationResult,
    NumericalValidationResult,
    SemanticValidationResult,
    ValidationStatus,
    WorkflowValidationReport,
    build_step_validation_result,
)
from src.agent.workflow.graph import build_analytical_graph
from src.agent.workflow.state import AgentState


class MockCodeGenerator(CodeGenerator):
    """Deterministic generator for unit testing workflow routing."""

    def __init__(self, sql_map: dict[str, str] | None = None):
        self.sql_map = sql_map or {}
        self.call_count = 0

    def generate_sql(
        self,
        question: str,
        profile=None,
        intent=None,
        error_feedback=None,
        prior_step_context=None,
        **kwargs,
    ) -> str:
        self.call_count += 1
        if question in self.sql_map:
            return self.sql_map[question]
        return "SELECT SUM(sales) AS total_sales FROM superstore"


@pytest.fixture
def workflow_setup(data_dir):
    """Fixture providing initialized database and profile."""
    db = SessionDatabase()
    db.register_csv(data_dir / "sample_superstore.csv", "superstore")
    profile = profile_table(db, "superstore")
    yield db, profile
    db.close()


def test_workflow_compilation(workflow_setup):
    """Verify that the 7-node LangGraph workflow compiles without schema errors."""
    db, _ = workflow_setup
    graph = build_analytical_graph(db=db)
    assert graph is not None


def test_workflow_single_step_execution(workflow_setup):
    """Verify single-step flow: Planner -> CodeGen -> Executor -> Assembler -> Validator."""
    db, profile = workflow_setup
    mock_gen = MockCodeGenerator(
        {
            "What is total sales?": "SELECT SUM(sales) AS total_sales FROM superstore",
        }
    )
    graph = build_analytical_graph(db=db, generator=mock_gen)

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_session_1",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    # In Stage 3.1 semantics, structured_intent is extracted, verifying Layer 1 and Layer 2
    assert final_state["status"] == "execution_verified"
    assert final_state["plan"] is not None
    assert final_state["plan"].is_multi_step is False
    assert final_state["assembled_result"] is not None
    assert final_state["assembled_result"].all_steps_succeeded is True
    assert final_state["assembled_result"].primary_scalar is not None
    assert final_state["assembled_result"].primary_scalar > 0
    assert final_state["validation_report"] is not None
    latest = final_state["validation_report"].latest_step_results[1]
    assert latest.execution_result.status == ValidationStatus.VERIFIED
    assert latest.numerical_result.status == ValidationStatus.VERIFIED


def test_workflow_multi_step_loop_execution(workflow_setup):
    """Verify multi-step planning loop through multiple steps before reaching Assembler."""
    db, profile = workflow_setup

    step1 = PlanStep(step_id=1, question="Sales 2016", target_table="superstore", depends_on=[])
    step2 = PlanStep(step_id=2, question="Sales 2017", target_table="superstore", depends_on=[])
    mock_plan = AnalyticalPlan(
        original_question="Compare 2016 vs 2017 sales",
        steps=[step1, step2],
        execution_order=[1, 2],
        is_multi_step=True,
    )

    class MockPlanner(QueryPlanner):
        def plan(self, *args, **kwargs):
            return mock_plan

    mock_gen = MockCodeGenerator(
        {
            "Sales 2016": "SELECT SUM(sales) FROM superstore WHERE order_date < '2017-01-01'",
            "Sales 2017": "SELECT SUM(sales) FROM superstore WHERE order_date >= '2017-01-01'",
        }
    )

    graph = build_analytical_graph(db=db, generator=mock_gen, planner=MockPlanner())

    initial_state: AgentState = {
        "question": "Compare 2016 vs 2017 sales",
        "session_id": "test_multi_1",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    # In Option C / Option B semantics, missing structured_intent yields INCONCLUSIVE
    assert final_state["status"] == "inconclusive"
    assert len(final_state["step_results"]) == 2
    assert final_state["assembled_result"] is not None
    assert final_state["assembled_result"].is_multi_step is True
    assert final_state["assembled_result"].all_steps_succeeded is True
    assert len(final_state["assembled_result"].primary_dataframe) == 2
    assert final_state["validation_report"] is not None
    assert final_state["validation_report"].workflow_status == ValidationStatus.INCONCLUSIVE


def test_workflow_error_routing_structural_boundary(workflow_setup):
    """Verify routing to Error Analyzer and Corrector structural boundary in Phase 4."""
    db, profile = workflow_setup

    class FailingGenerator(CodeGenerator):
        def generate_sql(self, *args, **kwargs):
            return "SELECT invalid_column FROM superstore"

    graph = build_analytical_graph(db=db, generator=FailingGenerator())

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_fail_1",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    # In Stage 5, persistent failure exhausts retries and terminates as validation_failed
    assert final_state["status"] == "validation_failed"
    assert final_state["retry_count"] == 2
    assert final_state["execution_result"].success is False
    assert final_state["execution_result"].error_type == "ColumnError"
    assert final_state["error_feedback"] is not None


def test_workflow_execution_verified_status(workflow_setup):
    """Verify execution_verified status when Layer 1 passes with valid QuerySpecification."""
    db, profile = workflow_setup

    intent = QuerySpecification(
        raw_question="What is total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(
        step_id=1,
        question="What is total sales?",
        target_table="superstore",
        structured_intent=intent,
    )
    mock_plan = AnalyticalPlan(
        original_question="What is total sales?",
        steps=[step],
        execution_order=[1],
        is_multi_step=False,
    )

    class MockPlanner(QueryPlanner):
        def plan(self, *args, **kwargs):
            return mock_plan

    mock_gen = MockCodeGenerator(
        {
            "What is total sales?": "SELECT SUM(sales) AS total_sales FROM superstore",
        }
    )

    graph = build_analytical_graph(db=db, generator=mock_gen, planner=MockPlanner())

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_exec_verified_1",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    # In Stage 3, Layer 1 & 2 are verified, while Layer 3 remains UNVERIFIED
    assert final_state["status"] == "execution_verified"
    val_report: WorkflowValidationReport = final_state["validation_report"]
    assert val_report is not None
    assert val_report.all_steps_verified is False
    step_val = val_report.latest_step_results[1]
    assert step_val.execution_result.passed is True
    assert step_val.execution_result.status == ValidationStatus.VERIFIED
    assert step_val.execution_result.schema_matched is True
    assert step_val.numerical_result.status == ValidationStatus.VERIFIED
    assert step_val.semantic_result.status == ValidationStatus.UNVERIFIED


def test_workflow_validated_status(workflow_setup):
    """Verify validated status is assigned only when all 3 layers pass across all steps."""
    db, profile = workflow_setup

    raw_ok = ExecutionResult(success=True, query="SELECT SUM(sales) AS total_sales FROM superstore")
    step_val = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="Execution verified",
            schema_matched=True,
        ),
        numerical_res=NumericalValidationResult(
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="Numerical verified",
        ),
        semantic_res=SemanticValidationResult(
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="Semantic verified",
        ),
        raw_execution_result=raw_ok,
    )
    val_report = WorkflowValidationReport()
    val_report.add_step_attempt(step_val)
    assert val_report.all_steps_verified is True

    class FullyVerifiedValidator:
        def validate_step(self, **kwargs):
            return step_val

        def evaluate_routing(self, **kwargs):
            return "assembler", None

    mock_gen = MockCodeGenerator(
        {
            "What is total sales?": "SELECT SUM(sales) AS total_sales FROM superstore",
        }
    )

    graph = build_analytical_graph(
        db=db,
        generator=mock_gen,
        validator=FullyVerifiedValidator(),
    )

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_validated_1",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "validated"
    assert final_state["validation_report"].all_steps_verified is True


def test_workflow_validation_failed_status(workflow_setup):
    """Verify validation_failed status when any layer fails or assembly fails."""
    db, profile = workflow_setup

    step_val_failed = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=False,
            status=ValidationStatus.FAILED,
            message="Column schema mismatch",
        ),
        numerical_res=NumericalValidationResult(
            passed=False,
            status=ValidationStatus.UNVERIFIED,
            message="Unverified",
        ),
        semantic_res=SemanticValidationResult(
            passed=False,
            status=ValidationStatus.UNVERIFIED,
            message="Unverified",
        ),
    )

    class FailingValidator:
        def validate_step(self, **kwargs):
            return step_val_failed

        def evaluate_routing(self, **kwargs):
            # Allow step to reach assembler/validator to test validator_node failure status
            return "assembler", "Validation failed"

    mock_gen = MockCodeGenerator(
        {
            "What is total sales?": "SELECT SUM(sales) AS total_sales FROM superstore",
        }
    )

    graph = build_analytical_graph(
        db=db,
        generator=mock_gen,
        validator=FailingValidator(),
    )

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_failed_1",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "validation_failed"


def test_workflow_status_priority_coexistence():
    """Verify Option B priority order:
    validation_failed -> inconclusive -> execution_verified -> validated.
    """
    from src.agent.workflow.graph import build_analytical_graph

    db = SessionDatabase()
    graph = build_analytical_graph(db=db)
    validator_node = graph.nodes["validator"].bound.func

    # Priority 1: validation_failed beats inconclusive
    report_failed_and_inconclusive = WorkflowValidationReport()
    step_fail = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=False,
            status=ValidationStatus.FAILED,
            message="Execution failed",
        ),
        numerical_res=NumericalValidationResult(
            passed=False,
            status=ValidationStatus.UNVERIFIED,
            message="Unverified",
        ),
        semantic_res=SemanticValidationResult(
            passed=False,
            status=ValidationStatus.UNVERIFIED,
            message="Unverified",
        ),
    )
    report_failed_and_inconclusive.add_step_attempt(step_fail)

    step_2_inconclusive = build_step_validation_result(
        step_id=2,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="Execution verified",
        ),
        numerical_res=NumericalValidationResult(
            passed=False,
            status=ValidationStatus.INCONCLUSIVE,
            message="Inconclusive",
        ),
        semantic_res=SemanticValidationResult(
            passed=False,
            status=ValidationStatus.INCONCLUSIVE,
            message="Inconclusive",
        ),
        raw_execution_result=ExecutionResult(success=True, query="SELECT 1"),
        has_meaningful_output=True,
    )
    report_failed_and_inconclusive.add_step_attempt(step_2_inconclusive)

    state_1: AgentState = {
        "assembled_result": AssembledResult(
            original_question="test",
            is_multi_step=False,
            all_steps_succeeded=True,
        ),
        "validation_report": report_failed_and_inconclusive,
    }
    out_1 = validator_node(state_1)
    assert out_1["status"] == "validation_failed"

    # Priority 2: inconclusive beats execution_verified
    report_inconclusive_and_verified = WorkflowValidationReport()
    step_1_inconclusive = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="Execution verified",
        ),
        numerical_res=NumericalValidationResult(
            passed=False,
            status=ValidationStatus.INCONCLUSIVE,
            message="Inconclusive",
        ),
        semantic_res=SemanticValidationResult(
            passed=False,
            status=ValidationStatus.INCONCLUSIVE,
            message="Inconclusive",
        ),
        raw_execution_result=ExecutionResult(success=True, query="SELECT 1"),
        has_meaningful_output=True,
    )
    report_inconclusive_and_verified.add_step_attempt(step_1_inconclusive)

    step_exec_verified = build_step_validation_result(
        step_id=2,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="Execution verified",
            schema_matched=True,
        ),
        numerical_res=NumericalValidationResult(
            passed=False,
            status=ValidationStatus.UNVERIFIED,
            message="Unverified",
        ),
        semantic_res=SemanticValidationResult(
            passed=False,
            status=ValidationStatus.UNVERIFIED,
            message="Unverified",
        ),
        raw_execution_result=ExecutionResult(success=True, query="SELECT 1"),
        has_meaningful_output=True,
    )
    report_inconclusive_and_verified.add_step_attempt(step_exec_verified)

    state_2: AgentState = {
        "assembled_result": AssembledResult(
            original_question="test",
            is_multi_step=False,
            all_steps_succeeded=True,
        ),
        "validation_report": report_inconclusive_and_verified,
    }
    out_2 = validator_node(state_2)
    assert out_2["status"] == "inconclusive"

    # Priority 1: assembly failure yields validation_failed even if report is clean
    state_3: AgentState = {
        "assembled_result": AssembledResult(
            original_question="test",
            is_multi_step=False,
            all_steps_succeeded=False,
        ),
        "validation_report": WorkflowValidationReport(),
    }
    out_3 = validator_node(state_3)
    assert out_3["status"] == "validation_failed"

    db.close()
