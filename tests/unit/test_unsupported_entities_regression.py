"""Regression tests: Unsupported Entity Requests Must Not Become Row Counts.

Critical Bug Verification:
    - User asked: 'how many students names are there?'
    - Previous faulty behavior:
        * Agent converted unsupported student-entity to COUNT(*) row count.
        * Returned 18.00 with VALIDATED status and HIGH verification confidence.
    - Required behavior:
        1. Identify requested entities and attributes in the natural-language question.
        2. Verify requested entities are supported by active dataset schema.
        3. Do NOT silently replace an unsupported entity with a generic table row count.
        4. Do NOT interpret 'how many X' as COUNT(*) unless X is correctly grounded
           or user explicitly asks for row/record count.
        5. If requested entity cannot be grounded, return INCONCLUSIVE with clear explanation.
        6. Do NOT generate a misleading numerical answer.
        7. Do NOT assign HIGH confidence to an unsupported entity mapping.
        8. Preserve legitimate row-count queries ('how many rows...', 'how many records...').
        9. Preserve legitimate entity-count queries ('unique customers', 'orders placed').
        10. Verify that unsupported entity requests cannot reach a VERIFIED final status
            merely because a generated COUNT(*) query returns a number.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.agent.codegen.generator import CodeGenerator
from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import TableProfile, profile_table
from src.agent.execution.engine import ControlledExecutionEngine, ExecutionResult
from src.agent.planning.assembler import ResultAssembler
from src.agent.planning.intent import AggregationType, IntentReliability
from src.agent.planning.planner import QueryPlanner
from src.agent.validation.contracts import (
    ConfidenceLevel,
    ValidationStatus,
    WorkflowValidationReport,
    build_step_validation_result,
)
from src.agent.validation.execution import ExecutionValidator
from src.agent.validation.numerical import NumericalValidator
from src.agent.validation.semantic import SemanticValidator


@pytest.fixture
def superstore_db() -> tuple[SessionDatabase, TableProfile]:
    """Create a SessionDatabase with a representative Superstore table."""
    db = SessionDatabase()
    df = pd.DataFrame(
        {
            "order_id": [f"ORD-{i}" for i in range(18)],
            "order_date": ["2017-01-15"] * 9 + ["2018-02-20"] * 9,
            "customer_id": [f"CUST-{i % 5}" for i in range(18)],
            "customer_name": [f"Customer {i % 5}" for i in range(18)],
            "sales": [100.0 + i for i in range(18)],
            "profit": [10.0 + i for i in range(18)],
            "region": ["East", "West"] * 9,
            "category": ["Technology", "Furniture"] * 9,
            "sub_category": ["Phones", "Chairs"] * 9,
        }
    )
    tbl = db.register_dataframe(df, "superstore")
    prof = profile_table(db, tbl)
    return db, prof


def test_unsupported_students_cannot_become_row_count(
    superstore_db: tuple[SessionDatabase, TableProfile],
):
    """'How many students are there?' on Superstore must NOT become COUNT(*) or return 18."""
    db, profile = superstore_db
    planner = QueryPlanner()
    plan = planner.plan(
        question="How many students are there?",
        active_table="superstore",
        profile=profile,
    )

    assert plan is not None
    assert len(plan.steps) == 1
    step = plan.steps[0]
    intent = step.structured_intent

    # Intent must reflect ungrounded entity
    assert intent is not None
    assert intent.is_grounded is False
    assert intent.intent_reliability == IntentReliability.UNGROUNDED
    assert "students" in intent.unsupported_entities

    # Code generation must not synthesize COUNT(*)
    generator = CodeGenerator(allow_fallback=True)
    sql = generator.generate_sql(
        question=step.question,
        profile=profile,
        intent=intent,
    )
    # The generated SQL must not be SELECT COUNT(*) FROM superstore
    assert "COUNT(*)" not in sql

    # Execution Engine runs the generated query
    engine = ControlledExecutionEngine(db)
    exec_res = engine.execute(sql)

    # Layer 1 Execution Validator
    l1 = ExecutionValidator()
    step_val = l1.validate_step(
        raw_execution_result=exec_res,
        structured_intent=intent,
        step=step,
        step_id=1,
        executed_sql=sql,
    )
    assert step_val.execution_result.status == ValidationStatus.INCONCLUSIVE
    assert step_val.confidence == ConfidenceLevel.LOW

    # Layer 2 Numerical Validator
    l2 = NumericalValidator()
    source_df = db.get_table_dataframe("superstore")
    num_res = l2.validate_step(
        step=step,
        execution_result=exec_res,
        source_df=source_df,
        table_profile=profile,
    )
    assert num_res.status == ValidationStatus.INCONCLUSIVE
    assert num_res.inconclusive_reason == "unsupported_schema_entity"

    # Layer 3 Semantic Validator
    l3 = SemanticValidator()
    sem_res = l3.validate_step(
        step=step,
        executed_sql=sql,
        execution_result=exec_res,
        table_profile=profile,
    )
    assert sem_res.status == ValidationStatus.INCONCLUSIVE

    # Result Assembler must not return 18.00 as primary scalar
    assembler = ResultAssembler()
    assembled = assembler.assemble(plan=plan, step_results={1: exec_res})
    assert assembled.primary_scalar is None
    assert assembled.all_steps_succeeded is False
    assert len(assembled.assembly_notes) > 0
    assert "absent from dataset schema" in assembled.assembly_notes[0]

    # Overall consolidated StepValidationResult
    consolidated = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=step_val.execution_result,
        numerical_res=num_res,
        semantic_res=sem_res,
        executed_sql=sql,
        raw_execution_result=exec_res,
        has_meaningful_output=False,
    )
    assert consolidated.status == ValidationStatus.INCONCLUSIVE
    assert consolidated.confidence == ConfidenceLevel.LOW
    assert consolidated.overall_passed is False

    report = WorkflowValidationReport()
    report.add_step_attempt(consolidated)
    assert report.workflow_status == ValidationStatus.INCONCLUSIVE
    assert report.workflow_confidence == ConfidenceLevel.LOW
    assert report.all_steps_verified is False


def test_unsupported_student_names_cannot_reach_verified_even_if_count_star_executed(
    superstore_db: tuple[SessionDatabase, TableProfile],
):
    """Verify that even if an execution engine returns a row count (18), validation rejects it."""
    db, profile = superstore_db
    planner = QueryPlanner()
    plan = planner.plan(
        question="how many students names are there?",
        active_table="superstore",
        profile=profile,
    )
    step = plan.steps[0]
    intent = step.structured_intent

    assert intent is not None
    assert not intent.is_grounded
    assert intent.intent_reliability == IntentReliability.UNGROUNDED
    assert any("student" in e for e in intent.unsupported_entities)

    # Simulate an external query returning 18 rows or scalar 18 (e.g. from a raw COUNT(*) query)
    simulated_count_star_sql = "SELECT COUNT(*) AS total_count FROM superstore"
    simulated_res = ExecutionResult(
        query=simulated_count_star_sql,
        success=True,
        row_count=1,
        column_count=1,
        is_scalar=True,
        scalar_value=18,
        dataframe=pd.DataFrame({"total_count": [18]}),
        raw_rows=[(18,)],
    )

    # Layer 1 must NOT mark it as VERIFIED
    l1 = ExecutionValidator()
    step_val = l1.validate_step(
        raw_execution_result=simulated_res,
        structured_intent=intent,
        step=step,
        step_id=1,
        executed_sql=simulated_count_star_sql,
    )
    assert step_val.execution_result.status == ValidationStatus.INCONCLUSIVE
    assert step_val.execution_result.passed is False

    # Layer 2 must NOT verify
    l2 = NumericalValidator()
    source_df = db.get_table_dataframe("superstore")
    num_res = l2.validate_step(
        step=step,
        execution_result=simulated_res,
        source_df=source_df,
        table_profile=profile,
    )
    assert num_res.status == ValidationStatus.INCONCLUSIVE
    assert num_res.passed is False

    # Layer 3 must NOT verify
    l3 = SemanticValidator()
    sem_res = l3.validate_step(
        step=step,
        executed_sql=simulated_count_star_sql,
        execution_result=simulated_res,
        table_profile=profile,
    )
    assert sem_res.status == ValidationStatus.INCONCLUSIVE
    assert sem_res.passed is False

    # Consolidated step
    consolidated = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=step_val.execution_result,
        numerical_res=num_res,
        semantic_res=sem_res,
        executed_sql=simulated_count_star_sql,
        raw_execution_result=simulated_res,
        has_meaningful_output=False,
    )
    assert consolidated.status == ValidationStatus.INCONCLUSIVE
    assert consolidated.confidence == ConfidenceLevel.LOW
    assert consolidated.overall_passed is False

    # Result Assembler must NOT pass through 18.00
    assembler = ResultAssembler()
    assembled = assembler.assemble(plan=plan, step_results={1: simulated_res})
    assert assembled.primary_scalar is None
    assert assembled.all_steps_succeeded is False


def test_legitimate_row_count_queries_preserved(
    superstore_db: tuple[SessionDatabase, TableProfile],
):
    """Verify legitimate row-count queries are correctly grounded and reach VERIFIED."""
    db, profile = superstore_db
    engine = ControlledExecutionEngine(db)
    l1 = ExecutionValidator()
    l2 = NumericalValidator()
    l3 = SemanticValidator()
    assembler = ResultAssembler()
    source_df = db.get_table_dataframe("superstore")

    for q in ["How many rows are in this dataset?", "How many records are there?"]:
        planner = QueryPlanner()
        plan = planner.plan(question=q, active_table="superstore", profile=profile)
        step = plan.steps[0]
        intent = step.structured_intent

        assert intent is not None
        assert intent.is_grounded is True
        assert intent.intent_reliability == IntentReliability.HIGH
        assert len(intent.unsupported_entities) == 0
        assert intent.metrics[0].aggregation == AggregationType.COUNT
        assert intent.metrics[0].column is None

        # Generate SQL
        generator = CodeGenerator(allow_fallback=True)
        sql = generator.generate_sql(question=q, profile=profile, intent=intent)
        exec_res = engine.execute(sql)
        assert exec_res.success is True
        assert exec_res.scalar_value == 18

        # L1, L2, L3
        s1 = l1.validate_step(exec_res, intent, step, 1, executed_sql=sql)
        assert s1.execution_result.passed is True

        s2 = l2.validate_step(step, exec_res, source_df, profile)
        assert s2.status == ValidationStatus.VERIFIED
        assert s2.passed is True

        s3 = l3.validate_step(step, sql, exec_res, profile)
        assert s3.status == ValidationStatus.VERIFIED
        assert s3.passed is True

        assembled = assembler.assemble(plan, {1: exec_res})
        assert assembled.primary_scalar == 18
        assert assembled.all_steps_succeeded is True


def test_legitimate_entity_count_queries_preserved(
    superstore_db: tuple[SessionDatabase, TableProfile],
):
    """Verify legitimate entity-count queries ('unique customers', 'orders') are grounded."""
    db, profile = superstore_db
    engine = ControlledExecutionEngine(db)
    l1 = ExecutionValidator()
    l2 = NumericalValidator()
    l3 = SemanticValidator()
    assembler = ResultAssembler()
    source_df = db.get_table_dataframe("superstore")

    # 1. Unique customers
    q1 = "How many unique customers are there?"
    plan1 = QueryPlanner().plan(q1, "superstore", profile)
    intent1 = plan1.steps[0].structured_intent
    assert intent1 is not None
    assert intent1.is_grounded is True
    assert intent1.metrics[0].aggregation == AggregationType.DISTINCT_COUNT
    assert intent1.metrics[0].column == "customer_id"

    generator = CodeGenerator(allow_fallback=True)
    sql1 = generator.generate_sql(q1, profile, intent1)
    res1 = engine.execute(sql1)
    assert res1.success is True
    # 5 unique customers
    assert res1.scalar_value == 5

    s1_l1 = l1.validate_step(res1, intent1, plan1.steps[0], 1, executed_sql=sql1)
    assert s1_l1.execution_result.passed is True
    s1_l2 = l2.validate_step(plan1.steps[0], res1, source_df, profile)
    assert s1_l2.status == ValidationStatus.VERIFIED
    s1_l3 = l3.validate_step(plan1.steps[0], sql1, res1, profile)
    assert s1_l3.status == ValidationStatus.VERIFIED
    assembled1 = assembler.assemble(plan1, {1: res1})
    assert assembled1.primary_scalar == 5
    assert assembled1.all_steps_succeeded is True

    # 2. Orders placed
    q2 = "How many orders were placed?"
    plan2 = QueryPlanner().plan(q2, "superstore", profile)
    intent2 = plan2.steps[0].structured_intent
    assert intent2 is not None
    assert intent2.is_grounded is True
    assert intent2.metrics[0].aggregation == AggregationType.DISTINCT_COUNT
    assert intent2.metrics[0].column == "order_id"

    sql2 = generator.generate_sql(q2, profile, intent2)
    res2 = engine.execute(sql2)
    assert res2.success is True
    assert res2.scalar_value == 18

    s2_l1 = l1.validate_step(res2, intent2, plan2.steps[0], 1, executed_sql=sql2)
    assert s2_l1.execution_result.passed is True
    s2_l2 = l2.validate_step(plan2.steps[0], res2, source_df, profile)
    assert s2_l2.status == ValidationStatus.VERIFIED
    s2_l3 = l3.validate_step(plan2.steps[0], sql2, res2, profile)
    assert s2_l3.status == ValidationStatus.VERIFIED
    assembled2 = assembler.assemble(plan2, {1: res2})
    assert assembled2.primary_scalar == 18
    assert assembled2.all_steps_succeeded is True


def test_how_many_customers_are_there_supported(
    superstore_db: tuple[SessionDatabase, TableProfile],
):
    """'How many customers are there?' on Superstore -> customer_id exists -> VERIFIED."""
    db, profile = superstore_db
    engine = ControlledExecutionEngine(db)
    l1 = ExecutionValidator()
    l2 = NumericalValidator()
    l3 = SemanticValidator()
    assembler = ResultAssembler()
    source_df = db.get_table_dataframe("superstore")

    q = "How many customers are there?"
    plan = QueryPlanner().plan(q, "superstore", profile)
    step = plan.steps[0]
    intent = step.structured_intent

    assert intent is not None
    assert intent.is_grounded is True
    assert intent.metrics[0].column == "customer_id"

    generator = CodeGenerator(allow_fallback=True)
    sql = generator.generate_sql(q, profile, intent)
    res = engine.execute(sql)
    assert res.success is True

    s_l1 = l1.validate_step(res, intent, step, 1, executed_sql=sql)
    assert s_l1.execution_result.passed is True
    s_l2 = l2.validate_step(step, res, source_df, profile)
    assert s_l2.status == ValidationStatus.VERIFIED
    s_l3 = l3.validate_step(step, sql, res, profile)
    assert s_l3.status == ValidationStatus.VERIFIED

    assembled = assembler.assemble(plan, {1: res})
    assert assembled.primary_scalar is not None
    assert assembled.all_steps_succeeded is True


def test_what_is_the_average_student_age_unsupported(
    superstore_db: tuple[SessionDatabase, TableProfile],
):
    """'What is the average student age?' on Superstore -> no student age column -> INCONCLUSIVE."""
    _, profile = superstore_db
    planner = QueryPlanner()
    plan = planner.plan("What is the average student age?", "superstore", profile)
    step = plan.steps[0]
    intent = step.structured_intent

    assert intent is not None
    assert intent.is_grounded is False
    assert intent.intent_reliability == IntentReliability.UNGROUNDED
    assert any("student" in e or "age" in e for e in intent.unsupported_entities)


def test_what_is_the_total_sales_supported(
    superstore_db: tuple[SessionDatabase, TableProfile],
):
    """'What is the total sales?' on Superstore -> sales column exists -> VERIFIED."""
    db, profile = superstore_db
    engine = ControlledExecutionEngine(db)
    l1 = ExecutionValidator()
    l2 = NumericalValidator()
    l3 = SemanticValidator()
    assembler = ResultAssembler()
    source_df = db.get_table_dataframe("superstore")

    q = "What is the total sales?"
    plan = QueryPlanner().plan(q, "superstore", profile)
    step = plan.steps[0]
    intent = step.structured_intent

    assert intent is not None
    assert intent.is_grounded is True
    assert intent.metrics[0].aggregation == AggregationType.SUM
    assert intent.metrics[0].column == "sales"

    generator = CodeGenerator(allow_fallback=True)
    sql = generator.generate_sql(q, profile, intent)
    res = engine.execute(sql)
    assert res.success is True

    s_l1 = l1.validate_step(res, intent, step, 1, executed_sql=sql)
    assert s_l1.execution_result.passed is True
    s_l2 = l2.validate_step(step, res, source_df, profile)
    assert s_l2.status == ValidationStatus.VERIFIED
    s_l3 = l3.validate_step(step, sql, res, profile)
    assert s_l3.status == ValidationStatus.VERIFIED

    assembled = assembler.assemble(plan, {1: res})
    assert assembled.primary_scalar is not None
    assert assembled.all_steps_succeeded is True


def test_how_many_people_are_there_ambiguous(
    superstore_db: tuple[SessionDatabase, TableProfile],
):
    """'How many people are there?' on Superstore -> ambiguous entity -> INCONCLUSIVE."""
    _, profile = superstore_db
    planner = QueryPlanner()
    plan = planner.plan("How many people are there?", "superstore", profile)
    step = plan.steps[0]
    intent = step.structured_intent

    assert intent is not None
    assert intent.is_grounded is False
    assert intent.intent_reliability == IntentReliability.UNGROUNDED
    assert "people" in intent.unsupported_entities


def test_llm_generated_incorrect_entity_to_column_mapping_rejected(
    superstore_db: tuple[SessionDatabase, TableProfile],
):
    """LLM hallucinating customer_id for 'students' must be rejected during intent validation."""
    _, profile = superstore_db
    from src.agent.planning.intent import MetricSpec, QuerySpecification
    from src.agent.planning.intent_parser import IntentParser

    # Simulate an LLM producing a QuerySpecification that mapped 'students' -> customer_id
    hallucinated_spec = QuerySpecification(
        raw_question="How many students are there?",
        table_name="superstore",
        metrics=[
            MetricSpec(
                name="student_count",
                aggregation=AggregationType.COUNT,
                column="customer_id",
            )
        ],
        dimensions=[],
        is_grounded=True,
    )

    parser = IntentParser()
    validated_spec = parser._validate_intent(hallucinated_spec, profile)

    # _validate_intent must detect the lack of grounding evidence for customer_id
    assert validated_spec is not None
    assert validated_spec.is_grounded is False
    assert validated_spec.intent_reliability == IntentReliability.UNGROUNDED
    assert (
        "students" in validated_spec.unsupported_entities
        or "customer_id" in validated_spec.unsupported_entities
    )


def test_valid_sql_answering_different_question_rejected_by_semantic_validator(
    superstore_db: tuple[SessionDatabase, TableProfile],
):
    """A valid SQL query answering a different question must be rejected by Layer 3."""
    _, profile = superstore_db
    from src.agent.planning.intent import MetricSpec, QuerySpecification
    from src.agent.planning.planner import PlanStep

    # User asked for students, but SQL selected customer_id
    step = PlanStep(
        step_id=1,
        question="How many students are there?",
        target_table="superstore",
        structured_intent=QuerySpecification(
            raw_question="How many students are there?",
            table_name="superstore",
            metrics=[
                MetricSpec(
                    name="student_count", aggregation=AggregationType.COUNT, column="customer_id"
                )
            ],
            dimensions=[],
            is_grounded=False,
            unsupported_entities=["students"],
        ),
    )

    sql = "SELECT COUNT(customer_id) AS student_count FROM superstore"
    res = ExecutionResult(
        query=sql,
        success=True,
        row_count=1,
        column_count=1,
        is_scalar=True,
        scalar_value=18,
        dataframe=pd.DataFrame({"student_count": [18]}),
        raw_rows=[(18,)],
    )

    l3 = SemanticValidator()
    sem_res = l3.validate_step(step, sql, res, profile)

    # Layer 3 must return INCONCLUSIVE and never VERIFIED
    assert sem_res.status == ValidationStatus.INCONCLUSIVE
    assert sem_res.passed is False


def test_self_correction_skipped_for_ungrounded_intent(
    superstore_db: tuple[SessionDatabase, TableProfile],
):
    """Self-correction must be skipped for ungrounded or inconclusive intents."""
    _, profile = superstore_db
    planner = QueryPlanner()
    plan = planner.plan("How many students are there?", "superstore", profile)
    step = plan.steps[0]

    l1 = ExecutionValidator()
    exec_res = ExecutionResult(
        query="SELECT NULL AS students FROM superstore WHERE 1 = 0",
        success=True,
        row_count=0,
        column_count=1,
        is_scalar=False,
        dataframe=pd.DataFrame({"students": []}),
        raw_rows=[],
    )

    step_val = l1.validate_step(exec_res, step.structured_intent, step, 1)
    route, reason = l1.evaluate_routing(plan, 1, step_val, 0)

    # Evaluates routing: must route to assembler without triggering error_analyzer/corrector
    assert route == "assembler"
