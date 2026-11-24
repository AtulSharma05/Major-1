"""Regression tests for QuerySpecification intent propagation across multi-layer validation.

Verifies:
    1. Scalar aggregation query: intent propagated through planning and graph,
       Layer 1, 2, and 3 report VERIFIED.
    2. Grouped aggregation query: intent propagated, all dimensions and metrics
       grounded and verified by Layers 1, 2, and 3.
    3. Top-N query with date filtering: intent retained with limit=5, ranking_spec,
       and time_constraints, independently verified by Layer 2 (Pandas) and Layer 3 (Semantic).
    4. Multi-step query: independent ranking query decomposed into distinct steps,
       each step retains its structured intent throughout graph execution.
    5. Genuine missing intent queries: ambiguous questions safely retain intent=None
       and report INCONCLUSIVE without false VERIFIED.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.agent.codegen.generator import CodeGenerator
from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import TableProfile, profile_table
from src.agent.execution.engine import ControlledExecutionEngine, ExecutionResult
from src.agent.planning.intent import (
    AggregationType,
    MetricSpec,
    OrderBySpec,
    QuerySpecification,
    RankingMode,
    RankingSpec,
    SortDirection,
)
from src.agent.planning.planner import PlanStep, QueryPlanner
from src.agent.validation.contracts import ValidationStatus
from src.agent.validation.execution import ExecutionValidator
from src.agent.validation.numerical import NumericalValidator
from src.agent.validation.semantic import SemanticValidator
from src.agent.workflow.graph import build_analytical_graph


class MockCodeGenerator(CodeGenerator):
    """Deterministic generator for unit testing workflow routing without Ollama."""

    def __init__(self, sql_map: dict[str, str] | None = None):
        self.sql_map = sql_map or {}

    def generate_sql(self, question: str, *args, **kwargs) -> str:
        for k, sql in self.sql_map.items():
            if k.lower() in question.lower():
                return sql
        return (
            "SELECT SUM(sales) AS total_sales FROM superstore "
            "WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'"
        )


@pytest.fixture
def superstore_db_and_profile() -> tuple[SessionDatabase, TableProfile]:
    """Create test session database and schema profile for superstore data."""
    db = SessionDatabase()
    df = pd.DataFrame(
        {
            "order_id": [f"CA-{i}" for i in range(1, 13)],
            "order_date": [
                "2016-03-15",
                "2016-07-20",
                "2017-01-10",
                "2017-02-15",
                "2017-05-20",
                "2017-06-25",
                "2017-08-10",
                "2017-09-15",
                "2017-10-20",
                "2017-11-25",
                "2018-01-05",
                "2018-04-12",
            ],
            "customer_id": [f"C{i}" for i in range(1, 13)],
            "region": [
                "East",
                "West",
                "East",
                "West",
                "Central",
                "South",
                "East",
                "West",
                "Central",
                "South",
                "East",
                "West",
            ],
            "category": [
                "Technology",
                "Furniture",
                "Technology",
                "Office Supplies",
                "Technology",
                "Furniture",
                "Office Supplies",
                "Technology",
                "Furniture",
                "Office Supplies",
                "Technology",
                "Furniture",
            ],
            "sub_category": [
                "Phones",
                "Chairs",
                "Accessories",
                "Storage",
                "Phones",
                "Chairs",
                "Paper",
                "Accessories",
                "Tables",
                "Paper",
                "Phones",
                "Chairs",
            ],
            "sales": [
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
                600.0,
                150.0,
                250.0,
                350.0,
                450.0,
                550.0,
                650.0,
            ],
            "profit": [20.0, -10.0, 80.0, 50.0, 120.0, 30.0, 15.0, 60.0, -25.0, 45.0, 110.0, 70.0],
            "quantity": [2, 3, 4, 1, 5, 2, 3, 2, 4, 1, 3, 2],
        }
    )
    tbl = db.register_dataframe(df, "superstore")
    prof = profile_table(db, tbl)
    return db, prof


# --- 1. Scalar Aggregation Query ---


def test_intent_propagation_scalar_aggregation(superstore_db_and_profile):
    """Test 1: Scalar aggregation retains QuerySpecification and verifies across all 3 layers."""
    db, prof = superstore_db_and_profile
    planner = QueryPlanner()
    q = "What were total sales in 2017?"
    plan = planner.plan(q, "superstore", profile=prof)

    assert plan.is_multi_step is False
    assert len(plan.steps) == 1
    step = plan.steps[0]

    # Verify intent presence and grounding
    assert step.structured_intent is not None
    intent = step.structured_intent
    assert intent.table_name == "superstore"
    assert len(intent.metrics) == 1
    assert intent.metrics[0].aggregation == AggregationType.SUM
    assert intent.metrics[0].column == "sales"
    assert intent.dimensions == []
    assert len(intent.time_constraints) == 1
    assert intent.time_constraints[0].start_time == "2017-01-01"
    assert intent.is_scalar() is True

    # Execute SQL
    engine = ControlledExecutionEngine(db)
    sql = (
        "SELECT SUM(sales) AS total_sales FROM superstore "
        "WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'"
    )
    res = engine.execute(sql)
    assert res.success is True

    # Layer 1
    exec_val = ExecutionValidator()
    l1_res = exec_val.validate_step(
        res, structured_intent=intent, step=step, step_id=1, attempt_number=1, executed_sql=sql
    )
    assert l1_res.execution_result.status == ValidationStatus.VERIFIED

    # Layer 2
    num_val = NumericalValidator()
    df = db.get_table_dataframe("superstore")
    l2_res = num_val.validate_step(step, res, df, table_profile=prof)
    assert l2_res.status == ValidationStatus.VERIFIED
    assert l2_res.passed is True

    # Layer 3
    sem_val = SemanticValidator()
    l3_res = sem_val.validate_step(
        step,
        executed_sql=sql,
        execution_result=res,
        table_profile=prof,
        available_tables=db.get_allowed_tables(),
    )
    assert l3_res.status == ValidationStatus.VERIFIED
    assert l3_res.passed is True


# --- 2. Grouped Aggregation Query ---


def test_intent_propagation_grouped_aggregation(superstore_db_and_profile):
    """Test 2: Grouped aggregation retains dimensions and verifies across all 3 layers."""
    db, prof = superstore_db_and_profile
    planner = QueryPlanner()
    q = "Average profit by region"
    plan = planner.plan(q, "superstore", profile=prof)

    step = plan.steps[0]
    assert step.structured_intent is not None
    intent = step.structured_intent
    assert intent.dimensions == ["region"]
    assert intent.metrics[0].aggregation == AggregationType.AVG
    assert intent.metrics[0].column == "profit"
    assert intent.is_grouped() is True

    engine = ControlledExecutionEngine(db)
    sql = "SELECT region, AVG(profit) AS avg_profit FROM superstore GROUP BY region"
    res = engine.execute(sql)
    assert res.success is True

    # Layer 1
    exec_val = ExecutionValidator()
    l1_res = exec_val.validate_step(
        res, structured_intent=intent, step=step, step_id=1, attempt_number=1, executed_sql=sql
    )
    assert l1_res.execution_result.status == ValidationStatus.VERIFIED

    # Layer 2
    num_val = NumericalValidator()
    df = db.get_table_dataframe("superstore")
    l2_res = num_val.validate_step(step, res, df, table_profile=prof)
    assert l2_res.status == ValidationStatus.VERIFIED
    assert l2_res.passed is True

    # Layer 3
    sem_val = SemanticValidator()
    l3_res = sem_val.validate_step(
        step,
        executed_sql=sql,
        execution_result=res,
        table_profile=prof,
        available_tables=db.get_allowed_tables(),
    )
    assert l3_res.status == ValidationStatus.VERIFIED
    assert l3_res.passed is True


# --- 3. Top-N Query with Date Filtering ---


def test_intent_propagation_top_n_with_date_filter(superstore_db_and_profile):
    """Test 3: Top-N query retains ranking_spec, limit=5, time_constraints."""
    db, prof = superstore_db_and_profile
    planner = QueryPlanner()
    q = "What were the top 5 sub-categories by total sales in 2017?"
    plan = planner.plan(q, "superstore", profile=prof)

    step = plan.steps[0]
    assert step.structured_intent is not None
    intent = step.structured_intent

    # Verify intent structure
    assert intent.dimensions == ["sub_category"]
    assert intent.metrics[0].aggregation == AggregationType.SUM
    assert intent.metrics[0].column == "sales"
    assert intent.limit == 5
    assert len(intent.time_constraints) == 1
    assert intent.time_constraints[0].start_time == "2017-01-01"
    assert intent.ranking_spec is not None
    assert intent.ranking_spec.mode == RankingMode.TOP_N_EXACT
    assert intent.ranking_spec.limit == 5
    assert intent.ranking_spec.direction == SortDirection.DESC
    assert any("DESC" in ob for ob in intent.order_by)

    # Execute deterministic SQL with secondary tiebreaker
    engine = ControlledExecutionEngine(db)
    sql = (
        "SELECT sub_category, SUM(sales) AS total_sales "
        "FROM superstore "
        "WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01' "
        "GROUP BY sub_category "
        "ORDER BY total_sales DESC, sub_category ASC "
        "LIMIT 5"
    )
    res = engine.execute(sql)
    assert res.success is True
    assert len(res.raw_rows) == 5

    # Layer 1
    exec_val = ExecutionValidator()
    l1_res = exec_val.validate_step(
        res, structured_intent=intent, step=step, step_id=1, attempt_number=1, executed_sql=sql
    )
    assert l1_res.execution_result.status == ValidationStatus.VERIFIED

    # Layer 2: Independent Pandas calculation verifies top 5 rows
    num_val = NumericalValidator()
    df = db.get_table_dataframe("superstore")
    l2_res = num_val.validate_step(step, res, df, table_profile=prof)
    assert l2_res.status == ValidationStatus.VERIFIED
    assert l2_res.passed is True
    assert l2_res.compared_rows == 5

    # Layer 3: All 8 canonical semantic checks verified
    sem_val = SemanticValidator()
    l3_res = sem_val.validate_step(
        step,
        executed_sql=sql,
        execution_result=res,
        table_profile=prof,
        available_tables=db.get_allowed_tables(),
    )
    assert l3_res.status == ValidationStatus.VERIFIED
    assert l3_res.passed is True
    assert len(l3_res.checks) == 8
    assert all(c.status == ValidationStatus.VERIFIED for c in l3_res.checks)


# --- 4. Multi-Step Query Decomposition & Intent Retention ---


def test_intent_propagation_multi_step_query(superstore_db_and_profile):
    """Test 4: Multi-step independent rankings retain structured intent on every step."""
    db, prof = superstore_db_and_profile
    planner = QueryPlanner()
    q = "Find product with highest sales and highest profit"
    plan = planner.plan(q, "superstore", profile=prof)

    assert plan.is_multi_step is True
    assert len(plan.steps) == 2
    assert plan.execution_order == [1, 2]

    # Step 1: highest sales
    s1 = plan.get_step(1)
    assert s1 is not None
    assert s1.structured_intent is not None
    assert s1.structured_intent.metrics[0].column == "sales"
    assert s1.structured_intent.limit == 1
    assert s1.structured_intent.ranking_spec is not None
    assert s1.structured_intent.ranking_spec.direction == SortDirection.DESC

    # Step 2: highest profit
    s2 = plan.get_step(2)
    assert s2 is not None
    assert s2.structured_intent is not None
    assert s2.structured_intent.metrics[0].column == "profit"
    assert s2.structured_intent.limit == 1
    assert s2.structured_intent.ranking_spec is not None
    assert s2.structured_intent.ranking_spec.direction == SortDirection.DESC


# --- 5. Genuine Missing Intent Safely Remains INCONCLUSIVE ---


def test_genuine_missing_intent_safely_inconclusive(superstore_db_and_profile):
    """Test 5: Unparseable/ambiguous queries safely return INCONCLUSIVE."""
    db, prof = superstore_db_and_profile
    planner = QueryPlanner()
    q = "Why did the store close?"
    plan = planner.plan(q, "superstore", profile=prof)

    step = plan.steps[0]
    # Intent MUST be None (no fabrication)
    assert step.structured_intent is None

    # Simulate execution of fallback SQL
    engine = ControlledExecutionEngine(db)
    sql = "SELECT 1"
    res = engine.execute(sql)

    # Layer 2 must safely return INCONCLUSIVE with missing_structured_intent
    num_val = NumericalValidator()
    df = db.get_table_dataframe("superstore")
    l2_res = num_val.validate_step(step, res, df, table_profile=prof)
    assert l2_res.status == ValidationStatus.INCONCLUSIVE
    assert l2_res.inconclusive_reason == "missing_structured_intent"
    assert l2_res.passed is False

    # Layer 3 must safely return INCONCLUSIVE with missing_structured_intent
    sem_val = SemanticValidator()
    l3_res = sem_val.validate_step(
        step,
        executed_sql=sql,
        execution_result=res,
        table_profile=prof,
        available_tables=db.get_allowed_tables(),
    )
    assert l3_res.status == ValidationStatus.INCONCLUSIVE
    assert l3_res.inconclusive_reason == "missing_structured_intent"
    assert l3_res.passed is False


# --- 6. End-to-End Graph Execution with SemanticValidator ---


def test_graph_execution_retains_intent_and_verifies_stage4(superstore_db_and_profile):
    """Test 6: Full LangGraph execution retains intent and verifies Layer 3."""
    db, prof = superstore_db_and_profile
    sem_val = SemanticValidator()
    mock_gen = MockCodeGenerator(
        {
            "What were total sales in 2017?": (
                "SELECT SUM(sales) AS total_sales FROM superstore "
                "WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'"
            ),
        }
    )
    graph = build_analytical_graph(db=db, generator=mock_gen, semantic_validator=sem_val)

    state_input = {
        "question": "What were total sales in 2017?",
        "active_table": "superstore",
        "schema_profile": prof,
    }

    final_state = graph.invoke(state_input)
    assert final_state["status"] == "validated"

    # Verify plan step structured_intent in graph state
    plan = final_state["plan"]
    assert plan is not None
    assert plan.steps[0].structured_intent is not None

    # Verify validation report
    val_report = final_state.get("validation_report")
    assert val_report is not None
    assert 1 in val_report.step_history
    attempts = val_report.step_history[1]
    assert len(attempts) >= 1
    last_attempt = attempts[-1]

    # Verify Layer 1, Layer 2, Layer 3
    assert last_attempt.execution_result.status == ValidationStatus.VERIFIED
    assert last_attempt.numerical_result.status == ValidationStatus.VERIFIED
    assert last_attempt.semantic_result.status == ValidationStatus.VERIFIED
    assert last_attempt.semantic_result.passed is True
    assert last_attempt.status == ValidationStatus.VERIFIED


# ==============================================================================
# Phase 5 Stage 4 Remediation: Ascending Top-N & Deterministic Ranking Tests
# ==============================================================================


def test_ascending_top_n_grouped_ranking_verifies_all_layers(superstore_db_and_profile):
    """Test 7: Ascending Top-5 grouped ranking (lowest sales) succeeds across L1, L2, L3."""
    db, prof = superstore_db_and_profile
    planner = QueryPlanner()
    question = "What were the 5 sub-categories with the lowest total sales in 2017?"
    plan = planner.plan(question, "superstore", profile=prof)
    step = plan.steps[0]
    intent = step.structured_intent
    assert intent is not None
    assert intent.ranking_spec is not None
    assert intent.ranking_spec.direction == SortDirection.ASC
    assert intent.limit == 5
    assert intent.dimensions == ["sub_category"]

    sql = (
        "SELECT sub_category, SUM(sales) AS total_sales FROM superstore "
        "WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01' "
        "GROUP BY sub_category "
        "ORDER BY total_sales ASC, sub_category ASC "
        "LIMIT 5"
    )
    engine = ControlledExecutionEngine(db)
    res = engine.execute(sql)
    assert res.success is True
    assert res.row_count == 5

    # Layer 1
    exec_val = ExecutionValidator()
    l1_res = exec_val.validate_step(res, structured_intent=intent, step=step, executed_sql=sql)
    assert l1_res.execution_result.status == ValidationStatus.VERIFIED

    # Layer 2
    num_val = NumericalValidator()
    source_df = db.get_table_dataframe("superstore")
    l2_res = num_val.validate_step(step, res, source_df, table_profile=prof)
    assert l2_res.status == ValidationStatus.VERIFIED, l2_res.message
    assert l2_res.passed is True

    # Layer 3
    sem_val = SemanticValidator()
    l3_res = sem_val.validate_step(step, executed_sql=sql, execution_result=res, table_profile=prof)
    assert l3_res.status == ValidationStatus.VERIFIED, l3_res.message
    assert l3_res.passed is True


def test_descending_top_n_grouped_ranking_verifies_all_layers(superstore_db_and_profile):
    """Test 8: Descending Top-5 grouped ranking (highest sales) succeeds across L1, L2, L3."""
    db, prof = superstore_db_and_profile
    planner = QueryPlanner()
    question = "What were the top 5 sub-categories by total sales in 2017?"
    plan = planner.plan(question, "superstore", profile=prof)
    step = plan.steps[0]
    intent = step.structured_intent
    assert intent is not None
    assert intent.ranking_spec is not None
    assert intent.ranking_spec.direction == SortDirection.DESC
    assert intent.limit == 5

    sql = (
        "SELECT sub_category, SUM(sales) AS total_sales FROM superstore "
        "WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01' "
        "GROUP BY sub_category "
        "ORDER BY total_sales DESC, sub_category ASC "
        "LIMIT 5"
    )
    engine = ControlledExecutionEngine(db)
    res = engine.execute(sql)
    assert res.success is True
    assert res.row_count == 5

    # Layer 1, 2, 3
    l1 = ExecutionValidator().validate_step(
        res, structured_intent=intent, step=step, executed_sql=sql
    )
    assert l1.execution_result.status == ValidationStatus.VERIFIED

    source_df = db.get_table_dataframe("superstore")
    l2 = NumericalValidator().validate_step(step, res, source_df, table_profile=prof)
    assert l2.status == ValidationStatus.VERIFIED, l2.message

    l3 = SemanticValidator().validate_step(
        step, executed_sql=sql, execution_result=res, table_profile=prof
    )
    assert l3.status == ValidationStatus.VERIFIED, l3.message


def test_tied_metric_values_at_cutoff_with_and_without_tiebreaker():
    """Test 9: Tied metric values at fifth-place cutoff:
    - With deterministic tie-breaker -> VERIFIED across L2 and L3.
    - Without tie-breaker -> INCONCLUSIVE (unproven_ranking_uniqueness) in L3.
    """
    db = SessionDatabase()
    # 7 groups: 3 tie at exactly sales=50.0 around cutoff
    df = pd.DataFrame(
        {
            "sub_category": ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf"],
            "sales": [10.0, 20.0, 30.0, 50.0, 50.0, 50.0, 100.0],
        }
    )
    db.register_dataframe(df, "tied_data")
    prof = profile_table(db, "tied_data")

    # Case A: With deterministic tie-breaker
    sql_with_tie = (
        "SELECT sub_category, SUM(sales) AS total_sales FROM tied_data "
        "GROUP BY sub_category "
        "ORDER BY total_sales ASC, sub_category ASC "
        "LIMIT 5"
    )
    engine = ControlledExecutionEngine(db)
    res_with_tie = engine.execute(sql_with_tie)

    intent_with_tie = QuerySpecification(
        raw_question="Bottom 5 sub_category by sales",
        table_name="tied_data",
        metrics=[MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales")],
        dimensions=["sub_category"],
        order_by=["total_sales ASC", "sub_category ASC"],
        limit=5,
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            metric="total_sales",
            direction=SortDirection.ASC,
            limit=5,
            secondary_keys=[OrderBySpec(column="sub_category", direction=SortDirection.ASC)],
        ),
    )
    step_with_tie = PlanStep(
        step_id=1, question="q", target_table="tied_data", structured_intent=intent_with_tie
    )

    l2_a = NumericalValidator().validate_step(step_with_tie, res_with_tie, df, table_profile=prof)
    assert l2_a.status == ValidationStatus.VERIFIED, l2_a.message

    l3_a = SemanticValidator().validate_step(
        step_with_tie, executed_sql=sql_with_tie, execution_result=res_with_tie, table_profile=prof
    )
    assert l3_a.status == ValidationStatus.VERIFIED, l3_a.message

    # Case B: Without tie-breaker
    sql_without_tie = (
        "SELECT sub_category, SUM(sales) AS total_sales FROM tied_data "
        "GROUP BY sub_category "
        "ORDER BY total_sales ASC "
        "LIMIT 5"
    )
    res_without_tie = engine.execute(sql_without_tie)
    intent_without_tie = QuerySpecification(
        raw_question="Bottom 5 sub_category by sales",
        table_name="tied_data",
        metrics=[MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales")],
        dimensions=["sub_category"],
        order_by=["total_sales ASC"],
        limit=5,
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            metric="total_sales",
            direction=SortDirection.ASC,
            limit=5,
        ),
    )
    step_without_tie = PlanStep(
        step_id=1, question="q", target_table="tied_data", structured_intent=intent_without_tie
    )

    l3_b = SemanticValidator().validate_step(
        step_without_tie,
        executed_sql=sql_without_tie,
        execution_result=res_without_tie,
        table_profile=prof,
    )
    assert l3_b.status == ValidationStatus.INCONCLUSIVE
    assert l3_b.inconclusive_reason == "unproven_ranking_uniqueness"


def test_numerical_mismatch_in_selected_group_fails_layer_2():
    """Test 10: Numerical mismatch in a selected group is caught by Layer 2 (FAILED)."""
    db = SessionDatabase()
    df = pd.DataFrame(
        {
            "sub_category": ["Art", "Binders", "Chairs", "Labels", "Paper"],
            "sales": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    db.register_dataframe(df, "mismatch_data")
    prof = profile_table(db, "mismatch_data")

    # Corrupt DuckDB result: Labels reports 999.0 instead of 40.0
    corrupted_df = pd.DataFrame(
        {
            "sub_category": ["Art", "Binders", "Chairs", "Labels", "Paper"],
            "total_sales": [10.0, 20.0, 30.0, 999.0, 50.0],
        }
    )
    sql_mismatch = (
        "SELECT sub_category, SUM(sales) AS total_sales FROM mismatch_data "
        "GROUP BY sub_category "
        "ORDER BY total_sales ASC, sub_category ASC "
        "LIMIT 5"
    )
    exec_res = ExecutionResult(
        query=sql_mismatch,
        success=True,
        row_count=5,
        dataframe=corrupted_df,
        raw_rows=corrupted_df.to_records(index=False).tolist(),
    )

    intent = QuerySpecification(
        raw_question="Bottom 5 by sales",
        table_name="mismatch_data",
        metrics=[MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales")],
        dimensions=["sub_category"],
        order_by=["total_sales ASC", "sub_category ASC"],
        limit=5,
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            metric="total_sales",
            direction=SortDirection.ASC,
            limit=5,
            secondary_keys=[OrderBySpec(column="sub_category", direction=SortDirection.ASC)],
        ),
    )
    step = PlanStep(step_id=1, question="q", target_table="mismatch_data", structured_intent=intent)

    num_val = NumericalValidator()
    l2_res = num_val.validate_step(step, exec_res, df, table_profile=prof)
    assert l2_res.status == ValidationStatus.FAILED
    assert l2_res.passed is False
    assert l2_res.details["failure_type"] == "metric_mismatch"
    assert l2_res.details["failing_group_key"] == ("Labels",)


def test_missing_secondary_ordering_flagged_inconclusive_by_layer_3():
    """Test 11: Missing secondary ordering is flagged as INCONCLUSIVE by Layer 3."""
    db = SessionDatabase()
    df = pd.DataFrame({"sub_category": ["Art", "Binders"], "sales": [10.0, 20.0]})
    db.register_dataframe(df, "test_table")
    prof = profile_table(db, "test_table")

    sql = (
        "SELECT sub_category, SUM(sales) AS total_sales FROM test_table "
        "GROUP BY sub_category ORDER BY total_sales DESC LIMIT 2"
    )
    exec_res = ControlledExecutionEngine(db).execute(sql)

    intent = QuerySpecification(
        raw_question="Top 2 by sales",
        table_name="test_table",
        metrics=[MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales")],
        dimensions=["sub_category"],
        order_by=["total_sales DESC"],
        limit=2,
    )
    step = PlanStep(step_id=1, question="q", target_table="test_table", structured_intent=intent)

    l3 = SemanticValidator().validate_step(
        step, executed_sql=sql, execution_result=exec_res, table_profile=prof
    )
    assert l3.status == ValidationStatus.INCONCLUSIVE
    assert l3.inconclusive_reason == "unproven_ranking_uniqueness"


def test_correct_deterministic_secondary_ordering_verified_by_layer_3():
    """Test 12: Correct deterministic secondary ordering is verified by Layer 3."""
    db = SessionDatabase()
    df = pd.DataFrame({"sub_category": ["Art", "Binders"], "sales": [10.0, 20.0]})
    db.register_dataframe(df, "test_table")
    prof = profile_table(db, "test_table")

    sql = (
        "SELECT sub_category, SUM(sales) AS total_sales FROM test_table "
        "GROUP BY sub_category ORDER BY total_sales DESC, sub_category ASC LIMIT 2"
    )
    exec_res = ControlledExecutionEngine(db).execute(sql)

    intent = QuerySpecification(
        raw_question="Top 2 by sales",
        table_name="test_table",
        metrics=[MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales")],
        dimensions=["sub_category"],
        order_by=["total_sales DESC", "sub_category ASC"],
        limit=2,
    )
    step = PlanStep(step_id=1, question="q", target_table="test_table", structured_intent=intent)

    l3 = SemanticValidator().validate_step(
        step, executed_sql=sql, execution_result=exec_res, table_profile=prof
    )
    assert l3.status == ValidationStatus.VERIFIED
    assert l3.passed is True


def test_multiple_grouping_dimensions_include_all_in_deterministic_ordering():
    """Test 13: Multiple grouping dimensions include all dimensions in secondary ordering."""
    db = SessionDatabase()
    df = pd.DataFrame(
        {
            "region": ["East", "West", "East", "West", "Central"],
            "sub_category": ["Art", "Art", "Binders", "Binders", "Chairs"],
            "sales": [100.0, 50.0, 50.0, 300.0, 20.0],
        }
    )
    db.register_dataframe(df, "multi_dim_table")
    prof = profile_table(db, "multi_dim_table")

    sql = (
        "SELECT region, sub_category, SUM(sales) AS total_sales FROM multi_dim_table "
        "GROUP BY region, sub_category "
        "ORDER BY total_sales ASC, region ASC, sub_category ASC "
        "LIMIT 3"
    )
    exec_res = ControlledExecutionEngine(db).execute(sql)

    intent = QuerySpecification(
        raw_question="Bottom 3 region and sub_category by sales",
        table_name="multi_dim_table",
        metrics=[MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales")],
        dimensions=["region", "sub_category"],
        order_by=["total_sales ASC", "region ASC", "sub_category ASC"],
        limit=3,
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            metric="total_sales",
            direction=SortDirection.ASC,
            limit=3,
            secondary_keys=[
                OrderBySpec(column="region", direction=SortDirection.ASC),
                OrderBySpec(column="sub_category", direction=SortDirection.ASC),
            ],
        ),
    )
    step = PlanStep(
        step_id=1, question="q", target_table="multi_dim_table", structured_intent=intent
    )

    l2 = NumericalValidator().validate_step(step, exec_res, df, table_profile=prof)
    assert l2.status == ValidationStatus.VERIFIED, l2.message

    l3 = SemanticValidator().validate_step(
        step, executed_sql=sql, execution_result=exec_res, table_profile=prof
    )
    assert l3.status == ValidationStatus.VERIFIED, l3.message


def test_limit_applied_strictly_after_aggregation_and_sorting():
    """Test 14: Verification that LIMIT is applied strictly after aggregation and sorting."""
    db = SessionDatabase()
    # 10 rows: group 'X' has 3 small rows [5, 5, 5] = 15 total; group 'Y' has 1 row [100] = 100
    # If LIMIT was applied before aggregation, group 'X' would have incomplete rows or wrong sums.
    df = pd.DataFrame(
        {
            "grp": ["X", "X", "X", "Y", "Z", "W", "V", "U", "T", "S"],
            "val": [5.0, 5.0, 5.0, 100.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0],
        }
    )
    db.register_dataframe(df, "post_agg_table")
    prof = profile_table(db, "post_agg_table")

    sql = (
        "SELECT grp, SUM(val) AS total_val FROM post_agg_table "
        "GROUP BY grp "
        "ORDER BY total_val DESC, grp ASC "
        "LIMIT 2"
    )
    exec_res = ControlledExecutionEngine(db).execute(sql)
    assert exec_res.row_count == 2
    # Group 'Y' has 100.0, Group 'X' has 15.0
    assert exec_res.dataframe.iloc[0]["grp"] == "Y"
    assert exec_res.dataframe.iloc[0]["total_val"] == 100.0
    assert exec_res.dataframe.iloc[1]["grp"] == "X"
    assert exec_res.dataframe.iloc[1]["total_val"] == 15.0

    intent = QuerySpecification(
        raw_question="Top 2 grp by val",
        table_name="post_agg_table",
        metrics=[MetricSpec(name="total_val", aggregation=AggregationType.SUM, column="val")],
        dimensions=["grp"],
        order_by=["total_val DESC", "grp ASC"],
        limit=2,
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            metric="total_val",
            direction=SortDirection.DESC,
            limit=2,
            secondary_keys=[OrderBySpec(column="grp", direction=SortDirection.ASC)],
        ),
    )
    step = PlanStep(
        step_id=1, question="q", target_table="post_agg_table", structured_intent=intent
    )

    l2 = NumericalValidator().validate_step(step, exec_res, df, table_profile=prof)
    assert l2.status == ValidationStatus.VERIFIED, l2.message


def test_null_and_nan_handling_in_ranking():
    """Test 15: Verification that NULL/NaN handling remains unchanged and sorts NULLs last."""
    db = SessionDatabase()
    df = pd.DataFrame(
        {
            "cat": ["A", "B", "C", "D"],
            "val": [10.0, None, float("nan"), 5.0],
        }
    )
    db.register_dataframe(df, "null_table")
    prof = profile_table(db, "null_table")

    # DuckDB sorts NULLs last in ASC: 5.0 (D), 10.0 (A), then NULLs
    sql_asc = (
        "SELECT cat, SUM(val) AS total_val FROM null_table "
        "GROUP BY cat ORDER BY total_val ASC, cat ASC LIMIT 2"
    )
    res_asc = ControlledExecutionEngine(db).execute(sql_asc)
    assert res_asc.dataframe.iloc[0]["cat"] == "D"
    assert res_asc.dataframe.iloc[0]["total_val"] == 5.0
    assert res_asc.dataframe.iloc[1]["cat"] == "A"
    assert res_asc.dataframe.iloc[1]["total_val"] == 10.0

    intent = QuerySpecification(
        raw_question="Bottom 2 cat by val",
        table_name="null_table",
        metrics=[MetricSpec(name="total_val", aggregation=AggregationType.SUM, column="val")],
        dimensions=["cat"],
        order_by=["total_val ASC", "cat ASC"],
        limit=2,
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            metric="total_val",
            direction=SortDirection.ASC,
            limit=2,
            secondary_keys=[OrderBySpec(column="cat", direction=SortDirection.ASC)],
        ),
    )
    step = PlanStep(step_id=1, question="q", target_table="null_table", structured_intent=intent)

    l2 = NumericalValidator().validate_step(step, res_asc, df, table_profile=prof)
    assert l2.status == ValidationStatus.VERIFIED, l2.message
