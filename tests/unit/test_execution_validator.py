"""Unit tests for Phase 5 Stage 2: Layer 1 Execution Validator and Workflow Routing.

Validates:
    1. Scalar results (valid values, NULL scalars, column alias matching, 1x1 tabular projections).
    2. Single-row multi-metric results (expected metrics, rejection of unexpected columns).
    3. Grouped tabular results (schema matching, missing columns).
    4. Structurally valid empty DataFrames (pass Layer 1, no automatic meaningful output).
    5. Inconclusive handling when QuerySpecification is missing (Option C).
    6. Generic tabular validation (structure verification or accurate inconclusive reporting).
    7. Inconsistency detection between ExecutionResult metadata and DataFrame shape.
    8. Error categorization and correctability for syntax, schema, missing, and timeout errors.
    9. Workflow routing safety gates (dependent blocking, independent advance, final assembly).
    10. Attempt history preservation and retained successful results.
"""

import pandas as pd
import pytest

from src.agent.data.ingestion import SessionDatabase
from src.agent.execution.engine import ControlledExecutionEngine, ExecutionResult
from src.agent.planning.intent import AggregationType, MetricSpec, QuerySpecification
from src.agent.planning.planner import AnalyticalPlan, PlanStep
from src.agent.validation.contracts import (
    ConfidenceLevel,
    ErrorCategory,
    ValidationStatus,
    WorkflowValidationReport,
)
from src.agent.validation.execution import ExecutionValidator


@pytest.fixture
def db_session(data_dir):
    """Fixture providing initialized database registered with sample data."""
    db = SessionDatabase()
    db.register_csv(data_dir / "sample_superstore.csv", "superstore")
    yield db
    db.close()


@pytest.fixture
def engine(db_session):
    """Fixture providing ControlledExecutionEngine."""
    return ControlledExecutionEngine(db_session, default_timeout_seconds=5.0)


@pytest.fixture
def validator():
    """Fixture providing ExecutionValidator."""
    return ExecutionValidator()


# ---------------------------------------------------------------------------
# 1. Scalar Result Validation (Valid value, NULL scalar, Alias, 1x1 Tabular)
# ---------------------------------------------------------------------------


def test_execution_validator_scalar_success(engine, validator):
    """Verify Layer 1 verification of standard scalar result with matching alias."""
    sql = "SELECT SUM(sales) AS total_sales FROM superstore"
    raw_res = engine.execute(sql)
    assert raw_res.success is True
    assert raw_res.is_scalar is True

    intent = QuerySpecification(
        raw_question="Total sales",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales")],
        dimensions=[],
    )

    step_val = validator.validate_step(
        raw_execution_result=raw_res,
        structured_intent=intent,
        step_id=1,
        attempt_number=1,
        executed_sql=sql,
    )

    assert step_val.execution_result.passed is True
    assert step_val.execution_result.status == ValidationStatus.VERIFIED
    assert step_val.execution_result.is_scalar is True
    assert step_val.execution_result.is_empty is False
    assert step_val.execution_result.schema_matched is True
    assert step_val.execution_result.row_count == 1
    assert step_val.execution_result.column_count == 1


def test_execution_validator_scalar_alias_mismatch(engine, validator):
    """Finding 2: 1x1 scalar query with mismatched column alias must fail schema validation."""
    # Query aliases output as 'wrong_alias', but QuerySpecification expects 'total_sales'
    sql = "SELECT SUM(sales) AS wrong_alias FROM superstore"
    raw_res = engine.execute(sql)
    assert raw_res.success is True
    assert raw_res.row_count == 1
    assert raw_res.column_count == 1

    intent = QuerySpecification(
        raw_question="Total sales",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales")],
        dimensions=[],
    )

    step_val = validator.validate_step(
        raw_execution_result=raw_res,
        structured_intent=intent,
        step_id=1,
        attempt_number=1,
        executed_sql=sql,
    )

    assert step_val.execution_result.passed is False
    assert step_val.execution_result.status == ValidationStatus.FAILED
    assert step_val.execution_result.schema_matched is False
    assert "Scalar schema mismatch" in (step_val.execution_result.message or "")
    assert step_val.diagnostic is not None
    assert step_val.diagnostic.error_category == ErrorCategory.SCHEMA_ERROR


def test_execution_validator_null_scalar_result(engine, validator):
    """Verify NULL scalar result with matching alias is recognized as valid scalar shape."""
    # Query matches 0 rows; SQL standard scalar aggregate produces 1 row with NULL
    sql = "SELECT SUM(sales) AS total_sales FROM superstore WHERE 1 = 0"
    raw_res = engine.execute(sql)
    assert raw_res.success is True
    assert raw_res.row_count == 1
    assert raw_res.column_count == 1
    assert raw_res.scalar_value is None

    intent = QuerySpecification(
        raw_question="Total sales when empty",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales")],
        dimensions=[],
    )

    step_val = validator.validate_step(
        raw_execution_result=raw_res,
        structured_intent=intent,
        step_id=1,
        attempt_number=1,
        executed_sql=sql,
    )

    assert step_val.execution_result.passed is True
    assert step_val.execution_result.status == ValidationStatus.VERIFIED
    assert step_val.execution_result.is_scalar is True
    assert step_val.execution_result.is_empty is True
    assert step_val.execution_result.schema_matched is True
    assert step_val.execution_result.details["is_null_scalar"] is True


def test_execution_validator_1x1_tabular_projection_vs_scalar(engine, validator):
    """Verify 1-row, 1-column entity projection is classified as tabular, NOT scalar."""
    sql = "SELECT category FROM superstore LIMIT 1"
    raw_res = engine.execute(sql)
    assert raw_res.success is True
    assert raw_res.row_count == 1
    assert raw_res.column_count == 1

    # Intent specifies dimension (entity projection), not a scalar metric
    intent = QuerySpecification(
        raw_question="Single category",
        table_name="superstore",
        metrics=[],
        dimensions=["category"],
        limit=1,
    )

    step_val = validator.validate_step(
        raw_execution_result=raw_res,
        structured_intent=intent,
        step_id=1,
        attempt_number=1,
        executed_sql=sql,
    )

    assert step_val.execution_result.passed is True
    assert step_val.execution_result.status == ValidationStatus.VERIFIED
    # Crucial: 1x1 projection with dimensional intent is tabular, not a primitive scalar
    assert step_val.execution_result.is_scalar is False


# ---------------------------------------------------------------------------
# 2. Single-Row Multi-Metric Grain Validation
# ---------------------------------------------------------------------------


def test_execution_validator_single_row_multi_metric_success(engine, validator):
    """Verify multi-aggregate query without dimensions is treated as single-row multi-metric."""
    sql = "SELECT SUM(sales) AS total_sales, SUM(profit) AS total_profit FROM superstore"
    raw_res = engine.execute(sql)
    assert raw_res.success is True
    assert raw_res.row_count == 1
    assert raw_res.column_count == 2
    assert raw_res.is_scalar is False

    intent = QuerySpecification(
        raw_question="Total sales and total profit",
        table_name="superstore",
        metrics=[
            MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales"),
            MetricSpec(name="total_profit", aggregation=AggregationType.SUM, column="profit"),
        ],
        dimensions=[],
    )

    step_val = validator.validate_step(
        raw_execution_result=raw_res,
        structured_intent=intent,
        step_id=1,
        attempt_number=1,
        executed_sql=sql,
    )

    assert step_val.execution_result.passed is True
    assert step_val.execution_result.status == ValidationStatus.VERIFIED
    assert step_val.execution_result.is_scalar is False
    assert step_val.execution_result.row_count == 1
    assert step_val.execution_result.column_count == 2
    assert step_val.execution_result.schema_matched is True


def test_execution_validator_single_row_multi_metric_missing_column(engine, validator):
    """Verify failure when query omits one of the expected multi-metric aliases."""
    sql = "SELECT SUM(sales) AS total_sales FROM superstore"
    raw_res = engine.execute(sql)

    intent = QuerySpecification(
        raw_question="Total sales and total profit",
        table_name="superstore",
        metrics=[
            MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales"),
            MetricSpec(name="total_profit", aggregation=AggregationType.SUM, column="profit"),
        ],
        dimensions=[],
    )

    step_val = validator.validate_step(
        raw_execution_result=raw_res,
        structured_intent=intent,
        step_id=1,
        attempt_number=1,
        executed_sql=sql,
    )

    assert step_val.execution_result.passed is False
    assert step_val.execution_result.status == ValidationStatus.FAILED
    assert "Missing expected metric column" in (step_val.execution_result.message or "")
    assert step_val.diagnostic is not None
    assert step_val.diagnostic.error_category == ErrorCategory.SCHEMA_ERROR


def test_execution_validator_single_row_multi_metric_unexpected_column(engine, validator):
    """Finding 4: Extra unexpected columns in multi-metric summary must fail schema validation."""
    # Query returns total_sales and total_profit plus an unexpected extra column
    sql = (
        "SELECT SUM(sales) AS total_sales, SUM(profit) AS total_profit, 1 AS extra FROM superstore"
    )
    raw_res = engine.execute(sql)
    assert raw_res.success is True
    assert raw_res.column_count == 3

    intent = QuerySpecification(
        raw_question="Total sales and total profit",
        table_name="superstore",
        metrics=[
            MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales"),
            MetricSpec(name="total_profit", aggregation=AggregationType.SUM, column="profit"),
        ],
        dimensions=[],
    )

    step_val = validator.validate_step(
        raw_execution_result=raw_res,
        structured_intent=intent,
        step_id=1,
        attempt_number=1,
        executed_sql=sql,
    )

    assert step_val.execution_result.passed is False
    assert step_val.execution_result.status == ValidationStatus.FAILED
    assert step_val.execution_result.schema_matched is False
    assert "Unexpected column(s)" in (step_val.execution_result.message or "")
    assert step_val.diagnostic is not None
    assert step_val.diagnostic.error_category == ErrorCategory.SCHEMA_ERROR


# ---------------------------------------------------------------------------
# 3. Grouped Tabular and Empty DataFrame Validation
# ---------------------------------------------------------------------------


def test_execution_validator_grouped_summary_success(engine, validator):
    """Verify Layer 1 verification of standard grouped summary."""
    sql = "SELECT region, SUM(sales) AS total_sales FROM superstore GROUP BY region"
    raw_res = engine.execute(sql)
    assert raw_res.success is True

    intent = QuerySpecification(
        raw_question="Sales by region",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales")],
        dimensions=["region"],
    )

    step_val = validator.validate_step(
        raw_execution_result=raw_res,
        structured_intent=intent,
        step_id=1,
        attempt_number=1,
        executed_sql=sql,
    )

    assert step_val.execution_result.passed is True
    assert step_val.execution_result.status == ValidationStatus.VERIFIED
    assert step_val.execution_result.is_scalar is False
    assert step_val.execution_result.schema_matched is True


def test_execution_validator_structurally_valid_empty_dataframe(engine, validator):
    """Finding 3: Valid empty DataFrame passes Layer 1, but meaningful output is False."""
    sql = (
        "SELECT region, SUM(sales) AS total_sales FROM superstore "
        "WHERE order_date > '2099-01-01' GROUP BY region"
    )
    raw_res = engine.execute(sql)
    assert raw_res.success is True
    assert raw_res.row_count == 0
    assert set(raw_res.dataframe.columns) == {"region", "total_sales"}

    intent = QuerySpecification(
        raw_question="Future sales by region",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales")],
        dimensions=["region"],
    )

    step_val = validator.validate_step(
        raw_execution_result=raw_res,
        structured_intent=intent,
        step_id=1,
        attempt_number=1,
        executed_sql=sql,
    )

    # Empty DataFrame is structurally valid at Layer 1
    assert step_val.execution_result.passed is True
    assert step_val.execution_result.status == ValidationStatus.VERIFIED
    assert step_val.execution_result.is_empty is True
    assert step_val.execution_result.schema_matched is True
    # Finding 3: Meaningful output is NOT automatically claimed for empty results
    assert step_val.confidence == ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# 4. Inconclusive Handling When QuerySpecification is Missing (Option C)
# ---------------------------------------------------------------------------


def test_execution_validator_missing_structured_intent_option_c(engine, validator):
    """Finding 1 (Option C): Missing QuerySpecification sets schema_matched=False."""
    sql = "SELECT SUM(sales) AS total_sales FROM superstore"
    raw_res = engine.execute(sql)

    step_val = validator.validate_step(
        raw_execution_result=raw_res,
        structured_intent=None,
        step_id=1,
        attempt_number=1,
        executed_sql=sql,
    )

    # Layer 1 represents physical execution success
    assert step_val.execution_result.passed is True
    assert step_val.execution_result.status == ValidationStatus.VERIFIED
    # Schema verification was not performed
    assert step_val.execution_result.schema_matched is False
    assert step_val.execution_result.details["schema_verification_performed"] is False
    assert "schema verification was not performed" in (step_val.execution_result.message or "")

    # Downstream layers mark INCONCLUSIVE
    assert step_val.numerical_result.status == ValidationStatus.INCONCLUSIVE
    assert step_val.semantic_result.status == ValidationStatus.INCONCLUSIVE

    # Overall step remains INCONCLUSIVE with MEDIUM confidence (never VERIFIED)
    assert step_val.status == ValidationStatus.INCONCLUSIVE
    assert step_val.confidence == ConfidenceLevel.MEDIUM
    assert step_val.overall_passed is False


# ---------------------------------------------------------------------------
# 5. Generic Tabular Validation
# ---------------------------------------------------------------------------


def test_execution_validator_tabular_with_dimensions(engine, validator):
    """Finding 5: Generic tabular query verifies expected dimensions when present."""
    sql = "SELECT region, category FROM superstore LIMIT 5"
    raw_res = engine.execute(sql)

    intent = QuerySpecification(
        raw_question="Region and category sample",
        table_name="superstore",
        dimensions=["region", "category"],
    )

    step_val = validator.validate_step(
        raw_execution_result=raw_res,
        structured_intent=intent,
        step_id=1,
        attempt_number=1,
        executed_sql=sql,
    )

    assert step_val.execution_result.passed is True
    assert step_val.execution_result.status == ValidationStatus.VERIFIED
    assert step_val.execution_result.schema_matched is True


def test_execution_validator_tabular_without_expected_structure(engine, validator):
    """Finding 5: Generic tabular query without defined structure reports schema_matched=False."""
    sql = "SELECT * FROM superstore LIMIT 5"
    raw_res = engine.execute(sql)

    # Empty QuerySpecification without defined dimensions or metrics
    intent = QuerySpecification(
        raw_question="Raw sample",
        table_name="superstore",
        dimensions=[],
        metrics=[],
    )

    step_val = validator.validate_step(
        raw_execution_result=raw_res,
        structured_intent=intent,
        step_id=1,
        attempt_number=1,
        executed_sql=sql,
    )

    assert step_val.execution_result.passed is True
    assert step_val.execution_result.status == ValidationStatus.VERIFIED
    assert step_val.execution_result.schema_matched is False
    assert step_val.status == ValidationStatus.INCONCLUSIVE


# ---------------------------------------------------------------------------
# 6. Additional Correctness Checks (Inconsistency, Missing, Timeout)
# ---------------------------------------------------------------------------


def test_execution_validator_shape_inconsistency(validator):
    """Finding 6a: Inconsistency between counts and DataFrame shape fails validation."""
    raw_res = ExecutionResult(
        success=True,
        query="SELECT 1",
        dataframe=pd.DataFrame({"x": [1, 2, 3]}),  # 3 rows, 1 col
        row_count=10,  # Reported 10 rows (mismatch!)
        column_count=1,
    )

    step_val = validator.validate_step(raw_execution_result=raw_res, step_id=1)

    assert step_val.execution_result.passed is False
    assert step_val.execution_result.status == ValidationStatus.FAILED
    assert "shape inconsistency" in (step_val.execution_result.message or "")
    assert step_val.diagnostic is not None
    assert step_val.diagnostic.error_category == ErrorCategory.FATAL_SYSTEM
    assert step_val.diagnostic.is_correctable is False


def test_execution_validator_column_count_inconsistency(validator):
    """Verify DataFrame column-count mismatch against ExecutionResult.column_count fails."""
    raw_res = ExecutionResult(
        success=True,
        query="SELECT 1",
        dataframe=pd.DataFrame({"x": [1]}),  # 1 row, 1 col
        row_count=1,
        column_count=5,  # Reported 5 columns (mismatch!)
    )

    step_val = validator.validate_step(raw_execution_result=raw_res, step_id=1)

    assert step_val.execution_result.passed is False
    assert step_val.execution_result.status == ValidationStatus.FAILED
    assert "shape inconsistency" in (step_val.execution_result.message or "")
    assert step_val.diagnostic is not None
    assert step_val.diagnostic.error_category == ErrorCategory.FATAL_SYSTEM
    assert step_val.diagnostic.is_correctable is False


def test_execution_validator_missing_execution_result(validator):
    """Finding 6b: Missing execution result produces FATAL_SYSTEM and is_correctable=False."""
    step_val = validator.validate_step(raw_execution_result=None, step_id=1)

    assert step_val.execution_result.passed is False
    assert step_val.execution_result.status == ValidationStatus.FAILED
    assert step_val.diagnostic is not None
    assert step_val.diagnostic.error_category == ErrorCategory.FATAL_SYSTEM
    assert step_val.diagnostic.is_correctable is False


def test_execution_validator_column_error_diagnostic(engine, validator):
    """Verify column error produces SCHEMA_ERROR diagnostic with is_correctable=True."""
    sql = "SELECT non_existent_column FROM superstore"
    raw_res = engine.execute(sql)
    assert raw_res.success is False
    assert raw_res.error_type == "ColumnError"

    step_val = validator.validate_step(
        raw_execution_result=raw_res,
        structured_intent=None,
        step_id=1,
        attempt_number=1,
        executed_sql=sql,
    )

    assert step_val.execution_result.passed is False
    assert step_val.execution_result.status == ValidationStatus.FAILED
    assert step_val.diagnostic is not None
    assert step_val.diagnostic.error_category == ErrorCategory.SCHEMA_ERROR
    assert step_val.diagnostic.is_correctable is True
    assert "non_existent_column" in step_val.diagnostic.error_summary


def test_execution_validator_timeout_diagnostic(validator):
    """Finding 6c: Timeout error produces FATAL_SYSTEM diagnostic with is_correctable=False."""
    raw_res = ExecutionResult(
        success=False,
        query="SELECT * FROM massive_table",
        error_type="TimeoutError",
        error_message="Query exceeded 5.0s timeout.",
        execution_time_ms=5001.0,
    )

    step_val = validator.validate_step(
        raw_execution_result=raw_res,
        step_id=1,
        attempt_number=1,
        executed_sql=raw_res.query,
    )

    assert step_val.execution_result.passed is False
    assert step_val.diagnostic is not None
    assert step_val.diagnostic.error_category == ErrorCategory.FATAL_SYSTEM
    assert step_val.diagnostic.is_correctable is False


# ---------------------------------------------------------------------------
# 7. Workflow Routing Safety Gates
# ---------------------------------------------------------------------------


def test_routing_inconclusive_blocks_dependent_step(validator):
    """Verify INCONCLUSIVE step with downstream dependent steps routes to error_analyzer."""
    step1 = PlanStep(
        step_id=1, question="Filter customers", target_table="superstore", depends_on=[]
    )
    step2 = PlanStep(step_id=2, question="Sum orders", target_table="superstore", depends_on=[1])
    plan = AnalyticalPlan(
        original_question="Filtered customer orders",
        steps=[step1, step2],
        execution_order=[1, 2],
        is_multi_step=True,
    )

    # Inconclusive Step 1 result
    raw_res = ExecutionResult(
        success=True,
        query="SELECT customer_id FROM superstore",
        dataframe=pd.DataFrame({"customer_id": ["C1", "C2"]}),
        row_count=2,
        column_count=1,
    )
    step1_val = validator.validate_step(
        raw_execution_result=raw_res, structured_intent=None, step_id=1
    )

    route, reason = validator.evaluate_routing(
        plan=plan,
        step_id=1,
        step_validation=step1_val,
        current_step_index=0,
    )

    assert route == "error_analyzer"
    assert reason is not None
    assert "cascading corruption" in reason or "inter-step column requirements" in reason


def test_routing_inconclusive_allows_independent_step(validator):
    """Verify INCONCLUSIVE step does not block subsequent independent steps."""
    step1 = PlanStep(step_id=1, question="Highest sales", target_table="superstore", depends_on=[])
    step2 = PlanStep(step_id=2, question="Highest profit", target_table="superstore", depends_on=[])
    plan = AnalyticalPlan(
        original_question="Highest sales and profit",
        steps=[step1, step2],
        execution_order=[1, 2],
        is_multi_step=True,
    )

    raw_res = ExecutionResult(
        success=True,
        query="SELECT product_name FROM superstore ORDER BY sales DESC LIMIT 1",
        dataframe=pd.DataFrame({"product_name": ["Widget A"]}),
        row_count=1,
        column_count=1,
    )
    step1_val = validator.validate_step(
        raw_execution_result=raw_res, structured_intent=None, step_id=1
    )

    route, reason = validator.evaluate_routing(
        plan=plan,
        step_id=1,
        step_validation=step1_val,
        current_step_index=0,
    )

    # Step 2 is independent (depends_on=[]); Step 1 allows proceeding to next_step
    assert route == "next_step"
    assert reason is None


def test_routing_inconclusive_final_step_routes_to_assembler_never_verified(validator):
    """Verify final INCONCLUSIVE step reaches assembler but cannot become VERIFIED."""
    step1 = PlanStep(step_id=1, question="Total sales", target_table="superstore", depends_on=[])
    plan = AnalyticalPlan(
        original_question="Total sales",
        steps=[step1],
        execution_order=[1],
        is_multi_step=False,
    )

    raw_res = ExecutionResult(
        success=True,
        query="SELECT SUM(sales) FROM superstore",
        dataframe=pd.DataFrame({"sum": [100.0]}),
        is_scalar=True,
        scalar_value=100.0,
        row_count=1,
        column_count=1,
    )
    step1_val = validator.validate_step(
        raw_execution_result=raw_res, structured_intent=None, step_id=1
    )

    route, reason = validator.evaluate_routing(
        plan=plan,
        step_id=1,
        step_validation=step1_val,
        current_step_index=0,
    )

    assert route == "assembler"
    assert reason is None

    # Verify report status is INCONCLUSIVE and never VERIFIED
    report = WorkflowValidationReport()
    report.add_step_attempt(step1_val)
    assert report.workflow_status == ValidationStatus.INCONCLUSIVE
    assert report.workflow_confidence == ConfidenceLevel.MEDIUM
    assert report.all_steps_verified is False


# ---------------------------------------------------------------------------
# 8. Attempt History Preservation and Result Retention
# ---------------------------------------------------------------------------


def test_attempt_history_preservation_and_retained_results(validator):
    """Verify multiple attempts are preserved chronologically and successful results retained."""
    report = WorkflowValidationReport()

    raw_ok = ExecutionResult(
        success=True,
        query="SELECT 1",
        dataframe=pd.DataFrame({"x": [1]}),
        is_scalar=True,
        scalar_value=1,
        row_count=1,
        column_count=1,
    )
    raw_fail = ExecutionResult(
        success=False,
        query="SELECT bad",
        error_type="SyntaxError",
        error_message="Invalid syntax",
    )

    attempt1 = validator.validate_step(
        raw_execution_result=raw_ok,
        step_id=1,
        attempt_number=1,
        executed_sql="SELECT 1",
    )
    report.add_step_attempt(attempt1)

    attempt2 = validator.validate_step(
        raw_execution_result=raw_fail,
        step_id=1,
        attempt_number=2,
        executed_sql="SELECT bad",
    )
    report.add_step_attempt(attempt2)

    # 1. Complete history preserved
    assert len(report.step_history[1]) == 2
    assert report.step_history[1][0].attempt_number == 1
    assert report.step_history[1][1].attempt_number == 2

    # 2. Latest result points to attempt 2
    assert report.latest_step_results[1].attempt_number == 2
    assert report.latest_step_results[1].status == ValidationStatus.FAILED

    # 3. Successful execution result from attempt 1 is retained
    assert 1 in report.retained_execution_results
    assert report.retained_execution_results[1].success is True
    assert report.retained_execution_results[1].scalar_value == 1


# ---------------------------------------------------------------------------
# 9. Additional Regression Checks (None DataFrame, Duplicates, Extra Columns)
# ---------------------------------------------------------------------------


def test_execution_validator_missing_dataframe_fails_scalar(validator):
    """Finding 1: Successful ExecutionResult with dataframe=None cannot be verified as scalar."""
    raw_res = ExecutionResult(
        success=True,
        query="SELECT 42",
        dataframe=None,  # Missing DataFrame!
        is_scalar=True,
        scalar_value=42,
        row_count=1,
        column_count=1,
    )

    intent = QuerySpecification(
        raw_question="Total",
        table_name="superstore",
        metrics=[MetricSpec(name="total", aggregation=AggregationType.SUM, column="sales")],
    )

    step_val = validator.validate_step(raw_execution_result=raw_res, structured_intent=intent)

    assert step_val.execution_result.passed is False
    assert step_val.execution_result.status == ValidationStatus.FAILED
    assert "no DataFrame was returned" in (step_val.execution_result.message or "")
    assert step_val.diagnostic is not None
    assert step_val.diagnostic.error_category == ErrorCategory.FATAL_SYSTEM
    assert step_val.diagnostic.is_correctable is False


def test_execution_validator_missing_dataframe_fails_missing_intent(validator):
    """Finding 1: Missing DataFrame with QuerySpecification=None fails Layer 1."""
    raw_res = ExecutionResult(
        success=True,
        query="SELECT 42",
        dataframe=None,  # Missing DataFrame!
        is_scalar=True,
        scalar_value=42,
        row_count=1,
        column_count=1,
    )

    step_val = validator.validate_step(raw_execution_result=raw_res, structured_intent=None)

    assert step_val.execution_result.passed is False
    assert step_val.execution_result.status == ValidationStatus.FAILED
    assert step_val.diagnostic is not None
    assert step_val.diagnostic.error_category == ErrorCategory.FATAL_SYSTEM


def test_execution_validator_duplicate_columns_case_insensitive(validator):
    """Finding 2: Duplicate column names (case-insensitive) must fail Layer 1."""
    # DataFrame with duplicate column names 'Region' and 'region'
    df = pd.DataFrame([[1, 2]], columns=["Region", "region"])
    raw_res = ExecutionResult(
        success=True,
        query="SELECT a.region AS Region, b.region FROM t a, t b",
        dataframe=df,
        is_scalar=False,
        row_count=1,
        column_count=2,
    )

    step_val = validator.validate_step(raw_execution_result=raw_res)

    assert step_val.execution_result.passed is False
    assert step_val.execution_result.status == ValidationStatus.FAILED
    assert step_val.execution_result.schema_matched is False
    assert "Duplicate column name(s)" in (step_val.execution_result.message or "")
    assert step_val.diagnostic is not None
    assert step_val.diagnostic.error_category == ErrorCategory.SCHEMA_ERROR
    assert step_val.diagnostic.is_correctable is True


def test_execution_validator_grouped_summary_unexpected_column(engine, validator):
    """Finding 3: Grouped summary with unexpected extra columns must fail schema validation."""
    # Query returns region, total_sales, AND unexpected category column
    sql = (
        "SELECT region, category, SUM(sales) AS total_sales "
        "FROM superstore GROUP BY region, category"
    )
    raw_res = engine.execute(sql)
    assert raw_res.success is True

    intent = QuerySpecification(
        raw_question="Sales by region",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales")],
        dimensions=["region"],  # Only region expected, category is unexpected!
    )

    step_val = validator.validate_step(raw_execution_result=raw_res, structured_intent=intent)

    assert step_val.execution_result.passed is False
    assert step_val.execution_result.status == ValidationStatus.FAILED
    assert step_val.execution_result.schema_matched is False
    assert "Unexpected column(s)" in (step_val.execution_result.message or "")
    assert step_val.diagnostic is not None
    assert step_val.diagnostic.error_category == ErrorCategory.SCHEMA_ERROR


def test_execution_validator_tabular_unexpected_column(engine, validator):
    """Finding 3: Tabular projection with unexpected extra columns must fail schema validation."""
    # Query returns region, category, AND unexpected sales column
    sql = "SELECT region, category, sales FROM superstore LIMIT 5"
    raw_res = engine.execute(sql)
    assert raw_res.success is True

    intent = QuerySpecification(
        raw_question="Region and category sample",
        table_name="superstore",
        dimensions=["region", "category"],  # sales is unexpected!
    )

    step_val = validator.validate_step(raw_execution_result=raw_res, structured_intent=intent)

    assert step_val.execution_result.passed is False
    assert step_val.execution_result.status == ValidationStatus.FAILED
    assert step_val.execution_result.schema_matched is False
    assert "Unexpected column(s)" in (step_val.execution_result.message or "")
    assert step_val.diagnostic is not None
    assert step_val.diagnostic.error_category == ErrorCategory.SCHEMA_ERROR
