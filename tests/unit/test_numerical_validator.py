"""Comprehensive unit tests for Layer 2 Independent Numerical Validator & Calculator.

Verifies:
    1. Scalar null detection and preservation of literal '<NA>' / 'null' strings.
    2. Strict boolean vs. integer separation (preventing bool-is-subclass-of-int leakage).
    3. Categorical exact float group-key matching (zero tolerance).
    4. Precision-safe large integer matching beyond 2^53 - 1.
    5. SQL NULL versus IEEE Floating NaN distinction.
    6. SQL aggregation semantics (COUNT(*), COUNT(col), COUNT(DISTINCT), empty/all-null sets).
    7. SQL LIKE wildcard, escaping, and case-sensitivity semantics.
    8. Safe AST formula evaluator with verified division-by-zero rules.
    9. Step validation across scalar, multi-metric, and grouped queries.
    10. History compaction lifecycle and non-mutating copy guarantees.
"""

import json
import math
from decimal import Decimal

import numpy as np
import pandas as pd

from src.agent.execution.engine import ExecutionResult
from src.agent.planning.intent import (
    AggregationType,
    FilterSpec,
    MetricSpec,
    QuerySpecification,
    TimeConstraint,
)
from src.agent.planning.planner import PlanStep
from src.agent.validation.contracts import (
    ExecutionValidationResult,
    NumericalValidationResult,
    SemanticValidationResult,
    ValidationStatus,
    WorkflowValidationReport,
    build_step_validation_result,
)
from src.agent.validation.numerical import (
    CANONICAL_FLOAT_NAN,
    CANONICAL_SQL_NULL,
    IndependentCalculator,
    NumericalValidator,
    SafeFormulaEvaluator,
    is_scalar_null,
    match_group_key_element,
    normalize_group_key_element,
)

# --- 1. Scalar Null Detection & Literal Text Tests ---


def test_is_scalar_null_genuine_nulls():
    """Verify that only genuine scalar missing representations return True."""
    assert is_scalar_null(None) is True
    assert is_scalar_null(np.nan) is True
    assert is_scalar_null(float("nan")) is True
    assert is_scalar_null(pd.NA) is True
    assert is_scalar_null(pd.NaT) is True


def test_is_scalar_null_literal_strings_and_bytes():
    """Verify that literal text '<NA>', 'null', and bytes are NEVER treated as null."""
    assert is_scalar_null("<NA>") is False
    assert is_scalar_null("null") is False
    assert is_scalar_null("NULL") is False
    assert is_scalar_null("") is False
    assert is_scalar_null(b"") is False
    assert is_scalar_null(b"null") is False


def test_is_scalar_null_non_scalar_collections():
    """Verify non-scalar arrays/lists do not trigger ambiguous truth-value errors."""
    assert is_scalar_null([1, 2]) is False
    assert is_scalar_null([]) is False
    assert is_scalar_null(np.array([1, 2])) is False
    assert is_scalar_null(pd.Series([1, 2])) is False


# --- 2. Strict Boolean vs. Integer Separation ---


def test_strict_boolean_vs_integer_differentiation():
    """Verify that boolean values do NOT match integer values through Python inheritance."""
    # True != 1, False != 0
    assert match_group_key_element(True, 1) is False
    assert match_group_key_element(1, True) is False
    assert match_group_key_element(False, 0) is False
    assert match_group_key_element(0, False) is False

    # True != 1.0, False != 0.0
    assert match_group_key_element(True, 1.0) is False
    assert match_group_key_element(False, 0.0) is False

    # Matching booleans
    assert match_group_key_element(True, True) is True
    assert match_group_key_element(False, False) is True
    assert match_group_key_element(True, False) is False


# --- 3. Exact Categorical Float Matching (Zero Tolerance) ---


def test_exact_float_group_key_matching_zero_tolerance():
    """Verify that group keys are categorical identities and reject any merging tolerance."""
    assert match_group_key_element(1.0000001, 1.0000000) is False
    assert match_group_key_element(10.5, 10.5) is True
    assert match_group_key_element(0.0, 0.0) is True


# --- 4. Precision-Safe Large Integer Matching ---


def test_large_integer_group_keys_beyond_safe_float_range():
    """Verify exact integer matching for values > 2^53 - 1 without 64-bit float loss."""
    large_id = 9007199254740993  # 2^53 + 1
    # Equal large ints match
    assert match_group_key_element(large_id, large_id) is True
    # Differing large ints fail
    assert match_group_key_element(large_id, large_id + 1) is False
    # Cross comparison with Decimal
    assert match_group_key_element(large_id, Decimal(str(large_id))) is True


# --- 5. SQL NULL vs. IEEE Floating NaN ---


def test_canonical_sentinels_null_vs_nan_distinction():
    """Verify that SQL NULL and IEEE Floating NaN are distinct categories."""
    assert match_group_key_element(CANONICAL_SQL_NULL, CANONICAL_SQL_NULL) is True
    assert match_group_key_element(CANONICAL_FLOAT_NAN, CANONICAL_FLOAT_NAN) is True
    assert match_group_key_element(CANONICAL_SQL_NULL, CANONICAL_FLOAT_NAN) is False
    assert match_group_key_element(CANONICAL_FLOAT_NAN, CANONICAL_SQL_NULL) is False


def test_normalize_group_key_element():
    """Verify normalization distinguishes source nulls from floating NaNs."""
    assert normalize_group_key_element(None) is CANONICAL_SQL_NULL
    assert normalize_group_key_element(pd.NA) is CANONICAL_SQL_NULL
    assert normalize_group_key_element(float("nan"), is_source_data=True) is CANONICAL_SQL_NULL
    assert normalize_group_key_element(float("nan"), is_source_data=False) is CANONICAL_FLOAT_NAN
    assert normalize_group_key_element("<NA>") == "<NA>"


# --- 6. Aggregation Semantics ---


def test_aggregation_count_semantics():
    """Verify COUNT(*), COUNT(col), and COUNT(DISTINCT col) with null values."""
    calc = IndependentCalculator()
    df = pd.DataFrame({"a": [1, 2, None, None, 2], "b": [None, None, None, None, None]})

    # COUNT(*) counts all rows including all-null rows
    assert calc.compute_atomic_metric(df, AggregationType.COUNT, "*") == 5
    # COUNT(a) counts only non-null values of a
    assert calc.compute_atomic_metric(df, AggregationType.COUNT, "a") == 3
    # COUNT(DISTINCT a) counts distinct non-null values
    assert calc.compute_atomic_metric(df, AggregationType.DISTINCT_COUNT, "a") == 2
    # COUNT(b) on all-null column returns 0
    assert calc.compute_atomic_metric(df, AggregationType.COUNT, "b") == 0
    assert calc.compute_atomic_metric(df, AggregationType.DISTINCT_COUNT, "b") == 0


def test_entirely_null_and_empty_aggregations():
    """Verify SUM, AVG, MIN, MAX yield None on empty or all-null columns."""
    calc = IndependentCalculator()
    df_empty = pd.DataFrame({"x": []})
    df_nulls = pd.DataFrame({"x": [None, None, None]})

    for df in [df_empty, df_nulls]:
        assert calc.compute_atomic_metric(df, AggregationType.SUM, "x") is None
        assert calc.compute_atomic_metric(df, AggregationType.AVG, "x") is None
        assert calc.compute_atomic_metric(df, AggregationType.MIN, "x") is None
        assert calc.compute_atomic_metric(df, AggregationType.MAX, "x") is None
        # COUNT yields 0 on empty/all-null
        assert calc.compute_atomic_metric(df, AggregationType.COUNT, "x") == 0


# --- 7. SQL LIKE Wildcard & Case-Sensitivity ---


def test_sql_like_wildcard_and_case_sensitivity():
    """Verify SQL LIKE wildcard matching and case-sensitivity."""
    calc = IndependentCalculator()
    df = pd.DataFrame({"s": ["Apple", "apple", "banana", "app", None]})

    # %app% is case-sensitive: matches 'apple' and 'app', rejects 'Apple' and None
    filters = [FilterSpec(column="s", operator="LIKE", value="%app%")]
    filtered, err = calc.apply_filters(df, filters, [])
    assert err is None
    assert set(filtered["s"]) == {"apple", "app"}

    # Exact wildcard _ (single character)
    filters_under = [FilterSpec(column="s", operator="LIKE", value="ap_")]
    filtered_under, err = calc.apply_filters(df, filters_under, [])
    assert err is None
    assert list(filtered_under["s"]) == ["app"]


# --- 8. Safe AST Formula Evaluator ---


def test_safe_formula_evaluator_valid_expressions():
    """Verify AST evaluation of derived metrics including division by zero."""
    evaluator = SafeFormulaEvaluator()

    # Standard expression
    context = {"SUM(profit)": 100.0, "SUM(sales)": 500.0}
    res = evaluator.evaluate("SUM(profit) / SUM(sales)", context)
    assert res == 0.2

    # Division by zero: 1.0 / 0.0 -> inf
    context_zero = {"a": 10.0, "b": 0.0}
    assert evaluator.evaluate("a / b", context_zero) == float("inf")

    # Division by zero: -10.0 / 0.0 -> -inf
    context_neg = {"a": -10.0, "b": 0.0}
    assert evaluator.evaluate("a / b", context_neg) == float("-inf")

    # 0.0 / 0.0 -> nan
    context_nan = {"a": 0.0, "b": 0.0}
    nan_res = evaluator.evaluate("a / b", context_nan)
    assert math.isnan(nan_res)

    # Null propagation: None / 10 -> None
    context_null = {"a": None, "b": 10.0}
    assert evaluator.evaluate("a / b", context_null) is None


def test_safe_formula_evaluator_rejects_unsafe_expressions():
    """Verify rejection of arbitrary Python function calls, imports, or attributes."""
    evaluator = SafeFormulaEvaluator()
    assert evaluator.is_safe("__import__('os').system('ls')") is False
    assert evaluator.is_safe("open('/etc/passwd')") is False
    assert evaluator.is_safe("lambda x: x") is False
    assert evaluator.is_safe("x.__class__") is False


# --- 9. NumericalValidator Step Tests ---


def test_numerical_validator_scalar_metric_success():
    """Verify scalar numerical validation within tolerance."""
    validator = NumericalValidator(atol=1e-4, rtol=1e-4)
    df = pd.DataFrame({"sales": [100.0, 200.0, 300.0]})

    spec = QuerySpecification(
        raw_question="Total sales",
        table_name="t",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(step_id=1, question="Total sales", target_table="t", structured_intent=spec)

    # DuckDB returned 600.00001 (diff = 1e-5 <= atol)
    exec_res = ExecutionResult(
        success=True,
        query="SELECT SUM(sales) AS total_sales FROM t",
        raw_rows=[(600.00001,)],
        is_scalar=True,
        row_count=1,
        column_count=1,
    )

    result = validator.validate_step(step, exec_res, df)
    assert result.passed is True
    assert result.status == ValidationStatus.VERIFIED
    assert result.absolute_difference is not None
    assert result.absolute_difference < 1e-4


def test_numerical_validator_scalar_metric_failure():
    """Verify scalar numerical validation failure when diff exceeds tolerance."""
    validator = NumericalValidator(atol=1e-4, rtol=1e-4)
    df = pd.DataFrame({"sales": [100.0, 200.0, 300.0]})

    spec = QuerySpecification(
        raw_question="Total sales",
        table_name="t",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(step_id=1, question="Total sales", target_table="t", structured_intent=spec)

    # DuckDB returned 500.0 (expected is 600.0)
    exec_res = ExecutionResult(
        success=True,
        query="SELECT SUM(sales) AS total_sales FROM t",
        raw_rows=[(500.0,)],
        is_scalar=True,
        row_count=1,
        column_count=1,
    )

    result = validator.validate_step(step, exec_res, df)
    assert result.passed is False
    assert result.status == ValidationStatus.FAILED
    assert result.absolute_difference == 100.0


def test_numerical_validator_grouped_summary_success():
    """Verify grouped summary validation with exact categorical keys."""
    validator = NumericalValidator()
    df = pd.DataFrame(
        {
            "region": ["East", "West", "East"],
            "sales": [100.0, 200.0, 300.0],
        }
    )

    spec = QuerySpecification(
        raw_question="Sales by region",
        table_name="t",
        dimensions=["region"],
        metrics=[MetricSpec(name="reg_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(step_id=1, question="Sales by region", target_table="t", structured_intent=spec)

    # DuckDB returned [('East', 400.0), ('West', 200.0)]
    exec_res = ExecutionResult(
        success=True,
        query="SELECT region, SUM(sales) AS reg_sales FROM t GROUP BY region",
        dataframe=pd.DataFrame({"region": ["East", "West"], "reg_sales": [400.0, 200.0]}),
        raw_rows=[("East", 400.0), ("West", 200.0)],
        is_scalar=False,
        row_count=2,
        column_count=2,
    )

    result = validator.validate_step(step, exec_res, df)
    assert result.passed is True
    assert result.status == ValidationStatus.VERIFIED


def test_numerical_validator_duplicate_group_rejection():
    """Verify that duplicate group keys in DuckDB query output trigger FAILED."""
    validator = NumericalValidator()
    df = pd.DataFrame({"region": ["East", "West"], "sales": [100.0, 200.0]})

    spec = QuerySpecification(
        raw_question="Sales by region",
        table_name="t",
        dimensions=["region"],
        metrics=[MetricSpec(name="reg_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(step_id=1, question="Sales by region", target_table="t", structured_intent=spec)

    # Faulty DuckDB output with duplicate 'East' key
    exec_res = ExecutionResult(
        success=True,
        query="SELECT ...",
        dataframe=pd.DataFrame({"region": ["East", "East"], "reg_sales": [100.0, 100.0]}),
        raw_rows=[("East", 100.0), ("East", 100.0)],
        is_scalar=False,
    )

    result = validator.validate_step(step, exec_res, df)
    assert result.passed is False
    assert result.status == ValidationStatus.FAILED
    assert "Duplicate group key" in result.message


def test_numerical_validator_inconclusive_conditions():
    """Verify INCONCLUSIVE is returned when source_df or structured_intent is missing."""
    validator = NumericalValidator()
    exec_res = ExecutionResult(success=True, query="SELECT 1", raw_rows=[(1,)])

    step = PlanStep(step_id=1, question="Q", target_table="t", structured_intent=None)

    # Missing structured intent
    res_no_intent = validator.validate_step(step, exec_res, pd.DataFrame({"a": [1]}))
    assert res_no_intent.status == ValidationStatus.INCONCLUSIVE

    # Missing source_df
    spec = QuerySpecification(
        raw_question="Q",
        table_name="t",
        metrics=[MetricSpec(name="a", column="a", aggregation=AggregationType.COUNT)],
    )
    step.structured_intent = spec
    res_no_df = validator.validate_step(step, exec_res, None)
    assert res_no_df.status == ValidationStatus.INCONCLUSIVE


# --- 10. History Compaction Lifecycle Tests ---


def test_history_compaction_non_mutating_copy_and_retention():
    """Verify safe history compaction creates replacement without mutating active references."""
    val_report = WorkflowValidationReport()

    # Attempt 1: SQL succeeds, but numerical validation fails
    res_1 = ExecutionResult(
        success=True,
        query="SELECT SUM(sales) FROM t",
        raw_rows=[(100.0,)],
        is_scalar=True,
    )
    step_val_1 = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=True, status=ValidationStatus.VERIFIED, message="Execution ok"
        ),
        numerical_res=NumericalValidationResult(
            layer_name="num", passed=False, status=ValidationStatus.FAILED, message="Diff"
        ),
        semantic_res=SemanticValidationResult(
            layer_name="sem", passed=False, status=ValidationStatus.UNVERIFIED, message="Unverified"
        ),
        raw_execution_result=res_1,
        executed_sql="SELECT SUM(sales) FROM t",
    )
    val_report.add_step_attempt(step_val_1)

    # Check Stage 2 Result Availability Policy: res_1 is retained because SQL succeeded
    assert val_report.retained_execution_results[1] is res_1
    assert val_report.retained_execution_results[1].raw_rows is not None

    # Attempt 2: SQL fails (e.g. timeout)
    res_2_failed_sql = ExecutionResult(success=False, query="SELECT syntax error", raw_rows=None)
    step_val_2 = build_step_validation_result(
        step_id=1,
        attempt_number=2,
        execution_res=ExecutionValidationResult(
            passed=False, status=ValidationStatus.FAILED, message="Syntax error"
        ),
        numerical_res=NumericalValidationResult(
            layer_name="num", passed=False, status=ValidationStatus.UNVERIFIED, message="Skipped"
        ),
        semantic_res=SemanticValidationResult(
            layer_name="sem", passed=False, status=ValidationStatus.UNVERIFIED, message="Skipped"
        ),
        raw_execution_result=res_2_failed_sql,
    )
    val_report.add_step_attempt(step_val_2)

    # Invariant: res_1 remains active retained execution result and is NOT destroyed
    assert val_report.retained_execution_results[1] is res_1
    assert val_report.retained_execution_results[1].raw_rows == [(100.0,)]

    # Attempt 3: SQL succeeds and numerical validation passes
    res_3 = ExecutionResult(
        success=True,
        query="SELECT SUM(sales) AS total_sales FROM t",
        raw_rows=[(600.0,)],
        is_scalar=True,
    )
    step_val_3 = build_step_validation_result(
        step_id=1,
        attempt_number=3,
        execution_res=ExecutionValidationResult(
            passed=True, status=ValidationStatus.VERIFIED, message="Execution ok"
        ),
        numerical_res=NumericalValidationResult(
            layer_name="num", passed=True, status=ValidationStatus.VERIFIED, message="Verified"
        ),
        semantic_res=SemanticValidationResult(
            layer_name="sem", passed=False, status=ValidationStatus.UNVERIFIED, message="Unverified"
        ),
        raw_execution_result=res_3,
    )
    val_report.add_step_attempt(step_val_3)

    # Now res_3 is the retained result
    assert val_report.retained_execution_results[1] is res_3
    assert val_report.retained_execution_results[1].raw_rows == [(600.0,)]

    # Invariant: Attempt 1 in step_history is now compacted (raw_rows is None via model_copy)
    history = val_report.step_history[1]
    assert len(history) == 3
    assert history[0].raw_execution_result.raw_rows is None
    # But original res_1 was NOT mutated
    assert res_1.raw_rows == [(100.0,)]
    # And latest attempt retains raw_rows
    assert history[2].raw_execution_result.raw_rows == [(600.0,)]


# --- 11. Additional Tests for Review Requirements ---


def test_grouped_query_with_null_dimension_values():
    """Verify grouped query with SQL NULL in dimension matches DuckDB None."""
    validator = NumericalValidator()
    df = pd.DataFrame(
        {
            "region": ["East", np.nan, "East"],
            "sales": [100.0, 200.0, 300.0],
        }
    )

    spec = QuerySpecification(
        raw_question="Sales by region with nulls",
        table_name="t",
        dimensions=["region"],
        metrics=[MetricSpec(name="reg_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(
        step_id=1,
        question="Sales by region with nulls",
        target_table="t",
        structured_intent=spec,
    )

    # DuckDB returns [('East', 400.0), (None, 200.0)]
    exec_res = ExecutionResult(
        success=True,
        query="SELECT region, SUM(sales) AS reg_sales FROM t GROUP BY region",
        dataframe=pd.DataFrame({"region": ["East", None], "reg_sales": [400.0, 200.0]}),
        raw_rows=[("East", 400.0), (None, 200.0)],
        is_scalar=False,
        row_count=2,
        column_count=2,
    )

    result = validator.validate_step(step, exec_res, df)
    assert result.passed is True
    assert result.status == ValidationStatus.VERIFIED


def test_grouped_query_rejects_duckdb_ieee_nan_matching_source_null():
    """Verify that DuckDB IEEE NaN in group key does NOT match source-data SQL NULL."""
    validator = NumericalValidator()
    df = pd.DataFrame(
        {
            "region": [np.nan],
            "sales": [100.0],
        }
    )

    spec = QuerySpecification(
        raw_question="Sales by region",
        table_name="t",
        dimensions=["region"],
        metrics=[MetricSpec(name="reg_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(step_id=1, question="Q", target_table="t", structured_intent=spec)

    # DuckDB falsely returned IEEE NaN instead of SQL NULL (None)
    exec_res = ExecutionResult(
        success=True,
        query="SELECT ...",
        dataframe=pd.DataFrame({"region": [float("nan")], "reg_sales": [100.0]}),
        raw_rows=[(float("nan"), 100.0)],
        is_scalar=False,
    )

    result = validator.validate_step(step, exec_res, df)
    assert result.passed is False
    assert result.status == ValidationStatus.FAILED


def test_formula_evaluator_rejects_keyword_and_extra_args():
    """Verify formula evaluator rejects keyword arguments and extra arguments."""
    evaluator = SafeFormulaEvaluator()
    assert evaluator.is_safe("SUM(sales, round=2)") is False
    assert evaluator.is_safe("SUM(sales, profit)") is False
    assert evaluator.is_safe("SUM()") is False
    assert evaluator.is_safe("SUM(sales) + __import__('os')") is False
    assert evaluator.is_safe("SUM(sales) / SUM(profit)") is True


def test_unsupported_formula_returns_inconclusive():
    """Verify unsupported formula returns INCONCLUSIVE in step validation."""
    validator = NumericalValidator()
    df = pd.DataFrame({"sales": [100.0, 200.0]})
    spec = QuerySpecification(
        raw_question="Derived",
        table_name="t",
        metrics=[
            MetricSpec(
                name="bad_metric",
                formula="SUM(sales, round=True)",
                aggregation=AggregationType.CUSTOM,
            )
        ],
    )
    step = PlanStep(step_id=1, question="Q", target_table="t", structured_intent=spec)
    exec_res = ExecutionResult(success=True, query="SELECT 1", raw_rows=[(100.0,)])

    res = validator.validate_step(step, exec_res, df)
    assert res.status == ValidationStatus.INCONCLUSIVE
    assert "Unsupported or malformed formula" in res.message


def test_large_integer_precision_beyond_float_safe_range():
    """Verify large integer sum beyond 2^53 - 1 retains exact precision."""
    validator = NumericalValidator()
    # 2^53 - 1 = 9007199254740991
    val1 = 9_007_199_254_740_993
    val2 = 10
    expected_sum = 9_007_199_254_741_003

    df = pd.DataFrame({"huge_int": [val1, val2]})
    spec = QuerySpecification(
        raw_question="Sum huge ints",
        table_name="t",
        metrics=[MetricSpec(name="sum_val", column="huge_int", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(step_id=1, question="Q", target_table="t", structured_intent=spec)

    # Exact integer match
    exec_res_ok = ExecutionResult(
        success=True,
        query="SELECT SUM(huge_int) FROM t",
        raw_rows=[(expected_sum,)],
        is_scalar=True,
    )
    res_ok = validator.validate_step(step, exec_res_ok, df)
    assert res_ok.passed is True
    assert res_ok.status == ValidationStatus.VERIFIED

    # Off-by-one difference (which float64 would truncate away)
    exec_res_bad = ExecutionResult(
        success=True,
        query="SELECT SUM(huge_int) FROM t",
        raw_rows=[(expected_sum + 1,)],
        is_scalar=True,
    )
    res_bad = validator.validate_step(step, exec_res_bad, df)
    assert res_bad.passed is False
    assert res_bad.status == ValidationStatus.FAILED


def test_date_parsing_dmy_and_mdy_formats():
    """Verify reliable date parsing for DMY and MDY formats."""
    calc = IndependentCalculator()

    # DMY dates: 25/01/2020 and 28/01/2020
    df_dmy = pd.DataFrame({"dt": ["25/01/2020", "28/01/2020"], "sales": [10.0, 20.0]})
    tc = [TimeConstraint(column="dt", start_time="2020-01-26", is_half_open=False)]
    filtered, err = calc.apply_filters(df_dmy, [], tc)
    assert err is None
    assert len(filtered) == 1
    assert filtered.iloc[0]["sales"] == 20.0

    # MDY dates: 01/25/2020 and 01/28/2020
    df_mdy = pd.DataFrame({"dt": ["01/25/2020", "01/28/2020"], "sales": [30.0, 40.0]})
    filtered_m, err_m = calc.apply_filters(df_mdy, [], tc)
    assert err_m is None
    assert len(filtered_m) == 1
    assert filtered_m.iloc[0]["sales"] == 40.0


def test_ambiguous_date_returns_inconclusive():
    """Verify ambiguous date formats (both day and month <= 12) return INCONCLUSIVE."""
    validator = NumericalValidator()
    df_ambig = pd.DataFrame({"dt": ["01/02/2020", "03/04/2020"], "sales": [10.0, 20.0]})
    spec = QuerySpecification(
        raw_question="Sales in range",
        table_name="t",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        time_constraints=[TimeConstraint(column="dt", start_time="2020-01-01", is_half_open=False)],
    )
    step = PlanStep(step_id=1, question="Q", target_table="t", structured_intent=spec)
    exec_res = ExecutionResult(success=True, query="SELECT 1", raw_rows=[(30.0,)])

    res = validator.validate_step(step, exec_res, df_ambig)
    assert res.status == ValidationStatus.INCONCLUSIVE
    assert "Ambiguous date format" in res.message


def test_aggregation_edge_cases():
    """Verify COUNT(*), COUNT(col), COUNT(DISTINCT), empty sets, and entirely NULL columns."""
    calc = IndependentCalculator()

    # Empty DataFrame
    df_empty = pd.DataFrame(columns=["a", "b"])
    assert calc.compute_atomic_metric(df_empty, AggregationType.COUNT, "*") == 0
    assert calc.compute_atomic_metric(df_empty, AggregationType.COUNT, "a") == 0
    assert calc.compute_atomic_metric(df_empty, AggregationType.DISTINCT_COUNT, "a") == 0
    assert calc.compute_atomic_metric(df_empty, AggregationType.SUM, "a") is None
    assert calc.compute_atomic_metric(df_empty, AggregationType.AVG, "a") is None
    assert calc.compute_atomic_metric(df_empty, AggregationType.MIN, "a") is None
    assert calc.compute_atomic_metric(df_empty, AggregationType.MAX, "a") is None

    # Entirely NULL column
    df_nulls = pd.DataFrame({"a": [None, np.nan, None]})
    assert calc.compute_atomic_metric(df_nulls, AggregationType.COUNT, "*") == 3
    assert calc.compute_atomic_metric(df_nulls, AggregationType.COUNT, "a") == 0
    assert calc.compute_atomic_metric(df_nulls, AggregationType.DISTINCT_COUNT, "a") == 0
    assert calc.compute_atomic_metric(df_nulls, AggregationType.SUM, "a") is None
    assert calc.compute_atomic_metric(df_nulls, AggregationType.AVG, "a") is None
    assert calc.compute_atomic_metric(df_nulls, AggregationType.MIN, "a") is None
    assert calc.compute_atomic_metric(df_nulls, AggregationType.MAX, "a") is None


def test_grouped_formula_division_by_zero():
    """Verify grouped formula evaluation correctly matches DuckDB division-by-zero rules."""
    validator = NumericalValidator()
    # Three groups: East (pos / 0 -> inf), West (neg / 0 -> -inf), Central (0 / 0 -> nan)
    df = pd.DataFrame(
        {
            "region": ["East", "West", "Central"],
            "profit": [100.0, -100.0, 0.0],
            "sales": [0.0, 0.0, 0.0],
        }
    )

    spec = QuerySpecification(
        raw_question="Margin by region",
        table_name="t",
        dimensions=["region"],
        metrics=[
            MetricSpec(name="profit", column="profit", aggregation=AggregationType.SUM),
            MetricSpec(name="sales", column="sales", aggregation=AggregationType.SUM),
            MetricSpec(
                name="margin",
                formula="SUM(profit) / SUM(sales)",
                aggregation=AggregationType.CUSTOM,
            ),
        ],
    )
    step = PlanStep(step_id=1, question="Q", target_table="t", structured_intent=spec)

    # DuckDB output matching division-by-zero semantics
    exec_res = ExecutionResult(
        success=True,
        query="SELECT ...",
        dataframe=pd.DataFrame(
            {
                "region": ["East", "West", "Central"],
                "profit": [100.0, -100.0, 0.0],
                "sales": [0.0, 0.0, 0.0],
                "margin": [float("inf"), float("-inf"), float("nan")],
            }
        ),
        raw_rows=[
            ("East", 100.0, 0.0, float("inf")),
            ("West", -100.0, 0.0, float("-inf")),
            ("Central", 0.0, 0.0, float("nan")),
        ],
        is_scalar=False,
    )

    res = validator.validate_step(step, exec_res, df)
    assert res.passed is True
    assert res.status == ValidationStatus.VERIFIED


def test_source_snapshot_consistency_and_immutability():
    """Requirement C: Verify registered DuckDB table matches snapshot and snapshot is immutable."""
    import datetime

    from src.agent.data.ingestion import SessionDatabase

    db = SessionDatabase()
    df_orig = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "val": [10.5, np.nan, 30.5],
            "order_date": [
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2),
                datetime.date(2020, 1, 3),
            ],
            "region": ["East", None, "West"],
        }
    )

    tbl_name = db.register_dataframe(df_orig, "test_snap")

    # Verify snapshot exists and is consistent with registered DuckDB table
    snap1 = db.get_table_dataframe(tbl_name)
    assert snap1 is not None
    assert len(snap1) == 3
    assert list(snap1.columns) == ["id", "val", "order_date", "region"]

    # Verify DuckDB execution returns matching rows and nulls
    duck_rows = db.connection.sql(f"SELECT * FROM {tbl_name} ORDER BY id").fetchall()
    assert len(duck_rows) == 3
    # Row 1: id=1, val=10.5, date=2020-01-01, region='East'
    assert duck_rows[0][0] == 1
    assert duck_rows[0][1] == 10.5
    assert duck_rows[0][3] == "East"
    # Row 2: id=2, val=None (SQL NULL), region=None (SQL NULL)
    assert duck_rows[1][0] == 2
    assert duck_rows[1][1] is None
    assert duck_rows[1][3] is None

    # Immutability Check: verify mutating the retrieved snapshot does NOT corrupt internal store
    snap1.iloc[0, 1] = 99999.9
    snap1["extra_col"] = "leak"

    snap2 = db.get_table_dataframe(tbl_name)
    assert snap2 is not None
    assert len(snap2) == 3  # Not dropped
    assert "extra_col" not in snap2.columns  # Not leaked
    assert snap2.iloc[0]["val"] == 10.5  # Not mutated


def test_workflow_history_compaction_full_lifecycle():
    """Requirement D: Retained latest successful, superseded compaction,
    immutability, and failed retries preserving last successful result.
    """
    val_report = WorkflowValidationReport()

    # Step attempt 1: SQL succeeds, numerical validation fails
    res_1 = ExecutionResult(
        success=True,
        query="SELECT 100",
        raw_rows=[(100,)],
        is_scalar=True,
    )
    s1 = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=True, status=ValidationStatus.VERIFIED, message="ok"
        ),
        numerical_res=NumericalValidationResult(
            layer_name="num", passed=False, status=ValidationStatus.FAILED, message="diff"
        ),
        semantic_res=SemanticValidationResult(
            layer_name="sem", passed=False, status=ValidationStatus.UNVERIFIED, message="unv"
        ),
        raw_execution_result=res_1,
        executed_sql="SELECT 100",
    )
    val_report.add_step_attempt(s1)

    # 1. Latest successful SQL result retained
    assert val_report.retained_execution_results[1] is res_1
    assert val_report.retained_execution_results[1].raw_rows == [(100,)]

    # Step attempt 2: SQL fails (e.g. timeout / syntax error)
    res_2_failed = ExecutionResult(
        success=False,
        query="SELECT syntax error",
        raw_rows=None,
        error_type="SyntaxError",
    )
    s2 = build_step_validation_result(
        step_id=1,
        attempt_number=2,
        execution_res=ExecutionValidationResult(
            passed=False, status=ValidationStatus.FAILED, message="Syntax error"
        ),
        numerical_res=NumericalValidationResult(
            layer_name="num", passed=False, status=ValidationStatus.UNVERIFIED, message="skipped"
        ),
        semantic_res=SemanticValidationResult(
            layer_name="sem", passed=False, status=ValidationStatus.UNVERIFIED, message="skipped"
        ),
        raw_execution_result=res_2_failed,
    )
    val_report.add_step_attempt(s2)

    # 4. Failed retry preserves last successful result; attempt 1 is NOT compacted
    assert val_report.retained_execution_results[1] is res_1
    assert val_report.retained_execution_results[1].raw_rows == [(100,)]
    assert val_report.step_history[1][0].raw_execution_result.raw_rows == [(100,)]

    # Step attempt 3: SQL succeeds and numerical validation passes
    res_3 = ExecutionResult(
        success=True,
        query="SELECT 200",
        raw_rows=[(200,)],
        is_scalar=True,
    )
    s3 = build_step_validation_result(
        step_id=1,
        attempt_number=3,
        execution_res=ExecutionValidationResult(
            passed=True, status=ValidationStatus.VERIFIED, message="ok"
        ),
        numerical_res=NumericalValidationResult(
            layer_name="num", passed=True, status=ValidationStatus.VERIFIED, message="verified"
        ),
        semantic_res=SemanticValidationResult(
            layer_name="sem", passed=False, status=ValidationStatus.UNVERIFIED, message="unv"
        ),
        raw_execution_result=res_3,
    )
    val_report.add_step_attempt(s3)

    # 1. Latest successful SQL result is now res_3
    assert val_report.retained_execution_results[1] is res_3
    assert val_report.retained_execution_results[1].raw_rows == [(200,)]

    # 2. Superseded attempt 1 is now compacted (raw_rows is None in history)
    assert val_report.step_history[1][0].raw_execution_result.raw_rows is None
    # Latest attempt retains raw_rows intact
    assert val_report.step_history[1][2].raw_execution_result.raw_rows == [(200,)]

    # 3. Reference immutability: original res_1 was NOT mutated in-place
    assert res_1.raw_rows == [(100,)]


# --- 12. Grouped Query Regression Tests & Evidence Validation ---


def test_grouped_query_permuted_row_order_success():
    """Verify grouped validation succeeds when DuckDB rows are in a different order from Pandas."""
    validator = NumericalValidator()
    df = pd.DataFrame(
        {
            "region": ["East", "West", "East"],
            "sales": [100.0, 200.0, 300.0],
        }
    )

    spec = QuerySpecification(
        raw_question="Sales by region",
        table_name="t",
        dimensions=["region"],
        metrics=[MetricSpec(name="reg_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(step_id=1, question="Sales by region", target_table="t", structured_intent=spec)

    # DuckDB returns West first, then East (opposite of Pandas natural group order)
    exec_res = ExecutionResult(
        success=True,
        query="SELECT region, SUM(sales) AS reg_sales FROM t GROUP BY region ORDER BY region DESC",
        dataframe=pd.DataFrame({"region": ["West", "East"], "reg_sales": [200.0, 400.0]}),
        raw_rows=[("West", 200.0), ("East", 400.0)],
        is_scalar=False,
        row_count=2,
        column_count=2,
    )

    result = validator.validate_step(step, exec_res, df)
    assert result.passed is True
    assert result.status == ValidationStatus.VERIFIED
    assert result.compared_rows == 2
    assert "All 2 grouped categories and metrics independently verified." in result.message
    assert result.details["matched_groups"] == 2
    assert result.details["passed_comparisons"] == 2


def test_grouped_query_missing_group_fails():
    """Verify grouped validation fails when a group is missing from DuckDB output."""
    validator = NumericalValidator()
    df = pd.DataFrame(
        {
            "region": ["East", "West", "East"],
            "sales": [100.0, 200.0, 300.0],
        }
    )

    spec = QuerySpecification(
        raw_question="Sales by region",
        table_name="t",
        dimensions=["region"],
        metrics=[MetricSpec(name="reg_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(step_id=1, question="Sales by region", target_table="t", structured_intent=spec)

    # 1. Missing group causing group count mismatch (1 returned vs 2 expected)
    exec_res_count_mismatch = ExecutionResult(
        success=True,
        query="SELECT region, SUM(sales) FROM t WHERE region = 'East' GROUP BY region",
        dataframe=pd.DataFrame({"region": ["East"], "reg_sales": [400.0]}),
        raw_rows=[("East", 400.0)],
        is_scalar=False,
        row_count=1,
        column_count=2,
    )
    res1 = validator.validate_step(step, exec_res_count_mismatch, df)
    assert res1.passed is False
    assert res1.status == ValidationStatus.FAILED
    assert "Group count mismatch" in res1.message
    assert "DuckDB returned 1 groups, independent calculation expected 2 groups" in res1.message

    # 2. Same group count but different key (DuckDB returned 'Central' instead of 'West')
    exec_res_key_missing = ExecutionResult(
        success=True,
        query="SELECT region, SUM(sales) FROM t ...",
        dataframe=pd.DataFrame({"region": ["East", "Central"], "reg_sales": [400.0, 200.0]}),
        raw_rows=[("East", 400.0), ("Central", 200.0)],
        is_scalar=False,
        row_count=2,
        column_count=2,
    )
    res2 = validator.validate_step(step, exec_res_key_missing, df)
    assert res2.passed is False
    assert res2.status == ValidationStatus.FAILED
    assert "missing from DuckDB output" in res2.message
    assert "('West',)" in res2.message
    assert res2.details["failure_reason"] == "group_key_mismatch"


def test_grouped_query_extra_group_fails():
    """Verify grouped validation fails when DuckDB output contains extra unexpected groups."""
    validator = NumericalValidator()
    df = pd.DataFrame(
        {
            "region": ["East", "West"],
            "sales": [100.0, 200.0],
        }
    )

    spec = QuerySpecification(
        raw_question="Sales by region",
        table_name="t",
        dimensions=["region"],
        metrics=[MetricSpec(name="reg_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(step_id=1, question="Sales by region", target_table="t", structured_intent=spec)

    # DuckDB returned 3 groups (East, West, South) when source only has East and West
    exec_res = ExecutionResult(
        success=True,
        query="SELECT ...",
        dataframe=pd.DataFrame(
            {"region": ["East", "West", "South"], "reg_sales": [100.0, 200.0, 50.0]}
        ),
        raw_rows=[("East", 100.0), ("West", 200.0), ("South", 50.0)],
        is_scalar=False,
        row_count=3,
        column_count=2,
    )

    result = validator.validate_step(step, exec_res, df)
    assert result.passed is False
    assert result.status == ValidationStatus.FAILED
    assert "Group count mismatch" in result.message
    assert "DuckDB returned 3 groups, independent calculation expected 2 groups" in result.message


def test_grouped_query_incorrect_metric_value_fails():
    """Verify grouped validation fails when a group has an incorrect metric value."""
    validator = NumericalValidator()
    df = pd.DataFrame(
        {
            "region": ["East", "West"],
            "sales": [100.0, 200.0],
        }
    )

    spec = QuerySpecification(
        raw_question="Sales by region",
        table_name="t",
        dimensions=["region"],
        metrics=[MetricSpec(name="reg_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(step_id=1, question="Sales by region", target_table="t", structured_intent=spec)

    # East has wrong sales (999.0 instead of 100.0)
    exec_res = ExecutionResult(
        success=True,
        query="SELECT ...",
        dataframe=pd.DataFrame({"region": ["East", "West"], "reg_sales": [999.0, 200.0]}),
        raw_rows=[("East", 999.0), ("West", 200.0)],
        is_scalar=False,
        row_count=2,
        column_count=2,
    )

    result = validator.validate_step(step, exec_res, df)
    assert result.passed is False
    assert result.status == ValidationStatus.FAILED
    assert result.duckdb_value == 999.0
    assert result.independent_value == 100.0
    assert result.absolute_difference == 899.0
    assert result.relative_difference is not None
    assert "Group metric mismatch for key ('East',), metric 'reg_sales'" in result.message


def test_grouped_validation_evidence_retention():
    """Verify successful grouped validation retains detailed comparison evidence and counts."""
    validator = NumericalValidator()
    df = pd.DataFrame(
        {
            "region": ["East", "West", "East"],
            "sales": [150.0, 250.0, 350.0],
        }
    )

    spec = QuerySpecification(
        raw_question="Sales by region",
        table_name="t",
        dimensions=["region"],
        metrics=[MetricSpec(name="reg_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(step_id=1, question="Sales by region", target_table="t", structured_intent=spec)

    exec_res = ExecutionResult(
        success=True,
        query="SELECT region, SUM(sales) AS reg_sales FROM t GROUP BY region",
        dataframe=pd.DataFrame({"region": ["East", "West"], "reg_sales": [500.0, 250.0]}),
        raw_rows=[("East", 500.0), ("West", 250.0)],
        is_scalar=False,
        row_count=2,
        column_count=2,
    )

    result = validator.validate_step(step, exec_res, df)
    assert result.passed is True
    assert result.status == ValidationStatus.VERIFIED

    # Check aggregate counts
    details = result.details
    assert details["expected_groups"] == 2
    assert details["matched_groups"] == 2
    assert details["compared_metrics"] == 2
    assert details["passed_comparisons"] == 2

    # Check group comparisons
    group_comps = details["group_comparisons"]
    assert len(group_comps) == 2
    keys = {g["group_key"] for g in group_comps}
    assert ("East",) in keys
    assert ("West",) in keys

    # Check flat comparisons
    comps = details["comparisons"]
    assert len(comps) == 2
    for c in comps:
        assert "group_key" in c
        assert "metric" in c
        assert "duckdb_value" in c
        assert "independent_value" in c
        assert "absolute_difference" in c
        assert "relative_difference" in c
        assert c["passed"] is True


def test_grouped_validation_evidence_values_match_calculations():
    """Verify that evidence values accurately match actual DuckDB and Pandas values."""
    validator = NumericalValidator()
    df = pd.DataFrame(
        {
            "region": ["East", "West", "East"],
            "profit": [10.5, 20.25, 30.5],
        }
    )

    spec = QuerySpecification(
        raw_question="Avg profit by region",
        table_name="t",
        dimensions=["region"],
        metrics=[MetricSpec(name="avg_profit", column="profit", aggregation=AggregationType.AVG)],
    )
    step = PlanStep(
        step_id=1, question="Avg profit by region", target_table="t", structured_intent=spec
    )

    # East mean: (10.5 + 30.5) / 2 = 20.5; West mean: 20.25
    exec_res = ExecutionResult(
        success=True,
        query="SELECT region, AVG(profit) AS avg_profit FROM t GROUP BY region",
        dataframe=pd.DataFrame({"region": ["East", "West"], "avg_profit": [20.5, 20.25]}),
        raw_rows=[("East", 20.5), ("West", 20.25)],
        is_scalar=False,
        row_count=2,
        column_count=2,
    )

    result = validator.validate_step(step, exec_res, df)
    assert result.passed is True

    details = result.details
    east_comp = next(c for c in details["comparisons"] if c["group_key"] == ("East",))
    assert east_comp["metric"] == "avg_profit"
    assert east_comp["duckdb_value"] == 20.5
    assert east_comp["independent_value"] == 20.5
    assert east_comp["absolute_difference"] == 0.0
    assert east_comp["relative_difference"] == 0.0
    assert east_comp["passed"] is True

    west_comp = next(c for c in details["comparisons"] if c["group_key"] == ("West",))
    assert west_comp["metric"] == "avg_profit"
    assert west_comp["duckdb_value"] == 20.25
    assert west_comp["independent_value"] == 20.25
    assert west_comp["absolute_difference"] == 0.0
    assert west_comp["passed"] is True


def test_grouped_validation_evidence_multiple_metrics():
    """Verify evidence structure when groups contain multiple metrics."""
    validator = NumericalValidator()
    df = pd.DataFrame(
        {
            "region": ["North", "South"],
            "sales": [100.0, 200.0],
            "profit": [15.0, 30.0],
        }
    )

    spec = QuerySpecification(
        raw_question="Sales and profit by region",
        table_name="t",
        dimensions=["region"],
        metrics=[
            MetricSpec(name="tot_sales", column="sales", aggregation=AggregationType.SUM),
            MetricSpec(name="tot_profit", column="profit", aggregation=AggregationType.SUM),
        ],
    )
    step = PlanStep(step_id=1, question="Q", target_table="t", structured_intent=spec)

    exec_res = ExecutionResult(
        success=True,
        query=(
            "SELECT region, SUM(sales) AS tot_sales, SUM(profit) AS tot_profit "
            "FROM t GROUP BY region"
        ),
        dataframe=pd.DataFrame(
            {"region": ["North", "South"], "tot_sales": [100.0, 200.0], "tot_profit": [15.0, 30.0]}
        ),
        raw_rows=[("North", 100.0, 15.0), ("South", 200.0, 30.0)],
        is_scalar=False,
        row_count=2,
        column_count=3,
    )

    result = validator.validate_step(step, exec_res, df)
    assert result.passed is True
    assert result.status == ValidationStatus.VERIFIED

    details = result.details
    assert details["expected_groups"] == 2
    assert details["matched_groups"] == 2
    assert details["compared_metrics"] == 4  # 2 groups * 2 metrics
    assert details["passed_comparisons"] == 4

    # Both metrics exist in each group's metrics map
    for g in details["group_comparisons"]:
        assert "tot_sales" in g["metrics"]
        assert "tot_profit" in g["metrics"]
        assert g["metrics"]["tot_sales"]["passed"] is True
        assert g["metrics"]["tot_profit"]["passed"] is True

    assert len(details["comparisons"]) == 4


def test_grouped_validation_evidence_serialization():
    """Verify that validation results with detailed evidence serialize cleanly."""
    validator = NumericalValidator()
    df = pd.DataFrame({"region": ["East"], "sales": [100.0]})

    spec = QuerySpecification(
        raw_question="Q",
        table_name="t",
        dimensions=["region"],
        metrics=[MetricSpec(name="sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(step_id=1, question="Q", target_table="t", structured_intent=spec)

    exec_res = ExecutionResult(
        success=True,
        query="SELECT region, SUM(sales) AS sales FROM t GROUP BY region",
        dataframe=pd.DataFrame({"region": ["East"], "sales": [100.0]}),
        raw_rows=[("East", 100.0)],
        is_scalar=False,
    )

    num_res = validator.validate_step(step, exec_res, df)
    assert num_res.passed is True

    step_val = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=True, status=ValidationStatus.VERIFIED, message="ok"
        ),
        numerical_res=num_res,
        semantic_res=SemanticValidationResult(
            layer_name="sem", passed=True, status=ValidationStatus.VERIFIED, message="ok"
        ),
        raw_execution_result=exec_res,
    )

    val_report = WorkflowValidationReport(
        workflow_status=ValidationStatus.VERIFIED,
        all_steps_verified=True,
        total_steps=1,
    )
    val_report.add_step_attempt(step_val)

    # 1. Pydantic model_dump and model_dump_json on NumericalValidationResult
    num_dump = num_res.model_dump()
    assert num_dump["status"] == ValidationStatus.VERIFIED
    assert num_dump["details"]["expected_groups"] == 1
    assert num_dump["details"]["matched_groups"] == 1
    assert num_dump["details"]["compared_metrics"] == 1
    assert num_dump["details"]["passed_comparisons"] == 1

    num_json = num_res.model_dump_json()
    assert isinstance(num_json, str)
    parsed_num = json.loads(num_json)
    assert parsed_num["status"] == "VERIFIED"
    assert parsed_num["details"]["expected_groups"] == 1
    assert parsed_num["details"]["group_comparisons"][0]["group_key"] == ["East"]

    # 2. Existing report serialization mechanism (serialize_validation_report_for_json)
    from src.ui.helpers import serialize_validation_report_for_json

    serialized = serialize_validation_report_for_json(val_report)
    report_json = json.dumps(serialized)
    assert isinstance(report_json, str)
    parsed_report = json.loads(report_json)
    step_details = parsed_report["latest_step_results"]["1"]["numerical_result"]["details"]
    assert step_details["expected_groups"] == 1
    assert step_details["matched_groups"] == 1
    assert step_details["compared_metrics"] == 1
    assert step_details["passed_comparisons"] == 1
    assert step_details["group_comparisons"][0]["group_key"] == ["East"]


def test_grouped_metric_failure_after_successful_comparison():
    """Verify that a metric failure preserves comparisons completed before and at failure."""
    validator = NumericalValidator()
    df = pd.DataFrame(
        {
            "region": ["East", "West"],
            "sales": [100.0, 200.0],
        }
    )

    spec = QuerySpecification(
        raw_question="Sales by region",
        table_name="t",
        dimensions=["region"],
        metrics=[MetricSpec(name="sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(step_id=1, question="Sales by region", target_table="t", structured_intent=spec)

    # East succeeds (100.0 == 100.0), West fails (999.0 != 200.0)
    exec_res = ExecutionResult(
        success=True,
        query="SELECT region, SUM(sales) AS sales FROM t GROUP BY region",
        dataframe=pd.DataFrame({"region": ["East", "West"], "sales": [100.0, 999.0]}),
        raw_rows=[("East", 100.0), ("West", 999.0)],
        is_scalar=False,
        row_count=2,
        column_count=2,
    )

    result = validator.validate_step(step, exec_res, df)
    assert result.passed is False
    assert result.status == ValidationStatus.FAILED
    assert result.duckdb_value == 999.0
    assert result.independent_value == 200.0
    assert result.absolute_difference == 799.0

    details = result.details
    assert details["failure_type"] == "metric_mismatch"
    assert details["failing_group_key"] == ("West",)
    assert details["failing_metric"] == "sales"
    assert details["expected_groups"] == 2
    assert details["actual_groups"] == 2
    assert details["matched_groups"] == 2
    assert details["compared_metrics"] == 2
    assert details["passed_comparisons"] == 1

    # Both groups are in group_comparisons: East passed, West failed
    assert len(details["group_comparisons"]) == 2
    assert details["group_comparisons"][0]["group_key"] == ("East",)
    assert details["group_comparisons"][0]["passed"] is True
    assert details["group_comparisons"][1]["group_key"] == ("West",)
    assert details["group_comparisons"][1]["passed"] is False

    # Flat comparisons include both
    assert len(details["comparisons"]) == 2
    assert details["comparisons"][0]["passed"] is True
    assert details["comparisons"][1]["passed"] is False
    assert details["comparisons"][1]["duckdb_value"] == 999.0


def test_grouped_metric_failure_with_multiple_metrics():
    """Verify evidence structure when a metric failure occurs in a multi-metric group."""
    validator = NumericalValidator()
    df = pd.DataFrame(
        {
            "region": ["North", "South"],
            "sales": [100.0, 200.0],
            "profit": [15.0, 30.0],
        }
    )

    spec = QuerySpecification(
        raw_question="Sales and profit by region",
        table_name="t",
        dimensions=["region"],
        metrics=[
            MetricSpec(name="tot_sales", column="sales", aggregation=AggregationType.SUM),
            MetricSpec(name="tot_profit", column="profit", aggregation=AggregationType.SUM),
        ],
    )
    step = PlanStep(step_id=1, question="Q", target_table="t", structured_intent=spec)

    # North: sales ok (100), profit ok (15); South: sales ok (200), profit fails (999 != 30)
    exec_res = ExecutionResult(
        success=True,
        query="SELECT ...",
        dataframe=pd.DataFrame(
            {
                "region": ["North", "South"],
                "tot_sales": [100.0, 200.0],
                "tot_profit": [15.0, 999.0],
            }
        ),
        raw_rows=[("North", 100.0, 15.0), ("South", 200.0, 999.0)],
        is_scalar=False,
        row_count=2,
        column_count=3,
    )

    result = validator.validate_step(step, exec_res, df)
    assert result.passed is False
    assert result.status == ValidationStatus.FAILED

    details = result.details
    assert details["failure_type"] == "metric_mismatch"
    assert details["failing_group_key"] == ("South",)
    assert details["failing_metric"] == "tot_profit"
    assert details["compared_metrics"] == 4
    assert details["passed_comparisons"] == 3

    comps = details["comparisons"]
    assert len(comps) == 4
    assert comps[0]["metric"] == "tot_sales" and comps[0]["passed"] is True
    assert comps[1]["metric"] == "tot_profit" and comps[1]["passed"] is True
    assert comps[2]["metric"] == "tot_sales" and comps[2]["passed"] is True
    assert comps[3]["metric"] == "tot_profit" and comps[3]["passed"] is False


def test_grouped_missing_groups_with_evidence_retained():
    """Verify that prior group comparisons are retained when a subsequent group is missing."""
    validator = NumericalValidator()
    df = pd.DataFrame(
        {
            "region": ["East", "West", "Central"],
            "sales": [100.0, 200.0, 300.0],
        }
    )

    spec = QuerySpecification(
        raw_question="Sales by region",
        table_name="t",
        dimensions=["region"],
        metrics=[MetricSpec(name="sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(step_id=1, question="Q", target_table="t", structured_intent=spec)

    # DuckDB only returns East and West; Central is missing
    exec_res = ExecutionResult(
        success=True,
        query="SELECT ...",
        dataframe=pd.DataFrame({"region": ["East", "West"], "sales": [100.0, 200.0]}),
        raw_rows=[("East", 100.0), ("West", 200.0)],
        is_scalar=False,
        row_count=2,
        column_count=2,
    )

    result = validator.validate_step(step, exec_res, df)
    assert result.passed is False
    assert result.status == ValidationStatus.FAILED

    details = result.details
    assert details["failure_type"] == "group_structure"
    assert details["failure_reason"] == "group_count_mismatch"
    assert details["expected_groups"] == 3
    assert details["actual_groups"] == 2
    assert details["matched_groups"] == 2
    assert details["missing_group_keys"] == (("Central",),)
    assert details["compared_metrics"] == 2
    assert details["passed_comparisons"] == 2
    assert len(details["group_comparisons"]) == 2
    assert len(details["comparisons"]) == 2


def test_grouped_extra_groups_with_evidence_retained():
    """Verify that valid group comparisons are retained when unexpected extra groups exist."""
    validator = NumericalValidator()
    df = pd.DataFrame(
        {
            "region": ["East", "West"],
            "sales": [100.0, 200.0],
        }
    )

    spec = QuerySpecification(
        raw_question="Sales by region",
        table_name="t",
        dimensions=["region"],
        metrics=[MetricSpec(name="sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(step_id=1, question="Q", target_table="t", structured_intent=spec)

    # DuckDB returns East and West (matching) plus extra South
    exec_res = ExecutionResult(
        success=True,
        query="SELECT ...",
        dataframe=pd.DataFrame(
            {"region": ["East", "West", "South"], "sales": [100.0, 200.0, 50.0]}
        ),
        raw_rows=[("East", 100.0), ("West", 200.0), ("South", 50.0)],
        is_scalar=False,
        row_count=3,
        column_count=2,
    )

    result = validator.validate_step(step, exec_res, df)
    assert result.passed is False
    assert result.status == ValidationStatus.FAILED

    details = result.details
    assert details["failure_type"] == "group_structure"
    assert details["expected_groups"] == 2
    assert details["actual_groups"] == 3
    assert details["matched_groups"] == 2
    assert details["unexpected_group_keys"] == (("South",),)
    assert details["compared_metrics"] == 2
    assert details["passed_comparisons"] == 2
    assert len(details["group_comparisons"]) == 2


def test_grouped_equal_count_key_mismatch_diagnostic():
    """Verify group_key_mismatch diagnostic when group counts match but keys differ."""
    validator = NumericalValidator()
    df = pd.DataFrame(
        {
            "region": ["East", "West"],
            "sales": [100.0, 200.0],
        }
    )

    spec = QuerySpecification(
        raw_question="Sales by region",
        table_name="t",
        dimensions=["region"],
        metrics=[MetricSpec(name="sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(step_id=1, question="Q", target_table="t", structured_intent=spec)

    # DuckDB returns East and South (count is 2, matching expected count 2, but West!=South)
    exec_res = ExecutionResult(
        success=True,
        query="SELECT ...",
        dataframe=pd.DataFrame({"region": ["East", "South"], "sales": [100.0, 200.0]}),
        raw_rows=[("East", 100.0), ("South", 200.0)],
        is_scalar=False,
        row_count=2,
        column_count=2,
    )

    result = validator.validate_step(step, exec_res, df)
    assert result.passed is False
    assert result.status == ValidationStatus.FAILED

    details = result.details
    assert details["failure_type"] == "group_structure"
    assert details["failure_reason"] == "group_key_mismatch"
    assert details["expected_groups"] == 2
    assert details["actual_groups"] == 2
    assert details["matched_groups"] == 1
    assert details["missing_group_keys"] == (("West",),)
    assert details["unexpected_group_keys"] == (("South",),)
    assert details["compared_metrics"] == 1
    assert details["passed_comparisons"] == 1
    assert "Group key mismatch" in result.message


def test_grouped_failed_and_partial_evidence_serialization():
    """Verify that failed and partial grouped evidence serializes cleanly without error."""
    validator = NumericalValidator()
    df = pd.DataFrame({"region": ["East", "West"], "sales": [100.0, 200.0]})

    spec = QuerySpecification(
        raw_question="Q",
        table_name="t",
        dimensions=["region"],
        metrics=[MetricSpec(name="sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = PlanStep(step_id=1, question="Q", target_table="t", structured_intent=spec)

    # Metric failure on West
    exec_res = ExecutionResult(
        success=True,
        query="SELECT ...",
        dataframe=pd.DataFrame({"region": ["East", "West"], "sales": [100.0, 999.0]}),
        raw_rows=[("East", 100.0), ("West", 999.0)],
        is_scalar=False,
    )

    num_res = validator.validate_step(step, exec_res, df)
    assert num_res.passed is False
    assert num_res.status == ValidationStatus.FAILED

    # 1. Direct NumericalValidationResult serialization
    dumped = num_res.model_dump()
    assert dumped["status"] == ValidationStatus.FAILED
    assert dumped["details"]["failure_type"] == "metric_mismatch"
    assert num_res.details["failing_group_key"] == ("West",)
    assert dumped["details"]["failing_group_key"] == ["West"]

    json_str = num_res.model_dump_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["status"] == "FAILED"
    assert parsed["details"]["failure_type"] == "metric_mismatch"
    assert parsed["details"]["failing_group_key"] == ["West"]
    assert len(parsed["details"]["comparisons"]) == 2

    # 2. Workflow report serialization with serialize_validation_report_for_json
    step_val = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=True, status=ValidationStatus.VERIFIED, message="ok"
        ),
        numerical_res=num_res,
        semantic_res=SemanticValidationResult(
            layer_name="sem", passed=False, status=ValidationStatus.UNVERIFIED, message="unv"
        ),
        raw_execution_result=exec_res,
    )
    val_report = WorkflowValidationReport(total_steps=1)
    val_report.add_step_attempt(step_val)

    from src.ui.helpers import serialize_validation_report_for_json

    serialized = serialize_validation_report_for_json(val_report)
    report_json = json.dumps(serialized)
    assert isinstance(report_json, str)
    parsed_report = json.loads(report_json)
    num_details = parsed_report["latest_step_results"]["1"]["numerical_result"]["details"]
    assert num_details["failure_type"] == "metric_mismatch"
    assert num_details["passed_comparisons"] == 1
    assert num_details["compared_metrics"] == 2
