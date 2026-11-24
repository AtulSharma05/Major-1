"""Comprehensive Unit and Integration Tests for Phase 5 Stage 4 Semantic Validation.

Covers:
    1. Canonical Checks: exactly 8 checks, duplicate/unknown/missing checks, status precedence.
    2. Intent Reliability: HIGH, AMBIGUOUS, SUSPECT, UNGROUNDED.
    3. Metric & Aggregation: correct/incorrect aggregates, formulas, AOV whitelist.
    4. Filter Semantics: operators, boundary constraints, NULL behavior, AND/OR tree.
    5. Join Integrity: single-table, Cartesian join detection, join conditions.
    6. Ordering & Ranking: superkeys, secondary keys, tie policies, ungrounded transforms.
    7. Conversion & Rounding: round-half-away, truncate, round-half-even, integral casts, NULL/NaN.
    8. Projection Contracts & Immutability: case-insensitive aliases, scalar allowlist, isolation.
    9. Multi-Layer Workflow Integration: independent L1, L2, L3 execution, 'validated' status.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import duckdb
import pandas as pd
import pytest

from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import ColumnProfile, TableProfile
from src.agent.execution.engine import ExecutionResult
from src.agent.planning.intent import (
    AggregationType,
    ConversionPolicy,
    DimensionSpec,
    FilterSpec,
    IntentReliability,
    MetricSpec,
    OrderBySpec,
    OutputTypeRequirement,
    QuerySpecification,
    RankingMode,
    RankingSpec,
    SortDirection,
    TieBreakerPolicy,
    TimeConstraint,
)
from src.agent.planning.planner import PlanStep
from src.agent.validation.contracts import (
    CANONICAL_SEMANTIC_CHECKS,
    ConfidenceLevel,
    SemanticCheckDetail,
    ValidationStatus,
    WorkflowValidationReport,
)
from src.agent.validation.projection import (
    CanonicalProjectionView,
    ProjectionNormalizationStatus,
    ProjectionNormalizer,
)
from src.agent.validation.semantic import (
    SemanticValidator,
    resolve_semantic_validation_status,
)
from src.agent.workflow.graph import build_analytical_graph

# --- Fixtures & Helpers ---


def _make_col_profile(name: str, dtype: str, distinct: int = 10, **kwargs) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        duckdb_type=dtype,
        total_rows=100,
        null_count=0,
        null_percentage=0.0,
        distinct_count=distinct,
        **kwargs,
    )


@pytest.fixture
def sample_table_profile() -> TableProfile:
    """Standard table profile fixture for tests."""
    columns = {
        "sales": _make_col_profile(
            "sales", "DOUBLE", min_val="10.0", max_val="500.0", mean_val=100.0
        ),
        "profit": _make_col_profile(
            "profit", "DOUBLE", min_val="-50.0", max_val="200.0", mean_val=25.0
        ),
        "quantity": _make_col_profile(
            "quantity", "BIGINT", min_val="1", max_val="10", mean_val=3.0
        ),
        "region": _make_col_profile("region", "VARCHAR", distinct=4),
        "category": _make_col_profile("category", "VARCHAR", distinct=3),
        "order_date": _make_col_profile("order_date", "TIMESTAMP", is_temporal=True),
        "order_id": _make_col_profile("order_id", "VARCHAR", distinct=100),
    }
    return TableProfile(
        table_name="superstore",
        row_count=100,
        column_count=len(columns),
        columns=columns,
    )


def _make_dummy_check(
    name: str, status: ValidationStatus, passed: bool = True
) -> SemanticCheckDetail:
    return SemanticCheckDetail(
        check_name=name,
        status=status,
        passed=passed,
        diagnostic_code=None if passed else f"failed_{name}",
        explanation=f"Dummy check {name}",
    )


def _make_step(
    intent: QuerySpecification, question: str = "Test question", step_id: int = 1
) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        question=question,
        target_table=intent.table_name or "superstore",
        structured_intent=intent,
    )


def _make_all_8_checks(
    status: ValidationStatus = ValidationStatus.VERIFIED, passed: bool = True
) -> list[SemanticCheckDetail]:
    return [
        _make_dummy_check(name, status, passed=passed)
        for name in sorted(list(CANONICAL_SEMANTIC_CHECKS))
    ]


# =============================================================================
# 1. Canonical Checks & Status Resolution Tests
# =============================================================================


def test_canonical_checks_set_has_exactly_eight():
    """Verify CANONICAL_SEMANTIC_CHECKS defines exactly eight canonical check names."""
    assert len(CANONICAL_SEMANTIC_CHECKS) == 8
    expected = {
        "check_table_grounding",
        "check_column_grounding",
        "check_aggregation_level",
        "check_metric_correctness",
        "check_filter_semantics",
        "check_join_integrity",
        "check_ordering_and_ranking",
        "check_projection_contract",
    }
    assert CANONICAL_SEMANTIC_CHECKS == expected


def test_status_resolution_all_eight_verified():
    """Verify that all 8 canonical checks verified with HIGH reliability yields VERIFIED."""
    checks = _make_all_8_checks(ValidationStatus.VERIFIED, passed=True)
    status, passed, diag = resolve_semantic_validation_status(
        checks=checks,
        is_schema_grounded=True,
        intent_reliability=IntentReliability.HIGH,
    )
    assert status == ValidationStatus.VERIFIED
    assert passed is True
    assert diag is None


def test_status_resolution_single_failed_dominates():
    """Verify Priority 1: Any canonical check FAILED yields FAILED immediately."""
    checks = _make_all_8_checks(ValidationStatus.VERIFIED, passed=True)
    # Replace check_metric_correctness with FAILED
    checks[3] = _make_dummy_check("check_metric_correctness", ValidationStatus.FAILED, passed=False)

    status, passed, diag = resolve_semantic_validation_status(
        checks=checks,
        is_schema_grounded=True,
        intent_reliability=IntentReliability.HIGH,
    )
    assert status == ValidationStatus.FAILED
    assert passed is False
    assert diag == "failed_check_metric_correctness"


def test_status_resolution_failure_not_masked_by_ungrounded_intent():
    """Verify that demonstrable FAILED is never converted to INCONCLUSIVE by ungrounded intent."""
    checks = _make_all_8_checks(ValidationStatus.VERIFIED, passed=True)
    checks[0] = _make_dummy_check("check_table_grounding", ValidationStatus.FAILED, passed=False)

    status, passed, diag = resolve_semantic_validation_status(
        checks=checks,
        is_schema_grounded=False,  # Ungrounded intent
        intent_reliability=IntentReliability.UNGROUNDED,
    )
    assert status == ValidationStatus.FAILED
    assert passed is False


def test_status_resolution_unknown_check_forces_inconclusive():
    """Verify that unknown check records corrupt evidence and produce INCONCLUSIVE."""
    checks = _make_all_8_checks(ValidationStatus.VERIFIED, passed=True)
    checks.append(_make_dummy_check("check_magic_mystery", ValidationStatus.VERIFIED, passed=True))

    status, passed, diag = resolve_semantic_validation_status(
        checks=checks,
        is_schema_grounded=True,
        intent_reliability=IntentReliability.HIGH,
    )
    assert status == ValidationStatus.INCONCLUSIVE
    assert passed is False
    assert "malformed_evidence_unknown_checks" in (diag or "")


def test_status_resolution_duplicate_check_forces_inconclusive():
    """Verify that duplicate canonical check records produce INCONCLUSIVE."""
    checks = _make_all_8_checks(ValidationStatus.VERIFIED, passed=True)
    # Add a duplicate of check 0
    checks.append(_make_dummy_check(checks[0].check_name, ValidationStatus.VERIFIED, passed=True))

    status, passed, diag = resolve_semantic_validation_status(
        checks=checks,
        is_schema_grounded=True,
        intent_reliability=IntentReliability.HIGH,
    )
    assert status == ValidationStatus.INCONCLUSIVE
    assert passed is False
    assert "malformed_evidence_duplicate_checks" in (diag or "")


def test_status_resolution_missing_check_forces_inconclusive():
    """Verify that an incomplete check set (< 8 canonical checks) produces INCONCLUSIVE."""
    checks = _make_all_8_checks(ValidationStatus.VERIFIED, passed=True)
    checks.pop()  # Remove 8th check

    status, passed, diag = resolve_semantic_validation_status(
        checks=checks,
        is_schema_grounded=True,
        intent_reliability=IntentReliability.HIGH,
    )
    assert status == ValidationStatus.INCONCLUSIVE
    assert passed is False
    assert "incomplete_semantic_check_set" in (diag or "")


def test_status_resolution_non_high_reliability_forces_inconclusive():
    """Verify that AMBIGUOUS or SUSPECT reliability gates VERIFIED and yields INCONCLUSIVE."""
    checks = _make_all_8_checks(ValidationStatus.VERIFIED, passed=True)

    for rel in (
        IntentReliability.AMBIGUOUS,
        IntentReliability.SUSPECT,
        IntentReliability.UNGROUNDED,
    ):
        status, passed, diag = resolve_semantic_validation_status(
            checks=checks,
            is_schema_grounded=True,
            intent_reliability=rel,
        )
        assert status == ValidationStatus.INCONCLUSIVE
        assert passed is False
        assert "intent_reliability_not_high" in (diag or "")


# =============================================================================
# 2. Table and Column Grounding Tests
# =============================================================================


def test_check_table_grounding_valid(sample_table_profile):
    """Verify check_table_grounding passes when table matches intent and schema."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = _make_step(intent, "Total sales?")
    sql = "SELECT SUM(sales) AS total_sales FROM superstore"
    res = ExecutionResult(success=True, query=sql, columns=("total_sales",), raw_rows=[(100.0,)])

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    table_check = next(c for c in result.checks if c.check_name == "check_table_grounding")
    assert table_check.status == ValidationStatus.VERIFIED
    assert table_check.passed is True


def test_check_table_grounding_unauthorized_table(sample_table_profile):
    """Verify check_table_grounding fails when an unauthorized table is queried."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = _make_step(intent, "Total sales?")
    sql = "SELECT SUM(sales) AS total_sales FROM secret_orders"
    res = ExecutionResult(success=True, query=sql, columns=("total_sales",), raw_rows=[(100.0,)])

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    table_check = next(c for c in result.checks if c.check_name == "check_table_grounding")
    assert table_check.status == ValidationStatus.FAILED
    assert result.status == ValidationStatus.FAILED


def test_check_column_grounding_unknown_column(sample_table_profile):
    """Verify check_column_grounding fails when referencing non-existent column."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Total revenue?",
        table_name="superstore",
        metrics=[
            MetricSpec(name="total_revenue", column="revenue", aggregation=AggregationType.SUM)
        ],
    )
    step = _make_step(intent, "Total revenue?")
    sql = "SELECT SUM(non_existent_column) AS total_revenue FROM superstore"
    res = ExecutionResult(success=True, query=sql, columns=("total_revenue",), raw_rows=[(100.0,)])

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    col_check = next(c for c in result.checks if c.check_name == "check_column_grounding")
    assert col_check.status == ValidationStatus.FAILED
    assert result.status == ValidationStatus.FAILED


# =============================================================================
# 3. Aggregation Level & Metric Correctness Tests
# =============================================================================


def test_scalar_query_with_group_by_fails(sample_table_profile):
    """Verify scalar query with unauthorized GROUP BY fails check_aggregation_level."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        dimensions=[],  # Scalar query
    )
    step = _make_step(intent, "Total sales?")
    sql = "SELECT SUM(sales) AS total_sales FROM superstore GROUP BY region"
    res = ExecutionResult(success=True, query=sql, columns=("total_sales",), raw_rows=[(100.0,)])

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    agg_check = next(c for c in result.checks if c.check_name == "check_aggregation_level")
    assert agg_check.status == ValidationStatus.FAILED
    assert agg_check.diagnostic_code == "scalar_query_has_group_by"


def test_grouped_query_missing_group_by_fails(sample_table_profile):
    """Verify grouped query lacking GROUP BY fails check_aggregation_level."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Sales by region?",
        table_name="superstore",
        dimensions=["region"],
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = _make_step(intent, "Sales by region?")
    sql = "SELECT region, SUM(sales) AS total_sales FROM superstore"
    res = ExecutionResult(
        success=True, query=sql, columns=("region", "total_sales"), raw_rows=[("East", 100.0)]
    )

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    agg_check = next(c for c in result.checks if c.check_name == "check_aggregation_level")
    assert agg_check.status == ValidationStatus.FAILED


def test_aov_metric_contract_approved_denominator(sample_table_profile):
    """Verify AOV metric passes when using approved order_id denominator whitelist."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Average order value?",
        table_name="superstore",
        metrics=[
            MetricSpec(
                name="aov",
                column="sales",
                denominator_column="order_id",
                aggregation=AggregationType.AOV,
            )
        ],
    )
    step = _make_step(intent, "Average order value?")
    sql = "SELECT SUM(sales) / COUNT(DISTINCT order_id) AS aov FROM superstore"
    res = ExecutionResult(success=True, query=sql, columns=("aov",), raw_rows=[(54.2,)])

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    metric_check = next(c for c in result.checks if c.check_name == "check_metric_correctness")
    assert metric_check.status == ValidationStatus.VERIFIED


def test_aov_metric_contract_unapproved_denominator_fails(sample_table_profile):
    """Verify AOV metric fails when using an unapproved denominator column (e.g. customer_id)."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Average order value?",
        table_name="superstore",
        metrics=[
            MetricSpec(
                name="aov",
                column="sales",
                denominator_column="customer_id",  # Not in whitelist!
                aggregation=AggregationType.AOV,
            )
        ],
    )
    step = _make_step(intent, "Average order value?")
    sql = "SELECT SUM(sales) / COUNT(DISTINCT customer_id) AS aov FROM superstore"
    res = ExecutionResult(success=True, query=sql, columns=("aov",), raw_rows=[(54.2,)])

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    metric_check = next(c for c in result.checks if c.check_name == "check_metric_correctness")
    assert metric_check.status == ValidationStatus.FAILED
    assert metric_check.diagnostic_code == "aov_unauthorized_denominator"


# =============================================================================
# 4. Conversion & Rounding Semantics (Revision 16 Rules)
# =============================================================================


def test_conversion_integral_cast_verified(sample_table_profile):
    """Rule 1: Provably integral result with integer cast is verified as safe identity."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Total sales?",
        table_name="superstore",
        metrics=[
            MetricSpec(
                name="total_sales",
                column="sales",
                aggregation=AggregationType.SUM,
                output_type=OutputTypeRequirement.INTEGER,
                conversion_policy=ConversionPolicy.NONE,
            )
        ],
    )
    step = _make_step(intent, "Total sales?")
    sql = "SELECT CAST(SUM(sales) AS INT) AS total_sales FROM superstore"
    # Result is provably integral: 200.0
    res = ExecutionResult(
        success=True,
        query=sql,
        columns=("total_sales",),
        raw_rows=[(200,)],
        is_scalar=True,
        scalar_value=200,
    )

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    metric_check = next(c for c in result.checks if c.check_name == "check_metric_correctness")
    assert metric_check.status == ValidationStatus.VERIFIED


def test_conversion_non_integral_integer_unspecified_policy_inconclusive(sample_table_profile):
    """Rule 2: Non-integral result with integer req & NO conversion policy -> INCONCLUSIVE."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Average sales as integer?",
        table_name="superstore",
        metrics=[
            MetricSpec(
                name="avg_sales",
                column="sales",
                aggregation=AggregationType.AVG,
                output_type=OutputTypeRequirement.INTEGER,
                conversion_policy=ConversionPolicy.NONE,  # Ambiguous: round, truncate, etc.
            )
        ],
    )
    step = _make_step(intent, "Average sales as integer?")
    sql = "SELECT CAST(AVG(sales) AS INT) AS avg_sales FROM superstore"
    # Result is non-integral: 150.5
    res = ExecutionResult(
        success=True,
        query=sql,
        columns=("avg_sales",),
        raw_rows=[(151,)],
        is_scalar=True,
        scalar_value=150.5,
    )

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    metric_check = next(c for c in result.checks if c.check_name == "check_metric_correctness")
    assert metric_check.status == ValidationStatus.INCONCLUSIVE
    assert metric_check.diagnostic_code == "ambiguous_integer_conversion"


def test_conversion_explicit_round_half_away_verified(sample_table_profile):
    """Rule 3: Explicit ROUND_HALF_AWAY_FROM_ZERO with CAST or ROUND is verified."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Average sales rounded?",
        table_name="superstore",
        metrics=[
            MetricSpec(
                name="avg_sales",
                column="sales",
                aggregation=AggregationType.AVG,
                output_type=OutputTypeRequirement.INTEGER,
                conversion_policy=ConversionPolicy.ROUND_HALF_AWAY_FROM_ZERO,
            )
        ],
    )
    step = _make_step(intent, "Average sales rounded?")
    sql = "SELECT ROUND(AVG(sales))::INT AS avg_sales FROM superstore"
    res = ExecutionResult(
        success=True,
        query=sql,
        columns=("avg_sales",),
        raw_rows=[(151,)],
        is_scalar=True,
        scalar_value=150.5,
    )

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    metric_check = next(c for c in result.checks if c.check_name == "check_metric_correctness")
    assert metric_check.status == ValidationStatus.VERIFIED


def test_conversion_explicit_truncate_with_cast_fails(sample_table_profile):
    """Rule 5: Explicit TRUNCATE policy contradicts CAST which rounds half away -> FAILED."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Truncated average sales?",
        table_name="superstore",
        metrics=[
            MetricSpec(
                name="avg_sales",
                column="sales",
                aggregation=AggregationType.AVG,
                output_type=OutputTypeRequirement.INTEGER,
                conversion_policy=ConversionPolicy.TRUNCATE,
            )
        ],
    )
    step = _make_step(intent, "Truncated average sales?")
    sql = "SELECT CAST(AVG(sales) AS INT) AS avg_sales FROM superstore"  # Uses round, not trunc!
    res = ExecutionResult(
        success=True,
        query=sql,
        columns=("avg_sales",),
        raw_rows=[(151,)],
        is_scalar=True,
        scalar_value=150.9,
    )

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    metric_check = next(c for c in result.checks if c.check_name == "check_metric_correctness")
    assert metric_check.status == ValidationStatus.FAILED
    assert metric_check.diagnostic_code == "unauthorized_conversion_applied"


def test_conversion_explicit_truncate_with_trunc_primitive_verified(sample_table_profile):
    """Rule 4: Explicit TRUNCATE policy using TRUNC(...) is verified."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Truncated average sales?",
        table_name="superstore",
        metrics=[
            MetricSpec(
                name="avg_sales",
                column="sales",
                aggregation=AggregationType.AVG,
                output_type=OutputTypeRequirement.INTEGER,
                conversion_policy=ConversionPolicy.TRUNCATE,
            )
        ],
    )
    step = _make_step(intent, "Truncated average sales?")
    sql = "SELECT TRUNC(AVG(sales))::INT AS avg_sales FROM superstore"
    res = ExecutionResult(
        success=True,
        query=sql,
        columns=("avg_sales",),
        raw_rows=[(150,)],
        is_scalar=True,
        scalar_value=150.9,
    )

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    metric_check = next(c for c in result.checks if c.check_name == "check_metric_correctness")
    assert metric_check.status == ValidationStatus.VERIFIED


def test_conversion_explicit_float_with_integer_cast_fails(sample_table_profile):
    """Rule 6: Explicit FLOAT requirement with integer cast -> FAILED."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Average sales as float?",
        table_name="superstore",
        metrics=[
            MetricSpec(
                name="avg_sales",
                column="sales",
                aggregation=AggregationType.AVG,
                output_type=OutputTypeRequirement.FLOAT,
            )
        ],
    )
    step = _make_step(intent, "Average sales as float?")
    sql = "SELECT CAST(AVG(sales) AS INT) AS avg_sales FROM superstore"
    res = ExecutionResult(
        success=True,
        query=sql,
        columns=("avg_sales",),
        raw_rows=[(150,)],
        is_scalar=True,
        scalar_value=150.0,
    )

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    metric_check = next(c for c in result.checks if c.check_name == "check_metric_correctness")
    assert metric_check.status == ValidationStatus.FAILED
    assert metric_check.diagnostic_code == "output_type_violation"


def test_conversion_round_half_even_unsupported(sample_table_profile):
    """Rule 7: ROUND_HALF_EVEN is unsupported in DuckDB SQL -> INCONCLUSIVE."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Average sales banker rounded?",
        table_name="superstore",
        metrics=[
            MetricSpec(
                name="avg_sales",
                column="sales",
                aggregation=AggregationType.AVG,
                conversion_policy=ConversionPolicy.ROUND_HALF_EVEN,
            )
        ],
    )
    step = _make_step(intent, "Average sales banker rounded?")
    sql = "SELECT ROUND(AVG(sales)) AS avg_sales FROM superstore"
    res = ExecutionResult(
        success=True,
        query=sql,
        columns=("avg_sales",),
        raw_rows=[(150.0,)],
        is_scalar=True,
        scalar_value=150.5,
    )

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    metric_check = next(c for c in result.checks if c.check_name == "check_metric_correctness")
    assert metric_check.status == ValidationStatus.INCONCLUSIVE
    assert metric_check.diagnostic_code == "unsupported_rounding_primitive"


def test_duckdb_actual_conversion_behavior():
    """Empirically test DuckDB SQL native conversion primitives: half-away, trunc, null."""
    con = duckdb.connect()

    # Round-half-away-from-zero: 0.5 -> 1, -0.5 -> -1, 1.5 -> 2, -1.5 -> -2, 2.5 -> 3
    res_pos_half = con.execute(
        "SELECT CAST(0.5 AS INT), CAST(1.5 AS INT), CAST(2.5 AS INT)"
    ).fetchall()[0]
    assert res_pos_half == (1, 2, 3)

    res_neg_half = con.execute(
        "SELECT CAST(-0.5 AS INT), CAST(-1.5 AS INT), CAST(-2.5 AS INT)"
    ).fetchall()[0]
    assert res_neg_half == (-1, -2, -3)

    # TRUNC toward zero
    res_trunc = con.execute("SELECT TRUNC(1.9)::INT, TRUNC(-1.9)::INT").fetchall()[0]
    assert res_trunc == (1, -1)

    # NULL preservation
    res_null = con.execute("SELECT CAST(NULL AS INT)").fetchall()[0]
    assert res_null == (None,)


# =============================================================================
# 5. Filter Semantics Tests
# =============================================================================


def test_filter_semantics_missing_predicate_fails(sample_table_profile):
    """Verify check_filter_semantics fails when a required filter predicate is omitted."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Sales for Technology category?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        filters=[FilterSpec(column="category", operator="=", value="Technology")],
    )
    step = _make_step(intent, "Sales for Technology category?")
    sql = "SELECT SUM(sales) AS total_sales FROM superstore"  # Missing WHERE clause!
    res = ExecutionResult(success=True, query=sql, columns=("total_sales",), raw_rows=[(100.0,)])

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    filter_check = next(c for c in result.checks if c.check_name == "check_filter_semantics")
    assert filter_check.status == ValidationStatus.FAILED
    assert filter_check.diagnostic_code == "missing_filter_predicate"


def test_filter_semantics_half_open_time_constraint(sample_table_profile):
    """Verify check_filter_semantics verifies half-open interval time constraints."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Sales in 2017?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        time_constraints=[
            TimeConstraint(
                column="order_date",
                start_time="2017-01-01",
                end_time="2018-01-01",
                is_half_open=True,
            )
        ],
    )
    step = _make_step(intent, "Sales in 2017?")
    sql = (
        "SELECT SUM(sales) AS total_sales FROM superstore "
        "WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'"
    )
    res = ExecutionResult(success=True, query=sql, columns=("total_sales",), raw_rows=[(100.0,)])

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    filter_check = next(c for c in result.checks if c.check_name == "check_filter_semantics")
    assert filter_check.status == ValidationStatus.VERIFIED


# =============================================================================
# 6. Join Integrity Tests
# =============================================================================


def test_join_integrity_single_table_verified(sample_table_profile):
    """Verify check_join_integrity passes when single table query has no joins."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = _make_step(intent, "Total sales?")
    sql = "SELECT SUM(sales) AS total_sales FROM superstore"
    res = ExecutionResult(success=True, query=sql, columns=("total_sales",), raw_rows=[(100.0,)])

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    join_check = next(c for c in result.checks if c.check_name == "check_join_integrity")
    assert join_check.status == ValidationStatus.VERIFIED


def test_join_integrity_cartesian_join_fails(sample_table_profile):
    """Verify check_join_integrity fails when a JOIN lacks ON or USING conditions."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Orders and customers?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = _make_step(intent, "Orders and customers?")
    sql = "SELECT SUM(sales) AS total_sales FROM superstore JOIN customers"  # Cartesian product!
    res = ExecutionResult(success=True, query=sql, columns=("total_sales",), raw_rows=[(100.0,)])

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    join_check = next(c for c in result.checks if c.check_name == "check_join_integrity")
    assert join_check.status == ValidationStatus.FAILED
    assert join_check.diagnostic_code == "cartesian_product_join"


# =============================================================================
# 7. Ordering & Ranking Tests
# =============================================================================


def test_ranking_single_extremum_verified_independent_of_ties(sample_table_profile):
    """Verify SINGLE_EXTREMUM scalar aggregation is verified regardless of row ties."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Highest sales value?",
        table_name="superstore",
        metrics=[MetricSpec(name="max_sales", column="sales", aggregation=AggregationType.MAX)],
        ranking_spec=RankingSpec(mode=RankingMode.SINGLE_EXTREMUM, is_scalar=True),
    )
    step = _make_step(intent, "Highest sales value?")
    sql = "SELECT MAX(sales) AS max_sales FROM superstore"
    res = ExecutionResult(
        success=True,
        query=sql,
        columns=("max_sales",),
        raw_rows=[(500.0,)],
        is_scalar=True,
        scalar_value=500.0,
    )

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    rank_check = next(c for c in result.checks if c.check_name == "check_ordering_and_ranking")
    assert rank_check.status == ValidationStatus.VERIFIED


def test_ranking_secondary_key_omitted_fails(sample_table_profile):
    """Verify ranking check fails when SQL omits a specified secondary sort key."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Top 5 regions by sales, broken by region name?",
        table_name="superstore",
        dimensions=["region"],
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        order_by=["total_sales DESC", "region ASC"],
        limit=5,
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            limit=5,
            direction=SortDirection.DESC,
            secondary_keys=[OrderBySpec(column="region", direction=SortDirection.ASC)],
        ),
    )
    step = _make_step(intent, "Top 5 regions by sales?")
    # Missing secondary key in ORDER BY!
    sql = (
        "SELECT region, SUM(sales) AS total_sales FROM superstore "
        "GROUP BY region ORDER BY total_sales DESC LIMIT 5"
    )
    res = ExecutionResult(
        success=True, query=sql, columns=("region", "total_sales"), raw_rows=[("East", 100.0)]
    )

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    rank_check = next(c for c in result.checks if c.check_name == "check_ordering_and_ranking")
    assert rank_check.status == ValidationStatus.FAILED
    assert rank_check.diagnostic_code == "ordering_secondary_key_mismatch"


def test_ranking_ungrounded_dimension_transformation_inconclusive(sample_table_profile):
    """Verify ungrounded dimension transformation (e.g. YEAR(col)) returns INCONCLUSIVE."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Top years by sales?",
        table_name="superstore",
        dimensions=["order_date"],  # Day-grain specified, but not year transformation
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        dimension_specs=[DimensionSpec(name="order_date", expression="order_date")],
        limit=5,
        ranking_spec=RankingSpec(mode=RankingMode.TOP_N_EXACT, limit=5),
    )
    step = _make_step(intent, "Top years by sales?")
    sql = (
        "SELECT YEAR(order_date) AS yr, SUM(sales) AS total_sales "
        "FROM superstore GROUP BY yr ORDER BY total_sales DESC LIMIT 5"
    )
    res = ExecutionResult(
        success=True, query=sql, columns=("order_date", "total_sales"), raw_rows=[(2017, 100.0)]
    )

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    rank_check = next(c for c in result.checks if c.check_name == "check_ordering_and_ranking")
    assert rank_check.status == ValidationStatus.INCONCLUSIVE
    assert rank_check.diagnostic_code == "ungrounded_dimension_transformation"


def test_ranking_top_n_with_ties_returns_inconclusive(sample_table_profile):
    """Verify TOP_N_WITH_TIES returns INCONCLUSIVE (L2 cannot verify window ties)."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Top 5 regions by sales including ties?",
        table_name="superstore",
        dimensions=["region"],
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_WITH_TIES, limit=5, tie_policy=TieBreakerPolicy.INCLUDE_TIES
        ),
    )
    step = _make_step(intent, "Top 5 regions by sales including ties?")
    sql = (
        "SELECT region, SUM(sales) AS total_sales FROM superstore "
        "GROUP BY region ORDER BY total_sales DESC LIMIT 5"
    )
    res = ExecutionResult(
        success=True, query=sql, columns=("region", "total_sales"), raw_rows=[("East", 100.0)]
    )

    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    rank_check = next(c for c in result.checks if c.check_name == "check_ordering_and_ranking")
    assert rank_check.status == ValidationStatus.INCONCLUSIVE
    assert rank_check.diagnostic_code == "unsupported_ranking_mode"


# =============================================================================
# 8. Projection Contracts, Immutability & Allowlist Tests
# =============================================================================


def test_canonical_projection_view_exact_scalar_allowlist():
    """Verify exact-type scalar allowlist: primitives, Decimal, date/time, bytes, None."""
    now = datetime.datetime.now()
    valid_row = (1, 2.5, "text", True, b"raw", now.date(), now, now.time(), Decimal("10.5"), None)
    view = CanonicalProjectionView(
        canonical_columns=("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"),
        canonical_raw_rows=(valid_row,),
        permutation_map=tuple(range(10)),
        is_permuted=False,
    )
    assert len(view.canonical_raw_rows[0]) == 10


def test_canonical_projection_view_nested_list_rejected():
    """Verify that cell with nested list raises TypeError."""
    with pytest.raises(TypeError, match="Disallowed value .* list"):
        CanonicalProjectionView(
            canonical_columns=("col1",),
            canonical_raw_rows=(([1, 2],),),
            permutation_map=(0,),
            is_permuted=False,
        )


def test_canonical_projection_view_nested_tuple_rejected():
    """Verify that cell with nested tuple raises TypeError."""
    with pytest.raises(TypeError, match="Disallowed value .* tuple"):
        CanonicalProjectionView(
            canonical_columns=("col1",),
            canonical_raw_rows=(((1, 2),),),
            permutation_map=(0,),
            is_permuted=False,
        )


def test_canonical_projection_view_custom_object_rejected():
    """Verify that cell with arbitrary custom object raises TypeError."""

    class CustomObj:
        pass

    with pytest.raises(TypeError, match="Disallowed value .* CustomObj"):
        CanonicalProjectionView(
            canonical_columns=("col1",),
            canonical_raw_rows=((CustomObj(),),),
            permutation_map=(0,),
            is_permuted=False,
        )


def test_canonical_projection_view_case_insensitive_duplicate_aliases_rejected():
    """Verify duplicate column aliases under case-insensitive comparison raise ValueError."""
    with pytest.raises(ValueError, match="Duplicate column aliases under case insensitivity"):
        CanonicalProjectionView(
            canonical_columns=("Total", "total"),
            canonical_raw_rows=((1, 2),),
            permutation_map=(0, 1),
            is_permuted=False,
        )


def test_canonical_projection_view_bijection_proof():
    """Verify invalid permutation map raises ValueError."""
    # Non-bijection: duplicate index 0
    with pytest.raises(ValueError, match="not a valid bijection"):
        CanonicalProjectionView(
            canonical_columns=("col1", "col2"),
            canonical_raw_rows=((1, 2),),
            permutation_map=(0, 0),
            is_permuted=True,
        )


def test_canonical_projection_view_caller_dict_mutation_isolation():
    """Verify caller-owned input dictionary remains unmutated after model construction."""
    caller_dict = {
        "canonical_columns": ["sales", "profit"],
        "canonical_raw_rows": [[100.0, 20.0]],
        "permutation_map": [0, 1],
        "is_permuted": False,
    }
    view = CanonicalProjectionView(**caller_dict)

    # Mutate caller dict
    caller_dict["permutation_map"].append(999)
    caller_dict["canonical_columns"].append("extra")

    # Assert view is isolated
    assert view.permutation_map == (0, 1)
    assert view.canonical_columns == ("sales", "profit")


def test_canonical_projection_view_caller_rows_unmutated_on_rejection():
    """Verify caller-owned row list is not mutated when validation fails and raises error."""
    caller_rows = [[1, [2, 3]]]  # Disallowed nested list
    with pytest.raises(TypeError):
        CanonicalProjectionView(
            canonical_columns=("col1", "col2"),
            canonical_raw_rows=caller_rows,
            permutation_map=(0, 1),
            is_permuted=False,
        )
    # Caller rows must be intact
    assert caller_rows == [[1, [2, 3]]]


def test_projection_normalizer_permuted_columns():
    """Verify ProjectionNormalizer detects permuted columns and builds valid canonical view."""
    normalizer = ProjectionNormalizer()
    expected = ["region", "sales"]
    # Actual execution returned ["sales", "region"]
    res = ExecutionResult(
        success=True,
        query="SELECT sales, region FROM t",
        columns=("sales", "region"),
        raw_rows=[(100.0, "East"), (200.0, "West")],
    )

    norm_res = normalizer.normalize(expected_projections=expected, execution_result=res)
    assert norm_res.status == ProjectionNormalizationStatus.PERMUTED
    assert norm_res.permutation_map == (1, 0)
    assert norm_res.canonical_view is not None
    assert norm_res.canonical_view.canonical_columns == ("region", "sales")
    # Verify rows permuted to canonical order
    assert norm_res.canonical_view.canonical_raw_rows == (("East", 100.0), ("West", 200.0))


def test_projection_normalizer_contradiction_missing_column():
    """Verify ProjectionNormalizer returns CONTRADICTION when expected column is missing."""
    normalizer = ProjectionNormalizer()
    expected = ["region", "sales", "profit"]
    res = ExecutionResult(
        success=True,
        query="SELECT region, sales FROM t",
        columns=("region", "sales"),
        raw_rows=[("East", 100.0)],
    )

    norm_res = normalizer.normalize(expected_projections=expected, execution_result=res)
    assert norm_res.status == ProjectionNormalizationStatus.CONTRADICTION
    assert norm_res.canonical_view is None
    assert norm_res.permutation_map == ()
    assert "profit" in norm_res.missing_projections


def test_projection_normalizer_unsupported_nested_cell():
    """Verify ProjectionNormalizer returns UNSUPPORTED when row contains nested container."""
    normalizer = ProjectionNormalizer()
    expected = ["id", "tags"]
    res = ExecutionResult(
        success=True,
        query="SELECT id, tags FROM t",
        columns=("id", "tags"),
        raw_rows=[(1, ["tag1", "tag2"])],  # Nested list cell
    )

    norm_res = normalizer.normalize(expected_projections=expected, execution_result=res)
    assert norm_res.status == ProjectionNormalizationStatus.UNSUPPORTED
    assert norm_res.canonical_view is None
    assert norm_res.permutation_map == ()


# =============================================================================
# 9. Multi-Layer Workflow Integration Tests
# =============================================================================


def test_full_three_layer_workflow_verified():
    """Verify that when Layer 1, 2, and 3 all pass, workflow status is 'validated'."""
    db = SessionDatabase()
    df = pd.DataFrame({"sales": [10.0, 20.0, 30.0], "region": ["East", "West", "East"]})
    db.register_dataframe(df, "superstore")
    profile = TableProfile(
        table_name="superstore",
        row_count=3,
        column_count=2,
        columns={
            "sales": _make_col_profile(
                "sales", "DOUBLE", min_val="10.0", max_val="30.0", mean_val=20.0
            ),
            "region": _make_col_profile("region", "VARCHAR", distinct=2),
        },
    )

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

    class FixedPlanner:
        def plan(self, *args, **kwargs):
            from src.agent.planning.planner import AnalyticalPlan

            return AnalyticalPlan(
                original_question="What is total sales?",
                steps=[step],
                execution_order=[1],
                is_multi_step=False,
            )

    class FixedGenerator:
        def generate_sql(self, *args, **kwargs):
            return "SELECT SUM(sales) AS total_sales FROM superstore"

    # Instantiate analytical graph with SemanticValidator enabled
    graph = build_analytical_graph(
        db=db,
        generator=FixedGenerator(),
        planner=FixedPlanner(),
        semantic_validator=SemanticValidator(),
    )

    final_state = graph.invoke(
        {
            "question": "What is total sales?",
            "session_id": "integration_test_l1_l2_l3",
            "active_table": "superstore",
            "schema_profile": profile,
        }
    )

    # All three layers verified -> overall workflow status is 'validated'
    assert final_state["status"] == "validated"
    val_report: WorkflowValidationReport = final_state["validation_report"]
    assert val_report.workflow_status == ValidationStatus.VERIFIED
    assert val_report.workflow_confidence == ConfidenceLevel.HIGH
    assert val_report.all_steps_verified is True

    step_res = val_report.latest_step_results[1]
    assert step_res.execution_result.status == ValidationStatus.VERIFIED
    assert step_res.numerical_result.status == ValidationStatus.VERIFIED
    assert step_res.semantic_result.status == ValidationStatus.VERIFIED
    assert len(step_res.semantic_result.checks) == 8
    assert all(c.status == ValidationStatus.VERIFIED for c in step_res.semantic_result.checks)


# =============================================================================
# 10. Stage 4 Remediation: Primary Ranking Verification
# =============================================================================


def test_ranking_primary_direction_mismatch_top_n_fails(sample_table_profile):
    """Verify primary sort direction mismatch (ASC instead of DESC) returns FAILED."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Top 5 regions by sales?",
        table_name="superstore",
        dimensions=["region"],
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        limit=5,
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            metric="total_sales",
            direction=SortDirection.DESC,
            limit=5,
            secondary_keys=[OrderBySpec(column="region", direction=SortDirection.ASC)],
        ),
    )
    step = _make_step(intent, "Top 5 regions by sales?")
    # SQL uses ASC instead of DESC
    sql = (
        "SELECT region, SUM(sales) AS total_sales FROM superstore "
        "GROUP BY region ORDER BY total_sales ASC, region ASC LIMIT 5"
    )
    res = ExecutionResult(
        success=True, query=sql, columns=("region", "total_sales"), raw_rows=[("East", 100.0)]
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    rank_check = next(c for c in result.checks if c.check_name == "check_ordering_and_ranking")
    assert rank_check.status == ValidationStatus.FAILED
    assert rank_check.diagnostic_code == "ordering_direction_mismatch"


def test_ranking_primary_direction_mismatch_bottom_n_fails(sample_table_profile):
    """Verify primary sort direction mismatch (DESC instead of ASC) returns FAILED."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Bottom 5 regions by sales?",
        table_name="superstore",
        dimensions=["region"],
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        limit=5,
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            metric="total_sales",
            direction=SortDirection.ASC,
            limit=5,
        ),
    )
    step = _make_step(intent, "Bottom 5 regions by sales?")
    # SQL uses DESC instead of ASC
    sql = (
        "SELECT region, SUM(sales) AS total_sales FROM superstore "
        "GROUP BY region ORDER BY total_sales DESC LIMIT 5"
    )
    res = ExecutionResult(
        success=True, query=sql, columns=("region", "total_sales"), raw_rows=[("East", 100.0)]
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    rank_check = next(c for c in result.checks if c.check_name == "check_ordering_and_ranking")
    assert rank_check.status == ValidationStatus.FAILED
    assert rank_check.diagnostic_code == "ordering_direction_mismatch"


def test_ranking_primary_metric_mismatch_fails(sample_table_profile):
    """Verify sorting by unrelated metric returns FAILED."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Top 5 regions by sales?",
        table_name="superstore",
        dimensions=["region"],
        metrics=[
            MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM),
            MetricSpec(name="total_profit", column="profit", aggregation=AggregationType.SUM),
        ],
        limit=5,
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            metric="total_sales",
            direction=SortDirection.DESC,
            limit=5,
        ),
    )
    step = _make_step(intent, "Top 5 regions by sales?")
    # SQL sorts by profit instead of sales
    sql = (
        "SELECT region, SUM(sales) AS total_sales, SUM(profit) AS total_profit "
        "FROM superstore GROUP BY region ORDER BY total_profit DESC LIMIT 5"
    )
    res = ExecutionResult(
        success=True,
        query=sql,
        columns=("region", "total_sales", "total_profit"),
        raw_rows=[("East", 100.0, 50.0)],
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    rank_check = next(c for c in result.checks if c.check_name == "check_ordering_and_ranking")
    assert rank_check.status == ValidationStatus.FAILED
    assert rank_check.diagnostic_code == "ordering_primary_metric_mismatch"


def test_ranking_primary_metric_and_direction_verified(sample_table_profile):
    """Verify correct primary metric and direction with deterministic keys passes."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Top 5 regions by sales?",
        table_name="superstore",
        dimensions=["region"],
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        limit=5,
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            metric="total_sales",
            direction=SortDirection.DESC,
            limit=5,
            secondary_keys=[OrderBySpec(column="region", direction=SortDirection.ASC)],
        ),
    )
    step = _make_step(intent, "Top 5 regions by sales?")
    sql = (
        "SELECT region, SUM(sales) AS total_sales FROM superstore "
        "GROUP BY region ORDER BY total_sales DESC, region ASC LIMIT 5"
    )
    res = ExecutionResult(
        success=True, query=sql, columns=("region", "total_sales"), raw_rows=[("East", 100.0)]
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    rank_check = next(c for c in result.checks if c.check_name == "check_ordering_and_ranking")
    assert rank_check.status == ValidationStatus.VERIFIED
    assert rank_check.passed is True


# =============================================================================
# 11. Stage 4 Remediation: Grouped LIMIT 1 Ranking Tie Determinism
# =============================================================================


def test_ranking_grouped_limit_1_without_superkeys_inconclusive(sample_table_profile):
    """Verify grouped LIMIT 1 query without secondary tie-breaker returns INCONCLUSIVE."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Top region by sales?",
        table_name="superstore",
        dimensions=["region"],
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        limit=1,
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            metric="total_sales",
            direction=SortDirection.DESC,
            limit=1,
        ),
    )
    step = _make_step(intent, "Top region by sales?")
    # Missing region in sort keys: non-deterministic under ties even with LIMIT 1!
    sql = (
        "SELECT region, SUM(sales) AS total_sales FROM superstore "
        "GROUP BY region ORDER BY total_sales DESC LIMIT 1"
    )
    res = ExecutionResult(
        success=True, query=sql, columns=("region", "total_sales"), raw_rows=[("East", 100.0)]
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    rank_check = next(c for c in result.checks if c.check_name == "check_ordering_and_ranking")
    assert rank_check.status == ValidationStatus.INCONCLUSIVE
    assert rank_check.diagnostic_code == "unproven_ranking_uniqueness"


def test_ranking_grouped_limit_1_with_superkeys_verified(sample_table_profile):
    """Verify grouped LIMIT 1 query with dimension in sort keys passes determinism."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Top region by sales?",
        table_name="superstore",
        dimensions=["region"],
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        limit=1,
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            metric="total_sales",
            direction=SortDirection.DESC,
            limit=1,
            secondary_keys=[OrderBySpec(column="region", direction=SortDirection.ASC)],
        ),
    )
    step = _make_step(intent, "Top region by sales?")
    sql = (
        "SELECT region, SUM(sales) AS total_sales FROM superstore "
        "GROUP BY region ORDER BY total_sales DESC, region ASC LIMIT 1"
    )
    res = ExecutionResult(
        success=True, query=sql, columns=("region", "total_sales"), raw_rows=[("East", 100.0)]
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    rank_check = next(c for c in result.checks if c.check_name == "check_ordering_and_ranking")
    assert rank_check.status == ValidationStatus.VERIFIED


# =============================================================================
# 12. Stage 4 Remediation: Tabular & Special Value Integrality
# =============================================================================


def test_conversion_grouped_non_integral_continuous_cast_fails(sample_table_profile):
    """Verify grouped query casting non-integral continuous metric to INT fails."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Average sales by category as int?",
        table_name="superstore",
        dimensions=["category"],
        metrics=[
            MetricSpec(
                name="avg_sales",
                column="sales",
                aggregation=AggregationType.AVG,
                output_type=OutputTypeRequirement.UNSPECIFIED,
                conversion_policy=ConversionPolicy.NONE,
            )
        ],
    )
    step = _make_step(intent, "Average sales by category?")
    sql = "SELECT category, CAST(AVG(sales) AS INT) AS avg_sales FROM superstore GROUP BY category"
    # Result rows contain fractional average: 150.75
    res = ExecutionResult(
        success=True,
        query=sql,
        columns=("category", "avg_sales"),
        raw_rows=[("Furniture", 150.75), ("Technology", 200.0)],
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    metric_check = next(c for c in result.checks if c.check_name == "check_metric_correctness")
    assert metric_check.status == ValidationStatus.FAILED
    assert metric_check.diagnostic_code == "unauthorized_conversion_applied"


def test_conversion_grouped_integral_cast_verified(sample_table_profile):
    """Verify grouped query casting provably integral metric to INT passes."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Total items by category?",
        table_name="superstore",
        dimensions=["category"],
        metrics=[
            MetricSpec(
                name="total_items",
                column="quantity",
                aggregation=AggregationType.SUM,
                output_type=OutputTypeRequirement.INTEGER,
                conversion_policy=ConversionPolicy.NONE,
            )
        ],
    )
    step = _make_step(intent, "Total items by category?")
    sql = (
        "SELECT category, CAST(SUM(quantity) AS INT) AS total_items "
        "FROM superstore GROUP BY category"
    )
    # All rows are integral
    res = ExecutionResult(
        success=True,
        query=sql,
        columns=("category", "total_items"),
        raw_rows=[("Furniture", 10.0), ("Technology", 25.0)],
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    metric_check = next(c for c in result.checks if c.check_name == "check_metric_correctness")
    assert metric_check.status == ValidationStatus.VERIFIED


def test_conversion_sql_null_scalar_integrality_inconclusive(sample_table_profile):
    """Verify SQL NULL scalar cannot prove integrality and returns INCONCLUSIVE."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Average profit as integer?",
        table_name="superstore",
        metrics=[
            MetricSpec(
                name="avg_profit",
                column="profit",
                aggregation=AggregationType.AVG,
                output_type=OutputTypeRequirement.INTEGER,
                conversion_policy=ConversionPolicy.NONE,
            )
        ],
    )
    step = _make_step(intent, "Average profit as integer?")
    sql = "SELECT CAST(AVG(profit) AS INT) AS avg_profit FROM superstore"
    # Result is SQL NULL
    res = ExecutionResult(
        success=True, query=sql, columns=("avg_profit",), is_scalar=True, scalar_value=None
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    metric_check = next(c for c in result.checks if c.check_name == "check_metric_correctness")
    assert metric_check.status == ValidationStatus.INCONCLUSIVE
    assert metric_check.diagnostic_code == "unprovable_integrality"


def test_conversion_ieee_nan_scalar_integrality_inconclusive(sample_table_profile):
    """Verify IEEE NaN scalar detects non-integrality and returns INCONCLUSIVE."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Average profit as integer?",
        table_name="superstore",
        metrics=[
            MetricSpec(
                name="avg_profit",
                column="profit",
                aggregation=AggregationType.AVG,
                output_type=OutputTypeRequirement.INTEGER,
                conversion_policy=ConversionPolicy.NONE,
            )
        ],
    )
    step = _make_step(intent, "Average profit as integer?")
    sql = "SELECT CAST(AVG(profit) AS INT) AS avg_profit FROM superstore"
    # Result is IEEE NaN
    res = ExecutionResult(
        success=True,
        query=sql,
        columns=("avg_profit",),
        is_scalar=True,
        scalar_value=float("nan"),
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    metric_check = next(c for c in result.checks if c.check_name == "check_metric_correctness")
    assert metric_check.status == ValidationStatus.INCONCLUSIVE
    assert metric_check.diagnostic_code == "ambiguous_integer_conversion"


# =============================================================================
# 13. Stage 4 Remediation: Projection Derivation & Deduplication
# =============================================================================


def test_projection_derivation_both_dimensions_and_specs(sample_table_profile):
    """Verify projection derivation combines both dimensions and specs without duplication."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Sales by region and year?",
        table_name="superstore",
        dimensions=["region"],
        dimension_specs=[
            DimensionSpec(name="region", expression="region"),
            DimensionSpec(name="yr", expression="YEAR(order_date)"),
        ],
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = _make_step(intent, "Sales by region and year?")
    sql = (
        "SELECT region, YEAR(order_date) AS yr, SUM(sales) AS total_sales "
        "FROM superstore GROUP BY region, yr"
    )
    res = ExecutionResult(
        success=True,
        query=sql,
        columns=("region", "yr", "total_sales"),
        raw_rows=[("East", 2017, 100.0)],
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    proj_check = next(c for c in result.checks if c.check_name == "check_projection_contract")
    assert proj_check.status == ValidationStatus.VERIFIED
    assert result.canonical_projection is not None
    assert result.canonical_projection.canonical_columns == ("region", "yr", "total_sales")


def test_projection_derivation_only_dimension_specs(sample_table_profile):
    """Verify projection derivation works when only dimension_specs is populated."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Sales by year?",
        table_name="superstore",
        dimensions=[],  # Empty dimensions list
        dimension_specs=[DimensionSpec(name="yr", expression="YEAR(order_date)")],
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = _make_step(intent, "Sales by year?")
    sql = "SELECT YEAR(order_date) AS yr, SUM(sales) AS total_sales FROM superstore GROUP BY yr"
    res = ExecutionResult(
        success=True, query=sql, columns=("yr", "total_sales"), raw_rows=[(2017, 100.0)]
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    proj_check = next(c for c in result.checks if c.check_name == "check_projection_contract")
    assert proj_check.status == ValidationStatus.VERIFIED
    assert result.canonical_projection.canonical_columns == ("yr", "total_sales")


# =============================================================================
# 14. Stage 4 Remediation: Common Table Expression (CTE) Grounding
# =============================================================================


def test_table_grounding_valid_single_cte(sample_table_profile):
    """Verify single CTE query passes table grounding without flagging CTE as unauthorized."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Regional sales summary?",
        table_name="superstore",
        metrics=[MetricSpec(name="s", column="sales", aggregation=AggregationType.SUM)],
    )
    step = _make_step(intent, "Regional sales summary?")
    sql = (
        "WITH regional_summary AS ("
        "  SELECT region, SUM(sales) AS s FROM superstore GROUP BY region"
        ") SELECT s FROM regional_summary"
    )
    res = ExecutionResult(success=True, query=sql, columns=("s",), raw_rows=[(100.0,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    table_check = next(c for c in result.checks if c.check_name == "check_table_grounding")
    assert table_check.status == ValidationStatus.VERIFIED


def test_table_grounding_valid_nested_and_multiple_ctes(sample_table_profile):
    """Verify multiple and nested CTEs pass table grounding."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Multi-CTE summary?",
        table_name="superstore",
        metrics=[MetricSpec(name="final_val", column="sales", aggregation=AggregationType.SUM)],
    )
    step = _make_step(intent, "Multi-CTE summary?")
    sql = (
        "WITH cte1 AS (SELECT region, sales FROM superstore), "
        "cte2 AS (SELECT region, sales * 2 AS double_sales FROM cte1) "
        "SELECT double_sales AS final_val FROM cte2"
    )
    res = ExecutionResult(success=True, query=sql, columns=("final_val",), raw_rows=[(200.0,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    table_check = next(c for c in result.checks if c.check_name == "check_table_grounding")
    assert table_check.status == ValidationStatus.VERIFIED


def test_table_grounding_unauthorized_table_inside_cte_fails(sample_table_profile):
    """Verify unauthorized physical table inside a CTE is detected and fails grounding."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Secret CTE summary?",
        table_name="superstore",
        metrics=[MetricSpec(name="val", column="secret_col", aggregation=AggregationType.SUM)],
    )
    step = _make_step(intent, "Secret CTE summary?")
    sql = (
        "WITH secret_cte AS (SELECT secret_col FROM forbidden_table) "
        "SELECT secret_col AS val FROM secret_cte"
    )
    res = ExecutionResult(success=True, query=sql, columns=("val",), raw_rows=[(100.0,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    table_check = next(c for c in result.checks if c.check_name == "check_table_grounding")
    assert table_check.status == ValidationStatus.FAILED
    assert table_check.diagnostic_code == "unauthorized_table_referenced"


# =============================================================================
# 15. Stage 4 Remediation: Filter Predicate AST Semantics
# =============================================================================


def test_filter_semantics_mismatched_value_fails(sample_table_profile):
    """Verify filter check fails when predicate value does not match intent."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Sales in West region?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        filters=[FilterSpec(column="region", operator="=", value="West")],
    )
    step = _make_step(intent, "Sales in West region?")
    # SQL filters for East instead of West
    sql = "SELECT SUM(sales) AS total_sales FROM superstore WHERE region = 'East'"
    res = ExecutionResult(success=True, query=sql, columns=("total_sales",), raw_rows=[(100.0,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    filter_check = next(c for c in result.checks if c.check_name == "check_filter_semantics")
    assert filter_check.status == ValidationStatus.FAILED
    assert filter_check.diagnostic_code == "filter_value_mismatch"


def test_filter_semantics_mismatched_operator_fails(sample_table_profile):
    """Verify filter check fails when operator does not match intent."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Sales not in West region?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        filters=[FilterSpec(column="region", operator="!=", value="West")],
    )
    step = _make_step(intent, "Sales not in West region?")
    # SQL uses = instead of !=
    sql = "SELECT SUM(sales) AS total_sales FROM superstore WHERE region = 'West'"
    res = ExecutionResult(success=True, query=sql, columns=("total_sales",), raw_rows=[(100.0,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    filter_check = next(c for c in result.checks if c.check_name == "check_filter_semantics")
    assert filter_check.status == ValidationStatus.FAILED
    assert filter_check.diagnostic_code == "filter_operator_mismatch"


def test_filter_semantics_missing_upper_date_bound_fails(sample_table_profile):
    """Verify filter check fails when half-open upper date bound is missing."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Sales in 2017?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        time_constraints=[
            TimeConstraint(
                column="order_date",
                start_time="2017-01-01",
                end_time="2018-01-01",
                is_half_open=True,
            )
        ],
    )
    step = _make_step(intent, "Sales in 2017?")
    # Missing order_date < '2018-01-01'!
    sql = "SELECT SUM(sales) AS total_sales FROM superstore WHERE order_date >= '2017-01-01'"
    res = ExecutionResult(success=True, query=sql, columns=("total_sales",), raw_rows=[(100.0,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    filter_check = next(c for c in result.checks if c.check_name == "check_filter_semantics")
    assert filter_check.status == ValidationStatus.FAILED
    assert filter_check.diagnostic_code == "time_boundary_mismatch"


def test_filter_semantics_incorrect_upper_date_bound_operator_fails(sample_table_profile):
    """Verify half-open upper date bound with <= instead of < fails."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Sales in 2017?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        time_constraints=[
            TimeConstraint(
                column="order_date",
                start_time="2017-01-01",
                end_time="2018-01-01",
                is_half_open=True,
            )
        ],
    )
    step = _make_step(intent, "Sales in 2017?")
    # Uses <= instead of <
    sql = (
        "SELECT SUM(sales) AS total_sales FROM superstore "
        "WHERE order_date >= '2017-01-01' AND order_date <= '2018-01-01'"
    )
    res = ExecutionResult(success=True, query=sql, columns=("total_sales",), raw_rows=[(100.0,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    filter_check = next(c for c in result.checks if c.check_name == "check_filter_semantics")
    assert filter_check.status == ValidationStatus.FAILED
    assert filter_check.diagnostic_code == "time_boundary_mismatch"


# =============================================================================
# 16. Stage 4 Remediation: Dimension Expression Equivalence
# =============================================================================


def test_aggregation_level_expression_equivalence_verified(sample_table_profile):
    """Verify GROUP BY YEAR(order_date) matches DimensionSpec with expression."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Sales by year?",
        table_name="superstore",
        dimensions=["order_date"],
        dimension_specs=[DimensionSpec(name="yr", expression="YEAR(order_date)")],
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = _make_step(intent, "Sales by year?")
    # SQL groups by the expression YEAR(order_date) directly
    sql = (
        "SELECT YEAR(order_date) AS yr, SUM(sales) AS total_sales "
        "FROM superstore GROUP BY YEAR(order_date)"
    )
    res = ExecutionResult(
        success=True, query=sql, columns=("yr", "total_sales"), raw_rows=[(2017, 100.0)]
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    agg_check = next(c for c in result.checks if c.check_name == "check_aggregation_level")
    assert agg_check.status == ValidationStatus.VERIFIED


def test_aggregation_level_extra_ungrounded_grouping_fails(sample_table_profile):
    """Verify extra ungrounded grouping expression fails check_aggregation_level."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Sales by region?",
        table_name="superstore",
        dimensions=["region"],
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step = _make_step(intent, "Sales by region?")
    # Groups by extra unrequested dimension 'category'
    sql = (
        "SELECT region, category, SUM(sales) AS total_sales "
        "FROM superstore GROUP BY region, category"
    )
    res = ExecutionResult(
        success=True,
        query=sql,
        columns=("region", "category", "total_sales"),
        raw_rows=[("East", "Furniture", 100.0)],
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    agg_check = next(c for c in result.checks if c.check_name == "check_aggregation_level")
    assert agg_check.status == ValidationStatus.FAILED
    assert agg_check.diagnostic_code == "unexpected_group_by_dimension"


# =============================================================================
# 17. Stage 4 Remediation: Validator Statelessness
# =============================================================================


def test_validator_instance_statelessness(sample_table_profile):
    """Verify SemanticValidator carries no mutable instance state across executions."""
    validator = SemanticValidator()
    assert not hasattr(validator, "_last_canonical_view")

    intent1 = QuerySpecification(
        raw_question="Total sales?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
    )
    step1 = _make_step(intent1, "Total sales?")
    sql1 = "SELECT SUM(sales) AS total_sales FROM superstore"
    res1 = ExecutionResult(success=True, query=sql1, columns=("total_sales",), raw_rows=[(100.0,)])

    result1 = validator.validate_step(
        step=step1, executed_sql=sql1, execution_result=res1, table_profile=sample_table_profile
    )
    assert not hasattr(validator, "_last_canonical_view")
    assert result1.canonical_projection is not None


# =============================================================================
# 18. Stage 4 Post-Audit Remediation: Date Normalization, Conjunction, & Qualified Ranking
# =============================================================================


def test_filter_date_literal_standard_string_verified(sample_table_profile):
    """Verify standard string date literals in time constraints pass verification."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Sales in 2017?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        time_constraints=[
            TimeConstraint(
                column="order_date",
                start_time="2017-01-01",
                end_time="2018-01-01",
                is_half_open=True,
            )
        ],
    )
    step = _make_step(intent, "Sales in 2017?")
    sql = (
        "SELECT SUM(sales) AS total_sales FROM superstore "
        "WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'"
    )
    res = ExecutionResult(success=True, query=sql, columns=("total_sales",), raw_rows=[(100.0,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    check = next(c for c in result.checks if c.check_name == "check_filter_semantics")
    assert check.status == ValidationStatus.VERIFIED


def test_filter_date_literal_typed_date_prefix_verified(sample_table_profile):
    """Verify DATE 'YYYY-MM-DD' literal syntax is normalized and verified."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Sales in 2017?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        time_constraints=[
            TimeConstraint(
                column="order_date",
                start_time="2017-01-01",
                end_time="2018-01-01",
                is_half_open=True,
            )
        ],
    )
    step = _make_step(intent, "Sales in 2017?")
    sql = (
        "SELECT SUM(sales) AS total_sales FROM superstore "
        "WHERE order_date >= DATE '2017-01-01' AND order_date < DATE '2018-01-01'"
    )
    res = ExecutionResult(success=True, query=sql, columns=("total_sales",), raw_rows=[(100.0,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    check = next(c for c in result.checks if c.check_name == "check_filter_semantics")
    assert check.status == ValidationStatus.VERIFIED


def test_filter_date_literal_explicit_cast_to_date_verified(sample_table_profile):
    """Verify CAST('YYYY-MM-DD' AS DATE) syntax is normalized and verified."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Sales in 2017?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        time_constraints=[
            TimeConstraint(
                column="order_date",
                start_time="2017-01-01",
                end_time="2018-01-01",
                is_half_open=True,
            )
        ],
    )
    step = _make_step(intent, "Sales in 2017?")
    sql = (
        "SELECT SUM(sales) AS total_sales FROM superstore "
        "WHERE order_date >= CAST('2017-01-01' AS DATE) AND order_date < CAST('2018-01-01' AS DATE)"
    )
    res = ExecutionResult(success=True, query=sql, columns=("total_sales",), raw_rows=[(100.0,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    check = next(c for c in result.checks if c.check_name == "check_filter_semantics")
    assert check.status == ValidationStatus.VERIFIED


def test_filter_date_boundary_operator_mismatch_fails(sample_table_profile):
    """Verify operator mismatch on date boundaries fails check_filter_semantics."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Sales in 2017?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        time_constraints=[
            TimeConstraint(
                column="order_date",
                start_time="2017-01-01",
                end_time="2018-01-01",
                is_half_open=True,
            )
        ],
    )
    step = _make_step(intent, "Sales in 2017?")
    # SQL uses <= for upper bound instead of strictly < for half-open interval
    sql = (
        "SELECT SUM(sales) AS total_sales FROM superstore "
        "WHERE order_date >= DATE '2017-01-01' AND order_date <= DATE '2018-01-01'"
    )
    res = ExecutionResult(success=True, query=sql, columns=("total_sales",), raw_rows=[(100.0,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    check = next(c for c in result.checks if c.check_name == "check_filter_semantics")
    assert check.status == ValidationStatus.FAILED
    assert check.diagnostic_code == "time_boundary_mismatch"


def test_filter_date_boundary_value_mismatch_fails(sample_table_profile):
    """Verify mismatched date boundary values fail check_filter_semantics."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Sales in 2017?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        time_constraints=[
            TimeConstraint(
                column="order_date",
                start_time="2017-01-01",
                end_time="2018-01-01",
                is_half_open=True,
            )
        ],
    )
    step = _make_step(intent, "Sales in 2017?")
    # SQL uses 2019-01-01 instead of 2018-01-01
    sql = (
        "SELECT SUM(sales) AS total_sales FROM superstore "
        "WHERE order_date >= DATE '2017-01-01' AND order_date < DATE '2019-01-01'"
    )
    res = ExecutionResult(success=True, query=sql, columns=("total_sales",), raw_rows=[(100.0,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    check = next(c for c in result.checks if c.check_name == "check_filter_semantics")
    assert check.status == ValidationStatus.FAILED
    assert check.diagnostic_code == "time_boundary_mismatch"


def test_filter_date_ambiguous_dynamic_expression_inconclusive(sample_table_profile):
    """Verify dynamic date expressions return INCONCLUSIVE when value cannot be proven."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Sales in 2017?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", column="sales", aggregation=AggregationType.SUM)],
        time_constraints=[
            TimeConstraint(
                column="order_date",
                start_time="2017-01-01",
                end_time="2018-01-01",
                is_half_open=True,
            )
        ],
    )
    step = _make_step(intent, "Sales in 2017?")
    sql = (
        "SELECT SUM(sales) AS total_sales FROM superstore "
        "WHERE order_date >= CURRENT_DATE AND order_date < DATE '2018-01-01'"
    )
    res = ExecutionResult(success=True, query=sql, columns=("total_sales",), raw_rows=[(100.0,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    check = next(c for c in result.checks if c.check_name == "check_filter_semantics")
    assert check.status == ValidationStatus.INCONCLUSIVE
    assert check.diagnostic_code == "unresolvable_date_expression"


def test_filter_conjunction_two_required_with_and_verified(sample_table_profile):
    """Verify two required independent filters connected by AND pass verification."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Active west customers?",
        table_name="superstore",
        metrics=[MetricSpec(name="cnt", column="customer_id", aggregation=AggregationType.COUNT)],
        filters=[
            FilterSpec(column="status", operator="=", value="active"),
            FilterSpec(column="region", operator="=", value="West"),
        ],
    )
    step = _make_step(intent, "Active west customers?")
    sql = (
        "SELECT COUNT(customer_id) AS cnt FROM superstore "
        "WHERE status = 'active' AND region = 'West'"
    )
    res = ExecutionResult(success=True, query=sql, columns=("cnt",), raw_rows=[(42,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    check = next(c for c in result.checks if c.check_name == "check_filter_semantics")
    assert check.status == ValidationStatus.VERIFIED


def test_filter_conjunction_two_required_connected_by_or_fails(sample_table_profile):
    """Verify two required filters connected by OR fail filter_conjunction_mismatch."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Active west customers?",
        table_name="superstore",
        metrics=[MetricSpec(name="cnt", column="customer_id", aggregation=AggregationType.COUNT)],
        filters=[
            FilterSpec(column="status", operator="=", value="active"),
            FilterSpec(column="region", operator="=", value="West"),
        ],
    )
    step = _make_step(intent, "Active west customers?")
    sql = (
        "SELECT COUNT(customer_id) AS cnt FROM superstore "
        "WHERE status = 'active' OR region = 'West'"
    )
    res = ExecutionResult(success=True, query=sql, columns=("cnt",), raw_rows=[(42,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    check = next(c for c in result.checks if c.check_name == "check_filter_semantics")
    assert check.status == ValidationStatus.FAILED
    assert check.diagnostic_code == "filter_conjunction_mismatch"
    assert "conjoined via OR" in check.explanation


def test_filter_conjunction_parenthesized_equivalent_and_verified(sample_table_profile):
    """Verify parenthesized equivalent AND expressions pass verification."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Active west customers?",
        table_name="superstore",
        metrics=[MetricSpec(name="cnt", column="customer_id", aggregation=AggregationType.COUNT)],
        filters=[
            FilterSpec(column="status", operator="=", value="active"),
            FilterSpec(column="region", operator="=", value="West"),
        ],
    )
    step = _make_step(intent, "Active west customers?")
    sql = (
        "SELECT COUNT(customer_id) AS cnt FROM superstore "
        "WHERE ((status = 'active') AND (region = 'West'))"
    )
    res = ExecutionResult(success=True, query=sql, columns=("cnt",), raw_rows=[(42,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    check = next(c for c in result.checks if c.check_name == "check_filter_semantics")
    assert check.status == ValidationStatus.VERIFIED


def test_filter_conjunction_mixed_and_or_incorrect_grouping_fails(sample_table_profile):
    """Verify mixed AND/OR with incorrect grouping fails filter_conjunction_mismatch."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Active west customers?",
        table_name="superstore",
        metrics=[MetricSpec(name="cnt", column="customer_id", aggregation=AggregationType.COUNT)],
        filters=[
            FilterSpec(column="status", operator="=", value="active"),
            FilterSpec(column="region", operator="=", value="West"),
        ],
    )
    step = _make_step(intent, "Active west customers?")
    # status is weakened by OR with c = 3
    sql = (
        "SELECT COUNT(customer_id) AS cnt FROM superstore "
        "WHERE (status = 'active' OR category = 'Furniture') AND region = 'West'"
    )
    res = ExecutionResult(success=True, query=sql, columns=("cnt",), raw_rows=[(42,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    check = next(c for c in result.checks if c.check_name == "check_filter_semantics")
    assert check.status == ValidationStatus.FAILED
    assert check.diagnostic_code == "filter_conjunction_mismatch"


def test_filter_conjunction_single_filter_preserved_verified(sample_table_profile):
    """Verify single required filter behavior is preserved as VERIFIED."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Active customers?",
        table_name="superstore",
        metrics=[MetricSpec(name="cnt", column="customer_id", aggregation=AggregationType.COUNT)],
        filters=[FilterSpec(column="status", operator="=", value="active")],
    )
    step = _make_step(intent, "Active customers?")
    sql = "SELECT COUNT(customer_id) AS cnt FROM superstore WHERE status = 'active'"
    res = ExecutionResult(success=True, query=sql, columns=("cnt",), raw_rows=[(42,)])
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    check = next(c for c in result.checks if c.check_name == "check_filter_semantics")
    assert check.status == ValidationStatus.VERIFIED


def test_ranking_qualified_aggregate_expression_verified(sample_table_profile):
    """Verify ORDER BY SUM(o.sales) DESC matches intent with primary metric sales."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Top categories by sales?",
        table_name="orders",
        dimensions=["category"],
        primary_metric="sales",
        metrics=[MetricSpec(name="sales", column="sales", aggregation=AggregationType.SUM)],
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            direction=SortDirection.DESC,
            n=5,
            metric="sales",
        ),
        limit=5,
    )
    step = _make_step(intent, "Top categories by sales?")
    sql = (
        "SELECT category, SUM(o.sales) AS sales "
        "FROM orders o "
        "GROUP BY category "
        "ORDER BY SUM(o.sales) DESC, category ASC "
        "LIMIT 5;"
    )
    res = ExecutionResult(
        success=True,
        query=sql,
        columns=("category", "sales"),
        raw_rows=[("Technology", 500.0), ("Furniture", 400.0)],
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    check = next(c for c in result.checks if c.check_name == "check_ordering_and_ranking")
    assert check.status == ValidationStatus.VERIFIED


def test_ranking_qualified_column_in_order_by_verified(sample_table_profile):
    """Verify ORDER BY o.sales DESC matches intent for non-grouped ranking."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Top orders by sales?",
        table_name="orders",
        primary_metric="sales",
        metrics=[MetricSpec(name="sales", column="sales", aggregation=AggregationType.SUM)],
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            direction=SortDirection.DESC,
            n=5,
            metric="sales",
        ),
        limit=5,
    )
    step = _make_step(intent, "Top orders by sales?")
    sql = "SELECT o.order_id, o.sales FROM orders o ORDER BY o.sales DESC LIMIT 5;"
    res = ExecutionResult(
        success=True, query=sql, columns=("order_id", "sales"), raw_rows=[("CA-1", 500.0)]
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    check = next(c for c in result.checks if c.check_name == "check_ordering_and_ranking")
    assert check.status == ValidationStatus.VERIFIED


def test_ranking_qualified_aggregate_direction_mismatch_fails(sample_table_profile):
    """Verify ORDER BY SUM(o.sales) ASC fails when DESC is required."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Top categories by sales?",
        table_name="orders",
        dimensions=["category"],
        primary_metric="sales",
        metrics=[MetricSpec(name="sales", column="sales", aggregation=AggregationType.SUM)],
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            direction=SortDirection.DESC,
            n=5,
            metric="sales",
        ),
        limit=5,
    )
    step = _make_step(intent, "Top categories by sales?")
    # Uses ASC instead of DESC
    sql = (
        "SELECT category, SUM(o.sales) AS sales "
        "FROM orders o "
        "GROUP BY category "
        "ORDER BY SUM(o.sales) ASC, category ASC "
        "LIMIT 5;"
    )
    res = ExecutionResult(
        success=True, query=sql, columns=("category", "sales"), raw_rows=[("Office", 10.0)]
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    check = next(c for c in result.checks if c.check_name == "check_ordering_and_ranking")
    assert check.status == ValidationStatus.FAILED
    assert check.diagnostic_code == "ordering_direction_mismatch"


def test_ranking_qualified_aggregate_unrelated_metric_fails(sample_table_profile):
    """Verify ORDER BY SUM(o.quantity) fails when sales is expected."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Top categories by sales?",
        table_name="orders",
        dimensions=["category"],
        primary_metric="sales",
        metrics=[MetricSpec(name="sales", column="sales", aggregation=AggregationType.SUM)],
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            direction=SortDirection.DESC,
            n=5,
            metric="sales",
        ),
        limit=5,
    )
    step = _make_step(intent, "Top categories by sales?")
    # Orders by quantity instead of sales
    sql = (
        "SELECT category, SUM(o.sales) AS sales, SUM(o.quantity) AS qty "
        "FROM orders o "
        "GROUP BY category "
        "ORDER BY SUM(o.quantity) DESC, category ASC "
        "LIMIT 5;"
    )
    res = ExecutionResult(
        success=True,
        query=sql,
        columns=("category", "sales", "qty"),
        raw_rows=[("Technology", 500.0, 10)],
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    check = next(c for c in result.checks if c.check_name == "check_ordering_and_ranking")
    assert check.status == ValidationStatus.FAILED
    assert check.diagnostic_code == "ordering_primary_metric_mismatch"


def test_ranking_qualified_secondary_keys_verified(sample_table_profile):
    """Verify secondary keys with qualified columns pass validation."""
    validator = SemanticValidator()
    intent = QuerySpecification(
        raw_question="Top categories by sales?",
        table_name="orders",
        dimensions=["category"],
        primary_metric="sales",
        metrics=[MetricSpec(name="sales", column="sales", aggregation=AggregationType.SUM)],
        ranking_spec=RankingSpec(
            mode=RankingMode.TOP_N_EXACT,
            direction=SortDirection.DESC,
            n=5,
            metric="sales",
            secondary_keys=[OrderBySpec(column="category", direction=SortDirection.ASC)],
        ),
        limit=5,
    )
    step = _make_step(intent, "Top categories by sales?")
    sql = (
        "SELECT o.category, SUM(o.sales) AS sales "
        "FROM orders o "
        "GROUP BY o.category "
        "ORDER BY SUM(o.sales) DESC, o.category ASC "
        "LIMIT 5;"
    )
    res = ExecutionResult(
        success=True,
        query=sql,
        columns=("category", "sales"),
        raw_rows=[("Technology", 500.0), ("Furniture", 400.0)],
    )
    result = validator.validate_step(
        step=step, executed_sql=sql, execution_result=res, table_profile=sample_table_profile
    )
    check = next(c for c in result.checks if c.check_name == "check_ordering_and_ranking")
    assert check.status == ValidationStatus.VERIFIED
