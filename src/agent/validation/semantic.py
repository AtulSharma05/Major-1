"""Layer 3: Scope-Aware Semantic Validator.

Architectural Context (ADR-02, Phase 5 Stage 4):
    Validates that executed SQL and execution results faithfully implement the
    user's analytical intent, conform to schema grounding, preserve mathematical
    semantics, satisfy ranking determinism, and adhere to canonical projection contracts.

Core Guarantees:
    1. Zero LLM Calls: Deterministic verification using SQLGlot AST inspection,
       QuerySpecification intent, schema metadata, and execution result evidence.
    2. Exactly Eight Canonical Checks:
       - check_table_grounding
       - check_column_grounding
       - check_aggregation_level
       - check_metric_correctness
       - check_filter_semantics
       - check_join_integrity
       - check_ordering_and_ranking
       - check_projection_contract
    3. Strict Precedence Hierarchy:
       FAILED > INCONCLUSIVE > VERIFIED.
       Genuine canonical contradictions always dominate and are never masked by
       ungrounded intent or inconclusive checks.
    4. Layer 2 Independence: Layer 2 never consumes Layer 3 or ProjectionNormalizer output.
    5. Immutability: ExecutionResult.raw_rows and attempt histories are never mutated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import sqlglot
from sqlglot import exp

from src.agent.data.profiling import TableProfile
from src.agent.execution.engine import ExecutionResult
from src.agent.planning.grounding import (
    detect_ungrounded_entities,
)
from src.agent.planning.intent import (
    AggregationType,
    ConversionPolicy,
    IntentReliability,
    OutputTypeRequirement,
    QuerySpecification,
    RankingMode,
)
from src.agent.planning.planner import PlanStep
from src.agent.validation.contracts import (
    CANONICAL_SEMANTIC_CHECKS,
    SemanticCheckDetail,
    SemanticValidationResult,
    ValidationStatus,
)
from src.agent.validation.projection import (
    CanonicalProjectionView,
    ProjectionNormalizationResult,
    ProjectionNormalizationStatus,
    ProjectionNormalizer,
)

# Approved denominator column whitelist for Average Order Value (AOV)
AOV_DENOMINATOR_WHITELIST = frozenset({"order_id", "order_number", "id", "transaction_id"})


def _normalize_date_str(val: Any) -> str | None:
    """Normalize date or timestamp value to standard YYYY-MM-DD string."""
    if val is None:
        return None
    s = str(val).strip("'\"").strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return None


def _values_equal(val1: Any, val2: Any) -> bool:
    """Compare two literal or predicate values with loose type tolerance."""
    if val1 is None and val2 is None:
        return True
    if val1 is None or val2 is None:
        return False
    s1 = str(val1).strip("'\"").strip()
    s2 = str(val2).strip("'\"").strip()
    if s1.lower() == s2.lower():
        return True
    d1 = _normalize_date_str(s1)
    d2 = _normalize_date_str(s2)
    if d1 is not None and d2 is not None and d1 == d2:
        return True
    try:
        return float(s1) == float(s2)
    except (ValueError, TypeError):
        return False


def resolve_semantic_validation_status(
    checks: Sequence[SemanticCheckDetail],
    is_schema_grounded: bool,
    intent_reliability: IntentReliability,
) -> tuple[ValidationStatus, bool, str | None]:
    """Deterministically resolve overall semantic validation status.

    Precedence:
        FAILED > INCONCLUSIVE > VERIFIED

    Invariants:
        1. Any FAILED canonical check yields FAILED overall (Priority 1).
        2. Unknown check records indicate corrupted evidence and force INCONCLUSIVE.
        3. Duplicate canonical checks violate contract and force INCONCLUSIVE.
        4. Incomplete check sets (<8 canonical checks) force INCONCLUSIVE.
        5. Ungrounded schema or non-HIGH intent reliability forces INCONCLUSIVE
           unless already FAILED.
        6. Any INCONCLUSIVE canonical check forces INCONCLUSIVE overall.
        7. VERIFIED strictly requires all 8 canonical checks to be VERIFIED and passed.
    """
    canonical_checks: list[SemanticCheckDetail] = []
    unknown_checks: list[SemanticCheckDetail] = []
    seen_canonical_names: set[str] = set()
    duplicate_names: set[str] = set()

    for c in checks:
        if c.check_name not in CANONICAL_SEMANTIC_CHECKS:
            unknown_checks.append(c)
        else:
            if c.check_name in seen_canonical_names:
                duplicate_names.add(c.check_name)
            seen_canonical_names.add(c.check_name)
            canonical_checks.append(c)

    # Priority 1: Demonstrable Failure in Recognized Canonical Checks
    canonical_failures = [c for c in canonical_checks if c.status == ValidationStatus.FAILED]
    if canonical_failures:
        first_failure = canonical_failures[0]
        diagnostic = (
            first_failure.diagnostic_code or f"Semantic failure in {first_failure.check_name}"
        )
        return ValidationStatus.FAILED, False, diagnostic

    # Priority 2: Evidence Integrity & Check Set Completeness
    if unknown_checks:
        unknown_names = sorted(list({c.check_name for c in unknown_checks}))
        return (
            ValidationStatus.INCONCLUSIVE,
            False,
            f"malformed_evidence_unknown_checks: {unknown_names}",
        )

    if duplicate_names:
        return (
            ValidationStatus.INCONCLUSIVE,
            False,
            f"malformed_evidence_duplicate_checks: {sorted(list(duplicate_names))}",
        )

    if len(seen_canonical_names) != 8 or seen_canonical_names != CANONICAL_SEMANTIC_CHECKS:
        missing = sorted(list(CANONICAL_SEMANTIC_CHECKS - seen_canonical_names))
        return (
            ValidationStatus.INCONCLUSIVE,
            False,
            f"incomplete_semantic_check_set; missing: {missing}",
        )

    # Priority 3: Intent Grounding & Reliability Gate
    if not is_schema_grounded:
        return ValidationStatus.INCONCLUSIVE, False, "intent_not_schema_grounded"

    if intent_reliability != IntentReliability.HIGH:
        return (
            ValidationStatus.INCONCLUSIVE,
            False,
            f"intent_reliability_not_high: {intent_reliability.value}",
        )

    # Priority 4: Upstream Inconclusive Checks
    inconclusive_checks = [c for c in canonical_checks if c.status == ValidationStatus.INCONCLUSIVE]
    if inconclusive_checks:
        first_inconclusive = inconclusive_checks[0]
        diagnostic = (
            first_inconclusive.diagnostic_code
            or f"Check {first_inconclusive.check_name} inconclusive"
        )
        return ValidationStatus.INCONCLUSIVE, False, diagnostic

    # Priority 5: Universal Canonical Verification
    if all(c.status == ValidationStatus.VERIFIED and c.passed for c in canonical_checks):
        return ValidationStatus.VERIFIED, True, None

    return ValidationStatus.INCONCLUSIVE, False, "undetermined_semantic_state"


class SemanticValidator:
    """Layer 3: Scope-aware semantic validator for analytical queries."""

    def __init__(self, projection_normalizer: ProjectionNormalizer | None = None) -> None:
        self.projection_normalizer = projection_normalizer or ProjectionNormalizer()

    def validate_step(
        self,
        step: PlanStep | None,
        executed_sql: str | None,
        execution_result: ExecutionResult | None,
        table_profile: TableProfile | None,
        available_tables: Sequence[str] | None = None,
    ) -> SemanticValidationResult:
        """Execute all eight canonical semantic checks against the step intent and SQL.

        Args:
            step: Active PlanStep containing structured_intent.
            executed_sql: Executed SQL query string.
            execution_result: ExecutionResult produced by ControlledExecutionEngine.
            table_profile: Active TableProfile providing grounded schema and statistics.
            available_tables: Optional list of all registered database tables.

        Returns:
            SemanticValidationResult containing all 8 check details and overall status.
        """
        intent = step.structured_intent if step else None

        # Missing executed SQL
        if not executed_sql or not executed_sql.strip():
            return self._build_inconclusive_result(
                reason="executed_sql_missing",
                explanation="No SQL query string was provided for semantic validation.",
                intent=intent,
            )

        # Parse SQL using SQLGlot DuckDB dialect
        try:
            ast = sqlglot.parse_one(executed_sql, read="duckdb")
        except Exception as exc:
            return self._build_syntax_failure_result(
                error_msg=str(exc),
                executed_sql=executed_sql,
                intent=intent,
            )

        if not isinstance(ast, exp.Select):
            select_ast = ast.find(exp.Select)
            if not select_ast:
                return self._build_syntax_failure_result(
                    error_msg="Executed SQL is not a valid SELECT statement.",
                    executed_sql=executed_sql,
                    intent=intent,
                )
            ast = select_ast

        # Missing structured intent
        if intent is None:
            return self._build_inconclusive_result(
                reason="missing_structured_intent",
                explanation="Structured QuerySpecification intent is unavailable for this step.",
                intent=None,
                ast=ast,
            )

        # Check 8: Projection contract (stateless, returns detail and canonical view)
        proj_check, canonical_view = self.check_projection_contract(ast, intent, execution_result)

        # Evaluate the 8 canonical checks
        checks: list[SemanticCheckDetail] = [
            self.check_table_grounding(ast, intent, table_profile, available_tables),
            self.check_column_grounding(ast, intent, table_profile),
            self.check_aggregation_level(ast, intent),
            self.check_metric_correctness(ast, intent, execution_result, table_profile),
            self.check_filter_semantics(ast, intent),
            self.check_join_integrity(ast, intent, table_profile),
            self.check_ordering_and_ranking(ast, intent, table_profile),
            proj_check,
        ]

        is_grounded = intent.is_grounded
        reliability = getattr(intent, "intent_reliability", IntentReliability.HIGH)

        overall_status, overall_passed, diagnostic = resolve_semantic_validation_status(
            checks=checks,
            is_schema_grounded=is_grounded,
            intent_reliability=reliability,
        )

        metrics_cov = tuple(m.name for m in intent.metrics)
        dims_cov = tuple(intent.dimensions)
        filters_cov = tuple(f.column for f in intent.filters)

        return SemanticValidationResult(
            passed=overall_passed,
            status=overall_status,
            message=diagnostic or "Semantic validation verified.",
            metrics_covered=metrics_cov,
            dimensions_covered=dims_cov,
            filters_covered=filters_cov,
            temporal_boundary_valid=True,
            inconclusive_reason=diagnostic
            if overall_status == ValidationStatus.INCONCLUSIVE
            else None,
            checks=tuple(checks),
            canonical_projection=canonical_view,
            intent_reliability=reliability,
        )

    # -------------------------------------------------------------------------
    # 1. check_table_grounding
    # -------------------------------------------------------------------------
    def check_table_grounding(
        self,
        ast: exp.Expression,
        intent: QuerySpecification,
        profile: TableProfile | None,
        available_tables: Sequence[str] | None,
    ) -> SemanticCheckDetail:
        """Check 1: Verify all referenced physical tables exist in DB and match grounded intent."""
        # 1. Extract defined Common Table Expressions (CTEs)
        cte_names: set[str] = set()
        with_node = ast.find(exp.With)
        if with_node:
            for cte in with_node.find_all(exp.CTE):
                if cte.alias:
                    cte_names.add(cte.alias.lower())

        # 2. Extract physical table references (exclude CTE aliases)
        all_table_nodes = list(ast.find_all(exp.Table))
        physical_tables = [
            t.name.lower() for t in all_table_nodes if t.name and t.name.lower() not in cte_names
        ]
        expected_table = intent.table_name.lower() if intent.table_name else None

        known_tables = set(available_tables) if available_tables else set()
        if profile and profile.table_name:
            known_tables.add(profile.table_name.lower())
        if expected_table:
            known_tables.add(expected_table)

        intent_ev = {"expected_table": expected_table}
        ast_ev = {"physical_tables": physical_tables, "cte_names": sorted(list(cte_names))}

        if not physical_tables and not cte_names:
            return SemanticCheckDetail(
                check_name="check_table_grounding",
                status=ValidationStatus.FAILED,
                passed=False,
                diagnostic_code="no_table_referenced",
                explanation="SQL statement does not reference any FROM table.",
                intent_evidence=intent_ev,
                ast_evidence=ast_ev,
            )

        # Check for unknown / unauthorized physical tables (including those inside CTEs)
        unknown = [t for t in physical_tables if known_tables and t not in known_tables]
        if unknown:
            return SemanticCheckDetail(
                check_name="check_table_grounding",
                status=ValidationStatus.FAILED,
                passed=False,
                diagnostic_code="unauthorized_table_referenced",
                explanation=f"SQL references unknown or unauthorized table(s): {unknown}.",
                intent_evidence=intent_ev,
                ast_evidence=ast_ev,
            )

        # Verify target table presence among physical tables (or known tables)
        if expected_table and expected_table not in physical_tables:
            return SemanticCheckDetail(
                check_name="check_table_grounding",
                status=ValidationStatus.FAILED,
                passed=False,
                diagnostic_code="target_table_missing",
                explanation=(
                    f"Expected table '{expected_table}' not referenced in physical "
                    f"SQL tables: {physical_tables}."
                ),
                intent_evidence=intent_ev,
                ast_evidence=ast_ev,
            )

        return SemanticCheckDetail(
            check_name="check_table_grounding",
            status=ValidationStatus.VERIFIED,
            passed=True,
            diagnostic_code=None,
            explanation=f"All referenced physical tables {physical_tables} match grounded intent.",
            intent_evidence=intent_ev,
            ast_evidence=ast_ev,
        )

    # -------------------------------------------------------------------------
    # 2. check_column_grounding
    # -------------------------------------------------------------------------
    def check_column_grounding(
        self,
        ast: exp.Expression,
        intent: QuerySpecification,
        profile: TableProfile | None,
    ) -> SemanticCheckDetail:
        """Check 2: Verify all referenced columns exist in the active schema."""
        if profile is None or not profile.columns:
            return SemanticCheckDetail(
                check_name="check_column_grounding",
                status=ValidationStatus.INCONCLUSIVE,
                passed=False,
                diagnostic_code="missing_schema_profile",
                explanation="TableProfile is unavailable; cannot verify column existence.",
                intent_evidence={"is_grounded": intent.is_grounded},
                ast_evidence={},
            )

        profile_cols = {c.lower() for c in profile.columns.keys()}
        # Extract column references (excluding aliases and star)
        referenced_cols: set[str] = set()
        for col_node in ast.find_all(exp.Column):
            name = col_node.name
            if name and name != "*":
                referenced_cols.add(name.lower())

        # Collect SELECT alias names to distinguish aliases from real columns
        select_aliases: set[str] = set()
        for select_exp in ast.find_all(exp.Alias):
            if select_exp.alias:
                select_aliases.add(select_exp.alias.lower())

        unknown_cols = [
            c for c in referenced_cols if c not in profile_cols and c not in select_aliases
        ]

        intent_ev = {"schema_columns": sorted(list(profile_cols))}
        ast_ev = {"referenced_columns": sorted(list(referenced_cols))}

        if getattr(intent, "unsupported_entities", None):
            unsupp_str = ", ".join(intent.unsupported_entities)
            t_name = profile.table_name if profile else "unknown"
            return SemanticCheckDetail(
                check_name="check_column_grounding",
                status=ValidationStatus.INCONCLUSIVE,
                passed=False,
                diagnostic_code="unsupported_entity_requested",
                explanation=(
                    f"Requested entity/attribute '{unsupp_str}' is absent from schema "
                    f"in table `{t_name}` and cannot be grounded."
                ),
                intent_evidence={
                    "is_grounded": False,
                    "unsupported_entities": intent.unsupported_entities,
                },
                ast_evidence={"referenced_columns": sorted(list(referenced_cols))},
            )

        if unknown_cols:
            return SemanticCheckDetail(
                check_name="check_column_grounding",
                status=ValidationStatus.FAILED,
                passed=False,
                diagnostic_code="unknown_column_referenced",
                explanation=f"SQL references columns that do not exist in schema: {unknown_cols}.",
                intent_evidence=intent_ev,
                ast_evidence=ast_ev,
            )

        # Check semantic grounding evidence against natural language question
        raw_q = intent.raw_question if intent else ""
        unsupported_concepts: list[str] = []
        if raw_q and profile:
            unsupported_concepts = detect_ungrounded_entities(raw_q, profile)
        if getattr(intent, "unsupported_entities", None):
            for u in intent.unsupported_entities:
                if u not in unsupported_concepts:
                    unsupported_concepts.append(u)

        if unsupported_concepts:
            unsupp_str = ", ".join(unsupported_concepts)
            t_name = profile.table_name if profile else "unknown"
            return SemanticCheckDetail(
                check_name="check_column_grounding",
                status=ValidationStatus.INCONCLUSIVE,
                passed=False,
                diagnostic_code="unsupported_entity_column_substitution",
                explanation=(
                    f"Requested entity/attribute '{unsupp_str}' is absent from schema "
                    f"in table `{t_name}` and cannot be grounded."
                ),
                intent_evidence={
                    "is_grounded": False,
                    "unsupported_entities": unsupported_concepts,
                },
                ast_evidence={"referenced_columns": sorted(list(referenced_cols))},
            )

        return SemanticCheckDetail(
            check_name="check_column_grounding",
            status=ValidationStatus.VERIFIED,
            passed=True,
            diagnostic_code=None,
            explanation="All referenced columns exist in the registered schema.",
            intent_evidence=intent_ev,
            ast_evidence=ast_ev,
        )

    # -------------------------------------------------------------------------
    # 3. check_aggregation_level
    # -------------------------------------------------------------------------
    def check_aggregation_level(
        self,
        ast: exp.Expression,
        intent: QuerySpecification,
    ) -> SemanticCheckDetail:
        """Check 3: Verify GROUP BY expressions match expected grouping dimensions."""
        group_by = ast.find(exp.Group)
        sql_group_cols: list[str] = []
        if group_by:
            for g_expr in group_by.expressions:
                if isinstance(g_expr, exp.Column):
                    sql_group_cols.append(g_expr.name.lower())
                else:
                    sql_group_cols.append(g_expr.sql().lower())

        # Collect SELECT aliases and their expressions for equivalence
        select_alias_map: dict[str, str] = {}
        for alias_node in ast.find_all(exp.Alias):
            if alias_node.alias:
                select_alias_map[alias_node.alias.lower()] = alias_node.this.sql().lower()

        # Build expected dimensions and their acceptable representations
        expected_dim_map: dict[str, set[str]] = {}
        covered_cols: set[str] = set()
        if intent.dimension_specs:
            for ds in intent.dimension_specs:
                forms = {ds.name.lower(), ds.expression.lower()}
                if ds.name.lower() in select_alias_map:
                    forms.add(select_alias_map[ds.name.lower()])
                expected_dim_map[ds.name.lower()] = forms
                covered_cols.add(ds.name.lower())
                try:
                    expr_ast = sqlglot.parse_one(ds.expression, read="duckdb")
                    for c in expr_ast.find_all(exp.Column):
                        if c.name:
                            covered_cols.add(c.name.lower())
                except Exception:
                    covered_cols.add(ds.expression.lower())

        for d in intent.dimensions:
            d_l = d.lower()
            if d_l not in covered_cols and d_l not in expected_dim_map:
                forms = {d_l}
                if d_l in select_alias_map:
                    forms.add(select_alias_map[d_l])
                expected_dim_map[d_l] = forms

        intent_ev = {
            "is_scalar": intent.is_scalar(),
            "expected_dimensions": list(expected_dim_map.keys()),
        }
        ast_ev = {"sql_group_by": sql_group_cols}

        # Case 1: Expected scalar aggregation (no dimensions)
        if intent.is_scalar():
            if sql_group_cols:
                return SemanticCheckDetail(
                    check_name="check_aggregation_level",
                    status=ValidationStatus.FAILED,
                    passed=False,
                    diagnostic_code="scalar_query_has_group_by",
                    explanation=(
                        f"Expected scalar aggregation, but SQL has GROUP BY: {sql_group_cols}."
                    ),
                    intent_evidence=intent_ev,
                    ast_evidence=ast_ev,
                )
            return SemanticCheckDetail(
                check_name="check_aggregation_level",
                status=ValidationStatus.VERIFIED,
                passed=True,
                diagnostic_code=None,
                explanation="Scalar aggregation verified without GROUP BY.",
                intent_evidence=intent_ev,
                ast_evidence=ast_ev,
            )

        # Case 2: Grouped query
        if not sql_group_cols and expected_dim_map:
            return SemanticCheckDetail(
                check_name="check_aggregation_level",
                status=ValidationStatus.FAILED,
                passed=False,
                diagnostic_code="grouped_query_missing_group_by",
                explanation=(
                    f"Query specifies dimensions {list(expected_dim_map.keys())}, "
                    "but SQL has no GROUP BY clause."
                ),
                intent_evidence=intent_ev,
                ast_evidence=ast_ev,
            )

        # Check that each expected dimension is present in at least one acceptable form
        missing_dims: list[str] = []
        for dim_name, acceptable in expected_dim_map.items():
            if not any(g in acceptable for g in sql_group_cols):
                missing_dims.append(dim_name)

        if missing_dims:
            return SemanticCheckDetail(
                check_name="check_aggregation_level",
                status=ValidationStatus.FAILED,
                passed=False,
                diagnostic_code="missing_group_by_dimension",
                explanation=f"SQL GROUP BY is missing required dimensions: {missing_dims}.",
                intent_evidence=intent_ev,
                ast_evidence=ast_ev,
            )

        # Check for unexpected grouping expressions
        all_acceptable = set()
        for forms in expected_dim_map.values():
            all_acceptable.update(forms)

        unexpected_dims = [g for g in sql_group_cols if g not in all_acceptable]
        if unexpected_dims:
            return SemanticCheckDetail(
                check_name="check_aggregation_level",
                status=ValidationStatus.FAILED,
                passed=False,
                diagnostic_code="unexpected_group_by_dimension",
                explanation=f"SQL GROUP BY groups by unexpected dimensions: {unexpected_dims}.",
                intent_evidence=intent_ev,
                ast_evidence=ast_ev,
            )

        return SemanticCheckDetail(
            check_name="check_aggregation_level",
            status=ValidationStatus.VERIFIED,
            passed=True,
            diagnostic_code=None,
            explanation=f"Aggregation level matches dimensions: {list(expected_dim_map.keys())}.",
            intent_evidence=intent_ev,
            ast_evidence=ast_ev,
        )

    # -------------------------------------------------------------------------
    # 4. check_metric_correctness
    # -------------------------------------------------------------------------
    def check_metric_correctness(
        self,
        ast: exp.Expression,
        intent: QuerySpecification,
        execution_result: ExecutionResult | None,
        profile: TableProfile | None,
    ) -> SemanticCheckDetail:
        """Check 4: Verify metric aggregation formulas, output types, and conversion policies."""
        if getattr(intent, "unsupported_entities", None) or not getattr(
            intent, "is_grounded", True
        ):
            unsupp_str = (
                ", ".join(intent.unsupported_entities)
                if getattr(intent, "unsupported_entities", None)
                else "unsupported entity"
            )
            return SemanticCheckDetail(
                check_name="check_metric_correctness",
                status=ValidationStatus.INCONCLUSIVE,
                passed=False,
                diagnostic_code="unsupported_entity_metric",
                explanation=(
                    f"Cannot verify metric correctness for ungrounded entity: {unsupp_str}."
                ),
                intent_evidence={
                    "unsupported_entities": getattr(intent, "unsupported_entities", [])
                },
                ast_evidence={},
            )

        if not intent.metrics:
            return SemanticCheckDetail(
                check_name="check_metric_correctness",
                status=ValidationStatus.INCONCLUSIVE,
                passed=False,
                diagnostic_code="no_metrics_specified",
                explanation="No metrics were defined in structured intent.",
                intent_evidence={},
                ast_evidence={},
            )

        select_expressions = ast.expressions
        intent_ev = {"metrics": [m.model_dump() for m in intent.metrics]}
        ast_ev = {"select_expressions": [e.sql() for e in select_expressions]}

        for metric in intent.metrics:
            # 4a. AOV contract verification
            if metric.aggregation == AggregationType.AOV:
                aov_found = False
                for expr_node in select_expressions:
                    sql_text = expr_node.sql().upper()
                    if "/" in sql_text and "SUM" in sql_text and "COUNT" in sql_text:
                        denom = (metric.denominator_column or "order_id").lower()
                        if denom not in AOV_DENOMINATOR_WHITELIST:
                            return SemanticCheckDetail(
                                check_name="check_metric_correctness",
                                status=ValidationStatus.FAILED,
                                passed=False,
                                diagnostic_code="aov_unauthorized_denominator",
                                explanation=(
                                    f"AOV denominator column '{denom}' not in approved whitelist: "
                                    f"{AOV_DENOMINATOR_WHITELIST}."
                                ),
                                intent_evidence=intent_ev,
                                ast_evidence=ast_ev,
                            )
                        aov_found = True
                        break
                if not aov_found:
                    return SemanticCheckDetail(
                        check_name="check_metric_correctness",
                        status=ValidationStatus.FAILED,
                        passed=False,
                        diagnostic_code="aov_formula_mismatch",
                        explanation=(
                            "AOV metric requires SUM(sales) / COUNT(DISTINCT order_id) "
                            "formula structure."
                        ),
                        intent_evidence=intent_ev,
                        ast_evidence=ast_ev,
                    )

            # 4b. Standard aggregation verification
            elif metric.aggregation in (
                AggregationType.SUM,
                AggregationType.AVG,
                AggregationType.MIN,
                AggregationType.MAX,
                AggregationType.COUNT,
                AggregationType.DISTINCT_COUNT,
            ):
                agg_name = (
                    "COUNT"
                    if metric.aggregation in (AggregationType.COUNT, AggregationType.DISTINCT_COUNT)
                    else metric.aggregation.value
                )
                agg_nodes = list(ast.find_all(getattr(exp, agg_name.capitalize(), exp.Anonymous)))
                if not agg_nodes:
                    agg_nodes = [
                        f for f in ast.find_all(exp.Anonymous) if f.name.upper() == agg_name
                    ]

                if not agg_nodes:
                    return SemanticCheckDetail(
                        check_name="check_metric_correctness",
                        status=ValidationStatus.FAILED,
                        passed=False,
                        diagnostic_code="missing_aggregate_function",
                        explanation=f"Required aggregate function {agg_name} not found in SQL.",
                        intent_evidence=intent_ev,
                        ast_evidence=ast_ev,
                    )

            # 4c. Conversion policy and integer cast verification (Revision 16 rules)
            cast_nodes = list(ast.find_all(exp.Cast))
            has_int_cast = any(
                c.to.this
                in (
                    exp.DataType.Type.INT,
                    exp.DataType.Type.BIGINT,
                    exp.DataType.Type.SMALLINT,
                    exp.DataType.Type.TINYINT,
                )
                or str(c.to).upper()
                in (
                    "INT",
                    "INTEGER",
                    "BIGINT",
                    "SMALLINT",
                    "TINYINT",
                    "INT4",
                    "INT8",
                    "INT2",
                    "INT1",
                )
                for c in cast_nodes
            )

            # Rule 6: Explicit FLOAT requirement with integer cast -> FAILED
            if metric.output_type == OutputTypeRequirement.FLOAT and has_int_cast:
                return SemanticCheckDetail(
                    check_name="check_metric_correctness",
                    status=ValidationStatus.FAILED,
                    passed=False,
                    diagnostic_code="output_type_violation",
                    explanation=(
                        f"Metric '{metric.name}' requires FLOAT output type, "
                        "but SQL applies integer cast."
                    ),
                    intent_evidence=intent_ev,
                    ast_evidence=ast_ev,
                )

            # Rule 7: ROUND_HALF_EVEN unsupported in DuckDB
            if metric.conversion_policy == ConversionPolicy.ROUND_HALF_EVEN:
                return SemanticCheckDetail(
                    check_name="check_metric_correctness",
                    status=ValidationStatus.INCONCLUSIVE,
                    passed=False,
                    diagnostic_code="unsupported_rounding_primitive",
                    explanation=(
                        "ROUND_HALF_EVEN is not natively supported by DuckDB SQL; "
                        "conversion cannot be verified."
                    ),
                    intent_evidence=intent_ev,
                    ast_evidence=ast_ev,
                )

            # Determine whether result value is provably integral across scalar or tabular rows
            val_is_integral: bool | None = None
            if execution_result and execution_result.success:
                if execution_result.is_scalar:
                    if execution_result.scalar_value is not None:
                        try:
                            f_val = float(execution_result.scalar_value)
                            if math.isnan(f_val):
                                # IEEE NaN is not integral
                                val_is_integral = False
                            else:
                                val_is_integral = f_val % 1.0 == 0.0
                        except (ValueError, TypeError):
                            val_is_integral = None
                    else:
                        # SQL NULL cannot prove integrality
                        val_is_integral = None
                elif execution_result.raw_rows is not None and execution_result.columns:
                    col_names_lower = [c.lower() for c in execution_result.columns]
                    m_name_lower = metric.name.lower()
                    col_idx: int | None = None
                    if m_name_lower in col_names_lower:
                        col_idx = col_names_lower.index(m_name_lower)
                    elif metric.column and metric.column.lower() in col_names_lower:
                        col_idx = col_names_lower.index(metric.column.lower())
                    elif len(execution_result.columns) == 1:
                        col_idx = 0
                    else:
                        col_idx = len(execution_result.columns) - 1

                    if col_idx is not None and execution_result.raw_rows:
                        has_non_null = False
                        all_integral = True
                        for row in execution_result.raw_rows:
                            if col_idx < len(row):
                                cell = row[col_idx]
                                if cell is not None:
                                    try:
                                        f_cell = float(cell)
                                        has_non_null = True
                                        if math.isnan(f_cell) or f_cell % 1.0 != 0.0:
                                            all_integral = False
                                            break
                                    except (ValueError, TypeError):
                                        pass
                        if has_non_null:
                            val_is_integral = all_integral
                        else:
                            val_is_integral = None

            # Revision 16 Conversion Policy Evaluation
            if has_int_cast:
                # Rule: TRUNCATE conversion policy
                if metric.conversion_policy == ConversionPolicy.TRUNCATE:
                    trunc_nodes = list(ast.find_all(exp.Trunc)) + [
                        f
                        for f in ast.find_all(exp.Anonymous)
                        if f.name.upper() in ("TRUNC", "TRUNCATE")
                    ]
                    if not trunc_nodes:
                        if val_is_integral is False:
                            return SemanticCheckDetail(
                                check_name="check_metric_correctness",
                                status=ValidationStatus.FAILED,
                                passed=False,
                                diagnostic_code="unauthorized_conversion_applied",
                                explanation=(
                                    "User specified TRUNCATE policy, but SQL uses CAST "
                                    "which rounds half away from zero."
                                ),
                                intent_evidence=intent_ev,
                                ast_evidence=ast_ev,
                            )
                        elif val_is_integral is None:
                            return SemanticCheckDetail(
                                check_name="check_metric_correctness",
                                status=ValidationStatus.INCONCLUSIVE,
                                passed=False,
                                diagnostic_code="unprovable_integrality",
                                explanation=(
                                    "Cannot verify TRUNCATE conversion because integrality "
                                    "is unprovable from available evidence."
                                ),
                                intent_evidence=intent_ev,
                                ast_evidence=ast_ev,
                            )

                # Rule 2: INTEGER output type without conversion policy
                elif (
                    metric.output_type == OutputTypeRequirement.INTEGER
                    and metric.conversion_policy == ConversionPolicy.NONE
                ):
                    if val_is_integral is False:
                        return SemanticCheckDetail(
                            check_name="check_metric_correctness",
                            status=ValidationStatus.INCONCLUSIVE,
                            passed=False,
                            diagnostic_code="ambiguous_integer_conversion",
                            explanation=(
                                "Non-integral result cast to INTEGER without explicit rounding "
                                "or truncation policy."
                            ),
                            intent_evidence=intent_ev,
                            ast_evidence=ast_ev,
                        )
                    elif val_is_integral is None:
                        return SemanticCheckDetail(
                            check_name="check_metric_correctness",
                            status=ValidationStatus.INCONCLUSIVE,
                            passed=False,
                            diagnostic_code="unprovable_integrality",
                            explanation=(
                                "Result integrality cannot be proven; integer conversion is "
                                "ambiguous without explicit policy."
                            ),
                            intent_evidence=intent_ev,
                            ast_evidence=ast_ev,
                        )

                # Rule: UNSPECIFIED output type with integer cast
                elif (
                    metric.output_type == OutputTypeRequirement.UNSPECIFIED
                    and metric.conversion_policy == ConversionPolicy.NONE
                ):
                    if val_is_integral is False:
                        return SemanticCheckDetail(
                            check_name="check_metric_correctness",
                            status=ValidationStatus.FAILED,
                            passed=False,
                            diagnostic_code="unauthorized_conversion_applied",
                            explanation=(
                                "SQL applied unauthorized integer cast truncating continuous "
                                "metric value."
                            ),
                            intent_evidence=intent_ev,
                            ast_evidence=ast_ev,
                        )
                    elif val_is_integral is None:
                        return SemanticCheckDetail(
                            check_name="check_metric_correctness",
                            status=ValidationStatus.INCONCLUSIVE,
                            passed=False,
                            diagnostic_code="unprovable_integrality",
                            explanation=(
                                "Cannot prove whether integer cast preserves continuous metric "
                                "precision from available evidence."
                            ),
                            intent_evidence=intent_ev,
                            ast_evidence=ast_ev,
                        )

        return SemanticCheckDetail(
            check_name="check_metric_correctness",
            status=ValidationStatus.VERIFIED,
            passed=True,
            diagnostic_code=None,
            explanation="Metric aggregation and conversion policies verified.",
            intent_evidence=intent_ev,
            ast_evidence=ast_ev,
        )

    # -------------------------------------------------------------------------
    # 5. check_filter_semantics
    # -------------------------------------------------------------------------
    def check_filter_semantics(
        self,
        ast: exp.Expression,
        intent: QuerySpecification,
    ) -> SemanticCheckDetail:
        """Check 5: Verify filter columns, operators, boundaries, and boolean structure."""
        where_node = ast.find(exp.Where)

        intent_ev = {
            "filters": [f.model_dump() for f in intent.filters],
            "time_constraints": [t.model_dump() for t in intent.time_constraints],
        }
        ast_ev = {"where_clause": where_node.sql() if where_node else ""}

        has_requirements = bool(intent.filters or intent.time_constraints)
        if has_requirements and not where_node:
            return SemanticCheckDetail(
                check_name="check_filter_semantics",
                status=ValidationStatus.FAILED,
                passed=False,
                diagnostic_code="missing_filter_predicate",
                explanation=(
                    "Intent requires filters or time constraints, but SQL has no WHERE clause."
                ),
                intent_evidence=intent_ev,
                ast_evidence=ast_ev,
            )

        if not has_requirements:
            return SemanticCheckDetail(
                check_name="check_filter_semantics",
                status=ValidationStatus.VERIFIED,
                passed=True,
                diagnostic_code=None,
                explanation="No filter predicates or time constraints required.",
                intent_evidence=intent_ev,
                ast_evidence=ast_ev,
            )

        def _unwrap_paren(node: exp.Expression) -> exp.Expression:
            while isinstance(node, exp.Paren):
                node = node.this
            return node

        def _extract_top_conjuncts(node: exp.Expression) -> list[exp.Expression]:
            node = _unwrap_paren(node)
            if isinstance(node, exp.And):
                return _extract_top_conjuncts(node.left) + _extract_top_conjuncts(node.right)
            return [node]

        def _extract_predicate_value(val_node: exp.Expression | None) -> tuple[Any, bool]:
            if val_node is None:
                return (None, False)
            if isinstance(val_node, exp.Null):
                return (None, False)
            if isinstance(val_node, exp.Boolean):
                return (val_node.this, False)
            if isinstance(val_node, exp.Literal):
                return (val_node.this, False)
            if isinstance(val_node, exp.Cast):
                inner = val_node.this
                if isinstance(inner, exp.Literal):
                    return (inner.this, False)
                elif isinstance(inner, (exp.Null, exp.Boolean)):
                    return (inner.this, False)
                else:
                    return (val_node.sql(), True)
            if isinstance(val_node, (exp.Date, exp.TsOrDsToDate, exp.StrToDate)):
                if isinstance(val_node.this, exp.Literal):
                    return (val_node.this.this, False)
                return (val_node.sql(), True)
            if isinstance(
                val_node, (exp.CurrentDate, exp.CurrentTimestamp, exp.Anonymous, exp.Subquery)
            ):
                return (val_node.sql(), True)
            return (val_node.sql().strip("'\""), False)

        def _extract_comp(c: exp.Expression) -> tuple[str, str, Any, bool] | None:
            c = _unwrap_paren(c)
            if not isinstance(
                c, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.In)
            ):
                return None
            op_map = {
                exp.EQ: "=",
                exp.NEQ: "!=",
                exp.GTE: ">=",
                exp.GT: ">",
                exp.LTE: "<=",
                exp.LT: "<",
                exp.Like: "LIKE",
                exp.In: "IN",
            }
            op_str = op_map.get(type(c), c.key.upper())
            left_cols = [col.name.lower() for col in c.left.find_all(exp.Column) if col.name]
            right_cols = (
                [col.name.lower() for col in c.right.find_all(exp.Column) if col.name]
                if hasattr(c, "right")
                else []
            )
            if left_cols and not right_cols:
                col_name = left_cols[0]
                val_node = getattr(c, "right", None)
            elif right_cols and not left_cols:
                col_name = right_cols[0]
                val_node = getattr(c, "left", None)
            elif left_cols:
                col_name = left_cols[0]
                val_node = getattr(c, "right", None)
            else:
                return None
            lit_val, is_ambig = _extract_predicate_value(val_node)
            return (col_name, op_str, lit_val, is_ambig)

        # Extract top-level conjuncts from WHERE clause
        top_conjuncts = _extract_top_conjuncts(where_node.this)
        top_predicates: list[tuple[str, str, Any, bool]] = []
        unresolvable_top_conjuncts: list[exp.Expression] = []

        for conj in top_conjuncts:
            p = _extract_comp(conj)
            if p is not None:
                top_predicates.append(p)
            else:
                unresolvable_top_conjuncts.append(conj)

        # Extract all comparisons across entire WHERE clause
        all_comp_nodes = list(
            where_node.find_all(
                (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.In)
            )
        )
        all_parsed: list[tuple[str, str, Any, bool]] = []
        for c in all_comp_nodes:
            p = _extract_comp(c)
            if p is not None:
                all_parsed.append(p)

        # Identify comparisons that occur inside OR disjunctions
        or_nodes = list(where_node.find_all(exp.Or))
        disjoined_parsed: list[tuple[str, str, Any, bool]] = []
        for or_n in or_nodes:
            for c in or_n.find_all(
                (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.In)
            ):
                p = _extract_comp(c)
                if p is not None:
                    disjoined_parsed.append(p)

        # Check explicit filter conditions
        op_aliases = {"==": "=", "<>": "!="}
        for f in intent.filters:
            col_target = f.column.lower()
            expected_op = f.operator.upper() if f.operator else "="
            expected_op = op_aliases.get(expected_op, expected_op)

            # Look for top-level conjunct match
            top_matches = [
                p
                for p in top_predicates
                if p[0] == col_target and op_aliases.get(p[1], p[1]) == expected_op
            ]
            top_val_matches = (
                [p for p in top_matches if _values_equal(p[2], f.value)]
                if f.value is not None
                else top_matches
            )

            if top_val_matches:
                if any(p[3] for p in top_val_matches):
                    return SemanticCheckDetail(
                        check_name="check_filter_semantics",
                        status=ValidationStatus.INCONCLUSIVE,
                        passed=False,
                        diagnostic_code="unresolvable_filter_expression",
                        explanation=(
                            f"Filter on '{f.column}' uses unresolvable or dynamic expression."
                        ),
                        intent_evidence=intent_ev,
                        ast_evidence=ast_ev,
                    )
                # Satisfied conjunctively at top level
                continue

            # Diagnose failure cause:
            # Check if condition is inside an OR disjunction (where conjunctive AND was required)
            disjoined_matches = [p for p in disjoined_parsed if p[0] == col_target]
            if disjoined_matches:
                authorized_or = False
                if expected_op == "IN" and isinstance(f.value, (list, tuple, set)):
                    val_set = {str(v).lower() for v in f.value}
                    if all(str(p[2]).lower() in val_set for p in disjoined_matches):
                        authorized_or = True

                if not authorized_or:
                    return SemanticCheckDetail(
                        check_name="check_filter_semantics",
                        status=ValidationStatus.FAILED,
                        passed=False,
                        diagnostic_code="filter_conjunction_mismatch",
                        explanation=(
                            f"Required filter on '{f.column}' is conjoined via OR instead "
                            f"of required conjunctive AND semantics."
                        ),
                        intent_evidence=intent_ev,
                        ast_evidence=ast_ev,
                    )

            # Check if column is absent from WHERE entirely
            all_col_matches = [p for p in all_parsed if p[0] == col_target]
            if not all_col_matches:
                if unresolvable_top_conjuncts:
                    return SemanticCheckDetail(
                        check_name="check_filter_semantics",
                        status=ValidationStatus.INCONCLUSIVE,
                        passed=False,
                        diagnostic_code="unresolvable_filter_logic",
                        explanation=(
                            f"Expected filter on '{f.column}' could not be proven due to "
                            f"complex/unresolvable WHERE clause structure."
                        ),
                        intent_evidence=intent_ev,
                        ast_evidence=ast_ev,
                    )
                return SemanticCheckDetail(
                    check_name="check_filter_semantics",
                    status=ValidationStatus.FAILED,
                    passed=False,
                    diagnostic_code="missing_filter_predicate",
                    explanation=(
                        f"Expected filter on column '{f.column}' not found in WHERE clause."
                    ),
                    intent_evidence=intent_ev,
                    ast_evidence=ast_ev,
                )

            # Column is present in top-level conjuncts, but operator or value mismatched
            top_cols = [p for p in top_predicates if p[0] == col_target]
            if not top_matches:
                observed_ops = (
                    [p[1] for p in top_cols] if top_cols else [p[1] for p in all_col_matches]
                )
                return SemanticCheckDetail(
                    check_name="check_filter_semantics",
                    status=ValidationStatus.FAILED,
                    passed=False,
                    diagnostic_code="filter_operator_mismatch",
                    explanation=(
                        f"Filter on '{f.column}' expected operator '{f.operator}', "
                        f"but observed operator(s): {observed_ops}."
                    ),
                    intent_evidence=intent_ev,
                    ast_evidence=ast_ev,
                )

            observed_vals = [p[2] for p in top_matches]
            return SemanticCheckDetail(
                check_name="check_filter_semantics",
                status=ValidationStatus.FAILED,
                passed=False,
                diagnostic_code="filter_value_mismatch",
                explanation=(
                    f"Filter on '{f.column}' expected value '{f.value}', "
                    f"but observed value(s): {observed_vals}."
                ),
                intent_evidence=intent_ev,
                ast_evidence=ast_ev,
            )

        # Check time constraints (half-open intervals: col >= start AND col < end)
        for tc in intent.time_constraints:
            col_target = tc.column.lower()
            all_tc_cols = [p for p in all_parsed if p[0] == col_target]
            if not all_tc_cols:
                return SemanticCheckDetail(
                    check_name="check_filter_semantics",
                    status=ValidationStatus.FAILED,
                    passed=False,
                    diagnostic_code="missing_time_constraint",
                    explanation=(
                        f"Required time constraint on '{tc.column}' missing from WHERE clause."
                    ),
                    intent_evidence=intent_ev,
                    ast_evidence=ast_ev,
                )

            # Check if time constraint is disjoined via OR
            disjoined_tc = [p for p in disjoined_parsed if p[0] == col_target]
            if disjoined_tc and not [p for p in top_predicates if p[0] == col_target]:
                return SemanticCheckDetail(
                    check_name="check_filter_semantics",
                    status=ValidationStatus.FAILED,
                    passed=False,
                    diagnostic_code="filter_conjunction_mismatch",
                    explanation=(
                        f"Time constraint on '{tc.column}' is conjoined via OR instead "
                        f"of required conjunctive AND semantics."
                    ),
                    intent_evidence=intent_ev,
                    ast_evidence=ast_ev,
                )

            # Lower bound check
            if tc.start_time:
                expected_lower_op = ">="
                valid_lower = [
                    p
                    for p in top_predicates
                    if p[0] == col_target
                    and p[1] in (expected_lower_op, ">=")
                    and _values_equal(p[2], tc.start_time)
                ]
                if not valid_lower:
                    ambig_lower = [
                        p
                        for p in top_predicates
                        if p[0] == col_target and p[1] in (expected_lower_op, ">=") and p[3]
                    ]
                    if ambig_lower:
                        return SemanticCheckDetail(
                            check_name="check_filter_semantics",
                            status=ValidationStatus.INCONCLUSIVE,
                            passed=False,
                            diagnostic_code="unresolvable_date_expression",
                            explanation=(
                                f"Time constraint on '{tc.column}' uses unresolvable or dynamic "
                                f"lower bound expression: {ambig_lower[0][2]}."
                            ),
                            intent_evidence=intent_ev,
                            ast_evidence=ast_ev,
                        )
                    return SemanticCheckDetail(
                        check_name="check_filter_semantics",
                        status=ValidationStatus.FAILED,
                        passed=False,
                        diagnostic_code="time_boundary_mismatch",
                        explanation=(
                            f"Time constraint on '{tc.column}' requires lower bound >= "
                            f"'{tc.start_time}', not found in predicates: {top_predicates}."
                        ),
                        intent_evidence=intent_ev,
                        ast_evidence=ast_ev,
                    )

            # Upper bound check (half-open strictly requires '<')
            if tc.end_time:
                expected_upper_op = "<" if tc.is_half_open else "<="
                valid_upper = [
                    p
                    for p in top_predicates
                    if p[0] == col_target
                    and p[1] == expected_upper_op
                    and _values_equal(p[2], tc.end_time)
                ]
                if not valid_upper:
                    ambig_upper = [
                        p
                        for p in top_predicates
                        if p[0] == col_target and p[1] == expected_upper_op and p[3]
                    ]
                    if ambig_upper:
                        return SemanticCheckDetail(
                            check_name="check_filter_semantics",
                            status=ValidationStatus.INCONCLUSIVE,
                            passed=False,
                            diagnostic_code="unresolvable_date_expression",
                            explanation=(
                                f"Time constraint on '{tc.column}' uses unresolvable or dynamic "
                                f"upper bound expression: {ambig_upper[0][2]}."
                            ),
                            intent_evidence=intent_ev,
                            ast_evidence=ast_ev,
                        )
                    return SemanticCheckDetail(
                        check_name="check_filter_semantics",
                        status=ValidationStatus.FAILED,
                        passed=False,
                        diagnostic_code="time_boundary_mismatch",
                        explanation=(
                            f"Time constraint on '{tc.column}' requires upper bound "
                            f"{expected_upper_op} '{tc.end_time}', "
                            f"not found in top-level predicates: {top_predicates}."
                        ),
                        intent_evidence=intent_ev,
                        ast_evidence=ast_ev,
                    )

        return SemanticCheckDetail(
            check_name="check_filter_semantics",
            status=ValidationStatus.VERIFIED,
            passed=True,
            diagnostic_code=None,
            explanation="Filter predicates and temporal boundary semantics verified via AST.",
            intent_evidence=intent_ev,
            ast_evidence=ast_ev,
        )

    # -------------------------------------------------------------------------
    # 6. check_join_integrity
    # -------------------------------------------------------------------------
    def check_join_integrity(
        self,
        ast: exp.Expression,
        intent: QuerySpecification,
        profile: TableProfile | None,
    ) -> SemanticCheckDetail:
        """Check 6: Verify join conditions, metric-grain preservation, and join multiplication."""
        join_nodes = list(ast.find_all(exp.Join))
        intent_ev = {"table_name": intent.table_name}
        ast_ev = {"join_count": len(join_nodes), "joins": [j.sql() for j in join_nodes]}

        # If intent expects single table, check for unexpected joins
        if not join_nodes:
            return SemanticCheckDetail(
                check_name="check_join_integrity",
                status=ValidationStatus.VERIFIED,
                passed=True,
                diagnostic_code=None,
                explanation="Single table query verified without joins.",
                intent_evidence=intent_ev,
                ast_evidence=ast_ev,
            )

        # If joins are present, verify each join has an ON condition (prevent Cartesian products)
        for j in join_nodes:
            if not j.args.get("on") and not j.args.get("using"):
                return SemanticCheckDetail(
                    check_name="check_join_integrity",
                    status=ValidationStatus.FAILED,
                    passed=False,
                    diagnostic_code="cartesian_product_join",
                    explanation=(
                        f"JOIN clause '{j.sql()}' lacks ON/USING condition; "
                        "risk of Cartesian product."
                    ),
                    intent_evidence=intent_ev,
                    ast_evidence=ast_ev,
                )

        return SemanticCheckDetail(
            check_name="check_join_integrity",
            status=ValidationStatus.VERIFIED,
            passed=True,
            diagnostic_code=None,
            explanation=f"Verified {len(join_nodes)} join(s) with explicit join predicates.",
            intent_evidence=intent_ev,
            ast_evidence=ast_ev,
        )

    # -------------------------------------------------------------------------
    # 7. check_ordering_and_ranking
    # -------------------------------------------------------------------------
    def check_ordering_and_ranking(
        self,
        ast: exp.Expression,
        intent: QuerySpecification,
        profile: TableProfile | None,
    ) -> SemanticCheckDetail:
        """Check 7: Verify ORDER BY direction, LIMIT, tie handling, and superkey determinism."""
        order_node = ast.find(exp.Order)
        limit_node = ast.find(exp.Limit)

        sql_order_info: list[dict[str, Any]] = []
        sql_order_cols: list[tuple[str, str]] = []
        if order_node:
            for o_expr in order_node.expressions:
                direction = "DESC" if o_expr.args.get("desc") else "ASC"
                this_expr = o_expr.this
                raw_col = (
                    this_expr.name.lower()
                    if isinstance(this_expr, exp.Column)
                    else this_expr.sql().lower()
                )

                unqual_expr = this_expr.copy()
                for col in unqual_expr.find_all(exp.Column):
                    col.set("table", None)
                unqual_col = (
                    unqual_expr.name.lower()
                    if isinstance(unqual_expr, exp.Column)
                    else unqual_expr.sql().lower()
                )

                ref_cols = {c.name.lower() for c in this_expr.find_all(exp.Column) if c.name}
                agg_name = this_expr.key.lower() if isinstance(this_expr, exp.Func) else None

                sql_order_info.append(
                    {
                        "raw": raw_col,
                        "unqual": unqual_col,
                        "dir": direction,
                        "cols": ref_cols,
                        "agg": agg_name,
                        "expr": this_expr,
                    }
                )
                sql_order_cols.append((raw_col, direction))

        sql_limit = (
            int(limit_node.expression.this) if limit_node and limit_node.expression else None
        )

        ranking_spec = intent.ranking_spec
        mode = ranking_spec.mode if ranking_spec else RankingMode.NONE

        intent_ev = {
            "order_by": [
                o.model_dump() if hasattr(o, "model_dump") else str(o) for o in intent.order_by
            ]
            if intent.order_by
            else [],
            "limit": intent.limit,
            "ranking_mode": mode.value,
        }
        ast_ev = {"sql_order_by": sql_order_cols, "sql_limit": sql_limit}

        # 7a. Non-ranking query with no ordering requirement
        if not intent.order_by and intent.limit is None and mode == RankingMode.NONE:
            return SemanticCheckDetail(
                check_name="check_ordering_and_ranking",
                status=ValidationStatus.VERIFIED,
                passed=True,
                diagnostic_code=None,
                explanation="No ordering or ranking constraints required.",
                intent_evidence=intent_ev,
                ast_evidence=ast_ev,
            )

        # 7b. Scalar extremum (e.g. MAX(sales))
        if mode == RankingMode.SINGLE_EXTREMUM:
            return SemanticCheckDetail(
                check_name="check_ordering_and_ranking",
                status=ValidationStatus.VERIFIED,
                passed=True,
                diagnostic_code=None,
                explanation="Scalar extremum verified independent of row-level ties.",
                intent_evidence=intent_ev,
                ast_evidence=ast_ev,
            )

        # 7c. Top-N with ties (Layer 2 cannot verify window ties)
        if mode == RankingMode.TOP_N_WITH_TIES:
            return SemanticCheckDetail(
                check_name="check_ordering_and_ranking",
                status=ValidationStatus.INCONCLUSIVE,
                passed=False,
                diagnostic_code="unsupported_ranking_mode",
                explanation=(
                    "TOP_N_WITH_TIES cannot be independently validated without "
                    "window partition evidence."
                ),
                intent_evidence=intent_ev,
                ast_evidence=ast_ev,
            )

        # 7d. Primary Sort Key & Direction Verification
        expected_primary_col: str | None = None
        expected_primary_dir: str | None = None

        if ranking_spec and ranking_spec.mode not in (
            RankingMode.NONE,
            RankingMode.SINGLE_EXTREMUM,
        ):
            expected_primary_dir = ranking_spec.direction.value.upper()
            if ranking_spec.metric:
                expected_primary_col = ranking_spec.metric.lower()
            elif intent.metrics:
                expected_primary_col = intent.metrics[0].name.lower()
        elif intent.order_by:
            first_ob = intent.order_by[0]
            if hasattr(first_ob, "column"):
                expected_primary_col = first_ob.column.lower()
                expected_primary_dir = first_ob.direction.value.upper()
            else:
                parts = str(first_ob).split()
                expected_primary_col = parts[0].lower()
                expected_primary_dir = parts[1].upper() if len(parts) > 1 else "ASC"

        if expected_primary_dir is not None:
            if not sql_order_cols:
                return SemanticCheckDetail(
                    check_name="check_ordering_and_ranking",
                    status=ValidationStatus.FAILED,
                    passed=False,
                    diagnostic_code="missing_order_by",
                    explanation="Intent specifies ranking order, but SQL has no ORDER BY clause.",
                    intent_evidence=intent_ev,
                    ast_evidence=ast_ev,
                )

            primary_info = sql_order_info[0]
            actual_col = primary_info["raw"]
            actual_dir = primary_info["dir"]
            # Direction check
            if actual_dir.upper() != expected_primary_dir:
                return SemanticCheckDetail(
                    check_name="check_ordering_and_ranking",
                    status=ValidationStatus.FAILED,
                    passed=False,
                    diagnostic_code="ordering_direction_mismatch",
                    explanation=(
                        f"Expected primary sort direction '{expected_primary_dir}', "
                        f"observed '{actual_dir}'."
                    ),
                    intent_evidence=intent_ev,
                    ast_evidence=ast_ev,
                )

            # Column / Metric check
            if expected_primary_col:
                acceptable_primary = {expected_primary_col}
                for m in intent.metrics:
                    if m.name.lower() == expected_primary_col or (
                        m.column and m.column.lower() == expected_primary_col
                    ):
                        if m.column:
                            acceptable_primary.add(m.column.lower())
                        if m.aggregation:
                            acceptable_primary.add(
                                f"{m.aggregation.value.lower()}({m.column.lower()})"
                            )

                raw_match = primary_info["raw"] in acceptable_primary
                unqual_match = primary_info["unqual"] in acceptable_primary

                # Structural match for qualified aggregate functions
                structural_match = False
                for m in intent.metrics:
                    target_names = {m.name.lower()}
                    if m.column:
                        target_names.add(m.column.lower())
                    if expected_primary_col in target_names:
                        if m.aggregation and primary_info["agg"]:
                            if primary_info["agg"] == m.aggregation.value.lower():
                                if m.column and m.column.lower() in primary_info["cols"]:
                                    structural_match = True
                                    break
                        elif not primary_info["agg"]:
                            if m.column and m.column.lower() in primary_info["cols"]:
                                structural_match = True
                                break

                if not (raw_match or unqual_match or structural_match):
                    return SemanticCheckDetail(
                        check_name="check_ordering_and_ranking",
                        status=ValidationStatus.FAILED,
                        passed=False,
                        diagnostic_code="ordering_primary_metric_mismatch",
                        explanation=(
                            f"Expected primary sort on '{expected_primary_col}', "
                            f"observed '{actual_col}'."
                        ),
                        intent_evidence=intent_ev,
                        ast_evidence=ast_ev,
                    )

        # 7e. Check LIMIT match
        if intent.limit is not None:
            if sql_limit != intent.limit:
                return SemanticCheckDetail(
                    check_name="check_ordering_and_ranking",
                    status=ValidationStatus.FAILED,
                    passed=False,
                    diagnostic_code="limit_mismatch",
                    explanation=f"Expected LIMIT {intent.limit}, observed LIMIT {sql_limit}.",
                    intent_evidence=intent_ev,
                    ast_evidence=ast_ev,
                )

        # 7f. Check secondary sort keys in exact order
        if ranking_spec and ranking_spec.secondary_keys:
            sec_keys = ranking_spec.secondary_keys
            if len(sql_order_cols) < 1 + len(sec_keys):
                return SemanticCheckDetail(
                    check_name="check_ordering_and_ranking",
                    status=ValidationStatus.FAILED,
                    passed=False,
                    diagnostic_code="ordering_secondary_key_mismatch",
                    explanation=(
                        "SQL ORDER BY misses required secondary keys: "
                        f"{[k.column for k in sec_keys]}."
                    ),
                    intent_evidence=intent_ev,
                    ast_evidence=ast_ev,
                )
            for idx, sk in enumerate(sec_keys):
                sec_info = sql_order_info[1 + idx]
                actual_k = sec_info["raw"]
                actual_unqual = sec_info["unqual"]
                actual_dir = sec_info["dir"]
                sk_col = sk.column.lower()
                key_match = (
                    actual_k == sk_col or actual_unqual == sk_col or sk_col in sec_info["cols"]
                )
                if not key_match or actual_dir != sk.direction.value:
                    return SemanticCheckDetail(
                        check_name="check_ordering_and_ranking",
                        status=ValidationStatus.FAILED,
                        passed=False,
                        diagnostic_code="ordering_secondary_key_mismatch",
                        explanation=(
                            f"Secondary key {idx + 1} mismatch: "
                            f"expected {sk.column} {sk.direction.value}, "
                            f"observed {actual_k} {actual_dir}."
                        ),
                        intent_evidence=intent_ev,
                        ast_evidence=ast_ev,
                    )

        # 7g. Grouped query superkey proof: verify GROUP BY expressions match dimensions
        if intent.is_grouped():
            group_by_node = ast.find(exp.Group)
            group_exprs = (
                [g.sql().lower() for g in group_by_node.expressions] if group_by_node else []
            )

            # Check ungrounded transformations (e.g. YEAR(col) when day-grain col was specified)
            if intent.dimension_specs:
                for ds in intent.dimension_specs:
                    exp_sql = ds.expression.lower()
                    if exp_sql not in group_exprs and ds.name.lower() not in group_exprs:
                        return SemanticCheckDetail(
                            check_name="check_ordering_and_ranking",
                            status=ValidationStatus.INCONCLUSIVE,
                            passed=False,
                            diagnostic_code="ungrounded_dimension_transformation",
                            explanation=(
                                f"SQL grouping expression for dimension '{ds.name}' is ungrounded."
                            ),
                            intent_evidence=intent_ev,
                            ast_evidence=ast_ev,
                        )

            # Check if sort keys contain dimensions for determinism under ties
            order_col_names = {k["raw"] for k in sql_order_info} | {
                k["unqual"] for k in sql_order_info
            }
            for k in sql_order_info:
                order_col_names.update(k["cols"])
            expected_dim_names = {d.lower() for d in intent.dimensions}
            if intent.dimension_specs:
                expected_dim_names.update(ds.name.lower() for ds in intent.dimension_specs)

            if not expected_dim_names.issubset(order_col_names):
                # Superkey check applies to LIMIT 1 as well as LIMIT > 1
                if sql_limit is not None and sql_limit >= 1:
                    return SemanticCheckDetail(
                        check_name="check_ordering_and_ranking",
                        status=ValidationStatus.INCONCLUSIVE,
                        passed=False,
                        diagnostic_code="unproven_ranking_uniqueness",
                        explanation=(
                            "Sort keys do not contain all grouping dimensions; "
                            "ordering may be non-deterministic under ties."
                        ),
                        intent_evidence=intent_ev,
                        ast_evidence=ast_ev,
                    )

        return SemanticCheckDetail(
            check_name="check_ordering_and_ranking",
            status=ValidationStatus.VERIFIED,
            passed=True,
            diagnostic_code=None,
            explanation="Ordering, ranking limit, and determinism verified.",
            intent_evidence=intent_ev,
            ast_evidence=ast_ev,
        )

    # -------------------------------------------------------------------------
    # 8. check_projection_contract
    # -------------------------------------------------------------------------
    def check_projection_contract(
        self,
        ast: exp.Expression,
        intent: QuerySpecification,
        execution_result: ExecutionResult | None,
    ) -> tuple[SemanticCheckDetail, CanonicalProjectionView | None]:
        """Check 8: Validate projection contract using ProjectionNormalizer."""
        if getattr(intent, "unsupported_entities", None) or not getattr(
            intent, "is_grounded", True
        ):
            unsupp_str = (
                ", ".join(intent.unsupported_entities)
                if getattr(intent, "unsupported_entities", None)
                else "unsupported entity"
            )
            return (
                SemanticCheckDetail(
                    check_name="check_projection_contract",
                    status=ValidationStatus.INCONCLUSIVE,
                    passed=False,
                    diagnostic_code="unsupported_entity_projection",
                    explanation=(
                        f"Cannot establish projection contract for ungrounded entity: {unsupp_str}."
                    ),
                    intent_evidence={
                        "unsupported_entities": getattr(intent, "unsupported_entities", [])
                    },
                    ast_evidence={},
                ),
                None,
            )

        expected_dims: list[str] = []
        seen_dims: set[str] = set()
        covered_cols: set[str] = set()

        if intent.dimension_specs:
            for ds in intent.dimension_specs:
                ds_l = ds.name.lower()
                if ds_l not in seen_dims:
                    seen_dims.add(ds_l)
                    expected_dims.append(ds.name)
                covered_cols.add(ds_l)
                try:
                    expr_ast = sqlglot.parse_one(ds.expression, read="duckdb")
                    for c in expr_ast.find_all(exp.Column):
                        if c.name:
                            covered_cols.add(c.name.lower())
                except Exception:
                    covered_cols.add(ds.expression.lower())

        for d in intent.dimensions:
            d_l = d.lower()
            if d_l not in seen_dims and d_l not in covered_cols:
                seen_dims.add(d_l)
                expected_dims.append(d)
                covered_cols.add(d_l)

        expected_projections: list[str] = list(expected_dims)
        for m in intent.metrics:
            expected_projections.append(m.name)

        if not expected_projections:
            expected_projections = ["result"]

        proj_res: ProjectionNormalizationResult = self.projection_normalizer.normalize(
            expected_projections=expected_projections,
            execution_result=execution_result,
        )

        canonical_view = proj_res.canonical_view
        intent_ev = {"expected_projections": expected_projections}
        ast_ev = {
            "actual_columns": list(execution_result.columns)
            if execution_result and execution_result.columns
            else []
        }
        exp_vs_obs = {
            "status": proj_res.status.value,
            "permutation_map": list(proj_res.permutation_map),
            "has_canonical_view": canonical_view is not None,
        }

        if proj_res.status in (
            ProjectionNormalizationStatus.CANONICAL,
            ProjectionNormalizationStatus.PERMUTED,
        ):
            detail = SemanticCheckDetail(
                check_name="check_projection_contract",
                status=ValidationStatus.VERIFIED,
                passed=True,
                diagnostic_code=None,
                explanation=f"Projection contract verified with status {proj_res.status.value}.",
                intent_evidence=intent_ev,
                ast_evidence=ast_ev,
                expected_vs_observed=exp_vs_obs,
            )
            return detail, canonical_view

        if proj_res.status == ProjectionNormalizationStatus.CONTRADICTION:
            detail = SemanticCheckDetail(
                check_name="check_projection_contract",
                status=ValidationStatus.FAILED,
                passed=False,
                diagnostic_code="projection_contradiction",
                explanation=proj_res.diagnostic
                or "Projection output contradicts expected schema contract.",
                intent_evidence=intent_ev,
                ast_evidence=ast_ev,
                expected_vs_observed=exp_vs_obs,
            )
            return detail, None

        # UNSUPPORTED
        detail = SemanticCheckDetail(
            check_name="check_projection_contract",
            status=ValidationStatus.INCONCLUSIVE,
            passed=False,
            diagnostic_code="unsupported_projection_types",
            explanation=proj_res.diagnostic
            or "Projection output contains unsupported scalar types.",
            intent_evidence=intent_ev,
            ast_evidence=ast_ev,
            expected_vs_observed=exp_vs_obs,
        )
        return detail, None

    # -------------------------------------------------------------------------
    # Helper fallback builders
    # -------------------------------------------------------------------------
    def _build_inconclusive_result(
        self,
        reason: str,
        explanation: str,
        intent: QuerySpecification | None,
        ast: exp.Expression | None = None,
    ) -> SemanticValidationResult:
        """Construct a consistent INCONCLUSIVE result when prerequisites are missing."""
        checks = [
            SemanticCheckDetail(
                check_name=name,
                status=ValidationStatus.INCONCLUSIVE,
                passed=False,
                diagnostic_code=reason,
                explanation=explanation,
            )
            for name in sorted(list(CANONICAL_SEMANTIC_CHECKS))
        ]
        return SemanticValidationResult(
            passed=False,
            status=ValidationStatus.INCONCLUSIVE,
            message=explanation,
            inconclusive_reason=reason,
            checks=tuple(checks),
            intent_reliability=getattr(intent, "intent_reliability", IntentReliability.UNGROUNDED)
            if intent
            else IntentReliability.UNGROUNDED,
        )

    def _build_syntax_failure_result(
        self,
        error_msg: str,
        executed_sql: str,
        intent: QuerySpecification | None,
    ) -> SemanticValidationResult:
        """Construct a FAILED result when SQL syntax is malformed or unparseable."""
        checks = [
            SemanticCheckDetail(
                check_name=name,
                status=ValidationStatus.FAILED,
                passed=False,
                diagnostic_code="sql_syntax_unparseable",
                explanation=f"SQL syntax error: {error_msg}",
                ast_evidence={"executed_sql": executed_sql},
            )
            for name in sorted(list(CANONICAL_SEMANTIC_CHECKS))
        ]
        return SemanticValidationResult(
            passed=False,
            status=ValidationStatus.FAILED,
            message=f"SQLGlot parsing failed: {error_msg}",
            checks=tuple(checks),
            intent_reliability=getattr(intent, "intent_reliability", IntentReliability.HIGH)
            if intent
            else IntentReliability.HIGH,
        )
