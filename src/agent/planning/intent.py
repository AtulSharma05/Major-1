"""Structured question intent modeling and analytical specifications.

Architectural Context (ADR-02, Master Execution Plan):
    Natural language analytical questions are inherently ambiguous. To guarantee
    deterministic, three-layer validation and reproducible numerical verification,
    the agent converts every analytical request into a typed, structured
    `QuerySpecification`.

    This specification acts as the single source of truth for:
    - Code generation (guiding SQL generation for Qwen2.5-Coder 7B).
    - Layer 2 Numerical Validation (independent Pandas verification calculation).
    - Layer 3 Semantic Validation (verifying the generated SQL satisfies all requested
      metrics, dimensions, filters, and time intervals).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AggregationType(str, Enum):
    """Supported analytical aggregation functions."""

    SUM = "SUM"
    COUNT = "COUNT"
    DISTINCT_COUNT = "DISTINCT_COUNT"
    AVG = "AVG"
    AOV = "AOV"  # Average Order Value: SUM(sales) / COUNT(DISTINCT order_id)
    MIN = "MIN"
    MAX = "MAX"
    CUSTOM = "CUSTOM"


class NumericValuePrecision(str, Enum):
    """Mathematical continuous vs. discrete precision requirement."""

    CONTINUOUS = "CONTINUOUS"  # Formula evaluates continuously (e.g. AVG, division)
    DISCRETE = "DISCRETE"  # Formula evaluates to integers (e.g. COUNT, DATEDIFF)


class OutputTypeRequirement(str, Enum):
    """Explicit requested physical data representation."""

    UNSPECIFIED = "UNSPECIFIED"  # Output format not constrained
    FLOAT = "FLOAT"  # Explicit decimal / floating point
    INTEGER = "INTEGER"  # Explicit integer representation


class ConversionPolicy(str, Enum):
    """Explicit conversion and rounding policy."""

    NONE = "NONE"  # No rounding/truncation permitted
    ROUND_HALF_AWAY_FROM_ZERO = "ROUND_HALF_AWAY"  # DuckDB CAST/ROUND default: 0.5 -> 1, -0.5 -> -1
    TRUNCATE = "TRUNCATE"  # DuckDB TRUNC: 1.9 -> 1, -1.9 -> -1
    ROUND_HALF_EVEN = "ROUND_HALF_EVEN"  # Banker's rounding: UNSUPPORTED in DuckDB SQL
    UNSUPPORTED = "UNSUPPORTED"  # Unknown or unmapped policy


# Alias for backward-compatible rounding policy references
RoundingPolicy = ConversionPolicy


class MetricSpec(BaseModel):
    """Specification of an analytical metric to compute."""

    name: str = Field(description="Output alias name for the metric, e.g. 'total_sales'")
    aggregation: AggregationType = Field(description="Primary aggregation type")
    column: str | None = Field(
        default=None,
        description="Target column name, or None for COUNT(*)",
    )
    denominator_column: str | None = Field(
        default=None,
        description="Optional denominator column for derived metrics like AOV (e.g. order_id)",
    )
    formula: str | None = Field(
        default=None,
        description="Mathematical expression for custom metrics, e.g. 'SUM(profit) / SUM(sales)'",
    )
    precision: NumericValuePrecision = Field(
        default=NumericValuePrecision.CONTINUOUS,
        description="Whether metric requires continuous or discrete precision",
    )
    output_type: OutputTypeRequirement = Field(
        default=OutputTypeRequirement.UNSPECIFIED,
        description="Explicit physical output type requirement",
    )
    conversion_policy: ConversionPolicy = Field(
        default=ConversionPolicy.NONE,
        description="Explicit rounding or truncation policy",
    )


class FilterSpec(BaseModel):
    """Specification of a conditional filter predicate."""

    column: str = Field(description="Column identifier to filter on")
    operator: str = Field(
        default="=",
        description=(
            "SQL comparison operator: '=', '!=', '>', '<', '>=', '<=', "
            "'IN', 'LIKE', 'BETWEEN', 'IS NULL', 'IS NOT NULL'"
        ),
    )
    value: Any = Field(description="Filter value, or list of values for 'IN' / 'BETWEEN' operator")


class TimeConstraint(BaseModel):
    """Temporal boundary filter for analytical queries.

    Design Rule:
        Uses half-open intervals [start_time, end_time) for TIMESTAMP columns
        (e.g., '>= 2017-01-01' AND '< 2018-01-01' for calendar year 2017) to
        prevent sub-second boundary leaks. Genuine DATE columns may use inclusive bounds.
    """

    column: str = Field(description="Date or timestamp column identifier")
    start_time: str | None = Field(default=None, description="Start date/timestamp string")
    end_time: str | None = Field(default=None, description="End date/timestamp string")
    is_half_open: bool = Field(
        default=True,
        description="True if interval is [start, end), False if [start, end]",
    )


class SortDirection(str, Enum):
    """Sort direction for ordering and ranking."""

    ASC = "ASC"
    DESC = "DESC"


class OrderBySpec(BaseModel):
    """Specification of an ORDER BY sort key."""

    column: str = Field(description="Sort key column name or alias")
    direction: SortDirection = Field(default=SortDirection.ASC, description="Sort direction")


class DimensionSpec(BaseModel):
    """Specification of an analytical grouping dimension."""

    name: str = Field(description="Dimension name or output alias")
    expression: str = Field(
        description="Underlying column name or SQL expression, e.g. 'EXTRACT(year FROM order_date)'"
    )


class RankingMode(str, Enum):
    """Classification of ranking queries."""

    NONE = "NONE"  # Non-ranking query
    SINGLE_EXTREMUM = "SINGLE_EXTREMUM"  # Scalar extremum (e.g. highest sales value)
    ENTITY_EXTREMUM = "ENTITY_EXTREMUM"  # Single dimension entity (e.g. top region)
    TOP_N_EXACT = "TOP_N_EXACT"  # Strict top-N cut without ties (LIMIT N)
    TOP_N_WITH_TIES = "TOP_N_WITH_TIES"  # Dense rank / WITH TIES semantics


class TieBreakerPolicy(str, Enum):
    """Policy for handling ties in ranking queries."""

    ARBITRARY = "ARBITRARY"  # Engine non-deterministic or order by rowid
    EXPLICIT_KEY = "EXPLICIT_KEY"  # Secondary sort column specified
    INCLUDE_TIES = "INCLUDE_TIES"  # All tied entities at boundary included
    FAIL_ON_TIE = "FAIL_ON_TIE"  # Strict bijection required; tie is error


class RankingSpec(BaseModel):
    """Specification of ranking, ordering, and tie-breaking requirements."""

    mode: RankingMode = Field(default=RankingMode.NONE, description="Ranking mode classification")
    metric: MetricSpec | str | None = Field(default=None, description="Primary ranking metric")
    direction: SortDirection = Field(
        default=SortDirection.DESC, description="Primary sort direction"
    )
    limit: int | None = Field(default=None, description="Maximum number of entities/rows to return")
    tie_policy: TieBreakerPolicy = Field(
        default=TieBreakerPolicy.ARBITRARY, description="Tie-breaking policy"
    )
    secondary_keys: list[OrderBySpec] = Field(
        default_factory=list, description="Ordered secondary sort keys"
    )
    is_scalar: bool = Field(default=False, description="True if output is a 1x1 scalar extremum")


class IntentReliability(str, Enum):
    """Classification of the reliability of parsed analytical intent."""

    HIGH = "HIGH"  # All tokens mapped, schema-grounded, unambiguous
    AMBIGUOUS = "AMBIGUOUS"  # Multiple interpretations or unresolvable secondary ties
    SUSPECT = "SUSPECT"  # Unmapped analytical tokens or potential intent drift
    UNGROUNDED = "UNGROUNDED"  # Core entities not grounded in schema


class QuerySpecification(BaseModel):
    """Structured analytical specification representing the user's intent.

    Carried across the 7-node LangGraph workflow as the immutable ground-truth
    analytical specification.
    """

    raw_question: str = Field(description="Original user natural language question")
    table_name: str = Field(description="Target registered DuckDB table name")
    metrics: list[MetricSpec] = Field(
        default_factory=list,
        description="List of metrics to compute",
    )
    dimensions: list[str] = Field(
        default_factory=list,
        description="Grouping categorical dimensions (GROUP BY columns)",
    )
    dimension_specs: list[DimensionSpec] = Field(
        default_factory=list,
        description="Detailed dimension specifications with explicit transformation expressions",
    )
    filters: list[FilterSpec] = Field(
        default_factory=list,
        description="Filter conditions (WHERE clause predicates)",
    )
    time_constraints: list[TimeConstraint] = Field(
        default_factory=list,
        description="Temporal interval constraints",
    )
    order_by: list[str] = Field(
        default_factory=list,
        description="Columns to sort by, e.g. ['total_sales DESC']",
    )
    limit: int | None = Field(
        default=None,
        description="Optional maximum rows to return (e.g. Top N questions)",
    )
    ranking_spec: RankingSpec | None = Field(
        default=None,
        description="Structured ranking specification if query involves extrema or top-N cuts",
    )
    is_grounded: bool = Field(
        default=True,
        description="True if all referenced tables and columns exist in active schema",
    )
    intent_reliability: IntentReliability = Field(
        default=IntentReliability.HIGH,
        description="Reliability rating of the extracted intent",
    )
    unsupported_entities: list[str] = Field(
        default_factory=list,
        description="Entities or attributes requested in question that are absent from schema",
    )

    def is_grouped(self) -> bool:
        """Return True if this query specifies grouping dimensions."""
        return len(self.dimensions) > 0 or len(self.dimension_specs) > 0

    def is_scalar(self) -> bool:
        """Return True if this query produces a single scalar output (no dimensions)."""
        return (
            len(self.dimensions) == 0 and len(self.dimension_specs) == 0 and len(self.metrics) == 1
        )
