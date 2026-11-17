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


class FilterSpec(BaseModel):
    """Specification of a conditional filter predicate."""

    column: str = Field(description="Column identifier to filter on")
    operator: str = Field(
        default="=",
        description="SQL comparison operator: '=', '!=', '>', '<', '>=', '<=', 'IN', 'LIKE'",
    )
    value: Any = Field(description="Filter value, or list of values for 'IN' operator")


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

    def is_grouped(self) -> bool:
        """Return True if this query specifies grouping dimensions."""
        return len(self.dimensions) > 0

    def is_scalar(self) -> bool:
        """Return True if this query produces a single scalar output (no dimensions)."""
        return len(self.dimensions) == 0 and len(self.metrics) == 1

