"""Projection normalization, immutable canonical views, and strict scalar contracts.

Architectural Context (ADR-02, Phase 5 Stage 4):
    Provides deterministic projection validation and immutable canonical row views
    derived from raw execution results, without mutating ExecutionResult.raw_rows
    or attempt history.

Key Invariants:
    1. Exact-Type Scalar Allowlist: Every row cell must strictly satisfy
       type(val) in SUPPORTED_SCALAR_TYPES. Nested containers (list, tuple, dict,
       set) and arbitrary custom objects are rejected as UNSUPPORTED.
    2. Deep Defensive Isolation: Pre-validation shallow-copy of caller dictionaries
       and recursive conversion of rows into immutable tuples prevents mutation of
       caller-owned collections on both successful and rejected model construction.
    3. Mathematical Bijection Proof: permutation_map must be a valid bijection
       over range(k), where k is the canonical column count.
    4. Identity vs. Permutation Invariants:
       - CANONICAL requires is_permuted=False and identity permutation tuple(range(k)).
       - PERMUTED requires is_permuted=True and non-identity valid permutation.
       - CONTRADICTION and UNSUPPORTED strictly require canonical_view=None and
         permutation_map=().
    5. Case-Insensitive Unique Aliases: Canonical column names must be unique
       under case-insensitive comparison.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.agent.execution.engine import ExecutionResult

# Supported immutable scalar allowlist (exact types only; subclasses and containers rejected)
SUPPORTED_SCALAR_TYPES: frozenset[type] = frozenset(
    {
        int,
        float,
        str,
        bool,
        bytes,
        datetime.date,
        datetime.datetime,
        datetime.time,
        Decimal,
        type(None),
    }
)


class ProjectionNormalizationStatus(str, Enum):
    """Classification of projection normalization outcome."""

    CANONICAL = "CANONICAL"  # 1:1 identical column names and canonical order
    PERMUTED = "PERMUTED"  # Valid non-identity bijection permutation
    CONTRADICTION = "CONTRADICTION"  # Missing/extra cols, duplicate aliases, width mismatch
    UNSUPPORTED = "UNSUPPORTED"  # Disallowed mutable types or nested objects in cells


class CanonicalProjectionView(BaseModel):
    """Deeply frozen, immutable canonical view of query execution results."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    canonical_columns: tuple[str, ...]
    canonical_raw_rows: tuple[tuple[Any, ...], ...]
    permutation_map: tuple[int, ...]
    is_permuted: bool

    @model_validator(mode="before")
    @classmethod
    def defensively_isolate_and_freeze(cls, data: Any) -> dict[str, Any]:
        """Defensively isolate and freeze input collections without mutating caller data."""
        if not isinstance(data, dict):
            raise TypeError("CanonicalProjectionView initialization requires a dict mapping")

        # 1. Non-mutating shallow copy of caller mapping
        copied_data = dict(data)

        # 2. Convert column aliases to immutable tuple
        if "canonical_columns" in copied_data:
            copied_data["canonical_columns"] = tuple(copied_data["canonical_columns"])

        # 3. Convert permutation map to immutable tuple
        if "permutation_map" in copied_data:
            copied_data["permutation_map"] = tuple(copied_data["permutation_map"])

        # 4. Deep-freeze row collections into immutable tuples of tuples
        if "canonical_raw_rows" in copied_data:
            raw_input = copied_data["canonical_raw_rows"]
            if not isinstance(raw_input, (list, tuple)):
                raise TypeError("canonical_raw_rows must be a sequence of rows")

            frozen_rows: list[tuple[Any, ...]] = []
            for row in raw_input:
                if not isinstance(row, (list, tuple)):
                    raise TypeError(f"Row {row!r} must be a sequence, got {type(row).__name__}")
                frozen_rows.append(tuple(row))
            copied_data["canonical_raw_rows"] = tuple(frozen_rows)

        return copied_data

    @model_validator(mode="after")
    def validate_canonical_invariants(self) -> CanonicalProjectionView:
        """Validate case-insensitive alias uniqueness, bijection, and exact scalar types."""
        k = len(self.canonical_columns)

        # Invariant 1: Case-insensitive alias uniqueness
        lower_cols = [c.lower() for c in self.canonical_columns]
        if len(set(lower_cols)) != k:
            raise ValueError(
                f"Duplicate column aliases under case insensitivity: {self.canonical_columns}"
            )

        # Invariant 2: Bijection proof on permutation_map
        if len(self.permutation_map) != k or set(self.permutation_map) != set(range(k)):
            raise ValueError(
                f"permutation_map {self.permutation_map} is not a valid bijection over [0, {k - 1}]"
            )

        # Invariant 3: Permutation flag consistency
        is_id = self.permutation_map == tuple(range(k))
        if self.is_permuted == is_id:
            raise ValueError(f"is_permuted={self.is_permuted} contradicts identity status {is_id}")

        # Invariant 4: Exact scalar allowlist validation (no containers, no subclasses)
        for r_idx, row in enumerate(self.canonical_raw_rows):
            if len(row) != k:
                raise ValueError(f"Row {r_idx} width {len(row)} != canonical column width {k}")
            for c_idx, val in enumerate(row):
                if type(val) not in SUPPORTED_SCALAR_TYPES:
                    type_name = type(val).__name__
                    raise TypeError(
                        f"Disallowed value {val!r} of exact type {type_name} at "
                        f"row {r_idx}, col {c_idx}. Only exact SUPPORTED_SCALAR_TYPES "
                        "are permitted in CanonicalProjectionView."
                    )
        return self


class ProjectionNormalizationResult(BaseModel):
    """Result of attempting projection normalization against expected columns."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    status: ProjectionNormalizationStatus
    diagnostic: str | None = None
    canonical_view: CanonicalProjectionView | None = None
    missing_projections: tuple[str, ...] = Field(default_factory=tuple)
    unexpected_projections: tuple[str, ...] = Field(default_factory=tuple)
    duplicate_aliases: tuple[str, ...] = Field(default_factory=tuple)
    permutation_map: tuple[int, ...] = Field(default_factory=tuple)

    @model_validator(mode="before")
    @classmethod
    def freeze_result_collections(cls, data: Any) -> Any:
        """Ensure all collection fields are frozen immutable tuples."""
        if isinstance(data, dict):
            copied = dict(data)
            for field in (
                "missing_projections",
                "unexpected_projections",
                "duplicate_aliases",
                "permutation_map",
            ):
                if field in copied and copied[field] is not None:
                    copied[field] = tuple(copied[field])
            return copied
        return data

    @model_validator(mode="after")
    def validate_result_contract(self) -> ProjectionNormalizationResult:
        """Enforce strict consistency between status, permutation_map, and canonical_view."""
        if self.status == ProjectionNormalizationStatus.CANONICAL:
            if self.canonical_view is None:
                raise ValueError("Status CANONICAL requires non-null canonical_view")
            if self.canonical_view.is_permuted:
                raise ValueError("Status CANONICAL requires canonical_view.is_permuted=False")
            if self.permutation_map != tuple(range(len(self.canonical_view.canonical_columns))):
                raise ValueError("Status CANONICAL requires identity permutation_map")

        elif self.status == ProjectionNormalizationStatus.PERMUTED:
            if self.canonical_view is None:
                raise ValueError("Status PERMUTED requires non-null canonical_view")
            if not self.canonical_view.is_permuted:
                raise ValueError("Status PERMUTED requires canonical_view.is_permuted=True")
            if self.permutation_map != self.canonical_view.permutation_map:
                raise ValueError("Result permutation_map must equal canonical_view.permutation_map")
            if self.permutation_map == tuple(range(len(self.canonical_view.canonical_columns))):
                raise ValueError("Status PERMUTED forbids identity permutation_map")

        elif self.status in (
            ProjectionNormalizationStatus.CONTRADICTION,
            ProjectionNormalizationStatus.UNSUPPORTED,
        ):
            if self.canonical_view is not None:
                raise ValueError(f"Status {self.status} must never expose a canonical_view")
            if len(self.permutation_map) != 0:
                raise ValueError(
                    f"Status {self.status} must expose empty permutation_map, "
                    f"got {self.permutation_map}"
                )

        return self


class ProjectionNormalizer:
    """Deterministic projection normalizer validating execution output against intent."""

    def normalize(
        self,
        expected_projections: Sequence[str],
        execution_result: ExecutionResult | None,
    ) -> ProjectionNormalizationResult:
        """Validate column alignment, verify scalar cell types, and build canonical view.

        Args:
            expected_projections: Ordered canonical column names expected by intent.
            execution_result: Raw execution result from ControlledExecutionEngine.

        Returns:
            ProjectionNormalizationResult with status and optional CanonicalProjectionView.
        """
        if not expected_projections:
            return ProjectionNormalizationResult(
                status=ProjectionNormalizationStatus.CONTRADICTION,
                diagnostic="Expected projection list is empty.",
            )

        # Check for case-insensitive duplicate aliases in expected projections
        expected_tuple = tuple(expected_projections)
        expected_lower = [c.lower() for c in expected_tuple]
        if len(set(expected_lower)) != len(expected_tuple):
            return ProjectionNormalizationResult(
                status=ProjectionNormalizationStatus.CONTRADICTION,
                diagnostic=f"Duplicate aliases in expected projections: {expected_tuple}",
                duplicate_aliases=tuple(
                    c for i, c in enumerate(expected_tuple) if expected_lower.count(c.lower()) > 1
                ),
            )

        if execution_result is None or not execution_result.success:
            return ProjectionNormalizationResult(
                status=ProjectionNormalizationStatus.CONTRADICTION,
                diagnostic="Execution result is missing or failed.",
                missing_projections=expected_tuple,
            )

        # Get actual column names from execution result
        actual_cols: list[str] = []
        if execution_result.columns:
            actual_cols = list(execution_result.columns)
        elif execution_result.dataframe is not None:
            actual_cols = list(execution_result.dataframe.columns)
        elif execution_result.is_scalar:
            actual_cols = [expected_tuple[0]] if len(expected_tuple) == 1 else ["scalar_value"]

        actual_lower = [c.lower() for c in actual_cols]

        # Check duplicate aliases in actual columns
        if len(set(actual_lower)) != len(actual_cols):
            dup_cols = tuple(
                c for i, c in enumerate(actual_cols) if actual_lower.count(c.lower()) > 1
            )
            return ProjectionNormalizationResult(
                status=ProjectionNormalizationStatus.CONTRADICTION,
                diagnostic=f"Duplicate column aliases in execution result: {actual_cols}",
                duplicate_aliases=dup_cols,
            )

        # Check column count match
        if len(actual_cols) != len(expected_tuple):
            missing = tuple(
                e
                for e, el in zip(expected_tuple, expected_lower, strict=False)
                if el not in actual_lower
            )
            unexpected = tuple(
                a
                for a, al in zip(actual_cols, actual_lower, strict=False)
                if al not in expected_lower
            )
            return ProjectionNormalizationResult(
                status=ProjectionNormalizationStatus.CONTRADICTION,
                diagnostic=(
                    f"Column count mismatch: expected {len(expected_tuple)} columns, "
                    f"observed {len(actual_cols)} columns."
                ),
                missing_projections=missing,
                unexpected_projections=unexpected,
            )

        # Build permutation mapping: for each expected col j, find actual col i
        perm_map: list[int] = []
        for exp_l in expected_lower:
            if exp_l not in actual_lower:
                missing = tuple(
                    e
                    for e, el in zip(expected_tuple, expected_lower, strict=False)
                    if el not in actual_lower
                )
                unexpected = tuple(
                    a
                    for a, al in zip(actual_cols, actual_lower, strict=False)
                    if al not in expected_lower
                )
                return ProjectionNormalizationResult(
                    status=ProjectionNormalizationStatus.CONTRADICTION,
                    diagnostic=f"Expected column missing from execution result: {missing}",
                    missing_projections=missing,
                    unexpected_projections=unexpected,
                )
            perm_map.append(actual_lower.index(exp_l))

        permutation_map = tuple(perm_map)

        # Extract raw rows from execution result
        raw_rows_source: list[tuple[Any, ...]] = []
        if execution_result.raw_rows is not None:
            raw_rows_source = list(execution_result.raw_rows)
        elif execution_result.dataframe is not None:
            raw_rows_source = [
                tuple(row) for row in execution_result.dataframe.itertuples(index=False, name=None)
            ]
        elif execution_result.is_scalar:
            raw_rows_source = [(execution_result.scalar_value,)]

        # Permute and validate cells against exact scalar allowlist
        canonical_rows: list[tuple[Any, ...]] = []
        k = len(expected_tuple)

        for r_idx, row in enumerate(raw_rows_source):
            if len(row) != len(actual_cols):
                return ProjectionNormalizationResult(
                    status=ProjectionNormalizationStatus.CONTRADICTION,
                    diagnostic=(
                        f"Row {r_idx} width {len(row)} does not match actual "
                        f"column count {len(actual_cols)}."
                    ),
                )

            permuted_row: list[Any] = []
            for j in range(k):
                actual_idx = permutation_map[j]
                val = row[actual_idx]

                # Exact type scalar check (no containers, no subclasses)
                if type(val) not in SUPPORTED_SCALAR_TYPES:
                    return ProjectionNormalizationResult(
                        status=ProjectionNormalizationStatus.UNSUPPORTED,
                        diagnostic=(
                            f"Unsupported value {val!r} of exact type {type(val).__name__} "
                            f"at row {r_idx}, col {actual_idx}."
                        ),
                    )
                permuted_row.append(val)
            canonical_rows.append(tuple(permuted_row))

        is_permuted = permutation_map != tuple(range(k))
        status = (
            ProjectionNormalizationStatus.PERMUTED
            if is_permuted
            else ProjectionNormalizationStatus.CANONICAL
        )

        try:
            view = CanonicalProjectionView(
                canonical_columns=expected_tuple,
                canonical_raw_rows=tuple(canonical_rows),
                permutation_map=permutation_map,
                is_permuted=is_permuted,
            )
            return ProjectionNormalizationResult(
                status=status,
                canonical_view=view,
                permutation_map=permutation_map,
            )
        except Exception as exc:
            return ProjectionNormalizationResult(
                status=ProjectionNormalizationStatus.CONTRADICTION,
                diagnostic=f"Canonical projection invariant violation: {exc}",
            )
