"""Typed contracts, statuses, and validation models for Phase 5 Three-Layer Validation.

Architectural Context (ADR-02, ADR-03, Master Execution Plan Phase 5):
    Provides immutable (frozen), type-safe contracts for:
    - 3-Layer Validation: Layer 1 (Execution), Layer 2 (Numerical), Layer 3 (Semantic).
    - Result Availability & Confidence Policy: Separates validation status from confidence,
      retaining successfully executed outputs even when validation is inconclusive or unverified.
    - Full Attempt Audit Trail: Preserves complete per-step retry history and diagnostics,
      enforcing strict chronological order and preventing historical mutation.
"""

from __future__ import annotations

import datetime
import uuid
from collections.abc import Mapping, Sequence
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Any

try:
    import numpy as np

    _NUMPY_SCALAR_TYPES: tuple[type, ...] = (np.generic,)
except ImportError:
    _NUMPY_SCALAR_TYPES = ()

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from src.agent.execution.engine import ExecutionResult

_IMMUTABLE_PRIMITIVES: tuple[type, ...] = (
    type(None),
    bool,
    int,
    float,
    str,
    bytes,
    complex,
    Decimal,
    Fraction,
    datetime.date,
    datetime.time,
    datetime.datetime,
    datetime.timedelta,
    uuid.UUID,
    Enum,
)


def _freeze_key(k: Any) -> Any:
    """Freeze a mapping key recursively and ensure it is hashable."""
    frozen_k = _deep_freeze(k)
    try:
        hash(frozen_k)
    except TypeError:
        raise TypeError(f"Unhashable mapping key: {type(k).__name__}") from None
    return frozen_k


def _deep_freeze(v: Any) -> Any:
    """Recursively freeze mappings into MappingProxyType and sequences into tuples.

    Rejects unsupported mutable or unknown types with TypeError.
    Safely converts supported NumPy scalar types to Python scalars recursively.
    """
    if isinstance(v, Mapping):
        return MappingProxyType({_freeze_key(k): _deep_freeze(val) for k, val in v.items()})
    if isinstance(v, (list, tuple)):
        return tuple(_deep_freeze(x) for x in v)
    if isinstance(v, (set, frozenset)):
        return frozenset(_deep_freeze(x) for x in v)
    if _NUMPY_SCALAR_TYPES and isinstance(v, _NUMPY_SCALAR_TYPES):
        scalar_val = v.item() if hasattr(v, "item") else v
        return _deep_freeze(scalar_val)
    if isinstance(v, BaseModel):
        if getattr(v, "model_config", {}).get("frozen"):
            return v
        return MappingProxyType(
            {_freeze_key(k): _deep_freeze(val) for k, val in v.model_dump().items()}
        )
    if isinstance(v, _IMMUTABLE_PRIMITIVES):
        return v
    raise TypeError(f"Unsupported mutable or unfreezable type: {type(v).__name__}")


def _deep_unfreeze(v: Any) -> Any:
    """Recursively unfreeze MappingProxyType to dict and tuples to lists for serialization."""
    if isinstance(v, Mapping):
        return {
            str(k) if not isinstance(k, (str, int, float, bool)) else k: _deep_unfreeze(val)
            for k, val in v.items()
        }
    if isinstance(v, (list, tuple)):
        return [_deep_unfreeze(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return [_deep_unfreeze(x) for x in v]
    if isinstance(v, BaseModel):
        return v.model_dump()
    return v


class ValidationStatus(str, Enum):
    """Deterministic validation lifecycle status."""

    VERIFIED = "VERIFIED"  # All 3 layers passed with confirmed execution
    FAILED = "FAILED"  # At least one layer detected an explicit failure or mismatch
    INCONCLUSIVE = "INCONCLUSIVE"  # Non-correctable ambiguity detected; unprovable
    UNVERIFIED = "UNVERIFIED"  # Retries exhausted, missing evidence, or fatal error


class ConfidenceLevel(str, Enum):
    """Evidence-based confidence level independent of validation status."""

    HIGH = "HIGH"  # Assigned ONLY when ValidationStatus is VERIFIED
    MEDIUM = "MEDIUM"  # Inconclusive with confirmed execution and meaningful output
    LOW = "LOW"  # Validation failed, unverified after retries, or execution error


class ErrorCategory(str, Enum):
    """Diagnostic categorization for targeted code remediation."""

    SYNTAX_ERROR = "SYNTAX_ERROR"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    NUMERICAL_MISMATCH = "NUMERICAL_MISMATCH"
    SEMANTIC_OMISSION = "SEMANTIC_OMISSION"
    INTERVAL_LEAK = "INTERVAL_LEAK"
    AMBIGUOUS_DATA = "AMBIGUOUS_DATA"
    FATAL_SYSTEM = "FATAL_SYSTEM"


class LayerValidationResult(BaseModel):
    """Base model for individual validation layer outcomes (deeply frozen/immutable)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    layer_name: str
    passed: bool
    status: ValidationStatus
    message: str
    details: Mapping[Any, Any] = Field(default_factory=dict)

    @field_validator("details", mode="after")
    @classmethod
    def _freeze_details(cls, v: Any) -> Any:
        return _deep_freeze(v)

    @field_serializer("details")
    def _serialize_details(self, v: Any) -> dict[Any, Any]:
        return _deep_unfreeze(v)

    @model_validator(mode="after")
    def validate_layer_status_and_passed(self) -> LayerValidationResult:
        """Enforce strict consistency between layer status and passed flag."""
        if self.status == ValidationStatus.VERIFIED and not self.passed:
            raise ValueError(f"{self.layer_name}: status VERIFIED requires passed=True.")
        if self.status == ValidationStatus.FAILED and self.passed:
            raise ValueError(f"{self.layer_name}: status FAILED requires passed=False.")
        if self.status == ValidationStatus.INCONCLUSIVE and self.passed:
            raise ValueError(
                f"{self.layer_name}: status INCONCLUSIVE cannot be represented as "
                "passed=True (successful verification)."
            )
        if self.status == ValidationStatus.UNVERIFIED and self.passed:
            raise ValueError(f"{self.layer_name}: status UNVERIFIED requires passed=False.")
        return self


class ExecutionValidationResult(LayerValidationResult):
    """Layer 1: Execution status, dialect sanity, schema presence, and grain."""

    layer_name: str = "Layer 1: Execution"
    is_empty: bool = False
    is_scalar: bool = False
    row_count: int = 0
    column_count: int = 0
    schema_matched: bool = True
    execution_time_ms: float = 0.0


class NumericalValidationResult(LayerValidationResult):
    """Layer 2: Independent Pandas/NumPy calculation outcome."""

    layer_name: str = "Layer 2: Numerical"
    duckdb_value: Any = None
    independent_value: Any = None
    absolute_difference: float | None = None
    relative_difference: float | None = None
    atol: float = 1e-4
    rtol: float = 1e-4
    compared_rows: int = 0
    inconclusive_reason: str | None = None

    @field_validator("duckdb_value", "independent_value", mode="after")
    @classmethod
    def _freeze_calc_values(cls, v: Any) -> Any:
        return _deep_freeze(v)

    @field_serializer("duckdb_value", "independent_value")
    def _serialize_calc_values(self, v: Any) -> Any:
        return _deep_unfreeze(v)


class SemanticCheckDetail(BaseModel):
    """Result of an individual canonical semantic check (deeply frozen/immutable)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    check_name: str
    status: ValidationStatus
    passed: bool
    diagnostic_code: str | None = None
    explanation: str = ""
    intent_evidence: Mapping[Any, Any] = Field(default_factory=dict)
    ast_evidence: Mapping[Any, Any] = Field(default_factory=dict)
    expected_vs_observed: Mapping[Any, Any] = Field(default_factory=dict)

    @field_validator("intent_evidence", "ast_evidence", "expected_vs_observed", mode="after")
    @classmethod
    def _freeze_check_mappings(cls, v: Any) -> Any:
        return _deep_freeze(v)

    @field_serializer("intent_evidence", "ast_evidence", "expected_vs_observed")
    def _serialize_check_mappings(self, v: Any) -> dict[Any, Any]:
        return _deep_unfreeze(v)


CANONICAL_SEMANTIC_CHECKS: frozenset[str] = frozenset(
    {
        "check_table_grounding",
        "check_column_grounding",
        "check_aggregation_level",
        "check_metric_correctness",
        "check_filter_semantics",
        "check_join_integrity",
        "check_ordering_and_ranking",
        "check_projection_contract",
    }
)


class SemanticValidationResult(LayerValidationResult):
    """Layer 3: Scope-aware AST semantic coverage outcome."""

    layer_name: str = "Layer 3: Semantic"
    metrics_covered: tuple[str, ...] = Field(default_factory=tuple)
    missing_metrics: tuple[str, ...] = Field(default_factory=tuple)
    dimensions_covered: tuple[str, ...] = Field(default_factory=tuple)
    missing_dimensions: tuple[str, ...] = Field(default_factory=tuple)
    filters_covered: tuple[str, ...] = Field(default_factory=tuple)
    missing_filters: tuple[str, ...] = Field(default_factory=tuple)
    temporal_boundary_valid: bool = True
    inconclusive_reason: str | None = None
    checks: tuple[SemanticCheckDetail, ...] = Field(default_factory=tuple)
    canonical_projection: Any = None
    intent_reliability: Any = None

    @field_validator("checks", mode="after")
    @classmethod
    def _freeze_checks(cls, v: Any) -> Any:
        return tuple(v) if isinstance(v, (list, tuple)) else v


class CorrectionDiagnostic(BaseModel):
    """Remediation feedback delivered to the code generator on retry."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    error_category: ErrorCategory
    is_correctable: bool
    failed_layer: str
    error_summary: str
    remediation_instruction: str
    prior_attempt_sql: str
    attempt_number: int = Field(ge=1, description="1-based attempt sequence number")


class StepValidationResult(BaseModel):
    """Complete 3-layer validation outcome for a single plan step attempt (frozen/immutable)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    step_id: int = Field(ge=1, description="1-based unique identifier of the plan step")
    attempt_number: int = Field(
        ge=1, description="1-based sequence number of the validation attempt"
    )
    status: ValidationStatus
    confidence: ConfidenceLevel
    overall_passed: bool
    execution_result: ExecutionValidationResult
    numerical_result: NumericalValidationResult
    semantic_result: SemanticValidationResult
    diagnostic: CorrectionDiagnostic | None = None
    executed_sql: str | None = None
    raw_execution_result: ExecutionResult | None = None
    notes: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_confidence_and_status(self) -> StepValidationResult:
        """Enforce strict evidence-based confidence rules and 3-layer verification invariants."""
        layers = [self.execution_result, self.numerical_result, self.semantic_result]

        # Invariant 1: VERIFIED status requires all 3 layers to be VERIFIED and passed,
        # overall_passed=True, HIGH confidence, and confirmed successful raw execution.
        if self.status == ValidationStatus.VERIFIED:
            if not self.overall_passed:
                raise ValueError("overall_passed must be True when status is VERIFIED.")
            if self.confidence != ConfidenceLevel.HIGH:
                raise ValueError(
                    f"Step {self.step_id} status is VERIFIED but confidence is "
                    f"'{self.confidence}'. VERIFIED status strictly requires HIGH confidence."
                )
            if not all(
                layer.passed and layer.status == ValidationStatus.VERIFIED for layer in layers
            ):
                raise ValueError(
                    f"Step {self.step_id} cannot be marked VERIFIED because not all validation "
                    "layers passed with VERIFIED status. "
                    f"Execution={self.execution_result.status}, "
                    f"Numerical={self.numerical_result.status}, "
                    f"Semantic={self.semantic_result.status}."
                )
            if self.raw_execution_result is None or not self.raw_execution_result.success:
                raise ValueError(
                    f"Step {self.step_id} cannot be marked VERIFIED without confirmed raw "
                    "execution success."
                )

        # Invariant 2: FAILED status requires at least one explicitly FAILED layer,
        # overall_passed=False, and LOW confidence.
        elif self.status == ValidationStatus.FAILED:
            if self.overall_passed:
                raise ValueError("overall_passed must be False when status is FAILED.")
            if self.confidence != ConfidenceLevel.LOW:
                raise ValueError(
                    f"Step {self.step_id} status is FAILED but confidence is '{self.confidence}'. "
                    "FAILED status strictly requires LOW confidence."
                )
            if not any(layer.status == ValidationStatus.FAILED for layer in layers):
                raise ValueError(
                    f"Step {self.step_id} has status FAILED but no layer is marked FAILED."
                )

        # Invariant 3: INCONCLUSIVE status requires at least one INCONCLUSIVE layer,
        # NO FAILED layer, overall_passed=False, and confidence reflecting execution evidence.
        elif self.status == ValidationStatus.INCONCLUSIVE:
            if self.overall_passed:
                raise ValueError("overall_passed cannot be True when status is INCONCLUSIVE.")
            if any(layer.status == ValidationStatus.FAILED for layer in layers):
                raise ValueError(
                    f"Step {self.step_id} cannot be INCONCLUSIVE when a layer has FAILED. "
                    "Explicit failure takes precedence."
                )
            if not any(layer.status == ValidationStatus.INCONCLUSIVE for layer in layers):
                raise ValueError(
                    f"Step {self.step_id} is marked INCONCLUSIVE but no layer is INCONCLUSIVE."
                )
            if self.confidence == ConfidenceLevel.HIGH:
                raise ValueError(
                    f"Step {self.step_id} is INCONCLUSIVE and cannot have HIGH confidence. "
                    "HIGH confidence strictly requires VERIFIED status."
                )
            if self.confidence == ConfidenceLevel.MEDIUM:
                if (
                    self.raw_execution_result is None
                    or not self.raw_execution_result.success
                    or not self.execution_result.passed
                ):
                    raise ValueError(
                        f"Step {self.step_id} is INCONCLUSIVE with MEDIUM confidence, which "
                        "requires confirmed raw execution success and passing execution layer."
                    )

        # Invariant 4: UNVERIFIED status requires overall_passed=False and LOW confidence.
        elif self.status == ValidationStatus.UNVERIFIED:
            if self.overall_passed:
                raise ValueError("overall_passed cannot be True when status is UNVERIFIED.")
            if self.confidence != ConfidenceLevel.LOW:
                raise ValueError(
                    f"Step {self.step_id} is UNVERIFIED and must have LOW confidence, "
                    f"got '{self.confidence}'."
                )

        # Invariant 5: Raw execution failure must NEVER produce VERIFIED or HIGH confidence.
        if self.raw_execution_result is not None and not self.raw_execution_result.success:
            if self.status == ValidationStatus.VERIFIED or self.confidence == ConfidenceLevel.HIGH:
                raise ValueError(
                    f"Step {self.step_id} raw execution failed; status cannot be VERIFIED and "
                    "confidence cannot be HIGH."
                )

        return self


def determine_step_status(
    execution_res: ExecutionValidationResult,
    numerical_res: NumericalValidationResult,
    semantic_res: SemanticValidationResult,
    raw_execution_result: ExecutionResult | None = None,
) -> ValidationStatus:
    """Determine the consolidated validation status across 3 layers and execution evidence.

    Rules:
        - If raw execution result is missing: execution is unconfirmed; cannot be VERIFIED.
        - If raw execution result failed or execution_res did not pass -> FAILED.
        - If any layer reports FAILED -> FAILED.
        - If all 3 layers report VERIFIED and passed AND raw execution succeeded -> VERIFIED.
        - If any layer reports INCONCLUSIVE and no layer failed -> INCONCLUSIVE.
          (Non-correctable ambiguity remains INCONCLUSIVE).
        - Otherwise -> UNVERIFIED.
    """
    layers = [execution_res, numerical_res, semantic_res]

    # Explicit raw execution failure
    if raw_execution_result is not None and not raw_execution_result.success:
        return ValidationStatus.FAILED

    # Demonstrable failure in any validation layer
    if any(layer.status == ValidationStatus.FAILED for layer in layers):
        return ValidationStatus.FAILED

    # Inconclusive ambiguity (ambiguous dates, unsupported custom metric) takes precedence
    # over unconfirmed execution or verification
    if any(layer.status == ValidationStatus.INCONCLUSIVE for layer in layers):
        return ValidationStatus.INCONCLUSIVE

    # Missing raw execution evidence: execution is unconfirmed; cannot produce VERIFIED
    if raw_execution_result is None or not raw_execution_result.success:
        return ValidationStatus.UNVERIFIED

    # All three layers verified and raw execution succeeded
    if (
        execution_res.passed
        and execution_res.status == ValidationStatus.VERIFIED
        and numerical_res.passed
        and numerical_res.status == ValidationStatus.VERIFIED
        and semantic_res.passed
        and semantic_res.status == ValidationStatus.VERIFIED
    ):
        return ValidationStatus.VERIFIED

    return ValidationStatus.UNVERIFIED


def determine_step_confidence(
    status: ValidationStatus,
    execution_succeeded: bool = False,
    has_meaningful_output: bool = True,
) -> ConfidenceLevel:
    """Derive evidence-based confidence from validation status and execution evidence.

    Policy:
        - VERIFIED status -> HIGH confidence.
        - INCONCLUSIVE status with successful execution and meaningful output -> MEDIUM confidence.
        - INCONCLUSIVE without meaningful output or with failed execution -> LOW confidence.
        - FAILED or UNVERIFIED status -> LOW confidence.
    """
    if status == ValidationStatus.VERIFIED:
        return ConfidenceLevel.HIGH

    if status == ValidationStatus.INCONCLUSIVE and execution_succeeded and has_meaningful_output:
        return ConfidenceLevel.MEDIUM

    return ConfidenceLevel.LOW


def synthesize_diagnostic(
    execution_res: ExecutionValidationResult,
    numerical_res: NumericalValidationResult,
    semantic_res: SemanticValidationResult,
    raw_execution_result: ExecutionResult | None = None,
    diagnostic: CorrectionDiagnostic | None = None,
    executed_sql: str | None = None,
    attempt_number: int = 1,
) -> CorrectionDiagnostic | None:
    """Synthesize a structured CorrectionDiagnostic when diagnostic is None or needs completion."""
    if diagnostic is not None:
        return diagnostic

    status = determine_step_status(
        execution_res=execution_res,
        numerical_res=numerical_res,
        semantic_res=semantic_res,
        raw_execution_result=raw_execution_result,
    )

    if status == ValidationStatus.VERIFIED:
        return None

    sql_text = executed_sql or ""

    # Priority 1: Layer 1 Execution Failure
    if not execution_res.passed or execution_res.status == ValidationStatus.FAILED:
        err_msg = execution_res.message or "Execution failed"
        is_fatal = (
            raw_execution_result is None
            or "timeout" in err_msg.lower()
            or "fatal" in err_msg.lower()
        )
        cat = ErrorCategory.FATAL_SYSTEM if is_fatal else ErrorCategory.SYNTAX_ERROR
        return CorrectionDiagnostic(
            error_category=cat,
            is_correctable=not is_fatal,
            failed_layer="Layer 1: Execution",
            error_summary=err_msg,
            remediation_instruction=(
                "Fix SQL syntax, function calls, or schema references according to DuckDB dialect."
            ),
            prior_attempt_sql=sql_text,
            attempt_number=attempt_number,
        )

    # Priority 2: Layer 2 Numerical Failure
    if numerical_res.status == ValidationStatus.FAILED:
        err_msg = numerical_res.message or "Numerical validation failed"
        return CorrectionDiagnostic(
            error_category=ErrorCategory.NUMERICAL_MISMATCH,
            is_correctable=True,
            failed_layer="Layer 2: Numerical",
            error_summary=err_msg,
            remediation_instruction=(
                "Ensure calculation matches intent: verify aggregation expressions, "
                "filtering intervals, group-by dimensions, and sorting directions."
            ),
            prior_attempt_sql=sql_text,
            attempt_number=attempt_number,
        )

    # Priority 3: Layer 3 Semantic Failure
    if semantic_res.status == ValidationStatus.FAILED:
        failed_checks = [c for c in semantic_res.checks if not c.passed]
        check_names = [c.check_name for c in failed_checks]
        explanations = [c.explanation for c in failed_checks if c.explanation]
        summary = f"Semantic checks failed: {', '.join(check_names)}. " + (
            "; ".join(explanations) if explanations else semantic_res.message
        )
        has_temporal_fail = not semantic_res.temporal_boundary_valid or any(
            "filter" in c.check_name for c in failed_checks
        )
        cat = (
            ErrorCategory.INTERVAL_LEAK
            if has_temporal_fail and "interval" in summary.lower()
            else ErrorCategory.SEMANTIC_OMISSION
        )
        fix_target = "; ".join(explanations) or semantic_res.message
        return CorrectionDiagnostic(
            error_category=cat,
            is_correctable=True,
            failed_layer="Layer 3: Semantic",
            error_summary=summary,
            remediation_instruction=(f"Align SQL AST with structured intent: {fix_target}"),
            prior_attempt_sql=sql_text,
            attempt_number=attempt_number,
        )

    # Priority 4: Inconclusive
    if status == ValidationStatus.INCONCLUSIVE:
        inc_reason = (
            semantic_res.inconclusive_reason
            or numerical_res.inconclusive_reason
            or "Validation inconclusive due to ambiguous data or unverifiable intent."
        )
        inc_layer = (
            "Layer 3: Semantic"
            if semantic_res.status == ValidationStatus.INCONCLUSIVE
            else "Layer 2: Numerical"
        )
        return CorrectionDiagnostic(
            error_category=ErrorCategory.AMBIGUOUS_DATA,
            is_correctable=False,  # Inconclusive is strictly NOT correctable
            failed_layer=inc_layer,
            error_summary=inc_reason,
            remediation_instruction=(
                "Intent or data is ambiguous/unprovable. Automatic correction cannot resolve "
                "ambiguity."
            ),
            prior_attempt_sql=sql_text,
            attempt_number=attempt_number,
        )

    return None


def build_step_validation_result(
    step_id: int,
    attempt_number: int,
    execution_res: ExecutionValidationResult,
    numerical_res: NumericalValidationResult,
    semantic_res: SemanticValidationResult,
    diagnostic: CorrectionDiagnostic | None = None,
    executed_sql: str | None = None,
    raw_execution_result: ExecutionResult | None = None,
    has_meaningful_output: bool | None = None,
    notes: Sequence[str] | None = None,
) -> StepValidationResult:
    """Convenience factory constructing a validated StepValidationResult enforcing invariants."""
    # If raw execution result failed, ensure execution_res reflects this failure
    if raw_execution_result is not None and not raw_execution_result.success:
        if execution_res.passed or execution_res.status != ValidationStatus.FAILED:
            execution_res = ExecutionValidationResult(
                layer_name=execution_res.layer_name,
                passed=False,
                status=ValidationStatus.FAILED,
                message=(
                    f"Raw execution failed: "
                    f"{raw_execution_result.error_message or 'Execution failed'}"
                ),
            )

    status = determine_step_status(
        execution_res=execution_res,
        numerical_res=numerical_res,
        semantic_res=semantic_res,
        raw_execution_result=raw_execution_result,
    )

    # Missing raw execution result is NOT confirmed execution success
    raw_ok = raw_execution_result.success if raw_execution_result is not None else False
    exec_succeeded = execution_res.passed and raw_ok

    # Determine whether output is meaningful:
    # 1. Use explicit evidence when provided (e.g. from Stage 2 execution validator).
    # 2. Do not infer meaningfulness from is_empty=True alone; empty output can be valid
    #    only when explicit semantic/expected-shape evidence is supplied.
    # 3. If meaningfulness cannot be established, meaningful remains False,
    #    resulting in LOW confidence for an INCONCLUSIVE result.
    if has_meaningful_output is not None:
        meaningful = has_meaningful_output and exec_succeeded
    else:
        if not exec_succeeded or raw_execution_result is None:
            meaningful = False
        else:
            has_scalar = (
                raw_execution_result.is_scalar and raw_execution_result.scalar_value is not None
            )
            has_non_empty_df = (
                raw_execution_result.dataframe is not None and not raw_execution_result.is_empty()
            )
            meaningful = has_scalar or has_non_empty_df

    confidence = determine_step_confidence(
        status=status,
        execution_succeeded=exec_succeeded,
        has_meaningful_output=meaningful,
    )
    overall_passed = status == ValidationStatus.VERIFIED

    effective_diagnostic = diagnostic
    if effective_diagnostic is None:
        effective_diagnostic = synthesize_diagnostic(
            execution_res=execution_res,
            numerical_res=numerical_res,
            semantic_res=semantic_res,
            raw_execution_result=raw_execution_result,
            diagnostic=None,
            executed_sql=executed_sql,
            attempt_number=attempt_number,
        )

    return StepValidationResult(
        step_id=step_id,
        attempt_number=attempt_number,
        status=status,
        confidence=confidence,
        overall_passed=overall_passed,
        execution_result=execution_res,
        numerical_result=numerical_res,
        semantic_result=semantic_res,
        diagnostic=effective_diagnostic,
        executed_sql=executed_sql,
        raw_execution_result=raw_execution_result,
        notes=tuple(notes) if notes else (),
    )


class WorkflowValidationReport(BaseModel):
    """Consolidated validation summary across all plan steps and historical attempts (mutable)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workflow_status: ValidationStatus = ValidationStatus.UNVERIFIED
    workflow_confidence: ConfidenceLevel = ConfidenceLevel.LOW
    all_steps_verified: bool = False
    # Complete per-step retry history: step_id -> list of StepValidationResult (all attempts)
    step_history: dict[int, list[StepValidationResult]] = Field(default_factory=dict)
    # Convenience mapping of step_id -> latest StepValidationResult
    latest_step_results: dict[int, StepValidationResult] = Field(default_factory=dict)
    # Preserved successfully executed raw results (Result Availability Policy)
    retained_execution_results: dict[int, ExecutionResult] = Field(default_factory=dict)
    total_steps: int = 0
    total_retries: int = 0
    verification_warnings: list[str] = Field(default_factory=list)
    validation_timestamp: str | None = None

    def add_step_attempt(self, result: StepValidationResult) -> None:
        """Record a validation attempt for a step, updating history and summary.

        Enforces:
            - Chronological order of attempts per step.
            - Rejection of duplicate or out-of-order attempt numbers.
            - Latest step result is always set to the final history entry.
            - Retention of last successful raw execution result without overwriting
              on later failure.
        """
        step_id = result.step_id
        if step_id not in self.step_history:
            if result.attempt_number < 1:
                raise ValueError(
                    f"Invalid attempt number {result.attempt_number} for step {step_id}: "
                    "attempt numbers must start at 1 or greater."
                )
            self.step_history[step_id] = []
        else:
            last_attempt = self.step_history[step_id][-1].attempt_number
            if result.attempt_number <= last_attempt:
                raise ValueError(
                    f"Out-of-order or duplicate attempt number {result.attempt_number} "
                    f"for step {step_id}: last recorded attempt was {last_attempt}. "
                    "Attempts must be strictly increasing."
                )

        self.step_history[step_id].append(result)
        self.latest_step_results[step_id] = self.step_history[step_id][-1]

        # Retain the last successful raw execution result (Result Availability Policy)
        if result.raw_execution_result and result.raw_execution_result.success:
            self.retained_execution_results[step_id] = result.raw_execution_result

            # Safe history compaction: create a new collection rather than in-place mutation.
            # Only compact superseded historical attempts that are not the active retained result.
            current_retained = self.retained_execution_results.get(step_id)
            history_list = list(self.step_history[step_id])
            compacted_history: list[StepValidationResult] = []

            for past_attempt in history_list[:-1]:
                past_raw = past_attempt.raw_execution_result
                if (
                    past_raw is not None
                    and past_raw is not current_retained
                    and getattr(past_raw, "raw_rows", None) is not None
                ):
                    compacted_raw = past_raw.model_copy(update={"raw_rows": None})
                    compacted_history.append(
                        past_attempt.model_copy(update={"raw_execution_result": compacted_raw})
                    )
                else:
                    compacted_history.append(past_attempt)

            # Preserve latest attempt intact
            compacted_history.append(history_list[-1])
            self.step_history[step_id] = compacted_history

        # Recalculate summary metrics and consistency checks
        self.recompute_summary()

    def recompute_summary(self) -> None:
        """Derive consolidated workflow status and confidence from all latest step results."""
        if not self.latest_step_results:
            self.workflow_status = ValidationStatus.UNVERIFIED
            self.workflow_confidence = ConfidenceLevel.LOW
            self.all_steps_verified = False
            self.total_steps = 0
            self.total_retries = 0
            return

        # Consistency verification
        for step_id, history in self.step_history.items():
            if history:
                self.latest_step_results[step_id] = history[-1]

        self.total_steps = len(self.latest_step_results)
        # Total retries is the count of attempts beyond the initial attempt (attempt_number > 1)
        self.total_retries = sum(
            max(0, len(attempts) - 1) for attempts in self.step_history.values()
        )

        all_verified = all(
            res.status == ValidationStatus.VERIFIED for res in self.latest_step_results.values()
        )
        any_failed = any(
            res.status == ValidationStatus.FAILED for res in self.latest_step_results.values()
        )
        any_unverified = any(
            res.status == ValidationStatus.UNVERIFIED for res in self.latest_step_results.values()
        )
        any_inconclusive = any(
            res.status == ValidationStatus.INCONCLUSIVE for res in self.latest_step_results.values()
        )

        # Result Availability and Confidence Policy:
        # 1. If all 3 validation layers pass on all steps -> VERIFIED with HIGH confidence
        if all_verified:
            self.workflow_status = ValidationStatus.VERIFIED
            self.workflow_confidence = ConfidenceLevel.HIGH
            self.all_steps_verified = True
        # 2. If any step failed -> FAILED with LOW confidence
        elif any_failed:
            self.workflow_status = ValidationStatus.FAILED
            self.workflow_confidence = ConfidenceLevel.LOW
            self.all_steps_verified = False
        # 3. If any step is unverified (retries exhausted) -> UNVERIFIED with LOW confidence
        elif any_unverified:
            self.workflow_status = ValidationStatus.UNVERIFIED
            self.workflow_confidence = ConfidenceLevel.LOW
            self.all_steps_verified = False
        # 4. If inconclusive (ambiguous dates, unsupported custom formula)
        elif any_inconclusive:
            self.workflow_status = ValidationStatus.INCONCLUSIVE
            self.all_steps_verified = False

            # Workflow receives MEDIUM confidence ONLY when all required steps have confirmed
            # successful execution and meaningful output (non-LOW step confidence).
            # If any required step has LOW confidence, the workflow must receive LOW confidence.
            all_steps_qualified = all(
                res.confidence != ConfidenceLevel.LOW
                and res.execution_result.passed
                and res.raw_execution_result is not None
                and res.raw_execution_result.success
                for res in self.latest_step_results.values()
            )
            self.workflow_confidence = (
                ConfidenceLevel.MEDIUM if all_steps_qualified else ConfidenceLevel.LOW
            )
        else:
            self.workflow_status = ValidationStatus.UNVERIFIED
            self.workflow_confidence = ConfidenceLevel.LOW
            self.all_steps_verified = False
