"""Three-Layer Validation & Bounded Self-Correction package.

Architectural Context (ADR-02, ADR-03, Master Execution Plan Phase 5):
    Provides deterministic verification (Execution, Numerical, Semantic) and
    bounded self-correction for autonomous analytics workflows.
"""

from src.agent.validation.contracts import (
    ConfidenceLevel,
    CorrectionDiagnostic,
    ErrorCategory,
    ExecutionValidationResult,
    LayerValidationResult,
    NumericalValidationResult,
    SemanticValidationResult,
    StepValidationResult,
    ValidationStatus,
    WorkflowValidationReport,
    build_step_validation_result,
    determine_step_confidence,
    determine_step_status,
)
from src.agent.validation.execution import ExecutionValidator
from src.agent.validation.numerical import (
    CANONICAL_FLOAT_NAN,
    CANONICAL_SQL_NULL,
    IndependentCalculator,
    NumericalValidator,
    SafeFormulaEvaluator,
    is_scalar_null,
    match_group_key_element,
)

__all__ = [
    "CANONICAL_FLOAT_NAN",
    "CANONICAL_SQL_NULL",
    "ConfidenceLevel",
    "CorrectionDiagnostic",
    "ErrorCategory",
    "ExecutionValidationResult",
    "ExecutionValidator",
    "IndependentCalculator",
    "LayerValidationResult",
    "NumericalValidationResult",
    "NumericalValidator",
    "SafeFormulaEvaluator",
    "SemanticValidationResult",
    "StepValidationResult",
    "ValidationStatus",
    "WorkflowValidationReport",
    "build_step_validation_result",
    "determine_step_confidence",
    "determine_step_status",
    "is_scalar_null",
    "match_group_key_element",
]
