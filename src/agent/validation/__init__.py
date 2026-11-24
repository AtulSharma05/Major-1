"""Three-Layer Validation & Bounded Self-Correction package.

Architectural Context (ADR-02, ADR-03, Master Execution Plan Phase 5):
    Provides deterministic verification (Execution, Numerical, Semantic) and
    bounded self-correction for autonomous analytics workflows.
"""

from src.agent.validation.contracts import (
    CANONICAL_SEMANTIC_CHECKS,
    ConfidenceLevel,
    CorrectionDiagnostic,
    ErrorCategory,
    ExecutionValidationResult,
    LayerValidationResult,
    NumericalValidationResult,
    SemanticCheckDetail,
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
from src.agent.validation.projection import (
    SUPPORTED_SCALAR_TYPES,
    CanonicalProjectionView,
    ProjectionNormalizationResult,
    ProjectionNormalizationStatus,
    ProjectionNormalizer,
)
from src.agent.validation.semantic import (
    SemanticValidator,
    resolve_semantic_validation_status,
)

__all__ = [
    "CANONICAL_FLOAT_NAN",
    "CANONICAL_SEMANTIC_CHECKS",
    "CANONICAL_SQL_NULL",
    "CanonicalProjectionView",
    "ConfidenceLevel",
    "CorrectionDiagnostic",
    "ErrorCategory",
    "ExecutionValidationResult",
    "ExecutionValidator",
    "IndependentCalculator",
    "LayerValidationResult",
    "NumericalValidationResult",
    "NumericalValidator",
    "ProjectionNormalizationResult",
    "ProjectionNormalizationStatus",
    "ProjectionNormalizer",
    "SUPPORTED_SCALAR_TYPES",
    "SafeFormulaEvaluator",
    "SemanticCheckDetail",
    "SemanticValidationResult",
    "SemanticValidator",
    "StepValidationResult",
    "ValidationStatus",
    "WorkflowValidationReport",
    "build_step_validation_result",
    "determine_step_confidence",
    "determine_step_status",
    "is_scalar_null",
    "match_group_key_element",
    "resolve_semantic_validation_status",
]
