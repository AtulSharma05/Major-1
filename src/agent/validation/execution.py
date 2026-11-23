"""Layer 1: Execution Validation for the Self-Validating Autonomous Data Analytics Agent.

Architectural Context (ADR-02, ADR-03, Master Execution Plan Phase 5 Stage 2):
    Execution validation provides the first line of defense immediately following
    code execution. It verifies:
    1. Raw physical execution success, syntax sanity, and timeout enforcement.
    2. Internal shape consistency between ExecutionResult metadata and DataFrame shape.
    3. Analytical grain classification:
       - True scalar: 1 row, 1 column, is_scalar=True (including valid NULL scalars).
       - Single-row multi-metric: 1 row, multiple columns, is_scalar=False.
       - Grouped summary: >=0 rows, dimensional + metric columns, is_scalar=False.
       - 1x1 tabular projection: dimensional projection returning 1x1, is_scalar=False.
    4. Structural schema matching against QuerySpecification metric aliases and dimensions.
    5. Structurally valid empty DataFrames (pass Layer 1, but meaningful output requires evidence).
    6. Accurate representation of schema verification when QuerySpecification is unavailable.
    7. Inter-step dependency safety gate to prevent cascading corruption.
"""

from __future__ import annotations

import pandas as pd

from src.agent.execution.engine import ExecutionResult
from src.agent.planning.intent import QuerySpecification
from src.agent.planning.planner import AnalyticalPlan, PlanStep
from src.agent.validation.contracts import (
    CorrectionDiagnostic,
    ErrorCategory,
    ExecutionValidationResult,
    NumericalValidationResult,
    SemanticValidationResult,
    StepValidationResult,
    ValidationStatus,
    build_step_validation_result,
)


class ExecutionValidator:
    """Layer 1 Validator: Validates physical execution, output shape, and schema sanity."""

    def validate_step(
        self,
        raw_execution_result: ExecutionResult | None,
        structured_intent: QuerySpecification | None = None,
        step: PlanStep | None = None,
        step_id: int = 1,
        attempt_number: int = 1,
        executed_sql: str | None = None,
    ) -> StepValidationResult:
        """Execute Layer 1 validation on a step's execution result.

        Args:
            raw_execution_result: Raw outcome from ControlledExecutionEngine.
            structured_intent: Optional QuerySpecification ground truth.
            step: Optional PlanStep containing metadata and dependencies.
            step_id: 1-based step identifier.
            attempt_number: 1-based attempt sequence number.
            executed_sql: Exact SQL text executed.

        Returns:
            Validated StepValidationResult with Layer 1 details and deferred L2/L3 placeholders.
        """
        effective_step_id = step.step_id if step else step_id

        # 1. Handle missing raw execution result
        if raw_execution_result is None:
            exec_res = ExecutionValidationResult(
                layer_name="Layer 1: Execution",
                passed=False,
                status=ValidationStatus.FAILED,
                message="Execution result missing or execution was unconfirmed.",
                is_empty=True,
            )
            diag = CorrectionDiagnostic(
                error_category=ErrorCategory.FATAL_SYSTEM,
                is_correctable=False,
                failed_layer="Layer 1: Execution",
                error_summary="Execution result missing or unconfirmed.",
                remediation_instruction=(
                    "Verify database connection and execution engine; no result was returned."
                ),
                prior_attempt_sql=executed_sql or "",
                attempt_number=attempt_number,
            )
            return self._build_step_result(
                step_id=effective_step_id,
                attempt_number=attempt_number,
                execution_res=exec_res,
                raw_execution_result=None,
                diagnostic=diag,
                executed_sql=executed_sql,
                has_meaningful_output=False,
            )

        # 2. Handle raw physical execution failure (syntax, runtime, timeout)
        if not raw_execution_result.success:
            err_msg = raw_execution_result.error_message or "Execution failed"
            err_type = raw_execution_result.error_type or "ExecutionError"
            category = self._categorize_error(raw_execution_result)
            is_correctable = category != ErrorCategory.FATAL_SYSTEM

            exec_res = ExecutionValidationResult(
                layer_name="Layer 1: Execution",
                passed=False,
                status=ValidationStatus.FAILED,
                message=f"{err_type}: {err_msg}",
                is_empty=True,
                execution_time_ms=raw_execution_result.execution_time_ms,
            )
            diag = CorrectionDiagnostic(
                error_category=category,
                is_correctable=is_correctable,
                failed_layer="Layer 1: Execution",
                error_summary=f"{err_type}: {err_msg}",
                remediation_instruction=self._build_remediation(category, err_type, err_msg),
                prior_attempt_sql=executed_sql or "",
                attempt_number=attempt_number,
            )
            return self._build_step_result(
                step_id=effective_step_id,
                attempt_number=attempt_number,
                execution_res=exec_res,
                raw_execution_result=raw_execution_result,
                diagnostic=diag,
                executed_sql=executed_sql,
                has_meaningful_output=False,
            )

        # 3. Successful raw execution: extract actual dimensions and shape
        row_count = raw_execution_result.row_count
        column_count = raw_execution_result.column_count
        df = raw_execution_result.dataframe
        actual_columns = (
            list(raw_execution_result.columns)
            if raw_execution_result.columns
            else (list(df.columns) if df is not None else [])
        )
        exec_time = raw_execution_result.execution_time_ms

        # 3a. Require actual DataFrame for successful ExecutionResult
        if df is None:
            exec_res = ExecutionValidationResult(
                layer_name="Layer 1: Execution",
                passed=False,
                status=ValidationStatus.FAILED,
                message="Execution succeeded but no DataFrame was returned by the engine.",
                is_empty=True,
                is_scalar=raw_execution_result.is_scalar,
                row_count=row_count,
                column_count=column_count,
                schema_matched=False,
                execution_time_ms=exec_time,
            )
            diag = CorrectionDiagnostic(
                error_category=ErrorCategory.FATAL_SYSTEM,
                is_correctable=False,
                failed_layer="Layer 1: Execution",
                error_summary="Execution succeeded but DataFrame is None.",
                remediation_instruction=(
                    "Verify execution engine pipeline; successful queries must produce a DataFrame."
                ),
                prior_attempt_sql=executed_sql or "",
                attempt_number=attempt_number,
            )
            return self._build_step_result(
                step_id=effective_step_id,
                attempt_number=attempt_number,
                execution_res=exec_res,
                raw_execution_result=raw_execution_result,
                diagnostic=diag,
                executed_sql=executed_sql,
                has_meaningful_output=False,
            )

        # 3b. Shape consistency check between ExecutionResult metadata and DataFrame shape
        df_rows, df_cols = df.shape
        if df_rows != row_count or df_cols != column_count:
            exec_res = ExecutionValidationResult(
                layer_name="Layer 1: Execution",
                passed=False,
                status=ValidationStatus.FAILED,
                message=(
                    f"ExecutionResult shape inconsistency: reported counts "
                    f"({row_count} rows, {column_count} cols) do not match "
                    f"DataFrame shape ({df_rows} rows, {df_cols} cols)."
                ),
                is_empty=raw_execution_result.is_empty(),
                is_scalar=raw_execution_result.is_scalar,
                row_count=row_count,
                column_count=column_count,
                schema_matched=False,
                execution_time_ms=exec_time,
            )
            diag = CorrectionDiagnostic(
                error_category=ErrorCategory.FATAL_SYSTEM,
                is_correctable=False,
                failed_layer="Layer 1: Execution",
                error_summary="ExecutionResult shape mismatch with actual DataFrame.",
                remediation_instruction=(
                    "Ensure execution engine produces consistent row and column metadata."
                ),
                prior_attempt_sql=executed_sql or "",
                attempt_number=attempt_number,
            )
            return self._build_step_result(
                step_id=effective_step_id,
                attempt_number=attempt_number,
                execution_res=exec_res,
                raw_execution_result=raw_execution_result,
                diagnostic=diag,
                executed_sql=executed_sql,
                has_meaningful_output=False,
            )

        # 3c. Detect duplicate DataFrame column names, case-insensitively
        seen_cols: set[str] = set()
        duplicate_cols: list[str] = []
        for col in actual_columns:
            col_lower = str(col).lower()
            if col_lower in seen_cols:
                if str(col) not in duplicate_cols:
                    duplicate_cols.append(str(col))
            else:
                seen_cols.add(col_lower)

        if duplicate_cols:
            exec_res = ExecutionValidationResult(
                layer_name="Layer 1: Execution",
                passed=False,
                status=ValidationStatus.FAILED,
                message=(
                    f"Duplicate column name(s) detected in query output: {duplicate_cols}. "
                    "Projections must produce unique column names."
                ),
                is_empty=raw_execution_result.is_empty(),
                is_scalar=raw_execution_result.is_scalar,
                row_count=row_count,
                column_count=column_count,
                schema_matched=False,
                execution_time_ms=exec_time,
                details={"duplicate_columns": tuple(duplicate_cols)},
            )
            diag = CorrectionDiagnostic(
                error_category=ErrorCategory.SCHEMA_ERROR,
                is_correctable=True,
                failed_layer="Layer 1: Execution",
                error_summary=f"Duplicate column name(s) in projection: {duplicate_cols}.",
                remediation_instruction=(
                    "Ensure all projected columns have unique aliases in the SELECT clause."
                ),
                prior_attempt_sql=executed_sql or "",
                attempt_number=attempt_number,
            )
            return self._build_step_result(
                step_id=effective_step_id,
                attempt_number=attempt_number,
                execution_res=exec_res,
                raw_execution_result=raw_execution_result,
                diagnostic=diag,
                executed_sql=executed_sql,
                has_meaningful_output=False,
            )

        # 4. Missing QuerySpecification: Option C
        # Physical execution verified (passed=True, status=VERIFIED), but schema_matched=False
        # Downstream layers mark INCONCLUSIVE; overall step is INCONCLUSIVE (never VERIFIED).
        if structured_intent is None:
            exec_res = ExecutionValidationResult(
                layer_name="Layer 1: Execution",
                passed=True,
                status=ValidationStatus.VERIFIED,
                message=(
                    "Physical execution succeeded; schema verification was not performed "
                    "because structured intent (QuerySpecification) is unavailable."
                ),
                is_empty=raw_execution_result.is_empty(),
                is_scalar=raw_execution_result.is_scalar,
                row_count=row_count,
                column_count=column_count,
                schema_matched=False,
                execution_time_ms=exec_time,
                details={
                    "schema_verification_performed": False,
                    "schema_unverified_reason": "QuerySpecification unavailable",
                    "actual_columns": tuple(actual_columns),
                    "row_count": row_count,
                    "column_count": column_count,
                    "is_scalar": raw_execution_result.is_scalar,
                },
            )
            numerical_res = NumericalValidationResult(
                layer_name="Layer 2: Numerical",
                passed=False,
                status=ValidationStatus.INCONCLUSIVE,
                message=(
                    "Numerical verification cannot proceed because structured intent "
                    "(QuerySpecification) is unavailable."
                ),
                inconclusive_reason="QuerySpecification unavailable",
            )
            semantic_res = SemanticValidationResult(
                layer_name="Layer 3: Semantic",
                passed=False,
                status=ValidationStatus.INCONCLUSIVE,
                message=(
                    "Semantic coverage cannot be established because structured intent "
                    "(QuerySpecification) is unavailable."
                ),
                inconclusive_reason="QuerySpecification unavailable",
            )
            has_meaningful = raw_execution_result.is_scalar or (row_count > 0)
            return build_step_validation_result(
                step_id=effective_step_id,
                attempt_number=attempt_number,
                execution_res=exec_res,
                numerical_res=numerical_res,
                semantic_res=semantic_res,
                diagnostic=None,
                executed_sql=executed_sql,
                raw_execution_result=raw_execution_result,
                has_meaningful_output=has_meaningful,
            )

        # 5. Intent-guided analytical grain classification
        grain = self._classify_grain(structured_intent)

        # 5a. True scalar grain (0 dimensions, exactly 1 metric)
        if grain == "scalar":
            return self._validate_scalar(
                effective_step_id=effective_step_id,
                attempt_number=attempt_number,
                raw_res=raw_execution_result,
                structured_intent=structured_intent,
                executed_sql=executed_sql,
            )

        # 5b. Single-row multi-metric grain (0 dimensions, >1 metrics)
        if grain == "single_row_multi_metric":
            return self._validate_single_row_multi_metric(
                effective_step_id=effective_step_id,
                attempt_number=attempt_number,
                raw_res=raw_execution_result,
                structured_intent=structured_intent,
                executed_sql=executed_sql,
            )

        # 5c. Grouped summary grain (>0 dimensions)
        if grain == "grouped":
            return self._validate_grouped(
                effective_step_id=effective_step_id,
                attempt_number=attempt_number,
                raw_res=raw_execution_result,
                structured_intent=structured_intent,
                executed_sql=executed_sql,
            )

        # 5d. General tabular projection (fallback)
        return self._validate_tabular(
            effective_step_id=effective_step_id,
            attempt_number=attempt_number,
            raw_res=raw_execution_result,
            structured_intent=structured_intent,
            executed_sql=executed_sql,
        )

    def evaluate_routing(
        self,
        plan: AnalyticalPlan | None,
        step_id: int,
        step_validation: StepValidationResult,
        current_step_index: int,
    ) -> tuple[str, str | None]:
        """Determine next workflow routing step and evaluate inter-step safety gates.

        Rules:
            1. If Layer 1 execution failed -> route to 'error_analyzer'.
            2. If step is INCONCLUSIVE:
               - If dependent steps require this step's output -> block execution and route
                 to 'error_analyzer' because inter-step column requirements cannot be verified.
               - If no dependent steps require this output:
                 - Final step -> route to 'assembler' (preserves output without VERIFIED status).
                 - Independent step -> route to 'next_step'.
            3. If Layer 1 verified:
               - Final step -> route to 'assembler'.
               - More steps remaining -> route to 'next_step'.

        Returns:
            Tuple of (route_target, blocking_reason).
        """
        # Rule 1: Execution or numerical validation failure
        if (
            step_validation.status == ValidationStatus.FAILED
            or not step_validation.execution_result.passed
            or step_validation.execution_result.status == ValidationStatus.FAILED
            or step_validation.numerical_result.status == ValidationStatus.FAILED
        ):
            err_msg = (
                step_validation.numerical_result.message
                if step_validation.numerical_result.status == ValidationStatus.FAILED
                else (step_validation.execution_result.message or "Step validation failed")
            )
            return "error_analyzer", err_msg

        is_final_step = (plan is None) or (current_step_index + 1 >= len(plan.execution_order))

        # Rule 2: Inconclusive validation safety gate
        if (
            step_validation.status == ValidationStatus.INCONCLUSIVE
            or step_validation.execution_result.status == ValidationStatus.INCONCLUSIVE
        ):
            # Check for downstream dependent steps
            if plan is not None:
                dependent_steps = [s for s in plan.steps if step_id in s.depends_on]
                if dependent_steps:
                    dep_ids = [s.step_id for s in dependent_steps]
                    reason = (
                        f"Step {step_id} validation is INCONCLUSIVE. Dependent step(s) {dep_ids} "
                        "require its output, but inter-step column requirements cannot be "
                        "verified against the existing workflow. Execution blocked to prevent "
                        "cascading corruption."
                    )
                    return "error_analyzer", reason

            # No dependent steps
            if is_final_step:
                return "assembler", None
            return "next_step", None

        # Rule 3: Execution validation passed
        if is_final_step:
            return "assembler", None
        return "next_step", None

    # --- Private Validation Helpers ---

    def _classify_grain(self, intent: QuerySpecification) -> str:
        """Classify analytical intent into scalar, single_row_multi_metric, or grouped grain."""
        if intent.is_scalar():
            return "scalar"
        if intent.is_grouped():
            return "grouped"
        if len(intent.dimensions) == 0 and len(intent.metrics) > 1:
            return "single_row_multi_metric"
        return "tabular"

    def _validate_scalar(
        self,
        effective_step_id: int,
        attempt_number: int,
        raw_res: ExecutionResult,
        structured_intent: QuerySpecification,
        executed_sql: str | None,
    ) -> StepValidationResult:
        """Validate a true scalar result (1x1), supporting NULL scalars and alias matching."""
        row_count = raw_res.row_count
        col_count = raw_res.column_count
        val = raw_res.scalar_value
        df = raw_res.dataframe
        actual_cols = list(df.columns) if df is not None else []

        # 1. Physical shape check: scalar requires exactly 1 row and 1 column
        if row_count != 1 or col_count != 1:
            exec_res = ExecutionValidationResult(
                layer_name="Layer 1: Execution",
                passed=False,
                status=ValidationStatus.FAILED,
                message=(
                    f"Grain mismatch: expected scalar result (1x1), "
                    f"got {row_count} rows and {col_count} columns."
                ),
                is_empty=raw_res.is_empty(),
                is_scalar=False,
                row_count=row_count,
                column_count=col_count,
                schema_matched=False,
                execution_time_ms=raw_res.execution_time_ms,
            )
            diag = CorrectionDiagnostic(
                error_category=ErrorCategory.SCHEMA_ERROR,
                is_correctable=True,
                failed_layer="Layer 1: Execution",
                error_summary=f"Grain mismatch: expected 1x1 scalar, got {row_count}x{col_count}.",
                remediation_instruction=(
                    "Ensure query produces a single scalar aggregate without grouping dimensions."
                ),
                prior_attempt_sql=executed_sql or "",
                attempt_number=attempt_number,
            )
            return self._build_step_result(
                step_id=effective_step_id,
                attempt_number=attempt_number,
                execution_res=exec_res,
                raw_execution_result=raw_res,
                diagnostic=diag,
                executed_sql=executed_sql,
                has_meaningful_output=False,
            )

        # 2. Schema check: validate returned column against expected metric alias (Finding 2)
        expected_alias = structured_intent.metrics[0].name if structured_intent.metrics else None
        if expected_alias is not None and actual_cols:
            actual_col = actual_cols[0]
            if actual_col.lower() != expected_alias.lower():
                exec_res = ExecutionValidationResult(
                    layer_name="Layer 1: Execution",
                    passed=False,
                    status=ValidationStatus.FAILED,
                    message=(
                        f"Scalar schema mismatch: expected column alias '{expected_alias}', "
                        f"got '{actual_col}'."
                    ),
                    is_empty=raw_res.is_empty(),
                    is_scalar=True,
                    row_count=1,
                    column_count=1,
                    schema_matched=False,
                    execution_time_ms=raw_res.execution_time_ms,
                    details={
                        "expected_alias": expected_alias,
                        "actual_column": actual_col,
                    },
                )
                diag = CorrectionDiagnostic(
                    error_category=ErrorCategory.SCHEMA_ERROR,
                    is_correctable=True,
                    failed_layer="Layer 1: Execution",
                    error_summary=(
                        f"Expected scalar alias '{expected_alias}', got '{actual_col}'."
                    ),
                    remediation_instruction=(
                        f"Ensure the SELECT expression aliases the aggregate as '{expected_alias}'."
                    ),
                    prior_attempt_sql=executed_sql or "",
                    attempt_number=attempt_number,
                )
                return self._build_step_result(
                    step_id=effective_step_id,
                    attempt_number=attempt_number,
                    execution_res=exec_res,
                    raw_execution_result=raw_res,
                    diagnostic=diag,
                    executed_sql=executed_sql,
                    has_meaningful_output=False,
                )

        # 3. Shape and alias matched! NULL scalar distinction preserved
        is_null_scalar = val is None or pd.isna(val)
        metric_name = expected_alias or "scalar"

        exec_res = ExecutionValidationResult(
            layer_name="Layer 1: Execution",
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="Scalar execution shape, grain, and schema alias verified successfully.",
            is_empty=is_null_scalar,
            is_scalar=True,
            row_count=1,
            column_count=1,
            schema_matched=True,
            execution_time_ms=raw_res.execution_time_ms,
            details={
                "metric_name": metric_name,
                "is_null_scalar": is_null_scalar,
                "scalar_value": None if is_null_scalar else val,
            },
        )
        return self._build_step_result(
            step_id=effective_step_id,
            attempt_number=attempt_number,
            execution_res=exec_res,
            raw_execution_result=raw_res,
            diagnostic=None,
            executed_sql=executed_sql,
            has_meaningful_output=True,
        )

    def _validate_single_row_multi_metric(
        self,
        effective_step_id: int,
        attempt_number: int,
        raw_res: ExecutionResult,
        structured_intent: QuerySpecification,
        executed_sql: str | None,
    ) -> StepValidationResult:
        """Validate a 1-row, multi-metric summary result, rejecting unexpected columns."""
        row_count = raw_res.row_count
        df = raw_res.dataframe
        actual_cols = list(df.columns) if df is not None else []
        actual_cols_lower = {c.lower(): c for c in actual_cols}

        expected_metrics = [m.name for m in structured_intent.metrics]
        expected_metrics_lower = {m.name.lower() for m in structured_intent.metrics}

        missing_metrics = [m for m in expected_metrics if m.lower() not in actual_cols_lower]
        unexpected_cols = [c for c in actual_cols if c.lower() not in expected_metrics_lower]

        # Single-row summary requires exactly 1 row
        if row_count != 1:
            exec_res = ExecutionValidationResult(
                layer_name="Layer 1: Execution",
                passed=False,
                status=ValidationStatus.FAILED,
                message=(
                    f"Row count mismatch for multi-metric summary: expected 1 row, "
                    f"got {row_count} rows."
                ),
                is_empty=raw_res.is_empty(),
                is_scalar=False,
                row_count=row_count,
                column_count=raw_res.column_count,
                schema_matched=False,
                execution_time_ms=raw_res.execution_time_ms,
            )
            diag = CorrectionDiagnostic(
                error_category=ErrorCategory.SCHEMA_ERROR,
                is_correctable=True,
                failed_layer="Layer 1: Execution",
                error_summary=f"Expected 1-row multi-metric summary, got {row_count} rows.",
                remediation_instruction=(
                    "Remove grouping dimensions or LIMIT to ensure exactly 1 summary row."
                ),
                prior_attempt_sql=executed_sql or "",
                attempt_number=attempt_number,
            )
            return self._build_step_result(
                step_id=effective_step_id,
                attempt_number=attempt_number,
                execution_res=exec_res,
                raw_execution_result=raw_res,
                diagnostic=diag,
                executed_sql=executed_sql,
                has_meaningful_output=False,
            )

        # Check for missing metric columns
        if missing_metrics:
            exec_res = ExecutionValidationResult(
                layer_name="Layer 1: Execution",
                passed=False,
                status=ValidationStatus.FAILED,
                message=f"Missing expected metric column(s): {missing_metrics}",
                is_empty=raw_res.is_empty(),
                is_scalar=False,
                row_count=row_count,
                column_count=raw_res.column_count,
                schema_matched=False,
                execution_time_ms=raw_res.execution_time_ms,
            )
            diag = CorrectionDiagnostic(
                error_category=ErrorCategory.SCHEMA_ERROR,
                is_correctable=True,
                failed_layer="Layer 1: Execution",
                error_summary=f"Missing metric column(s): {missing_metrics}",
                remediation_instruction=(
                    f"Ensure SELECT aliases include all expected metrics: {expected_metrics}."
                ),
                prior_attempt_sql=executed_sql or "",
                attempt_number=attempt_number,
            )
            return self._build_step_result(
                step_id=effective_step_id,
                attempt_number=attempt_number,
                execution_res=exec_res,
                raw_execution_result=raw_res,
                diagnostic=diag,
                executed_sql=executed_sql,
                has_meaningful_output=False,
            )

        # Check for unexpected extra columns (Finding 4)
        if unexpected_cols:
            exec_res = ExecutionValidationResult(
                layer_name="Layer 1: Execution",
                passed=False,
                status=ValidationStatus.FAILED,
                message=f"Unexpected column(s) in multi-metric summary: {unexpected_cols}",
                is_empty=raw_res.is_empty(),
                is_scalar=False,
                row_count=row_count,
                column_count=raw_res.column_count,
                schema_matched=False,
                execution_time_ms=raw_res.execution_time_ms,
            )
            diag = CorrectionDiagnostic(
                error_category=ErrorCategory.SCHEMA_ERROR,
                is_correctable=True,
                failed_layer="Layer 1: Execution",
                error_summary=f"Unexpected column(s) in summary: {unexpected_cols}",
                remediation_instruction=(
                    f"Remove unexpected column(s) {unexpected_cols}. "
                    f"Return only expected metrics: {expected_metrics}."
                ),
                prior_attempt_sql=executed_sql or "",
                attempt_number=attempt_number,
            )
            return self._build_step_result(
                step_id=effective_step_id,
                attempt_number=attempt_number,
                execution_res=exec_res,
                raw_execution_result=raw_res,
                diagnostic=diag,
                executed_sql=executed_sql,
                has_meaningful_output=False,
            )

        exec_res = ExecutionValidationResult(
            layer_name="Layer 1: Execution",
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="Single-row multi-metric schema and shape verified successfully.",
            is_empty=False,
            is_scalar=False,
            row_count=1,
            column_count=len(actual_cols),
            schema_matched=True,
            execution_time_ms=raw_res.execution_time_ms,
            details={
                "expected_metrics": tuple(expected_metrics),
                "actual_columns": tuple(actual_cols),
            },
        )
        return self._build_step_result(
            step_id=effective_step_id,
            attempt_number=attempt_number,
            execution_res=exec_res,
            raw_execution_result=raw_res,
            diagnostic=None,
            executed_sql=executed_sql,
            has_meaningful_output=True,
        )

    def _validate_grouped(
        self,
        effective_step_id: int,
        attempt_number: int,
        raw_res: ExecutionResult,
        structured_intent: QuerySpecification,
        executed_sql: str | None,
    ) -> StepValidationResult:
        """Validate a grouped tabular result, including structurally valid empty DataFrames."""
        row_count = raw_res.row_count
        df = raw_res.dataframe
        actual_cols = list(df.columns) if df is not None else []
        actual_cols_lower = {c.lower(): c for c in actual_cols}

        expected_dims = list(structured_intent.dimensions)
        expected_metrics = [m.name for m in structured_intent.metrics]
        expected_all_lower = {d.lower() for d in expected_dims} | {
            m.lower() for m in expected_metrics
        }

        missing_dims = [d for d in expected_dims if d.lower() not in actual_cols_lower]
        missing_metrics = [m for m in expected_metrics if m.lower() not in actual_cols_lower]
        missing_all = missing_dims + missing_metrics
        unexpected_cols = [c for c in actual_cols if c.lower() not in expected_all_lower]

        if missing_all:
            exec_res = ExecutionValidationResult(
                layer_name="Layer 1: Execution",
                passed=False,
                status=ValidationStatus.FAILED,
                message=f"Schema mismatch: missing column(s) {missing_all}",
                is_empty=raw_res.is_empty(),
                is_scalar=False,
                row_count=row_count,
                column_count=raw_res.column_count,
                schema_matched=False,
                execution_time_ms=raw_res.execution_time_ms,
            )
            diag = CorrectionDiagnostic(
                error_category=ErrorCategory.SCHEMA_ERROR,
                is_correctable=True,
                failed_layer="Layer 1: Execution",
                error_summary=f"Missing expected grouping or metric column(s): {missing_all}",
                remediation_instruction=(
                    f"Include all grouping dimensions {expected_dims} and metric "
                    f"aliases {expected_metrics}."
                ),
                prior_attempt_sql=executed_sql or "",
                attempt_number=attempt_number,
            )
            return self._build_step_result(
                step_id=effective_step_id,
                attempt_number=attempt_number,
                execution_res=exec_res,
                raw_execution_result=raw_res,
                diagnostic=diag,
                executed_sql=executed_sql,
                has_meaningful_output=False,
            )

        if unexpected_cols:
            exec_res = ExecutionValidationResult(
                layer_name="Layer 1: Execution",
                passed=False,
                status=ValidationStatus.FAILED,
                message=f"Unexpected column(s) in grouped summary: {unexpected_cols}",
                is_empty=raw_res.is_empty(),
                is_scalar=False,
                row_count=row_count,
                column_count=raw_res.column_count,
                schema_matched=False,
                execution_time_ms=raw_res.execution_time_ms,
            )
            diag = CorrectionDiagnostic(
                error_category=ErrorCategory.SCHEMA_ERROR,
                is_correctable=True,
                failed_layer="Layer 1: Execution",
                error_summary=f"Unexpected column(s) in grouped summary: {unexpected_cols}",
                remediation_instruction=(
                    f"Remove unexpected column(s) {unexpected_cols}. "
                    f"Project only specified dimensions {expected_dims} and "
                    f"metrics {expected_metrics}."
                ),
                prior_attempt_sql=executed_sql or "",
                attempt_number=attempt_number,
            )
            return self._build_step_result(
                step_id=effective_step_id,
                attempt_number=attempt_number,
                execution_res=exec_res,
                raw_execution_result=raw_res,
                diagnostic=diag,
                executed_sql=executed_sql,
                has_meaningful_output=False,
            )

        # Distinguish 1x1 projection from scalar: if grouped query returns 1x1, it is tabular
        is_empty_df = row_count == 0

        # Finding 3: Structurally valid empty DataFrames pass Layer 1, but do NOT automatically
        # claim meaningful output without explicit evidence.
        has_meaningful = not is_empty_df

        exec_res = ExecutionValidationResult(
            layer_name="Layer 1: Execution",
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="Grouped summary schema and structure verified successfully.",
            is_empty=is_empty_df,
            is_scalar=False,  # Tabular grouping projection is never a primitive scalar
            row_count=row_count,
            column_count=len(actual_cols),
            schema_matched=True,
            execution_time_ms=raw_res.execution_time_ms,
            details={
                "matched_dimensions": tuple(expected_dims),
                "matched_metrics": tuple(expected_metrics),
                "row_count": row_count,
                "is_empty": is_empty_df,
            },
        )
        return self._build_step_result(
            step_id=effective_step_id,
            attempt_number=attempt_number,
            execution_res=exec_res,
            raw_execution_result=raw_res,
            diagnostic=None,
            executed_sql=executed_sql,
            has_meaningful_output=has_meaningful,
        )

    def _validate_tabular(
        self,
        effective_step_id: int,
        attempt_number: int,
        raw_res: ExecutionResult,
        structured_intent: QuerySpecification,
        executed_sql: str | None,
    ) -> StepValidationResult:
        """Validate generic tabular output. Verify expected structure, otherwise INCONCLUSIVE."""
        df = raw_res.dataframe
        actual_cols = list(df.columns) if df is not None else []
        actual_cols_lower = {c.lower(): c for c in actual_cols}
        row_count = raw_res.row_count

        expected_cols: list[str] = []
        if structured_intent.dimensions:
            expected_cols.extend(structured_intent.dimensions)
        if structured_intent.metrics:
            expected_cols.extend([m.name for m in structured_intent.metrics])

        # If expected structure can be established (Finding 5)
        if expected_cols:
            expected_cols_lower = {c.lower() for c in expected_cols}
            missing_cols = [c for c in expected_cols if c.lower() not in actual_cols_lower]
            unexpected_cols = [c for c in actual_cols if c.lower() not in expected_cols_lower]
            if missing_cols:
                exec_res = ExecutionValidationResult(
                    layer_name="Layer 1: Execution",
                    passed=False,
                    status=ValidationStatus.FAILED,
                    message=f"Tabular schema mismatch: missing column(s) {missing_cols}",
                    is_empty=raw_res.is_empty(),
                    is_scalar=False,
                    row_count=row_count,
                    column_count=len(actual_cols),
                    schema_matched=False,
                    execution_time_ms=raw_res.execution_time_ms,
                )
                diag = CorrectionDiagnostic(
                    error_category=ErrorCategory.SCHEMA_ERROR,
                    is_correctable=True,
                    failed_layer="Layer 1: Execution",
                    error_summary=f"Missing expected tabular column(s): {missing_cols}",
                    remediation_instruction=(
                        f"Include required columns in projection: {expected_cols}."
                    ),
                    prior_attempt_sql=executed_sql or "",
                    attempt_number=attempt_number,
                )
                return self._build_step_result(
                    step_id=effective_step_id,
                    attempt_number=attempt_number,
                    execution_res=exec_res,
                    raw_execution_result=raw_res,
                    diagnostic=diag,
                    executed_sql=executed_sql,
                    has_meaningful_output=False,
                )

            if unexpected_cols:
                exec_res = ExecutionValidationResult(
                    layer_name="Layer 1: Execution",
                    passed=False,
                    status=ValidationStatus.FAILED,
                    message=f"Tabular schema mismatch: unexpected column(s) {unexpected_cols}",
                    is_empty=raw_res.is_empty(),
                    is_scalar=False,
                    row_count=row_count,
                    column_count=len(actual_cols),
                    schema_matched=False,
                    execution_time_ms=raw_res.execution_time_ms,
                )
                diag = CorrectionDiagnostic(
                    error_category=ErrorCategory.SCHEMA_ERROR,
                    is_correctable=True,
                    failed_layer="Layer 1: Execution",
                    error_summary=f"Unexpected column(s) in projection: {unexpected_cols}",
                    remediation_instruction=(
                        f"Remove unexpected column(s) {unexpected_cols}. "
                        f"Project only specified columns: {expected_cols}."
                    ),
                    prior_attempt_sql=executed_sql or "",
                    attempt_number=attempt_number,
                )
                return self._build_step_result(
                    step_id=effective_step_id,
                    attempt_number=attempt_number,
                    execution_res=exec_res,
                    raw_execution_result=raw_res,
                    diagnostic=diag,
                    executed_sql=executed_sql,
                    has_meaningful_output=False,
                )

            exec_res = ExecutionValidationResult(
                layer_name="Layer 1: Execution",
                passed=True,
                status=ValidationStatus.VERIFIED,
                message="Tabular output and expected structure verified successfully.",
                is_empty=raw_res.is_empty(),
                is_scalar=False,
                row_count=row_count,
                column_count=len(actual_cols),
                schema_matched=True,
                execution_time_ms=raw_res.execution_time_ms,
                details={
                    "expected_columns": tuple(expected_cols),
                    "actual_columns": tuple(actual_cols),
                    "row_count": row_count,
                },
            )
            return self._build_step_result(
                step_id=effective_step_id,
                attempt_number=attempt_number,
                execution_res=exec_res,
                raw_execution_result=raw_res,
                diagnostic=None,
                executed_sql=executed_sql,
                has_meaningful_output=(row_count > 0),
            )

        # Expected structure cannot be established (Finding 5)
        # Follow Option C: physical execution passed, but schema verification is not performed
        exec_res = ExecutionValidationResult(
            layer_name="Layer 1: Execution",
            passed=True,
            status=ValidationStatus.VERIFIED,
            message=(
                "Physical execution succeeded; schema verification was not performed "
                "because no expected column structure was defined in the specification."
            ),
            is_empty=raw_res.is_empty(),
            is_scalar=False,
            row_count=row_count,
            column_count=len(actual_cols),
            schema_matched=False,
            execution_time_ms=raw_res.execution_time_ms,
            details={
                "schema_verification_performed": False,
                "schema_unverified_reason": "No expected columns defined in specification",
                "actual_columns": tuple(actual_cols),
                "row_count": row_count,
            },
        )
        numerical_res = NumericalValidationResult(
            layer_name="Layer 2: Numerical",
            passed=False,
            status=ValidationStatus.INCONCLUSIVE,
            message="Numerical verification inconclusive without defined column structure.",
            inconclusive_reason="Undefined column structure",
        )
        semantic_res = SemanticValidationResult(
            layer_name="Layer 3: Semantic",
            passed=False,
            status=ValidationStatus.INCONCLUSIVE,
            message="Semantic coverage inconclusive without defined column structure.",
            inconclusive_reason="Undefined column structure",
        )
        return build_step_validation_result(
            step_id=effective_step_id,
            attempt_number=attempt_number,
            execution_res=exec_res,
            numerical_res=numerical_res,
            semantic_res=semantic_res,
            diagnostic=None,
            executed_sql=executed_sql,
            raw_execution_result=raw_res,
            has_meaningful_output=(row_count > 0),
        )

    def _build_step_result(
        self,
        step_id: int,
        attempt_number: int,
        execution_res: ExecutionValidationResult,
        raw_execution_result: ExecutionResult | None,
        diagnostic: CorrectionDiagnostic | None,
        executed_sql: str | None,
        has_meaningful_output: bool,
    ) -> StepValidationResult:
        """Construct StepValidationResult with Stage 3 & 4 placeholders."""
        numerical_res = NumericalValidationResult(
            layer_name="Layer 2: Numerical",
            passed=False,
            status=ValidationStatus.UNVERIFIED,
            message="Numerical validation deferred to Stage 3.",
        )
        semantic_res = SemanticValidationResult(
            layer_name="Layer 3: Semantic",
            passed=False,
            status=ValidationStatus.UNVERIFIED,
            message="Semantic validation deferred to Stage 4.",
        )

        return build_step_validation_result(
            step_id=step_id,
            attempt_number=attempt_number,
            execution_res=execution_res,
            numerical_res=numerical_res,
            semantic_res=semantic_res,
            diagnostic=diagnostic,
            executed_sql=executed_sql,
            raw_execution_result=raw_execution_result,
            has_meaningful_output=has_meaningful_output,
        )

    def _categorize_error(self, raw_res: ExecutionResult) -> ErrorCategory:
        """Map raw ExecutionResult error type to typed ErrorCategory."""
        err_type = (raw_res.error_type or "").upper()
        if "COLUMN" in err_type or "TABLE" in err_type:
            return ErrorCategory.SCHEMA_ERROR
        if "TIMEOUT" in err_type:
            return ErrorCategory.FATAL_SYSTEM
        return ErrorCategory.SYNTAX_ERROR

    def _build_remediation(self, category: ErrorCategory, err_type: str, err_msg: str) -> str:
        """Generate actionable remediation instructions for the code generator."""
        if category == ErrorCategory.SCHEMA_ERROR:
            return f"Correct schema identifier: {err_msg}. Check active table column profile."
        if category == ErrorCategory.FATAL_SYSTEM:
            return f"Fatal execution failure ({err_type}): {err_msg}. Not correctable by SQL retry."
        return f"Fix SQL syntax error: {err_msg}. Follow standard DuckDB SQL dialect."
