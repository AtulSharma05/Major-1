"""Unit tests for Phase 5 Validation Contracts, Status Models, and Policy Invariants.

Validates:
    1. Exhaustive enum definitions for ValidationStatus, ConfidenceLevel, and ErrorCategory.
    2. Valid and invalid StepValidationResult construction enforcing Pydantic invariants.
    3. Status and confidence independence (e.g. MEDIUM confidence for INCONCLUSIVE).
    4. Strict invariant: VERIFIED status requires all three validation layers to pass.
    5. Strict invariant: HIGH confidence can NEVER be assigned unless status is VERIFIED.
    6. Non-correctable ambiguity remains INCONCLUSIVE (distinct from UNVERIFIED).
    7. Full per-step retry history preservation in WorkflowValidationReport.
    8. Result Availability Policy: retention of last successful execution without misrepresentation.
    9. Workflow summary consistency across single-step and multi-step combinations.
    10. Regression tests for raw execution success, layer pass/status invariants, and immutability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.agent.execution.engine import ExecutionResult
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


def _make_raw_ok(
    query: str = "SELECT 1",
    dataframe: pd.DataFrame | None = None,
    is_scalar: bool = False,
    scalar_value: Any = None,
) -> ExecutionResult:
    """Helper to create a successful ExecutionResult."""
    if dataframe is None and not is_scalar:
        dataframe = pd.DataFrame({"col": [100.0]})
    return ExecutionResult(
        success=True,
        query=query,
        dataframe=dataframe,
        is_scalar=is_scalar,
        scalar_value=scalar_value,
        row_count=len(dataframe) if dataframe is not None else 1,
        column_count=len(dataframe.columns) if dataframe is not None else 1,
    )


def _make_raw_fail(
    query: str = "SELECT bad_column FROM table",
    error_message: str = "Binder Error: column bad_column not found",
) -> ExecutionResult:
    """Helper to create a failed ExecutionResult."""
    return ExecutionResult(
        success=False,
        query=query,
        error_type="BinderError",
        error_message=error_message,
    )


def _make_l1(
    passed: bool = True,
    status: ValidationStatus = ValidationStatus.VERIFIED,
) -> ExecutionValidationResult:
    return ExecutionValidationResult(
        layer_name="Layer 1: Execution",
        passed=passed,
        status=status,
        message="Execution verified" if passed else "Execution failed",
        row_count=10 if passed else 0,
        column_count=2 if passed else 0,
    )


def _make_l2(
    passed: bool = True,
    status: ValidationStatus = ValidationStatus.VERIFIED,
) -> NumericalValidationResult:
    return NumericalValidationResult(
        layer_name="Layer 2: Numerical",
        passed=passed,
        status=status,
        message="Numerical match" if passed else "Tolerance breach",
        duckdb_value=500.0,
        independent_value=500.0 if passed else 450.0,
        absolute_difference=0.0 if passed else 50.0,
    )


def _make_l3(
    passed: bool = True,
    status: ValidationStatus = ValidationStatus.VERIFIED,
) -> SemanticValidationResult:
    return SemanticValidationResult(
        layer_name="Layer 3: Semantic",
        passed=passed,
        status=status,
        message="Semantic intent verified" if passed else "Filter dropped",
        metrics_covered=["total_sales"],
        dimensions_covered=["region"],
    )


def test_validation_statuses_and_confidence_enum_values():
    """Verify all required enum constants exist and have distinct identity."""
    assert ValidationStatus.VERIFIED == "VERIFIED"
    assert ValidationStatus.FAILED == "FAILED"
    assert ValidationStatus.INCONCLUSIVE == "INCONCLUSIVE"
    assert ValidationStatus.UNVERIFIED == "UNVERIFIED"

    assert ConfidenceLevel.HIGH == "HIGH"
    assert ConfidenceLevel.MEDIUM == "MEDIUM"
    assert ConfidenceLevel.LOW == "LOW"

    assert ErrorCategory.SYNTAX_ERROR == "SYNTAX_ERROR"
    assert ErrorCategory.SCHEMA_ERROR == "SCHEMA_ERROR"
    assert ErrorCategory.NUMERICAL_MISMATCH == "NUMERICAL_MISMATCH"
    assert ErrorCategory.SEMANTIC_OMISSION == "SEMANTIC_OMISSION"
    assert ErrorCategory.INTERVAL_LEAK == "INTERVAL_LEAK"
    assert ErrorCategory.AMBIGUOUS_DATA == "AMBIGUOUS_DATA"
    assert ErrorCategory.FATAL_SYSTEM == "FATAL_SYSTEM"


def test_step_validation_result_valid_construction():
    """Verify valid construction of StepValidationResult when all three layers pass."""
    raw_ok = _make_raw_ok()
    step_res = StepValidationResult(
        step_id=1,
        attempt_number=1,
        status=ValidationStatus.VERIFIED,
        confidence=ConfidenceLevel.HIGH,
        overall_passed=True,
        execution_result=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
        numerical_result=_make_l2(passed=True, status=ValidationStatus.VERIFIED),
        semantic_result=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
        raw_execution_result=raw_ok,
        executed_sql="SELECT region, SUM(sales) FROM superstore GROUP BY region",
    )
    assert step_res.step_id == 1
    assert step_res.status == ValidationStatus.VERIFIED
    assert step_res.confidence == ConfidenceLevel.HIGH
    assert step_res.overall_passed is True
    assert step_res.raw_execution_result is not None
    assert step_res.raw_execution_result.success is True


def test_confidence_high_rejected_if_not_verified():
    """Invariant: HIGH confidence can NEVER be assigned to non-VERIFIED results."""
    raw_ok = _make_raw_ok()

    # Attempting HIGH confidence on FAILED result
    with pytest.raises(ValidationError, match="FAILED status strictly requires LOW confidence"):
        StepValidationResult(
            step_id=1,
            attempt_number=1,
            status=ValidationStatus.FAILED,
            confidence=ConfidenceLevel.HIGH,  # Invalid!
            overall_passed=False,
            execution_result=_make_l1(passed=False, status=ValidationStatus.FAILED),
            numerical_result=_make_l2(passed=False, status=ValidationStatus.FAILED),
            semantic_result=_make_l3(passed=False, status=ValidationStatus.FAILED),
            raw_execution_result=raw_ok,
        )

    # Attempting HIGH confidence on INCONCLUSIVE result
    with pytest.raises(ValidationError, match="cannot have HIGH confidence"):
        StepValidationResult(
            step_id=1,
            attempt_number=1,
            status=ValidationStatus.INCONCLUSIVE,
            confidence=ConfidenceLevel.HIGH,  # Invalid!
            overall_passed=False,
            execution_result=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
            numerical_result=_make_l2(passed=False, status=ValidationStatus.INCONCLUSIVE),
            semantic_result=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
            raw_execution_result=raw_ok,
        )

    # Attempting HIGH confidence on UNVERIFIED result
    with pytest.raises(ValidationError, match="must have LOW confidence"):
        StepValidationResult(
            step_id=1,
            attempt_number=1,
            status=ValidationStatus.UNVERIFIED,
            confidence=ConfidenceLevel.HIGH,  # Invalid!
            overall_passed=False,
            execution_result=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
            numerical_result=_make_l2(passed=False, status=ValidationStatus.UNVERIFIED),
            semantic_result=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
            raw_execution_result=raw_ok,
        )


def test_verified_status_rejected_unless_all_three_layers_pass():
    """Invariant: VERIFIED status requires all three validation layers to pass."""
    raw_ok = _make_raw_ok()

    # Layer 2 failed (numerical mismatch), but marked VERIFIED
    with pytest.raises(ValidationError, match="cannot be marked VERIFIED"):
        StepValidationResult(
            step_id=1,
            attempt_number=1,
            status=ValidationStatus.VERIFIED,
            confidence=ConfidenceLevel.HIGH,
            overall_passed=True,
            execution_result=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
            numerical_result=_make_l2(passed=False, status=ValidationStatus.FAILED),
            semantic_result=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
            raw_execution_result=raw_ok,
        )

    # Layer 3 failed (semantic omission), but marked VERIFIED
    with pytest.raises(ValidationError, match="cannot be marked VERIFIED"):
        StepValidationResult(
            step_id=1,
            attempt_number=1,
            status=ValidationStatus.VERIFIED,
            confidence=ConfidenceLevel.HIGH,
            overall_passed=True,
            execution_result=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
            numerical_result=_make_l2(passed=True, status=ValidationStatus.VERIFIED),
            semantic_result=_make_l3(passed=False, status=ValidationStatus.FAILED),
            raw_execution_result=raw_ok,
        )


def test_status_and_confidence_independence():
    """Verify that status and confidence are independent fields evaluated deterministically."""
    # Inconclusive with successful execution and meaningful output -> MEDIUM confidence
    conf_med = determine_step_confidence(
        ValidationStatus.INCONCLUSIVE,
        execution_succeeded=True,
        has_meaningful_output=True,
    )
    assert conf_med == ConfidenceLevel.MEDIUM

    # Inconclusive with successful execution but no meaningful output -> LOW confidence
    conf_low_no_output = determine_step_confidence(
        ValidationStatus.INCONCLUSIVE,
        execution_succeeded=True,
        has_meaningful_output=False,
    )
    assert conf_low_no_output == ConfidenceLevel.LOW

    # Inconclusive with execution failure -> LOW confidence
    conf_low_inc = determine_step_confidence(
        ValidationStatus.INCONCLUSIVE,
        execution_succeeded=False,
    )
    assert conf_low_inc == ConfidenceLevel.LOW

    # Failed validation even if query technically ran without crash -> LOW confidence
    conf_failed = determine_step_confidence(ValidationStatus.FAILED, execution_succeeded=True)
    assert conf_failed == ConfidenceLevel.LOW

    # Unverified -> LOW confidence
    conf_unver = determine_step_confidence(ValidationStatus.UNVERIFIED, execution_succeeded=True)
    assert conf_unver == ConfidenceLevel.LOW


def test_inconclusive_distinct_from_unverified():
    """Requirement 1: Non-correctable ambiguity remains INCONCLUSIVE, never silently UNVERIFIED."""
    raw_ok = _make_raw_ok()
    l1 = _make_l1(passed=True, status=ValidationStatus.VERIFIED)
    l2 = NumericalValidationResult(
        layer_name="Layer 2: Numerical",
        passed=False,
        status=ValidationStatus.INCONCLUSIVE,
        message="Ambiguous date format detected; verification cannot proceed deterministically",
        inconclusive_reason="Ambiguous date column 'order_date' format requires user clarification",
    )
    l3 = _make_l3(passed=True, status=ValidationStatus.VERIFIED)

    status = determine_step_status(l1, l2, l3, raw_execution_result=raw_ok)
    assert status == ValidationStatus.INCONCLUSIVE
    assert status != ValidationStatus.UNVERIFIED

    # Confidence for INCONCLUSIVE with clean execution must be MEDIUM
    confidence = determine_step_confidence(
        status,
        execution_succeeded=True,
        has_meaningful_output=True,
    )
    assert confidence == ConfidenceLevel.MEDIUM

    # Built step result preserves INCONCLUSIVE with confirmed execution
    step_res = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=l1,
        numerical_res=l2,
        semantic_res=l3,
        raw_execution_result=raw_ok,
    )
    assert step_res.status == ValidationStatus.INCONCLUSIVE
    assert step_res.confidence == ConfidenceLevel.MEDIUM
    assert step_res.overall_passed is False


def test_workflow_report_preserves_full_retry_history():
    """Requirement 2: WorkflowValidationReport preserves complete attempt history for every step."""
    raw_ok = _make_raw_ok()
    report = WorkflowValidationReport()

    # Step 1, Attempt 1: Failed on syntax error
    l1_fail = _make_l1(passed=False, status=ValidationStatus.FAILED)
    diag1 = CorrectionDiagnostic(
        error_category=ErrorCategory.SYNTAX_ERROR,
        is_correctable=True,
        failed_layer="Layer 1: Execution",
        error_summary="Binder Error: Column not found",
        remediation_instruction="Use sub_category instead of product_name",
        prior_attempt_sql="SELECT product_name FROM superstore",
        attempt_number=1,
    )
    step1_att1 = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=l1_fail,
        numerical_res=_make_l2(passed=False, status=ValidationStatus.UNVERIFIED),
        semantic_res=_make_l3(passed=False, status=ValidationStatus.UNVERIFIED),
        diagnostic=diag1,
        executed_sql="SELECT product_name FROM superstore",
    )
    report.add_step_attempt(step1_att1)

    assert report.total_steps == 1
    assert report.total_retries == 0
    assert report.workflow_status == ValidationStatus.FAILED
    assert len(report.step_history[1]) == 1

    # Step 1, Attempt 2: Numerical mismatch
    l1_ok = _make_l1(passed=True, status=ValidationStatus.VERIFIED)
    l2_fail = _make_l2(passed=False, status=ValidationStatus.FAILED)
    diag2 = CorrectionDiagnostic(
        error_category=ErrorCategory.NUMERICAL_MISMATCH,
        is_correctable=True,
        failed_layer="Layer 2: Numerical",
        error_summary="Numerical tolerance breach",
        remediation_instruction="Use SUM(sales) instead of AVG(sales)",
        prior_attempt_sql="SELECT AVG(sales) FROM superstore",
        attempt_number=2,
    )
    step1_att2 = build_step_validation_result(
        step_id=1,
        attempt_number=2,
        execution_res=l1_ok,
        numerical_res=l2_fail,
        semantic_res=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
        diagnostic=diag2,
        executed_sql="SELECT AVG(sales) FROM superstore",
        raw_execution_result=raw_ok,
    )
    report.add_step_attempt(step1_att2)

    assert report.total_steps == 1
    assert report.total_retries == 1
    assert report.workflow_status == ValidationStatus.FAILED
    assert len(report.step_history[1]) == 2
    assert report.latest_step_results[1].attempt_number == 2

    # Step 1, Attempt 3: Verified!
    step1_att3 = build_step_validation_result(
        step_id=1,
        attempt_number=3,
        execution_res=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
        numerical_res=_make_l2(passed=True, status=ValidationStatus.VERIFIED),
        semantic_res=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
        executed_sql="SELECT SUM(sales) FROM superstore",
        raw_execution_result=raw_ok,
    )
    report.add_step_attempt(step1_att3)

    assert report.total_steps == 1
    assert report.total_retries == 2  # Two retries after initial attempt
    assert report.workflow_status == ValidationStatus.VERIFIED
    assert report.workflow_confidence == ConfidenceLevel.HIGH
    assert report.all_steps_verified is True
    # Verify that all 3 historical attempts are preserved intact
    assert len(report.step_history[1]) == 3
    assert report.step_history[1][0].attempt_number == 1
    assert report.step_history[1][0].status == ValidationStatus.FAILED
    assert report.step_history[1][1].attempt_number == 2
    assert report.step_history[1][1].status == ValidationStatus.FAILED
    assert report.step_history[1][2].attempt_number == 3
    assert report.step_history[1][2].status == ValidationStatus.VERIFIED


def test_result_availability_policy_retains_last_successful_execution():
    """Policy: retain last successful execution result without misrepresenting failed retries."""
    report = WorkflowValidationReport()

    # Step 1 Attempt 1: Execution succeeded (10 rows), but Layer 2 failed numerical verification
    raw_df = pd.DataFrame({"sales": [100.0, 200.0]})
    exec_success = ExecutionResult(
        success=True,
        query="SELECT sales FROM superstore",
        dataframe=raw_df,
        row_count=2,
        column_count=1,
    )
    att1 = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
        numerical_res=_make_l2(passed=False, status=ValidationStatus.FAILED),
        semantic_res=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
        executed_sql="SELECT sales FROM superstore",
        raw_execution_result=exec_success,
    )
    report.add_step_attempt(att1)

    # Retained execution results holds Attempt 1's DataFrame
    assert 1 in report.retained_execution_results
    assert report.retained_execution_results[1].row_count == 2

    # Step 1 Attempt 2: SQL syntax error caused execution failure
    exec_failure = _make_raw_fail(
        query="SELEC sales FORM superstore",
        error_message="Parser Error at or near 'SELEC'",
    )
    att2 = build_step_validation_result(
        step_id=1,
        attempt_number=2,
        execution_res=ExecutionValidationResult(
            layer_name="Layer 1: Execution",
            passed=False,
            status=ValidationStatus.FAILED,
            message="Execution error: Parser Error",
        ),
        numerical_res=_make_l2(passed=False, status=ValidationStatus.UNVERIFIED),
        semantic_res=_make_l3(passed=False, status=ValidationStatus.UNVERIFIED),
        executed_sql="SELEC sales FORM superstore",
        raw_execution_result=exec_failure,
    )
    report.add_step_attempt(att2)

    # Policy Assertion:
    # 1. Retained execution result remains Attempt 1's output (not wiped by Attempt 2)
    assert report.retained_execution_results[1].success is True
    assert report.retained_execution_results[1].row_count == 2
    # 2. Attempt 2 itself is labeled with its own failure status (not falsely claiming Attempt 1)
    assert report.latest_step_results[1].status == ValidationStatus.FAILED
    assert report.latest_step_results[1].raw_execution_result.success is False


def test_workflow_summary_consistency_multi_step_combinations():
    """Verify workflow status and confidence derivation across multi-step plans."""
    raw_ok = _make_raw_ok()

    # Scenario A: All steps verified -> VERIFIED, HIGH
    rep_a = WorkflowValidationReport()
    rep_a.add_step_attempt(
        build_step_validation_result(
            1, 1, _make_l1(), _make_l2(), _make_l3(), raw_execution_result=raw_ok
        )
    )
    rep_a.add_step_attempt(
        build_step_validation_result(
            2, 1, _make_l1(), _make_l2(), _make_l3(), raw_execution_result=raw_ok
        )
    )
    assert rep_a.workflow_status == ValidationStatus.VERIFIED
    assert rep_a.workflow_confidence == ConfidenceLevel.HIGH
    assert rep_a.all_steps_verified is True

    # Scenario B: Step 1 verified, Step 2 inconclusive (ambiguous date) -> INCONCLUSIVE, MEDIUM
    rep_b = WorkflowValidationReport()
    rep_b.add_step_attempt(
        build_step_validation_result(
            1, 1, _make_l1(), _make_l2(), _make_l3(), raw_execution_result=raw_ok
        )
    )
    l2_inc = NumericalValidationResult(
        layer_name="Layer 2: Numerical",
        passed=False,
        status=ValidationStatus.INCONCLUSIVE,
        message="Ambiguous date",
    )
    rep_b.add_step_attempt(
        build_step_validation_result(
            2, 1, _make_l1(), l2_inc, _make_l3(), raw_execution_result=raw_ok
        )
    )
    assert rep_b.workflow_status == ValidationStatus.INCONCLUSIVE
    assert rep_b.workflow_confidence == ConfidenceLevel.MEDIUM
    assert rep_b.all_steps_verified is False

    # Scenario C: Step 1 verified, Step 2 failed -> FAILED, LOW
    rep_c = WorkflowValidationReport()
    rep_c.add_step_attempt(
        build_step_validation_result(
            1, 1, _make_l1(), _make_l2(), _make_l3(), raw_execution_result=raw_ok
        )
    )
    l2_fail = _make_l2(passed=False, status=ValidationStatus.FAILED)
    rep_c.add_step_attempt(
        build_step_validation_result(
            2, 1, _make_l1(), l2_fail, _make_l3(), raw_execution_result=raw_ok
        )
    )
    assert rep_c.workflow_status == ValidationStatus.FAILED
    assert rep_c.workflow_confidence == ConfidenceLevel.LOW
    assert rep_c.all_steps_verified is False

    # Scenario D: Step 1 verified, Step 2 unverified after 2 retries -> UNVERIFIED, LOW
    rep_d = WorkflowValidationReport()
    rep_d.add_step_attempt(
        build_step_validation_result(
            1, 1, _make_l1(), _make_l2(), _make_l3(), raw_execution_result=raw_ok
        )
    )
    unver_step = StepValidationResult(
        step_id=2,
        attempt_number=3,
        status=ValidationStatus.UNVERIFIED,
        confidence=ConfidenceLevel.LOW,
        overall_passed=False,
        execution_result=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
        numerical_result=_make_l2(passed=False, status=ValidationStatus.UNVERIFIED),
        semantic_result=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
        raw_execution_result=raw_ok,
    )
    rep_d.add_step_attempt(unver_step)
    assert rep_d.workflow_status == ValidationStatus.UNVERIFIED
    assert rep_d.workflow_confidence == ConfidenceLevel.LOW
    assert rep_d.all_steps_verified is False


# ---------------------------------------------------------------------------
# Specific Phase 5 Stage 1 Regression Scenarios Requested
# ---------------------------------------------------------------------------


def test_regression_raw_execution_failure_cannot_result_in_verified_or_high():
    """Regression 1: Raw ExecutionResult with success=False must never produce VERIFIED or HIGH."""
    raw_fail = _make_raw_fail()

    # Via factory:
    step_res = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
        numerical_res=_make_l2(passed=True, status=ValidationStatus.VERIFIED),
        semantic_res=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
        raw_execution_result=raw_fail,
    )
    assert step_res.status == ValidationStatus.FAILED
    assert step_res.confidence == ConfidenceLevel.LOW
    assert step_res.overall_passed is False

    # Via direct constructor marked VERIFIED with raw failure:
    with pytest.raises(ValidationError, match="without confirmed raw execution success"):
        StepValidationResult(
            step_id=1,
            attempt_number=1,
            status=ValidationStatus.VERIFIED,
            confidence=ConfidenceLevel.HIGH,
            overall_passed=True,
            execution_result=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
            numerical_result=_make_l2(passed=True, status=ValidationStatus.VERIFIED),
            semantic_result=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
            raw_execution_result=raw_fail,
        )

    # Via direct constructor with HIGH confidence on raw failure:
    with pytest.raises(ValidationError, match="(raw execution failed|must have LOW confidence)"):
        StepValidationResult(
            step_id=1,
            attempt_number=1,
            status=ValidationStatus.UNVERIFIED,
            confidence=ConfidenceLevel.HIGH,
            overall_passed=False,
            execution_result=_make_l1(passed=False, status=ValidationStatus.FAILED),
            numerical_result=_make_l2(passed=False, status=ValidationStatus.UNVERIFIED),
            semantic_result=_make_l3(passed=False, status=ValidationStatus.UNVERIFIED),
            raw_execution_result=raw_fail,
        )


def test_regression_missing_raw_execution_evidence_cannot_be_verified():
    """Regression 2: Missing raw execution evidence cannot be silently treated as success."""
    # When raw_execution_result is None, determine_step_status cannot return VERIFIED
    status = determine_step_status(
        execution_res=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
        numerical_res=_make_l2(passed=True, status=ValidationStatus.VERIFIED),
        semantic_res=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
        raw_execution_result=None,
    )
    assert status == ValidationStatus.UNVERIFIED

    # Factory without raw execution evidence yields UNVERIFIED and LOW confidence
    step_res = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
        numerical_res=_make_l2(passed=True, status=ValidationStatus.VERIFIED),
        semantic_res=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
        raw_execution_result=None,
    )
    assert step_res.status == ValidationStatus.UNVERIFIED
    assert step_res.confidence == ConfidenceLevel.LOW
    assert step_res.overall_passed is False

    # Direct constructor marked VERIFIED with missing raw result raises ValidationError
    with pytest.raises(ValidationError, match="without confirmed raw execution success"):
        StepValidationResult(
            step_id=1,
            attempt_number=1,
            status=ValidationStatus.VERIFIED,
            confidence=ConfidenceLevel.HIGH,
            overall_passed=True,
            execution_result=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
            numerical_result=_make_l2(passed=True, status=ValidationStatus.VERIFIED),
            semantic_result=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
            raw_execution_result=None,
        )


def test_regression_invalid_layer_status_pass_combinations_rejected():
    """Regression 3: Invalid layer status/passed combinations are rejected by Pydantic."""
    # VERIFIED requires passed=True
    with pytest.raises(ValidationError, match="status VERIFIED requires passed=True"):
        LayerValidationResult(
            layer_name="L1",
            passed=False,
            status=ValidationStatus.VERIFIED,
            message="invalid",
        )

    # FAILED requires passed=False
    with pytest.raises(ValidationError, match="status FAILED requires passed=False"):
        LayerValidationResult(
            layer_name="L1",
            passed=True,
            status=ValidationStatus.FAILED,
            message="invalid",
        )

    # INCONCLUSIVE requires passed=False
    with pytest.raises(
        ValidationError, match="status INCONCLUSIVE cannot be represented as passed=True"
    ):
        LayerValidationResult(
            layer_name="L1",
            passed=True,
            status=ValidationStatus.INCONCLUSIVE,
            message="invalid",
        )

    # UNVERIFIED requires passed=False
    with pytest.raises(ValidationError, match="status UNVERIFIED requires passed=False"):
        LayerValidationResult(
            layer_name="L1",
            passed=True,
            status=ValidationStatus.UNVERIFIED,
            message="invalid",
        )


def test_regression_inconclusive_cannot_have_overall_passed_true():
    """Regression 4: INCONCLUSIVE step cannot have overall_passed=True or HIGH confidence."""
    raw_ok = _make_raw_ok()
    l2_inc = NumericalValidationResult(
        layer_name="Layer 2: Numerical",
        passed=False,
        status=ValidationStatus.INCONCLUSIVE,
        message="Ambiguous",
    )

    # overall_passed=True on INCONCLUSIVE is rejected
    with pytest.raises(
        ValidationError, match="overall_passed cannot be True when status is INCONCLUSIVE"
    ):
        StepValidationResult(
            step_id=1,
            attempt_number=1,
            status=ValidationStatus.INCONCLUSIVE,
            confidence=ConfidenceLevel.MEDIUM,
            overall_passed=True,  # Invalid!
            execution_result=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
            numerical_result=l2_inc,
            semantic_result=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
            raw_execution_result=raw_ok,
        )

    # confidence=HIGH on INCONCLUSIVE is rejected
    with pytest.raises(ValidationError, match="cannot have HIGH confidence"):
        StepValidationResult(
            step_id=1,
            attempt_number=1,
            status=ValidationStatus.INCONCLUSIVE,
            confidence=ConfidenceLevel.HIGH,  # Invalid!
            overall_passed=False,
            execution_result=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
            numerical_result=l2_inc,
            semantic_result=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
            raw_execution_result=raw_ok,
        )


def test_regression_failed_step_requires_at_least_one_failed_layer():
    """Regression 5: FAILED step cannot be constructed without an explicitly failed layer."""
    raw_ok = _make_raw_ok()

    with pytest.raises(ValidationError, match="no layer is marked FAILED"):
        StepValidationResult(
            step_id=1,
            attempt_number=1,
            status=ValidationStatus.FAILED,
            confidence=ConfidenceLevel.LOW,
            overall_passed=False,
            execution_result=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
            numerical_result=_make_l2(passed=True, status=ValidationStatus.VERIFIED),
            semantic_result=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
            raw_execution_result=raw_ok,
        )


def test_regression_attempt_records_are_frozen_immutable():
    """Regression 6: Attempt records cannot be mutated after creation (frozen instance)."""
    raw_ok = _make_raw_ok()
    step_res = StepValidationResult(
        step_id=1,
        attempt_number=1,
        status=ValidationStatus.VERIFIED,
        confidence=ConfidenceLevel.HIGH,
        overall_passed=True,
        execution_result=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
        numerical_result=_make_l2(passed=True, status=ValidationStatus.VERIFIED),
        semantic_result=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
        raw_execution_result=raw_ok,
    )

    # Attempting to mutate a field on StepValidationResult raises ValidationError
    with pytest.raises(ValidationError, match="Instance is frozen"):
        step_res.status = ValidationStatus.FAILED  # type: ignore[misc]

    # Layer validation result is also frozen
    layer = _make_l1(passed=True, status=ValidationStatus.VERIFIED)
    with pytest.raises(ValidationError, match="Instance is frozen"):
        layer.passed = False  # type: ignore[misc]

    # Diagnostic is frozen
    diag = CorrectionDiagnostic(
        error_category=ErrorCategory.SYNTAX_ERROR,
        is_correctable=True,
        failed_layer="Layer 1",
        error_summary="syntax",
        remediation_instruction="fix syntax",
        prior_attempt_sql="SELECT",
        attempt_number=1,
    )
    with pytest.raises(ValidationError, match="Instance is frozen"):
        diag.is_correctable = False  # type: ignore[misc]

    # In contrast, WorkflowValidationReport is a mutable accumulator
    report = WorkflowValidationReport()
    report.verification_warnings.append("test warning")
    assert "test warning" in report.verification_warnings


def test_regression_nested_mutation_rejected_on_frozen_records():
    """Regression: Deep immutability prevents mutation of nested dicts, lists, and tuples."""
    raw_ok = _make_raw_ok()

    # 1. LayerValidationResult details mapping and nested structures
    l1 = ExecutionValidationResult(
        layer_name="Layer 1: Execution",
        passed=True,
        status=ValidationStatus.VERIFIED,
        message="ok",
        details={"query_grain": "daily", "nested": {"schema": "main"}, "tags": ["prod", "etl"]},
    )
    # Direct dictionary key assignment raises TypeError
    with pytest.raises(TypeError, match="does not support item assignment"):
        l1.details["query_grain"] = "monthly"  # type: ignore[index]

    # Nested dictionary mutation raises TypeError
    with pytest.raises(TypeError, match="does not support item assignment"):
        l1.details["nested"]["schema"] = "dev"  # type: ignore[index]

    # Nested list append raises AttributeError (converted to tuple)
    with pytest.raises(AttributeError, match="has no attribute 'append'"):
        l1.details["tags"].append("new_tag")  # type: ignore[attr-defined]

    # 2. SemanticValidationResult coverage collections are immutable tuples
    l3 = SemanticValidationResult(
        layer_name="Layer 3: Semantic",
        passed=True,
        status=ValidationStatus.VERIFIED,
        message="ok",
        metrics_covered=["sales", "profit"],
        dimensions_covered=["region"],
    )
    assert isinstance(l3.metrics_covered, tuple)
    with pytest.raises(AttributeError, match="has no attribute 'append'"):
        l3.metrics_covered.append("discount")  # type: ignore[attr-defined]

    with pytest.raises(TypeError, match="does not support item assignment"):
        l3.dimensions_covered[0] = "state"  # type: ignore[index]

    # 3. StepValidationResult notes are immutable tuples
    step_res = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=l1,
        numerical_res=_make_l2(),
        semantic_res=l3,
        raw_execution_result=raw_ok,
        notes=["audit_note_1"],
    )
    assert isinstance(step_res.notes, tuple)
    with pytest.raises(AttributeError, match="has no attribute 'append'"):
        step_res.notes.append("tampered_note")  # type: ignore[attr-defined]

    # 4. Straightforward serialization preserved
    dumped = l1.model_dump()
    assert isinstance(dumped["details"], dict)
    assert dumped["details"]["nested"]["schema"] == "main"
    assert "tags" in dumped["details"]
    json_str = l1.model_dump_json()
    assert '"query_grain":"daily"' in json_str


def test_regression_duplicate_and_out_of_order_attempts_rejected():
    """Regression 7: Duplicate and out-of-order attempt numbers are rejected in add_step_attempt."""
    report = WorkflowValidationReport()
    raw_ok = _make_raw_ok()

    att1 = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
        numerical_res=_make_l2(passed=False, status=ValidationStatus.FAILED),
        semantic_res=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
        raw_execution_result=raw_ok,
    )
    report.add_step_attempt(att1)

    # Duplicate attempt number (attempt 1 again)
    with pytest.raises(ValueError, match="Out-of-order or duplicate attempt number 1"):
        report.add_step_attempt(att1)

    # Record attempt 2
    att2 = build_step_validation_result(
        step_id=1,
        attempt_number=2,
        execution_res=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
        numerical_res=_make_l2(passed=False, status=ValidationStatus.FAILED),
        semantic_res=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
        raw_execution_result=raw_ok,
    )
    report.add_step_attempt(att2)

    # Adding attempt 1 when last attempt was 2 is rejected as out-of-order
    with pytest.raises(ValueError, match="Out-of-order or duplicate attempt number 1"):
        report.add_step_attempt(att1)


def test_regression_invalid_step_id_and_attempt_numbers_rejected_directly():
    """Regression: Invalid step_id and attempt_number (< 1) rejected in model constructors."""
    raw_ok = _make_raw_ok()

    # step_id < 1 rejected
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        StepValidationResult(
            step_id=0,
            attempt_number=1,
            status=ValidationStatus.VERIFIED,
            confidence=ConfidenceLevel.HIGH,
            overall_passed=True,
            execution_result=_make_l1(),
            numerical_result=_make_l2(),
            semantic_result=_make_l3(),
            raw_execution_result=raw_ok,
        )

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        StepValidationResult(
            step_id=-2,
            attempt_number=1,
            status=ValidationStatus.VERIFIED,
            confidence=ConfidenceLevel.HIGH,
            overall_passed=True,
            execution_result=_make_l1(),
            numerical_result=_make_l2(),
            semantic_result=_make_l3(),
            raw_execution_result=raw_ok,
        )

    # attempt_number < 1 rejected
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        StepValidationResult(
            step_id=1,
            attempt_number=0,
            status=ValidationStatus.VERIFIED,
            confidence=ConfidenceLevel.HIGH,
            overall_passed=True,
            execution_result=_make_l1(),
            numerical_result=_make_l2(),
            semantic_result=_make_l3(),
            raw_execution_result=raw_ok,
        )

    # CorrectionDiagnostic attempt_number < 1 rejected
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        CorrectionDiagnostic(
            error_category=ErrorCategory.SYNTAX_ERROR,
            is_correctable=True,
            failed_layer="Layer 1",
            error_summary="syntax",
            remediation_instruction="fix",
            prior_attempt_sql="SELECT",
            attempt_number=0,
        )


def test_regression_empty_output_not_automatically_medium_confidence():
    """Regression: Empty output (row_count == 0) does not get MEDIUM confidence automatically."""
    raw_empty = _make_raw_ok(dataframe=pd.DataFrame({"revenue": []}))
    l2_inc = NumericalValidationResult(
        layer_name="Layer 2: Numerical",
        passed=False,
        status=ValidationStatus.INCONCLUSIVE,
        message="Ambiguous date",
    )

    # When has_meaningful_output is None (no evidence supplied), meaningfulness is False -> LOW
    step_res = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
        numerical_res=l2_inc,
        semantic_res=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
        raw_execution_result=raw_empty,
        has_meaningful_output=None,
    )
    assert step_res.status == ValidationStatus.INCONCLUSIVE
    assert step_res.confidence == ConfidenceLevel.LOW


def test_regression_explicitly_valid_empty_output_handled_with_evidence():
    """Regression: Valid empty output receives MEDIUM confidence when evidence supplied."""
    raw_empty = _make_raw_ok(dataframe=pd.DataFrame({"revenue": []}))
    l2_inc = NumericalValidationResult(
        layer_name="Layer 2: Numerical",
        passed=False,
        status=ValidationStatus.INCONCLUSIVE,
        message="Ambiguous date",
    )

    # When has_meaningful_output=True is explicitly supplied, meaningfulness is True -> MEDIUM
    step_res = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
        numerical_res=l2_inc,
        semantic_res=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
        raw_execution_result=raw_empty,
        has_meaningful_output=True,
    )
    assert step_res.status == ValidationStatus.INCONCLUSIVE
    assert step_res.confidence == ConfidenceLevel.MEDIUM


def test_regression_workflow_inconclusive_low_confidence_step_does_not_receive_medium():
    """Regression: A workflow with an INCONCLUSIVE/LOW-confidence step receives LOW confidence."""
    report = WorkflowValidationReport()
    raw_ok = _make_raw_ok()
    raw_empty = _make_raw_ok(dataframe=pd.DataFrame({"revenue": []}))

    # Step 1: VERIFIED, HIGH confidence
    step1 = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=_make_l1(),
        numerical_res=_make_l2(),
        semantic_res=_make_l3(),
        raw_execution_result=raw_ok,
    )
    report.add_step_attempt(step1)

    # Step 2: INCONCLUSIVE, LOW confidence (empty output without explicit meaningfulness)
    l2_inc = NumericalValidationResult(
        layer_name="Layer 2: Numerical",
        passed=False,
        status=ValidationStatus.INCONCLUSIVE,
        message="Ambiguous date",
    )
    step2 = build_step_validation_result(
        step_id=2,
        attempt_number=1,
        execution_res=_make_l1(),
        numerical_res=l2_inc,
        semantic_res=_make_l3(),
        raw_execution_result=raw_empty,
        has_meaningful_output=None,
    )
    assert step2.confidence == ConfidenceLevel.LOW
    report.add_step_attempt(step2)

    # Workflow must be INCONCLUSIVE and LOW confidence because Step 2 has LOW confidence
    assert report.workflow_status == ValidationStatus.INCONCLUSIVE
    assert report.workflow_confidence == ConfidenceLevel.LOW


def test_regression_workflow_inconclusive_step_with_meaningful_output_receives_medium():
    """Regression: Workflow with inconclusive step receives MEDIUM when all steps qualify."""
    report = WorkflowValidationReport()
    raw_ok = _make_raw_ok()
    raw_empty = _make_raw_ok(dataframe=pd.DataFrame({"revenue": []}))

    # Step 1: VERIFIED, HIGH confidence
    step1 = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=_make_l1(),
        numerical_res=_make_l2(),
        semantic_res=_make_l3(),
        raw_execution_result=raw_ok,
    )
    report.add_step_attempt(step1)

    # Step 2: INCONCLUSIVE, MEDIUM confidence (empty output with explicit evidence)
    l2_inc = NumericalValidationResult(
        layer_name="Layer 2: Numerical",
        passed=False,
        status=ValidationStatus.INCONCLUSIVE,
        message="Ambiguous date",
    )
    step2 = build_step_validation_result(
        step_id=2,
        attempt_number=1,
        execution_res=_make_l1(),
        numerical_res=l2_inc,
        semantic_res=_make_l3(),
        raw_execution_result=raw_empty,
        has_meaningful_output=True,
    )
    assert step2.confidence == ConfidenceLevel.MEDIUM
    report.add_step_attempt(step2)

    # Workflow receives MEDIUM confidence because all steps qualify with non-LOW confidence
    assert report.workflow_status == ValidationStatus.INCONCLUSIVE
    assert report.workflow_confidence == ConfidenceLevel.MEDIUM


def test_regression_latest_result_always_matches_final_history_entry():
    """Regression 8: latest_step_results[step_id] always matches step_history[step_id][-1]."""
    report = WorkflowValidationReport()
    raw_ok = _make_raw_ok()

    for att_num in range(1, 4):
        step_att = build_step_validation_result(
            step_id=1,
            attempt_number=att_num,
            execution_res=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
            numerical_res=_make_l2(
                passed=(att_num == 3),
                status=ValidationStatus.VERIFIED if att_num == 3 else ValidationStatus.FAILED,
            ),
            semantic_res=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
            raw_execution_result=raw_ok,
        )
        report.add_step_attempt(step_att)
        assert report.latest_step_results[1] == report.step_history[1][-1]
        assert report.latest_step_results[1].attempt_number == att_num


def test_regression_later_failed_attempt_does_not_overwrite_retained_execution():
    """Regression 9: Later failed attempt does not overwrite last successful execution result."""
    report = WorkflowValidationReport()
    raw_success = _make_raw_ok(dataframe=pd.DataFrame({"revenue": [1000, 2000]}))
    raw_failure = _make_raw_fail(error_message="Runtime out of memory")

    att1 = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
        numerical_res=_make_l2(passed=False, status=ValidationStatus.FAILED),
        semantic_res=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
        raw_execution_result=raw_success,
    )
    report.add_step_attempt(att1)
    assert report.retained_execution_results[1].success is True
    assert "revenue" in report.retained_execution_results[1].dataframe.columns

    att2 = build_step_validation_result(
        step_id=1,
        attempt_number=2,
        execution_res=_make_l1(passed=False, status=ValidationStatus.FAILED),
        numerical_res=_make_l2(passed=False, status=ValidationStatus.UNVERIFIED),
        semantic_res=_make_l3(passed=False, status=ValidationStatus.UNVERIFIED),
        raw_execution_result=raw_failure,
    )
    report.add_step_attempt(att2)

    # Retained result is still Attempt 1's successful result
    assert report.retained_execution_results[1].success is True
    assert "revenue" in report.retained_execution_results[1].dataframe.columns
    # Latest step result reflects Attempt 2 failure
    assert report.latest_step_results[1].attempt_number == 2
    assert report.latest_step_results[1].status == ValidationStatus.FAILED


def test_regression_workflow_retry_and_step_counts_consistency():
    """Regression 10: Workflow retry and step counts remain consistent across multi-step retries."""
    report = WorkflowValidationReport()
    raw_ok = _make_raw_ok()

    # Step 1 Attempt 1
    report.add_step_attempt(
        build_step_validation_result(
            1, 1, _make_l1(), _make_l2(passed=False, status=ValidationStatus.FAILED), _make_l3(),
            raw_execution_result=raw_ok,
        )
    )
    assert report.total_steps == 1
    assert report.total_retries == 0

    # Step 1 Attempt 2 (Retry 1 for Step 1)
    report.add_step_attempt(
        build_step_validation_result(
            1, 2, _make_l1(), _make_l2(passed=True, status=ValidationStatus.VERIFIED), _make_l3(),
            raw_execution_result=raw_ok,
        )
    )
    assert report.total_steps == 1
    assert report.total_retries == 1

    # Step 2 Attempt 1
    report.add_step_attempt(
        build_step_validation_result(
            2, 1, _make_l1(), _make_l2(passed=False, status=ValidationStatus.FAILED), _make_l3(),
            raw_execution_result=raw_ok,
        )
    )
    assert report.total_steps == 2
    assert report.total_retries == 1

    # Step 2 Attempt 2 (Retry 1 for Step 2)
    report.add_step_attempt(
        build_step_validation_result(
            2, 2, _make_l1(), _make_l2(passed=False, status=ValidationStatus.FAILED), _make_l3(),
            raw_execution_result=raw_ok,
        )
    )
    assert report.total_steps == 2
    assert report.total_retries == 2

    # Step 2 Attempt 3 (Retry 2 for Step 2)
    report.add_step_attempt(
        build_step_validation_result(
            2, 3, _make_l1(), _make_l2(passed=True, status=ValidationStatus.VERIFIED), _make_l3(),
            raw_execution_result=raw_ok,
        )
    )
    assert report.total_steps == 2
    assert report.total_retries == 3
    assert report.workflow_status == ValidationStatus.VERIFIED
    assert report.workflow_confidence == ConfidenceLevel.HIGH
    assert report.all_steps_verified is True


def test_regression_inconclusive_medium_direct_construction_rules():
    """Inconclusive + MEDIUM direct construction rules enforced at model level."""
    raw_ok = _make_raw_ok()
    raw_fail = _make_raw_fail()
    l2_inc = NumericalValidationResult(
        layer_name="Layer 2: Numerical",
        passed=False,
        status=ValidationStatus.INCONCLUSIVE,
        message="Ambiguous format",
    )

    # 1. Direct construction of INCONCLUSIVE + MEDIUM with raw_execution_result=None is rejected
    with pytest.raises(ValidationError, match="INCONCLUSIVE with MEDIUM confidence"):
        StepValidationResult(
            step_id=1,
            attempt_number=1,
            status=ValidationStatus.INCONCLUSIVE,
            confidence=ConfidenceLevel.MEDIUM,
            overall_passed=False,
            execution_result=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
            numerical_result=l2_inc,
            semantic_result=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
            raw_execution_result=None,
        )

    # 2. Direct construction of INCONCLUSIVE + MEDIUM with failed raw execution is rejected
    with pytest.raises(ValidationError, match="INCONCLUSIVE with MEDIUM confidence"):
        StepValidationResult(
            step_id=1,
            attempt_number=1,
            status=ValidationStatus.INCONCLUSIVE,
            confidence=ConfidenceLevel.MEDIUM,
            overall_passed=False,
            execution_result=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
            numerical_result=l2_inc,
            semantic_result=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
            raw_execution_result=raw_fail,
        )

    # 3. Direct construction of INCONCLUSIVE + MEDIUM with non-passing execution layer is rejected
    with pytest.raises(ValidationError, match="INCONCLUSIVE with MEDIUM confidence"):
        StepValidationResult(
            step_id=1,
            attempt_number=1,
            status=ValidationStatus.INCONCLUSIVE,
            confidence=ConfidenceLevel.MEDIUM,
            overall_passed=False,
            execution_result=_make_l1(passed=False, status=ValidationStatus.UNVERIFIED),
            numerical_result=l2_inc,
            semantic_result=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
            raw_execution_result=raw_ok,
        )

    # 4. Direct construction of INCONCLUSIVE + MEDIUM with confirmed success is accepted
    step_med = StepValidationResult(
        step_id=1,
        attempt_number=1,
        status=ValidationStatus.INCONCLUSIVE,
        confidence=ConfidenceLevel.MEDIUM,
        overall_passed=False,
        execution_result=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
        numerical_result=l2_inc,
        semantic_result=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
        raw_execution_result=raw_ok,
    )
    assert step_med.status == ValidationStatus.INCONCLUSIVE
    assert step_med.confidence == ConfidenceLevel.MEDIUM

    # 5. INCONCLUSIVE + LOW without raw execution remains accepted
    step_low = StepValidationResult(
        step_id=1,
        attempt_number=1,
        status=ValidationStatus.INCONCLUSIVE,
        confidence=ConfidenceLevel.LOW,
        overall_passed=False,
        execution_result=_make_l1(passed=True, status=ValidationStatus.VERIFIED),
        numerical_result=l2_inc,
        semantic_result=_make_l3(passed=True, status=ValidationStatus.VERIFIED),
        raw_execution_result=None,
    )
    assert step_low.status == ValidationStatus.INCONCLUSIVE
    assert step_low.confidence == ConfidenceLevel.LOW


def test_regression_deep_freeze_rejects_unsupported_mutable_objects():
    """Deep freeze rejects unsupported mutable objects such as NumPy arrays."""
    # NumPy array inside details mapping is rejected with TypeError
    with pytest.raises(TypeError, match="Unsupported mutable or unfreezable type: ndarray"):
        ExecutionValidationResult(
            layer_name="Layer 1: Execution",
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="ok",
            details={"arr": np.array([1, 2, 3])},
        )

    # NumPy array as duckdb_value in NumericalValidationResult is rejected
    with pytest.raises(TypeError, match="Unsupported mutable or unfreezable type: ndarray"):
        NumericalValidationResult(
            layer_name="Layer 2: Numerical",
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="ok",
            duckdb_value=np.array([10.0, 20.0]),
        )

    # Arbitrary custom mutable class instance is rejected
    class UnfreezableObj:
        def __init__(self) -> None:
            self.data = [1, 2]

    with pytest.raises(TypeError, match="Unsupported mutable or unfreezable type: UnfreezableObj"):
        ExecutionValidationResult(
            layer_name="Layer 1: Execution",
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="ok",
            details={"custom": UnfreezableObj()},
        )

    # Supported NumPy scalars and nested mappings/lists continue working cleanly
    res_ok = NumericalValidationResult(
        layer_name="Layer 2: Numerical",
        passed=True,
        status=ValidationStatus.VERIFIED,
        message="ok",
        duckdb_value=np.float64(123.45),
        independent_value=np.int64(123),
        details={
            "scalar": np.int64(42),
            "nested": {"key": "val", "seq": [1, 2, 3]},
        },
    )
    assert res_ok.duckdb_value == 123.45
    assert res_ok.independent_value == 123
    assert res_ok.details["scalar"] == 42
    assert res_ok.details["nested"]["seq"] == (1, 2, 3)

    # Serialization preserves standard dict and list types without errors
    dumped = res_ok.model_dump()
    assert dumped["details"]["nested"]["seq"] == [1, 2, 3]
    json_str = res_ok.model_dump_json()
    assert '"scalar":42' in json_str


def test_regression_mapping_keys_freezing_and_numpy_scalars():
    """Mapping keys are recursively frozen and verified hashable; NumPy scalars converted safely."""
    # 1. Normal nested mappings with string/numeric keys remain supported
    res_keys = ExecutionValidationResult(
        layer_name="Layer 1: Execution",
        passed=True,
        status=ValidationStatus.VERIFIED,
        message="ok",
        details={
            "metric_name": "sales",
            42: "numeric_key",
            2026: {"month": 11, 1: "nested_int_key"},
        },
    )
    assert res_keys.details["metric_name"] == "sales"
    assert res_keys.details[42] == "numeric_key"
    assert res_keys.details[2026][1] == "nested_int_key"

    # Serialization preserves string and numeric keys
    dumped = res_keys.model_dump()
    assert dumped["details"][42] == "numeric_key"
    assert dumped["details"][2026][1] == "nested_int_key"
    json_str = res_keys.model_dump_json()
    assert '"numeric_key"' in json_str

    # 2. NumPy scalar keys and values are converted to native Python types recursively
    res_np = ExecutionValidationResult(
        layer_name="Layer 1: Execution",
        passed=True,
        status=ValidationStatus.VERIFIED,
        message="ok",
        details={
            np.int64(100): np.float64(99.9),
            "nested": {np.str_("dyn_key"): np.bool_(True)},
        },
    )
    # The keys and values must be native Python primitives
    assert type(list(res_np.details.keys())[0]) is int
    assert type(res_np.details[100]) is float
    assert res_np.details[100] == 99.9
    assert type(res_np.details["nested"]["dyn_key"]) is bool
    assert res_np.details["nested"]["dyn_key"] is True

    # 3. A mutable, hashable custom mapping key is rejected
    class MutableHashableKey:
        """Class with default __hash__ (hashable by id) but mutable attributes."""

        def __init__(self) -> None:
            self.mutable_field = [1, 2, 3]

    with pytest.raises(
        TypeError, match="Unsupported mutable or unfreezable type: MutableHashableKey"
    ):
        ExecutionValidationResult(
            layer_name="Layer 1: Execution",
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="ok",
            details={MutableHashableKey(): "val"},
        )

    # 4. A tuple key containing a list is safely converted into a hashable immutable tuple
    res_tuple_key = ExecutionValidationResult(
        layer_name="Layer 1: Execution",
        passed=True,
        status=ValidationStatus.VERIFIED,
        message="ok",
        details={("region", ("south", 1)): "valid_tuple_key"},
    )
    assert res_tuple_key.details[("region", ("south", 1))] == "valid_tuple_key"

    # 5. An unhashable mapping key (such as a nested dict) is rejected with clear TypeError
    from src.agent.validation.contracts import _freeze_key

    with pytest.raises(TypeError, match="Unhashable mapping key: dict"):
        _freeze_key({"inner": "dict"})
