"""UI-independent data ingestion and profiling presentation helpers.

Provides safe file-stream loading, collision resolution, Excel sheet inspection,
preview extraction without raw SQL execution, and display table transformations.
"""

from __future__ import annotations

import datetime
import io
import math
import re
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.agent.data.ingestion import (
    SessionDatabase,
    read_csv_with_fallback,
    sanitize_identifier,
)
from src.agent.data.profiling import TableProfile
from src.agent.validation.contracts import (
    ValidationStatus,
    WorkflowValidationReport,
)

__all__ = [
    "resolve_table_name_collision",
    "get_excel_sheet_names",
    "ingest_uploaded_file",
    "get_table_preview_df",
    "profile_to_display_df",
    "get_lifecycle_status_badge",
    "get_status_badge_html",
    "format_step_validation_summary",
    "format_numerical_evidence_table",
    "format_numerical_validation_display",
    "format_semantic_validation_display",
    "validation_report_to_display_rows",
    "serialize_validation_report_for_json",
    "format_data_quality_metrics",
    "format_correction_history_display",
    "get_workflow_nodes_summary",
    "get_workflow_execution_path",
    "prepare_comparison_chart_df",
    "prepare_chart_data",
]


def resolve_table_name_collision(existing_tables: list[str], raw_name: str) -> str:
    """Derive a unique table name given existing session tables.

    If the sanitized table name already exists in `existing_tables`, appends
    an incrementing numeric suffix (e.g., 'sales', 'sales_2', 'sales_3').
    """
    clean_base = sanitize_identifier(raw_name)
    existing_set = {t.lower() for t in existing_tables}

    if clean_base not in existing_set:
        return clean_base

    counter = 2
    while f"{clean_base}_{counter}" in existing_set:
        counter += 1
    return f"{clean_base}_{counter}"


def get_excel_sheet_names(file_bytes: bytes) -> list[str]:
    """Inspect and return available sheet names from Excel file bytes."""
    with io.BytesIO(file_bytes) as stream:
        excel_file = pd.ExcelFile(stream)
        return list(excel_file.sheet_names)


def ingest_uploaded_file(
    db: SessionDatabase,
    file_bytes: bytes,
    filename: str,
    sheet_name: str | int = 0,
) -> tuple[bool, str, str | None]:
    """Ingest uploaded CSV or Excel file bytes into the DuckDB session.

    Validates:
        - Allowed file extensions (.csv, .xlsx, .xls).
        - Non-empty byte payload.
        - Successful tabular parsing with at least 1 column.
        - Table name collision resolution against active session tables.

    Args:
        db: Active SessionDatabase.
        file_bytes: Raw bytes from Streamlit file uploader.
        filename: Original uploaded filename.
        sheet_name: Sheet name or index for Excel files.

    Returns:
        Tuple of (success: bool, table_name_or_message: str, error_details: str | None).
    """
    if not file_bytes:
        return False, "File is empty", "The uploaded file contains 0 bytes."

    ext = Path(filename).suffix.lower()
    if ext not in {".csv", ".xlsx", ".xls"}:
        return (
            False,
            f"Unsupported file type '{ext}'",
            "Supported formats are CSV (.csv) and Excel (.xlsx, .xls).",
        )

    # Parse into pandas DataFrame
    try:
        with io.BytesIO(file_bytes) as stream:
            if ext == ".csv":
                df = read_csv_with_fallback(stream, low_memory=False)
            else:
                df = pd.read_excel(stream, sheet_name=sheet_name)
    except pd.errors.EmptyDataError:
        return False, "File contains no data", "The uploaded file is empty or has no columns."
    except Exception as exc:
        return False, "Failed to parse file", str(exc)

    if df.empty and len(df.columns) == 0:
        return False, "File contains no data", "The dataset has zero columns."

    # Resolve table name collisions
    base_name = Path(filename).stem
    clean_table_name = resolve_table_name_collision(db.get_allowed_tables(), base_name)

    # Register in DuckDB
    try:
        registered_name = db.register_dataframe(df, clean_table_name, sanitize_columns=True)
        return True, registered_name, None
    except Exception as exc:
        return False, "Database registration failed", str(exc)


def get_table_preview_df(
    db: SessionDatabase,
    table_name: str,
    limit: int = 20,
) -> pd.DataFrame:
    """Retrieve the first N rows of a registered table safely without SQL execution.

    Uses DuckDB's relational table API directly (`db.connection.table().limit()`).
    """
    clean_name = table_name.lower()
    if not db.is_table_allowed(clean_name):
        raise ValueError(f"Table '{table_name}' is not registered in session allowlist.")

    # Safe relational limit access: support db.cursor(), db.connection.cursor(), or db.query_df
    if hasattr(db, "cursor"):
        cur = db.cursor()
    elif hasattr(db, "connection") and hasattr(db.connection, "cursor"):
        cur = db.connection.cursor()
    else:
        return db.query_df(f'SELECT * FROM "{clean_name}" LIMIT {limit}')

    rel = cur.table(clean_name).limit(limit)
    return rel.df()


def profile_to_display_df(profile: TableProfile) -> pd.DataFrame:
    """Transform TableProfile columns into a clean DataFrame for Streamlit display."""
    records: list[dict[str, Any]] = []

    for col in profile.columns.values():
        samples_str = (
            ", ".join(repr(s) for s in col.sample_values[:3]) if col.sample_values else "-"
        )
        mean_str = f"{col.mean_val:.2f}" if col.mean_val is not None else "-"
        date_fmt = col.date_format if col.is_temporal and col.date_format else "-"

        records.append(
            {
                "Column": col.name,
                "Type": col.duckdb_type,
                "Date Format": date_fmt,
                "Null Count": col.null_count,
                "Null %": f"{col.null_percentage:.1f}%",
                "Distinct": col.distinct_count,
                "Min": str(col.min_val) if col.min_val is not None else "-",
                "Max": str(col.max_val) if col.max_val is not None else "-",
                "Mean": mean_str,
                "Samples": samples_str,
            }
        )

    return pd.DataFrame(records)


def get_lifecycle_status_badge(status: str | None) -> tuple[str, str, str]:
    """Return display label, icon, and alert severity for a workflow lifecycle status."""
    mapping = {
        "validated": (
            "VALIDATED — All Steps Verified across Execution, Numerical & Semantic Layers",
            "",
            "success",
        ),
        "execution_verified": (
            "EXECUTION VERIFIED — Queries Executed & Schema Confirmed (Layer 1 Passed)",
            "",
            "info",
        ),
        "inconclusive": (
            "INCONCLUSIVE — Execution Succeeded, but Analytical Correctness Not Established",
            "",
            "warning",
        ),
        "validation_failed": (
            "VALIDATION FAILED — Execution or Layer Invariant Violation Detected",
            "",
            "error",
        ),
    }
    clean_status = (status or "unknown").lower()
    return mapping.get(
        clean_status,
        (f"STATUS: {clean_status.upper()}", "", "info"),
    )


def format_step_validation_summary(
    val_report: Any,
    step_id: int,
) -> dict[str, Any] | None:
    """Safely extract StepValidationResult details for UI presentation."""
    if not isinstance(val_report, WorkflowValidationReport):
        return None
    step_val = val_report.latest_step_results.get(step_id)
    if not step_val:
        return None

    # Derive shape description
    shape_desc = (
        "Scalar (1×1)"
        if step_val.execution_result.is_scalar
        else (
            f"Tabular ({step_val.execution_result.row_count}×"
            f"{step_val.execution_result.column_count})"
        )
    )

    # Derive schema match description
    if step_val.execution_result.status == ValidationStatus.VERIFIED:
        schema_matched = (
            "Matched"
            if step_val.execution_result.schema_matched
            else "Inconclusive (Missing Intent)"
        )
    elif step_val.execution_result.status == ValidationStatus.FAILED:
        schema_matched = "Mismatched"
    else:
        schema_matched = "Unverified"

    diagnostic_summary = None
    if step_val.diagnostic:
        diagnostic_summary = {
            "category": step_val.diagnostic.error_category.value,
            "summary": step_val.diagnostic.error_summary,
            "correctable": step_val.diagnostic.is_correctable,
            "remediation": step_val.diagnostic.remediation_instruction,
        }

    num_res = step_val.numerical_result
    num_details = dict(num_res.details) if num_res.details else {}

    sem_res = step_val.semantic_result
    sem_checks: list[dict[str, Any]] = []
    if sem_res and sem_res.checks:
        for c in sem_res.checks:
            sem_checks.append(
                {
                    "check_name": c.check_name,
                    "status": c.status.value,
                    "passed": c.passed,
                    "diagnostic_code": c.diagnostic_code or "None",
                    "explanation": c.explanation or "",
                    "intent_evidence": dict(c.intent_evidence) if c.intent_evidence else {},
                    "ast_evidence": dict(c.ast_evidence) if c.ast_evidence else {},
                }
            )

    sem_proj = None
    if sem_res and sem_res.canonical_projection:
        cp = sem_res.canonical_projection
        sem_proj = {
            "columns": list(cp.columns) if hasattr(cp, "columns") else [],
            "cardinality": getattr(cp, "cardinality", None),
            "is_scalar": getattr(cp, "is_scalar", False),
        }

    return {
        "step_status": step_val.status.value,
        "confidence": step_val.confidence.value,
        "execution_passed": step_val.execution_result.passed,
        "execution_status": step_val.execution_result.status.value,
        "is_scalar": step_val.execution_result.is_scalar,
        "row_count": step_val.execution_result.row_count,
        "column_count": step_val.execution_result.column_count,
        "shape_description": shape_desc,
        "schema_matched": schema_matched,
        "is_empty": step_val.execution_result.is_empty,
        "message": step_val.execution_result.message or "Executed successfully",
        "numerical_status": num_res.status.value,
        "numerical_passed": num_res.passed,
        "numerical_message": num_res.message,
        "numerical_inconclusive_reason": num_res.inconclusive_reason,
        "numerical_duckdb_value": num_res.duckdb_value,
        "numerical_independent_value": num_res.independent_value,
        "numerical_absolute_difference": num_res.absolute_difference,
        "numerical_relative_difference": num_res.relative_difference,
        "numerical_atol": num_res.atol,
        "numerical_rtol": num_res.rtol,
        "numerical_compared_rows": num_res.compared_rows,
        "numerical_details": num_details,
        "semantic_status": sem_res.status.value,
        "semantic_passed": sem_res.passed,
        "semantic_message": sem_res.message,
        "semantic_inconclusive_reason": sem_res.inconclusive_reason,
        "semantic_checks": sem_checks,
        "semantic_canonical_projection": sem_proj,
        "diagnostic": diagnostic_summary,
        "attempts": len(val_report.step_history.get(step_id, [])),
    }


def format_numerical_evidence_table(
    details: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Transform raw numerical validation comparisons into clean display rows."""
    if not details or not isinstance(details, dict):
        return []

    comparisons = details.get("comparisons") or []
    if not isinstance(comparisons, (list, tuple)):
        return []

    rows: list[dict[str, Any]] = []
    for comp in comparisons:
        if not isinstance(comp, (dict, pd.Series)) and not hasattr(comp, "get"):
            continue

        raw_key = comp.get("group_key")
        if isinstance(raw_key, (list, tuple)):
            key_str = ", ".join(str(k) for k in raw_key)
        else:
            key_str = str(raw_key)

        act_val = comp.get("duckdb_value")
        exp_val = comp.get("independent_value")
        abs_diff = comp.get("absolute_difference")
        rel_diff = comp.get("relative_difference")
        passed = comp.get("passed", False)

        rel_diff_str = (
            "N/A (zero denominator)"
            if rel_diff is None and exp_val == 0.0
            else ("None" if rel_diff is None else str(rel_diff))
        )

        rows.append(
            {
                "Group Key": key_str,
                "Metric": str(comp.get("metric", "")),
                "DuckDB Value": str(act_val),
                "Independent Value": str(exp_val),
                "Absolute Difference": str(abs_diff) if abs_diff is not None else "None",
                "Relative Difference": rel_diff_str,
                "Status": "PASSED" if passed else "FAILED",
            }
        )
    return rows


def format_numerical_validation_display(
    num_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Parse numerical validation outcome and details for dashboard presentation."""
    if not num_info or not isinstance(num_info, dict):
        return {"mode": "none", "status": "UNVERIFIED"}

    status = num_info.get("numerical_status", "UNVERIFIED")
    details = num_info.get("numerical_details") or {}
    if not isinstance(details, dict):
        details = dict(details) if hasattr(details, "items") else {}

    message = num_info.get("numerical_message") or ""
    inconclusive_reason = num_info.get("numerical_inconclusive_reason")

    is_grouped = bool(details.get("comparisons") or details.get("group_comparisons"))
    is_scalar = not is_grouped and (
        num_info.get("is_scalar", False)
        or num_info.get("numerical_duckdb_value") is not None
        or num_info.get("numerical_independent_value") is not None
    )

    if is_grouped:
        return {
            "mode": "grouped",
            "status": status,
            "passed": num_info.get("numerical_passed", False),
            "message": message,
            "inconclusive_reason": inconclusive_reason,
            "failure_type": details.get("failure_type"),
            "failure_reason": details.get("failure_reason"),
            "failing_group_key": details.get("failing_group_key"),
            "failing_metric": details.get("failing_metric"),
            "expected_groups": details.get("expected_groups", 0),
            "actual_groups": details.get("actual_groups", 0),
            "matched_groups": details.get("matched_groups", 0),
            "compared_metrics": details.get("compared_metrics", 0),
            "passed_comparisons": details.get("passed_comparisons", 0),
            "missing_group_keys": details.get("missing_group_keys", ()),
            "unexpected_group_keys": details.get("unexpected_group_keys", ()),
            "comparison_rows": format_numerical_evidence_table(details),
        }

    if is_scalar:
        return {
            "mode": "scalar",
            "status": status,
            "passed": num_info.get("numerical_passed", False),
            "message": message,
            "inconclusive_reason": inconclusive_reason,
            "duckdb_value": num_info.get("numerical_duckdb_value"),
            "independent_value": num_info.get("numerical_independent_value"),
            "absolute_difference": num_info.get("numerical_absolute_difference"),
            "relative_difference": num_info.get("numerical_relative_difference"),
            "atol": num_info.get("numerical_atol", 1e-4),
            "rtol": num_info.get("numerical_rtol", 1e-4),
            "compared_rows": num_info.get("numerical_compared_rows", 1),
        }

    return {
        "mode": "none",
        "status": status,
        "passed": num_info.get("numerical_passed", False),
        "message": message,
        "inconclusive_reason": inconclusive_reason,
    }


def format_semantic_validation_display(
    info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Parse Layer 3 semantic validation outcome and checks for dashboard presentation."""
    if not info or not isinstance(info, dict):
        return {
            "status": "UNVERIFIED",
            "checks": [],
            "passed_count": 0,
            "total_count": 0,
            "message": "",
            "inconclusive_reason": None,
            "canonical_projection": None,
        }

    status = info.get("semantic_status", "UNVERIFIED")
    raw_checks = info.get("semantic_checks") or []

    check_rows: list[dict[str, Any]] = []
    passed_count = 0
    for c in raw_checks:
        st_val = c.get("status", "UNVERIFIED")
        if st_val == "VERIFIED":
            status_badge = "VERIFIED"
            passed_count += 1
        elif st_val == "FAILED":
            status_badge = "FAILED"
        elif st_val == "INCONCLUSIVE":
            status_badge = "INCONCLUSIVE"
        else:
            status_badge = "UNVERIFIED"

        check_rows.append(
            {
                "Check Name": c.get("check_name", "unknown"),
                "Status": status_badge,
                "Diagnostic Code": c.get("diagnostic_code") or "None",
                "Explanation": c.get("explanation") or "",
                "intent_evidence": c.get("intent_evidence") or {},
                "ast_evidence": c.get("ast_evidence") or {},
            }
        )

    return {
        "status": status,
        "checks": check_rows,
        "passed_count": passed_count,
        "total_count": len(check_rows),
        "message": info.get("semantic_message") or "",
        "inconclusive_reason": info.get("semantic_inconclusive_reason"),
        "canonical_projection": info.get("semantic_canonical_projection"),
    }


def validation_report_to_display_rows(
    report: Any,
) -> list[dict[str, Any]]:
    """Transform WorkflowValidationReport step results into structured display rows."""
    if not isinstance(report, WorkflowValidationReport):
        return []
    rows: list[dict[str, Any]] = []
    for step_id, res in report.latest_step_results.items():
        if res.execution_result.status == ValidationStatus.VERIFIED:
            schema_label = (
                "Matched"
                if res.execution_result.schema_matched
                else "Inconclusive (Missing Intent)"
            )
        elif res.execution_result.status == ValidationStatus.FAILED:
            schema_label = "Mismatched"
        else:
            schema_label = "Unverified"

        rows.append(
            {
                "Step ID": f"Step {step_id}",
                "Step Status": res.status.value,
                "Confidence": res.confidence.value,
                "Layer 1 (Execution)": res.execution_result.status.value,
                "Layer 2 (Numerical)": res.numerical_result.status.value,
                "Layer 3 (Semantic)": res.semantic_result.status.value,
                "Schema Match": schema_label,
                "Shape": (
                    "1×1 Scalar"
                    if res.execution_result.is_scalar
                    else f"{res.execution_result.row_count}×{res.execution_result.column_count}"
                ),
                "Attempts": len(report.step_history.get(step_id, [])),
                "Details": res.execution_result.message or "Execution verified",
            }
        )
    return rows


def serialize_validation_report_for_json(
    obj: Any,
    max_preview_rows: int = 5,
) -> Any:
    """Recursively convert a validation report or arbitrary object to JSON-safe primitives.

    Handles:
        - WorkflowValidationReport and nested Pydantic BaseModel instances.
        - pandas.DataFrame: converts to structured metadata dict (shape, columns, preview).
        - pandas.Series: converts to metadata dict and values preview.
        - numpy numeric types (int64, float64, bool_): converted to native Python types.
        - NaN and infinite floats: safely converted to None or string representations.
        - Timestamps / datetimes: converted to ISO 8601 strings.
        - Enums: converted to their string values.
        - Integer dictionary keys: converted to strings for standard JSON compliance.
        - None, lists, tuples, and legacy report dictionaries: handled safely.
    """
    if obj is None:
        return None

    if isinstance(obj, BaseModel):
        return serialize_validation_report_for_json(obj.model_dump(), max_preview_rows)

    if isinstance(obj, pd.DataFrame):
        preview_records: list[dict[str, Any]] = []
        for row in obj.head(max_preview_rows).to_dict(orient="records"):
            preview_records.append(serialize_validation_report_for_json(row, max_preview_rows))
        return {
            "_type": "DataFrame",
            "row_count": len(obj),
            "column_count": len(obj.columns),
            "columns": [str(c) for c in obj.columns],
            "preview": preview_records,
        }

    if isinstance(obj, pd.Series):
        return {
            "_type": "Series",
            "name": str(obj.name) if obj.name is not None else None,
            "length": len(obj),
            "values": [
                serialize_validation_report_for_json(v, max_preview_rows)
                for v in obj.head(max_preview_rows).tolist()
            ],
        }

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return str(obj)
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return [serialize_validation_report_for_json(x, max_preview_rows) for x in obj.tolist()]

    if isinstance(obj, (datetime.datetime, datetime.date, pd.Timestamp)):
        return obj.isoformat()

    if isinstance(obj, (datetime.timedelta, pd.Timedelta)):
        return str(obj)

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, dict):
        return {
            str(k): serialize_validation_report_for_json(v, max_preview_rows)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple, set)):
        return [serialize_validation_report_for_json(item, max_preview_rows) for item in obj]

    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return str(obj)
        return obj

    if isinstance(obj, (str, int, bool)):
        return obj

    return str(obj)


def get_status_badge_html(status: str | None, label: str | None = None) -> str:
    """Generate subtle, restrained HTML status badge with professional enterprise styling."""
    clean_status = (status or "UNKNOWN").upper().strip()
    display_label = label or clean_status

    css_class = "status-unverified"
    if clean_status in {"VERIFIED", "VALIDATED", "SUCCESS", "PASSED"}:
        css_class = "status-verified"
    elif clean_status in {"FAILED", "ERROR", "FAILURE"}:
        css_class = "status-failed"
    elif clean_status in {"INCONCLUSIVE", "AMBIGUOUS", "WARNING"}:
        css_class = "status-inconclusive"
    elif clean_status in {"RUNNING", "EXECUTING", "CORRECTING", "PROCESSING"}:
        css_class = "status-running"
    elif clean_status in {"UNVERIFIED", "PENDING", "UNKNOWN"}:
        css_class = "status-unverified"

    return f'<span class="status-pill {css_class}">{display_label}</span>'


def format_data_quality_metrics(profile: TableProfile | None) -> dict[str, Any]:
    """Calculate data quality metrics and indicators from TableProfile."""
    if profile is None or not profile.columns:
        return {
            "total_cells": 0,
            "total_nulls": 0,
            "completeness_pct": 100.0,
            "columns_with_nulls": 0,
            "ambiguous_date_columns": [],
            "temporal_columns_count": 0,
            "numeric_columns_count": 0,
            "text_columns_count": 0,
        }

    total_cells = profile.row_count * profile.column_count
    total_nulls = sum(c.null_count for c in profile.columns.values())
    completeness = (1.0 - (total_nulls / total_cells)) * 100.0 if total_cells > 0 else 100.0
    cols_with_nulls = sum(1 for c in profile.columns.values() if c.null_count > 0)
    ambiguous_dates = [
        c.name for c in profile.columns.values() if c.date_format and "AMBIGUOUS" in c.date_format
    ]
    temporal_count = sum(1 for c in profile.columns.values() if c.is_temporal)
    numeric_count = sum(
        1
        for c in profile.columns.values()
        if c.duckdb_type
        in {
            "BIGINT",
            "INTEGER",
            "DOUBLE",
            "FLOAT",
            "DECIMAL",
            "HUGEINT",
            "TINYINT",
            "SMALLINT",
        }
    )
    text_count = sum(
        1 for c in profile.columns.values() if c.duckdb_type in {"VARCHAR", "TEXT", "STRING"}
    )

    return {
        "total_cells": total_cells,
        "total_nulls": total_nulls,
        "completeness_pct": completeness,
        "columns_with_nulls": cols_with_nulls,
        "ambiguous_date_columns": ambiguous_dates,
        "temporal_columns_count": temporal_count,
        "numeric_columns_count": numeric_count,
        "text_columns_count": text_count,
    }


def format_correction_history_display(
    val_report: Any,
    step_id: int | None = None,
) -> list[dict[str, Any]]:
    """Transform WorkflowValidationReport attempt history into structured display records."""
    if not isinstance(val_report, WorkflowValidationReport) or not val_report.step_history:
        return []

    target_steps = [step_id] if step_id is not None else sorted(val_report.step_history.keys())
    records: list[dict[str, Any]] = []

    for sid in target_steps:
        attempts = val_report.step_history.get(sid, [])
        total_attempts = len(attempts)
        for att in attempts:
            diag = att.diagnostic
            error_cat = diag.error_category.value if diag else None
            error_summary = diag.error_summary if diag else None
            remediation = diag.remediation_instruction if diag else None
            is_correctable = diag.is_correctable if diag else False

            exec_ok = (
                att.raw_execution_result.success
                if att.raw_execution_result is not None
                else att.execution_result.passed
            )

            # Determine decision outcome note
            if att.status == ValidationStatus.VERIFIED:
                decision_note = (
                    "Initial attempt verified across all layers (no correction needed)"
                    if att.attempt_number == 1
                    else f"Successfully corrected and verified on attempt {att.attempt_number}"
                )
            elif att.status == ValidationStatus.INCONCLUSIVE:
                decision_note = (
                    "Validation inconclusive due to ambiguous data or unprovable intent "
                    "(automated correction prohibited)"
                )
            elif not is_correctable:
                decision_note = (
                    f"Non-correctable failure ({error_cat or 'System fault'}); correction halted"
                )
            elif att.attempt_number >= 3:
                decision_note = "Correction retry budget exhausted (maximum of 2 retries used)"
            else:
                remaining = max(0, 3 - att.attempt_number)
                decision_note = (
                    f"Actionable failure diagnosed; queued for correction "
                    f"({remaining} retries remaining)"
                )

            records.append(
                {
                    "step_id": sid,
                    "attempt_number": att.attempt_number,
                    "is_latest": att.attempt_number == total_attempts,
                    "total_attempts": total_attempts,
                    "executed_sql": att.executed_sql or "-- No SQL recorded",
                    "execution_success": exec_ok,
                    "execution_status": att.execution_result.status.value,
                    "execution_message": (att.execution_result.message or "Executed successfully"),
                    "execution_time_ms": att.execution_result.execution_time_ms,
                    "numerical_status": att.numerical_result.status.value,
                    "numerical_message": att.numerical_result.message or "Verified",
                    "semantic_status": att.semantic_result.status.value,
                    "semantic_message": att.semantic_result.message or "Verified",
                    "step_status": att.status.value,
                    "confidence": att.confidence.value,
                    "has_diagnostic": diag is not None,
                    "error_category": error_cat,
                    "error_summary": error_summary,
                    "remediation_instruction": remediation,
                    "is_correctable": is_correctable,
                    "decision_note": decision_note,
                }
            )

    return records


def get_workflow_nodes_summary(
    final_state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Generate structured status trace for compiled LangGraph analytical state machine nodes."""
    if not final_state or not isinstance(final_state, dict):
        return []

    try:
        plan = final_state.get("plan")
        step_results = final_state.get("step_results") or {}
        val_report = final_state.get("validation_report")
        workflow_status = str(final_state.get("status", "unknown")).lower()
        retry_count = final_state.get("retry_count", 0)

        # 1. Planner
        plan_steps = getattr(plan, "steps", None) if plan else None
        plan_steps_count = (
            len(plan_steps) if plan_steps and isinstance(plan_steps, (list, dict)) else 0
        )
        is_multi_step = bool(getattr(plan, "is_multi_step", False)) if plan else False
        mode_label = "Multi-Step DAG" if is_multi_step else "Single-Step Direct"
        plan_desc = (
            f"{plan_steps_count} step(s) planned ({mode_label})" if plan else "No plan generated"
        )

        # 2. Code Generator
        step_sql = final_state.get("step_sql") or {}
        sql_count = len(step_sql) if isinstance(step_sql, (dict, list)) else 0
        codegen_desc = f"{sql_count} SQL statement(s) synthesized with schema grounding"

        # 3. Controlled Executor
        if isinstance(step_results, dict):
            exec_count = len(step_results)
            total_exec_time = sum(
                float(getattr(res, "execution_time_ms", 0.0) or 0.0)
                for res in step_results.values()
                if hasattr(res, "execution_time_ms")
            )
            all_exec_ok = (
                all(bool(getattr(res, "success", False)) for res in step_results.values())
                if step_results
                else False
            )
        else:
            exec_count = 0
            total_exec_time = 0.0
            all_exec_ok = False

        if exec_count > 0:
            if all_exec_ok:
                executor_status = "COMPLETED" if retry_count == 0 else "RETRY_RESOLVED"
            else:
                executor_status = "FAILED"
        else:
            executor_status = "UNVERIFIED"

        executor_desc = (
            f"{exec_count} query execution(s) completed in {total_exec_time:.2f} ms total"
            if exec_count > 0
            else "No execution results recorded"
        )

        # 4. Error Analyzer (Separate architectural node in compiled LangGraph graph)
        if retry_count > 0:
            error_analyzer_status = "TRIGGERED"
            error_analyzer_desc = (
                f"Diagnosed actionable failure on attempt {retry_count}; approved self-correction"
            )
        elif workflow_status in {"inconclusive"}:
            error_analyzer_status = "INELIGIBLE"
            error_analyzer_desc = "Validation is inconclusive; self-correction disallowed"
        elif workflow_status in {"validation_failed"} and retry_count == 0:
            error_analyzer_status = "TRIGGERED"
            error_analyzer_desc = (
                "Diagnosed terminal failure; non-correctable or retry budget exhausted"
            )
        else:
            error_analyzer_status = "SKIPPED"
            error_analyzer_desc = "Initial execution succeeded; Error Analyzer not invoked"

        # 5. Corrector (Separate architectural node in compiled LangGraph graph)
        if retry_count > 0:
            corrector_status = "REMEDIATED"
            corrector_desc = (
                f"Synthesized {retry_count} corrected query version(s) preserving analytical intent"
            )
        else:
            corrector_status = "SKIPPED"
            corrector_desc = "Initial execution succeeded; Corrector not invoked"

        # 6. Step Advance (Topological iterator node in compiled LangGraph graph)
        if is_multi_step:
            step_advance_status = "COMPLETED"
            step_advance_desc = (
                f"Iterated across {plan_steps_count} topological DAG steps with retry budget reset"
            )
        else:
            step_advance_status = "SKIPPED"
            step_advance_desc = "Single-step execution (topological iteration not required)"

        # 7. Result Assembler
        assembled = final_state.get("assembled_result")
        if assembled:
            succeeded = bool(getattr(assembled, "all_steps_succeeded", False))
            assembler_status = "COMPLETED" if succeeded else "FAILED"
            step_res_obj = getattr(assembled, "step_results", {})
            if isinstance(step_res_obj, (dict, list, tuple)):
                count = len(step_res_obj)
            else:
                count = 0
            assembler_desc = f"Consolidated {count} step output(s)"
        else:
            assembler_status = "SKIPPED"
            assembler_desc = "Result assembly not reached"

        # 8. Multi-Layer Validator
        val_status = "UNVERIFIED"
        val_desc = "Validation not executed"
        if isinstance(val_report, WorkflowValidationReport):
            val_status = val_report.workflow_status.value
            val_desc = (
                f"Workflow validation: {val_report.workflow_status.value} "
                f"({val_report.workflow_confidence.value} confidence, "
                f"{val_report.total_retries} total retries)"
            )
        elif val_report and hasattr(val_report, "workflow_status"):
            val_status = str(
                getattr(val_report.workflow_status, "value", val_report.workflow_status)
            )
            val_desc = f"Validation evaluated: {val_status}"

        return [
            {
                "node_id": "planner",
                "name": "Planner",
                "role": "Query Decomposition & Intent Grounding",
                "status": "COMPLETED" if plan else "FAILED",
                "details": plan_desc,
            },
            {
                "node_id": "codegen",
                "name": "Code Generator",
                "role": "Schema-Grounded DuckDB SQL Synthesis",
                "status": "COMPLETED" if sql_count > 0 else "FAILED",
                "details": codegen_desc,
            },
            {
                "node_id": "executor",
                "name": "Controlled Executor",
                "role": "AST-Safe Thread-Isolated Execution",
                "status": executor_status,
                "details": executor_desc,
            },
            {
                "node_id": "error_analyzer",
                "name": "Error Analyzer",
                "role": "Failure Diagnosis & Correction Eligibility",
                "status": error_analyzer_status,
                "details": error_analyzer_desc,
            },
            {
                "node_id": "corrector",
                "name": "Corrector",
                "role": "Intent-Preserving Query Remediation",
                "status": corrector_status,
                "details": corrector_desc,
            },
            {
                "node_id": "step_advance",
                "name": "Step Advance",
                "role": "Topological Step Iterator & Retry Reset",
                "status": step_advance_status,
                "details": step_advance_desc,
            },
            {
                "node_id": "assembler",
                "name": "Result Assembler",
                "role": "Multi-Step Intermediate Output Consolidation",
                "status": assembler_status,
                "details": assembler_desc,
            },
            {
                "node_id": "validator",
                "name": "Multi-Layer Validator",
                "role": "3-Layer Execution, Numerical & Semantic Verification",
                "status": val_status,
                "details": val_desc,
            },
        ]
    except Exception as err:
        return [
            {
                "node_id": node_id,
                "name": name,
                "role": "Workflow Node",
                "status": "UNVERIFIED",
                "details": f"Summary extraction fallback ({err})",
            }
            for node_id, name in [
                ("planner", "Planner"),
                ("codegen", "Code Generator"),
                ("executor", "Controlled Executor"),
                ("error_analyzer", "Error Analyzer"),
                ("corrector", "Corrector"),
                ("step_advance", "Step Advance"),
                ("assembler", "Result Assembler"),
                ("validator", "Multi-Layer Validator"),
            ]
        ]


def get_workflow_execution_path(
    final_state: dict[str, Any] | None,
) -> list[str]:
    """Derive the actual sequence of graph node transitions traversed during execution."""
    if not final_state or not isinstance(final_state, dict):
        return []

    path = ["planner"]
    plan = final_state.get("plan")
    val_report = final_state.get("validation_report")

    if isinstance(val_report, WorkflowValidationReport) and val_report.step_history:
        execution_order = getattr(plan, "execution_order", None) or sorted(
            val_report.step_history.keys()
        )
        for idx, sid in enumerate(execution_order):
            if idx > 0:
                path.append("step_advance")
            path.append("codegen")
            path.append("executor")
            attempts = val_report.step_history.get(sid, [])
            for _att in attempts[:-1]:
                path.extend(["error_analyzer", "corrector", "executor"])
            if (
                attempts
                and attempts[-1].status != ValidationStatus.VERIFIED
                and final_state.get("status") in {"validation_failed", "failed"}
            ):
                path.append("error_analyzer")
    else:
        path.extend(["codegen", "executor"])
        retry_count = int(final_state.get("retry_count", 0) or 0)
        is_multi_step = bool(getattr(plan, "is_multi_step", False)) if plan else False

        if retry_count > 0:
            path.extend(["error_analyzer", "corrector", "executor"])

        if is_multi_step:
            path.extend(["step_advance", "codegen", "executor"])

    path.extend(["assembler", "validator"])
    return path


def prepare_comparison_chart_df(
    assembled: Any,
    plan: Any = None,
) -> tuple[pd.DataFrame | None, str | None, str | None]:
    """Extract a clean, dedicated comparison chart DataFrame for comparative scalar queries.

    Replaces generic step_id by description tables with a clean Year vs Total Sales
    bar chart when the underlying results support it (e.g. comparing 2016 vs 2017).

    Returns:
        (chart_df, x_axis_label, y_axis_label) or (None, None, None).
    """
    if not assembled or not hasattr(assembled, "step_results"):
        return None, None, None

    step_results = assembled.step_results
    if not isinstance(step_results, dict) or len(step_results) < 2:
        return None, None, None

    # Check if all steps in step_results are scalar numbers
    all_scalar = all(
        getattr(r, "is_scalar", False)
        and isinstance(getattr(r, "scalar_value", None), (int, float))
        for r in step_results.values()
    )
    if not all_scalar:
        return None, None, None

    labels: list[str] = []
    values: list[float] = []

    order = getattr(plan, "execution_order", None) or sorted(step_results.keys())

    for sid in order:
        if sid not in step_results:
            continue
        res = step_results[sid]
        val = float(res.scalar_value)

        step_q = ""
        if plan and hasattr(plan, "get_step"):
            step_obj = plan.get_step(sid)
            if step_obj and hasattr(step_obj, "question"):
                step_q = step_obj.question

        if (
            not step_q
            and hasattr(assembled, "primary_dataframe")
            and assembled.primary_dataframe is not None
        ):
            sdf = assembled.primary_dataframe
            if "step_id" in sdf.columns and "description" in sdf.columns:
                match = sdf[sdf["step_id"] == sid]
                if not match.empty:
                    step_q = str(match["description"].iloc[0])

        m = re.search(r"\b(19\d\d|20\d\d)\b", step_q)
        if m:
            labels.append(m.group(1))
        else:
            labels.append(f"Step {sid}")
        values.append(val)

    # Check if distinct 4-digit years were identified across all steps
    has_all_years = all(bool(re.match(r"^(19\d\d|20\d\d)$", lab)) for lab in labels)
    distinct_years = len(set(labels)) == len(labels)

    if has_all_years and distinct_years:
        x_col = "Year"
        orig_q = str(
            getattr(assembled, "original_question", "")
            or (getattr(plan, "original_question", "") if plan else "")
        ).lower()
        if "profit" in orig_q:
            y_col = "Total Profit"
        elif "quantity" in orig_q:
            y_col = "Total Quantity"
        else:
            y_col = "Total Sales"

        chart_df = pd.DataFrame({x_col: labels, y_col: values}).set_index(x_col)
        return chart_df, x_col, y_col

    return None, None, None


def prepare_chart_data(
    df: pd.DataFrame | None,
) -> tuple[str, str | None, str | None]:
    """Determine whether a DataFrame has suitable grain for analytical chart visualization.

    Returns:
        Tuple of (chart_type, x_column, y_column) where chart_type is 'bar' | 'line' | 'table_only'.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df.columns) < 2:
        return "table_only", None, None

    # Guard: do NOT render generic step_id by description charts for scalar comparisons
    if set(df.columns).issubset({"step_id", "description", "value"}):
        return "table_only", None, None
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df.columns) < 2:
        return "table_only", None, None

    cols = list(df.columns)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in cols if c not in numeric_cols]

    if not numeric_cols:
        return "table_only", None, None

    # Priority 1: 1 categorical/date column + 1 numeric column
    if len(non_numeric_cols) == 1 and len(numeric_cols) >= 1:
        x_col = non_numeric_cols[0]
        y_col = numeric_cols[0]
        if (
            pd.api.types.is_datetime64_any_dtype(df[x_col])
            or "date" in str(x_col).lower()
            or "year" in str(x_col).lower()
        ):
            return "line", x_col, y_col
        if df[x_col].nunique() <= 30:
            return "bar", x_col, y_col

    # Priority 2: 2 numeric columns (e.g. year and sales)
    if len(numeric_cols) >= 2 and len(non_numeric_cols) == 0:
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        if (
            "year" in str(x_col).lower()
            or "date" in str(x_col).lower()
            or "month" in str(x_col).lower()
        ):
            return "line", x_col, y_col
        if df[x_col].nunique() <= 30:
            return "bar", x_col, y_col

    return "table_only", None, None
