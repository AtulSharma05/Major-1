"""UI-independent data ingestion and profiling presentation helpers.

Provides safe file-stream loading, collision resolution, Excel sheet inspection,
preview extraction without raw SQL execution, and display table transformations.
"""

from __future__ import annotations

import datetime
import io
import math
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
    "format_step_validation_summary",
    "validation_report_to_display_rows",
    "serialize_validation_report_for_json",
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

    # Safe relational limit access (no raw SQL string interpolation)
    rel = db.connection.table(clean_name).limit(limit)
    return rel.df()


def profile_to_display_df(profile: TableProfile) -> pd.DataFrame:
    """Transform TableProfile columns into a clean DataFrame for Streamlit display."""
    records: list[dict[str, Any]] = []

    for col in profile.columns.values():
        samples_str = (
            ", ".join(repr(s) for s in col.sample_values[:3])
            if col.sample_values
            else "-"
        )
        mean_str = f"{col.mean_val:.2f}" if col.mean_val is not None else "-"
        date_fmt = col.date_format if col.is_temporal and col.date_format else "-"

        records.append({
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
        })

    return pd.DataFrame(records)


def get_lifecycle_status_badge(status: str | None) -> tuple[str, str, str]:
    """Return display label, icon, and alert severity for a workflow lifecycle status."""
    mapping = {
        "validated": (
            "VALIDATED — All Steps Verified across Execution, Numerical & Semantic Layers",
            "🟢",
            "success",
        ),
        "execution_verified": (
            "EXECUTION VERIFIED — Queries Executed & Schema Confirmed (Layer 1 Passed)",
            "🔵",
            "info",
        ),
        "inconclusive": (
            "INCONCLUSIVE — Execution Succeeded, but Analytical Correctness Not Established",
            "🟡",
            "warning",
        ),
        "validation_failed": (
            "VALIDATION FAILED — Execution or Layer Invariant Violation Detected",
            "🔴",
            "error",
        ),
    }
    clean_status = (status or "unknown").lower()
    return mapping.get(
        clean_status,
        (f"STATUS: {clean_status.upper()}", "⚪", "info"),
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
        "numerical_status": step_val.numerical_result.status.value,
        "semantic_status": step_val.semantic_result.status.value,
        "diagnostic": diagnostic_summary,
        "attempts": len(val_report.step_history.get(step_id, [])),
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
                "✅ Matched"
                if res.execution_result.schema_matched
                else "⚠️ Inconclusive (Missing Intent)"
            )
        elif res.execution_result.status == ValidationStatus.FAILED:
            schema_label = "❌ Mismatched"
        else:
            schema_label = "⚪ Unverified"

        rows.append({
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
        })
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
            preview_records.append(
                serialize_validation_report_for_json(row, max_preview_rows)
            )
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
        return [
            serialize_validation_report_for_json(x, max_preview_rows)
            for x in obj.tolist()
        ]

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
        return [
            serialize_validation_report_for_json(item, max_preview_rows)
            for item in obj
        ]

    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return str(obj)
        return obj

    if isinstance(obj, (str, int, bool)):
        return obj

    return str(obj)
