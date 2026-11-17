"""UI-independent data ingestion and profiling presentation helpers.

Provides safe file-stream loading, collision resolution, Excel sheet inspection,
preview extraction without raw SQL execution, and display table transformations.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd

from src.agent.data.ingestion import (
    SessionDatabase,
    read_csv_with_fallback,
    sanitize_identifier,
)
from src.agent.data.profiling import TableProfile


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
