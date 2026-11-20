"""Unit tests for UI-independent ingestion and display helpers."""

import datetime
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import profile_table
from src.agent.execution.engine import ExecutionResult
from src.agent.validation.contracts import (
    CorrectionDiagnostic,
    ErrorCategory,
    ExecutionValidationResult,
    NumericalValidationResult,
    SemanticValidationResult,
    ValidationStatus,
    WorkflowValidationReport,
    build_step_validation_result,
)
from src.ui.helpers import (
    format_step_validation_summary,
    get_excel_sheet_names,
    get_lifecycle_status_badge,
    get_table_preview_df,
    ingest_uploaded_file,
    profile_to_display_df,
    resolve_table_name_collision,
    serialize_validation_report_for_json,
    validation_report_to_display_rows,
)


def test_resolve_table_name_collision():
    """Verify numeric suffix incrementing upon collision."""
    existing = ["sales", "sales_2", "customers"]
    assert resolve_table_name_collision(existing, "sales") == "sales_3"
    assert resolve_table_name_collision(existing, "new_table") == "new_table"
    assert resolve_table_name_collision(existing, "Customers") == "customers_2"


def test_ingest_uploaded_csv_utf8():
    """Verify ingesting a standard UTF-8 encoded CSV."""
    db = SessionDatabase()
    csv_bytes = "Customer,Revenue\nAditi,500.0\n".encode("utf-8")
    success, table_name, err = ingest_uploaded_file(db, csv_bytes, "customers.csv")

    assert success is True
    assert table_name == "customers"
    assert err is None
    assert db.is_table_allowed("customers")

    preview = get_table_preview_df(db, "customers")
    assert preview["customer"].iloc[0] == "Aditi"
    assert preview["revenue"].iloc[0] == 500.0
    db.close()


def test_ingest_uploaded_csv_utf8_sig():
    """Verify ingesting a UTF-8 CSV with Byte Order Mark (BOM).

    BOM (\\xef\\xbb\\xbf) must be stripped cleanly so headers do not contain
    the \\ufeff artifact.
    """
    db = SessionDatabase()
    raw_text = "\ufeffOrder ID,Sales\nORD-999,120.0\n"
    csv_bytes = raw_text.encode("utf-8-sig")
    success, table_name, err = ingest_uploaded_file(db, csv_bytes, "bom_orders.csv")

    assert success is True
    assert table_name == "bom_orders"
    assert err is None

    # Verify column name has no BOM prefix artifact
    preview = get_table_preview_df(db, "bom_orders")
    assert "order_id" in preview.columns
    assert preview["order_id"].iloc[0] == "ORD-999"
    db.close()


def test_ingest_uploaded_csv_cp1252_and_special_characters():
    """Verify ingesting a CP1252-encoded CSV with preserved original text.

    CP1252 includes characters like Euro '€' (0x80) and umlaut 'ö' (0xf6)
    which must be preserved intact without replacement or data loss.
    """
    db = SessionDatabase()
    raw_text = "Customer,City,Price\nBjörn,München,50€\n"
    cp1252_bytes = raw_text.encode("cp1252")
    success, table_name, err = ingest_uploaded_file(db, cp1252_bytes, "euro_sales.csv")

    assert success is True
    assert table_name == "euro_sales"
    assert err is None

    preview = get_table_preview_df(db, "euro_sales")
    assert preview["customer"].iloc[0] == "Björn"
    assert preview["city"].iloc[0] == "München"
    assert preview["price"].iloc[0] == "50€"
    db.close()


def test_ingest_uploaded_csv_filename_with_umlauts():
    """Verify ingesting a file whose filename contains special characters like 'ö' and 'ä'."""
    db = SessionDatabase()
    csv_bytes = "Product,Qty\nWidget,10\n".encode("utf-8")
    # Filename with umlaut 'ö' and 'ä'
    filename = "björn_verkäufe_2024.csv"
    success, table_name, err = ingest_uploaded_file(db, csv_bytes, filename)

    assert success is True
    # Sanitized identifier should cleanly transliterate to valid ASCII table name
    assert table_name == "bjorn_verkaufe_2024"
    assert db.is_table_allowed("bjorn_verkaufe_2024")

    preview = get_table_preview_df(db, "bjorn_verkaufe_2024")
    assert len(preview) == 1
    assert preview["product"].iloc[0] == "Widget"
    db.close()


def test_ingest_uploaded_excel_valid():
    """Verify ingesting an Excel spreadsheet byte payload and sheet inspection."""
    db = SessionDatabase()
    df = pd.DataFrame({"product": ["A", "B"], "qty": [10, 20]})
    stream = io.BytesIO()
    with pd.ExcelWriter(stream, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Inventory", index=False)
    excel_bytes = stream.getvalue()

    # Verify sheet inspection
    sheets = get_excel_sheet_names(excel_bytes)
    assert sheets == ["Inventory"]

    # Ingest Excel
    success, table_name, err = ingest_uploaded_file(
        db, excel_bytes, "stock.xlsx", sheet_name="Inventory"
    )
    assert success is True
    assert table_name == "stock"
    assert err is None
    assert db.is_table_allowed("stock")
    db.close()


def test_ingest_uploaded_file_edge_cases():
    """Verify rejection of empty, unsupported, or corrupted files."""
    db = SessionDatabase()

    # Empty bytes
    ok, msg, detail = ingest_uploaded_file(db, b"", "empty.csv")
    assert not ok
    assert "empty" in msg.lower()

    # Unsupported extension
    ok, msg, detail = ingest_uploaded_file(db, b"hello", "test.txt")
    assert not ok
    assert "unsupported" in msg.lower()

    # Corrupted / invalid Excel bytes
    ok, msg, detail = ingest_uploaded_file(db, b"not_an_excel_file", "bad.xlsx")
    assert not ok
    assert "failed" in msg.lower() or "error" in msg.lower()
    db.close()


def test_get_table_preview_df():
    """Verify safe relational table preview retrieval."""
    db = SessionDatabase()
    df = pd.DataFrame({"id": range(30), "val": [f"item_{i}" for i in range(30)]})
    db.register_dataframe(df, "large_table")

    # Safe preview limit
    preview = get_table_preview_df(db, "large_table", limit=20)
    assert len(preview) == 20
    assert list(preview.columns) == ["id", "val"]

    # Unregistered table rejection
    with pytest.raises(ValueError, match="not registered"):
        get_table_preview_df(db, "ghost_table")
    db.close()


def test_profile_to_display_df(data_dir: Path):
    """Verify conversion of TableProfile into a display DataFrame."""
    db = SessionDatabase()
    db.register_csv(data_dir / "sample_superstore.csv", "superstore")
    profile = profile_table(db, "superstore")
    display_df = profile_to_display_df(profile)

    assert not display_df.empty
    expected_cols = [
        "Column",
        "Type",
        "Date Format",
        "Null Count",
        "Null %",
        "Distinct",
        "Min",
        "Max",
        "Mean",
        "Samples",
    ]
    assert list(display_df.columns) == expected_cols

    # Ensure 'sales' row has numeric mean formatted
    sales_row = display_df[display_df["Column"] == "sales"].iloc[0]
    assert sales_row["Null Count"] == 0
    assert sales_row["Mean"] != "-"
    db.close()


def test_get_lifecycle_status_badge():
    """Verify lifecycle status mapping to icon and severity for all four statuses."""
    label_v, icon_v, alert_v = get_lifecycle_status_badge("validated")
    assert "VALIDATED" in label_v
    assert icon_v == "🟢"
    assert alert_v == "success"

    label_e, icon_e, alert_e = get_lifecycle_status_badge("execution_verified")
    assert "EXECUTION VERIFIED" in label_e
    assert icon_e == "🔵"
    assert alert_e == "info"

    label_i, icon_i, alert_i = get_lifecycle_status_badge("inconclusive")
    assert "INCONCLUSIVE" in label_i
    assert icon_i == "🟡"
    assert alert_i == "warning"

    label_f, icon_f, alert_f = get_lifecycle_status_badge("validation_failed")
    assert "VALIDATION FAILED" in label_f
    assert icon_f == "🔴"
    assert alert_f == "error"

    # None and unknown status handling
    label_none, icon_none, alert_none = get_lifecycle_status_badge(None)
    assert "UNKNOWN" in label_none
    assert icon_none == "⚪"
    assert alert_none == "info"

    label_u, icon_u, alert_u = get_lifecycle_status_badge("unknown_status")
    assert "UNKNOWN_STATUS" in label_u
    assert icon_u == "⚪"
    assert alert_u == "info"


def test_format_step_validation_summary_verified():
    """Verify per-step formatting when Layer 1 execution is verified."""
    report = WorkflowValidationReport()
    step_val = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="Execution verified with schema match",
            schema_matched=True,
            is_scalar=True,
            row_count=1,
            column_count=1,
        ),
        numerical_res=NumericalValidationResult(
            passed=False,
            status=ValidationStatus.UNVERIFIED,
            message="Numerical calculation pending Stage 3",
        ),
        semantic_res=SemanticValidationResult(
            passed=False,
            status=ValidationStatus.UNVERIFIED,
            message="Semantic coverage pending Stage 4",
        ),
        raw_execution_result=ExecutionResult(
            success=True,
            query="SELECT SUM(sales) AS total_sales FROM superstore",
            is_scalar=True,
            scalar_value=123.45,
        ),
        has_meaningful_output=True,
    )
    report.add_step_attempt(step_val)

    summary = format_step_validation_summary(report, 1)
    assert summary is not None
    assert summary["step_status"] == "UNVERIFIED"
    assert summary["execution_passed"] is True
    assert summary["execution_status"] == "VERIFIED"
    assert summary["shape_description"] == "Scalar (1×1)"
    assert summary["schema_matched"] == "Matched"
    assert summary["numerical_status"] == "UNVERIFIED"
    assert summary["semantic_status"] == "UNVERIFIED"
    assert summary["diagnostic"] is None
    assert summary["attempts"] == 1


def test_format_step_validation_summary_inconclusive():
    """Verify per-step formatting when structured intent is missing (Option C)."""
    report = WorkflowValidationReport()
    step_val = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="Physical execution verified; schema unverified (missing QuerySpecification)",
            schema_matched=False,
            is_scalar=False,
            row_count=10,
            column_count=2,
        ),
        numerical_res=NumericalValidationResult(
            passed=False,
            status=ValidationStatus.INCONCLUSIVE,
            message="Numerical calculation cannot proceed without QuerySpecification",
        ),
        semantic_res=SemanticValidationResult(
            passed=False,
            status=ValidationStatus.INCONCLUSIVE,
            message="Semantic coverage cannot proceed without QuerySpecification",
        ),
        raw_execution_result=ExecutionResult(success=True, query="SELECT 1"),
        has_meaningful_output=True,
    )
    report.add_step_attempt(step_val)

    summary = format_step_validation_summary(report, 1)
    assert summary is not None
    assert summary["step_status"] == "INCONCLUSIVE"
    assert summary["confidence"] == "MEDIUM"
    assert summary["execution_passed"] is True
    assert summary["schema_matched"] == "Inconclusive (Missing Intent)"
    assert summary["shape_description"] == "Tabular (10×2)"
    assert summary["numerical_status"] == "INCONCLUSIVE"
    assert summary["semantic_status"] == "INCONCLUSIVE"


def test_format_step_validation_summary_failed_with_diagnostic():
    """Verify per-step formatting when validation fails with diagnostic metadata."""
    report = WorkflowValidationReport()
    diag = CorrectionDiagnostic(
        error_category=ErrorCategory.SCHEMA_ERROR,
        is_correctable=True,
        failed_layer="Layer 1: Execution",
        error_summary="Column 'invalid_col' does not exist in superstore",
        remediation_instruction="Use valid column from profile",
        prior_attempt_sql="SELECT invalid_col FROM superstore",
        attempt_number=1,
    )
    step_val = build_step_validation_result(
        step_id=2,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=False,
            status=ValidationStatus.FAILED,
            message="Schema mismatch: column missing",
            schema_matched=False,
        ),
        numerical_res=NumericalValidationResult(
            passed=False,
            status=ValidationStatus.UNVERIFIED,
            message="Unverified",
        ),
        semantic_res=SemanticValidationResult(
            passed=False,
            status=ValidationStatus.UNVERIFIED,
            message="Unverified",
        ),
        diagnostic=diag,
    )
    report.add_step_attempt(step_val)

    summary = format_step_validation_summary(report, 2)
    assert summary is not None
    assert summary["step_status"] == "FAILED"
    assert summary["execution_passed"] is False
    assert summary["execution_status"] == "FAILED"
    assert summary["schema_matched"] == "Mismatched"
    assert summary["diagnostic"] is not None
    assert summary["diagnostic"]["category"] == "SCHEMA_ERROR"
    assert summary["diagnostic"]["correctable"] is True
    assert "invalid_col" in summary["diagnostic"]["summary"]


def test_format_step_validation_summary_missing_or_invalid():
    """Verify safe degradation when report or step is missing or of older type."""
    # None report
    assert format_step_validation_summary(None, 1) is None

    # Legacy dictionary report
    assert format_step_validation_summary({"execution_passed": True}, 1) is None

    # Step ID not found
    report = WorkflowValidationReport()
    assert format_step_validation_summary(report, 99) is None


def test_validation_report_to_display_rows():
    """Verify structured row formatting for WorkflowValidationReport display."""
    report = WorkflowValidationReport()
    step_val = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="Execution verified",
            schema_matched=True,
            is_scalar=True,
        ),
        numerical_res=NumericalValidationResult(
            passed=False,
            status=ValidationStatus.UNVERIFIED,
            message="Unverified",
        ),
        semantic_res=SemanticValidationResult(
            passed=False,
            status=ValidationStatus.UNVERIFIED,
            message="Unverified",
        ),
        raw_execution_result=ExecutionResult(success=True, query="SELECT 1"),
        has_meaningful_output=True,
    )
    report.add_step_attempt(step_val)

    rows = validation_report_to_display_rows(report)
    assert len(rows) == 1
    r = rows[0]
    assert r["Step ID"] == "Step 1"
    assert r["Step Status"] == "UNVERIFIED"
    assert r["Layer 1 (Execution)"] == "VERIFIED"
    assert r["Layer 2 (Numerical)"] == "UNVERIFIED"
    assert r["Layer 3 (Semantic)"] == "UNVERIFIED"
    assert r["Schema Match"] == "✅ Matched"
    assert r["Shape"] == "1×1 Scalar"
    assert r["Attempts"] == 1

    # Test with inconclusive step (missing intent)
    report_inconclusive = WorkflowValidationReport()
    step_val_inc = build_step_validation_result(
        step_id=2,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="Physical execution verified",
            schema_matched=False,
        ),
        numerical_res=NumericalValidationResult(
            passed=False,
            status=ValidationStatus.INCONCLUSIVE,
            message="Intent missing",
        ),
        semantic_res=SemanticValidationResult(
            passed=False,
            status=ValidationStatus.INCONCLUSIVE,
            message="Intent missing",
        ),
        raw_execution_result=ExecutionResult(success=True, query="SELECT 1"),
        has_meaningful_output=True,
    )
    report_inconclusive.add_step_attempt(step_val_inc)
    rows_inc = validation_report_to_display_rows(report_inconclusive)
    assert len(rows_inc) == 1
    assert rows_inc[0]["Step Status"] == "INCONCLUSIVE"
    assert rows_inc[0]["Schema Match"] == "⚠️ Inconclusive (Missing Intent)"
    assert rows_inc[0]["Layer 2 (Numerical)"] == "INCONCLUSIVE"
    assert rows_inc[0]["Layer 3 (Semantic)"] == "INCONCLUSIVE"

    # Safe handling of non-reports
    assert validation_report_to_display_rows(None) == []
    assert validation_report_to_display_rows({}) == []


def test_helpers_module_exports_and_imports():
    """Verify that all helpers expected by app.py are properly exported and importable."""
    import src.ui.helpers as helpers

    expected_exports = [
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

    for export_name in expected_exports:
        assert hasattr(helpers, export_name), f"Missing export: {export_name}"
        assert callable(getattr(helpers, export_name)), f"Export {export_name} must be callable"

    # Explicitly test module-level imports as in app.py
    from src.ui.helpers import (
        format_step_validation_summary,
        get_excel_sheet_names,
        get_lifecycle_status_badge,
        get_table_preview_df,
        ingest_uploaded_file,
        profile_to_display_df,
        serialize_validation_report_for_json,
        validation_report_to_display_rows,
    )

    assert callable(format_step_validation_summary)
    assert callable(get_lifecycle_status_badge)
    assert callable(validation_report_to_display_rows)
    assert callable(serialize_validation_report_for_json)
    assert callable(get_table_preview_df)
    assert callable(get_excel_sheet_names)
    assert callable(ingest_uploaded_file)
    assert callable(profile_to_display_df)


def test_serialize_validation_report_for_json_with_dataframe_and_numpy():
    """Verify JSON-safe serialization of report containing DataFrames and NumPy primitives."""
    df = pd.DataFrame({
        "region": ["Central", "East"],
        "sales": [np.float64(100.5), np.float64(200.75)],
        "int_col": [np.int64(10), np.int64(20)],
        "timestamp": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
    })
    raw_res = ExecutionResult(
        success=True,
        query="SELECT region, sales, int_col FROM superstore",
        dataframe=df,
        row_count=2,
        column_count=4,
    )

    step_val = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="Execution verified",
            schema_matched=True,
            row_count=2,
            column_count=4,
        ),
        numerical_res=NumericalValidationResult(
            passed=False,
            status=ValidationStatus.UNVERIFIED,
            message="Numerical pending Stage 3",
        ),
        semantic_res=SemanticValidationResult(
            passed=False,
            status=ValidationStatus.UNVERIFIED,
            message="Semantic pending Stage 4",
        ),
        raw_execution_result=raw_res,
        has_meaningful_output=True,
    )

    report = WorkflowValidationReport()
    report.add_step_attempt(step_val)

    # Serialize for JSON
    serialized = serialize_validation_report_for_json(report, max_preview_rows=5)

    # 1. Must be directly serializable with standard json.dumps without raising errors
    json_str = json.dumps(serialized, indent=2)
    assert isinstance(json_str, str)
    assert len(json_str) > 0

    # 2. DataFrame metadata must be structured and preserved
    raw_ser = serialized["latest_step_results"]["1"]["raw_execution_result"]
    assert raw_ser is not None
    df_ser = raw_ser["dataframe"]
    assert isinstance(df_ser, dict)
    assert df_ser["_type"] == "DataFrame"
    assert df_ser["row_count"] == 2
    assert df_ser["column_count"] == 4
    assert df_ser["columns"] == ["region", "sales", "int_col", "timestamp"]
    assert len(df_ser["preview"]) == 2
    assert df_ser["preview"][0]["region"] == "Central"
    assert df_ser["preview"][0]["sales"] == 100.5
    assert df_ser["preview"][0]["int_col"] == 10
    assert "2024-01-01" in df_ser["preview"][0]["timestamp"]

    # 3. Retained execution results must also be safely serialized
    retained_df_ser = serialized["retained_execution_results"]["1"]["dataframe"]
    assert retained_df_ser["_type"] == "DataFrame"
    assert retained_df_ser["row_count"] == 2

    # 4. Invariant: Original report must NOT be mutated
    assert isinstance(report.latest_step_results[1].raw_execution_result.dataframe, pd.DataFrame)
    assert isinstance(report.retained_execution_results[1].dataframe, pd.DataFrame)


def test_serialize_validation_report_for_json_edge_cases():
    """Verify safe handling of edge cases, Series, NaN/Inf, and arbitrary nested collections."""
    # None input
    assert serialize_validation_report_for_json(None) is None

    # Series input
    s = pd.Series([np.int64(1), np.int64(2)], name="test_series")
    ser_s = serialize_validation_report_for_json(s)
    assert ser_s["_type"] == "Series"
    assert ser_s["name"] == "test_series"
    assert ser_s["values"] == [1, 2]

    # NumPy scalars and special float values
    data = {
        "nan_val": np.float64(np.nan),
        "inf_val": np.float64(np.inf),
        "neg_inf_val": np.float64(-np.inf),
        "np_int": np.int64(42),
        "np_bool": np.bool_(True),
        "date_val": datetime.date(2024, 5, 1),
        "delta_val": datetime.timedelta(days=1),
        "set_val": {1, 2},
    }
    ser_data = serialize_validation_report_for_json(data)
    assert ser_data["nan_val"] is None
    assert ser_data["inf_val"] == "inf"
    assert ser_data["neg_inf_val"] == "-inf"
    assert ser_data["np_int"] == 42
    assert ser_data["np_bool"] is True
    assert ser_data["date_val"] == "2024-05-01"
    assert "1 day" in ser_data["delta_val"]
    assert isinstance(ser_data["set_val"], list)

    # Legacy dictionary input
    legacy = {"execution_passed": True, "details": {"inner_df": pd.DataFrame({"x": [1]})}}
    ser_legacy = serialize_validation_report_for_json(legacy)
    assert ser_legacy["execution_passed"] is True
    assert ser_legacy["details"]["inner_df"]["_type"] == "DataFrame"
