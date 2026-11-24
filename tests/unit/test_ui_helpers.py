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
from src.agent.planning.assembler import AssembledResult
from src.agent.validation.contracts import (
    CANONICAL_SEMANTIC_CHECKS,
    CorrectionDiagnostic,
    ErrorCategory,
    ExecutionValidationResult,
    NumericalValidationResult,
    SemanticCheckDetail,
    SemanticValidationResult,
    ValidationStatus,
    WorkflowValidationReport,
    build_step_validation_result,
)
from src.agent.validation.semantic import SemanticValidator
from src.agent.workflow.graph import build_analytical_graph
from src.ui.helpers import (
    format_correction_history_display,
    format_data_quality_metrics,
    format_numerical_evidence_table,
    format_numerical_validation_display,
    format_semantic_validation_display,
    format_step_validation_summary,
    get_excel_sheet_names,
    get_lifecycle_status_badge,
    get_status_badge_html,
    get_table_preview_df,
    get_workflow_execution_path,
    get_workflow_nodes_summary,
    ingest_uploaded_file,
    prepare_chart_data,
    prepare_comparison_chart_df,
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
    assert icon_v == ""
    assert alert_v == "success"

    label_e, icon_e, alert_e = get_lifecycle_status_badge("execution_verified")
    assert "EXECUTION VERIFIED" in label_e
    assert icon_e == ""
    assert alert_e == "info"

    label_i, icon_i, alert_i = get_lifecycle_status_badge("inconclusive")
    assert "INCONCLUSIVE" in label_i
    assert icon_i == ""
    assert alert_i == "warning"

    label_f, icon_f, alert_f = get_lifecycle_status_badge("validation_failed")
    assert "VALIDATION FAILED" in label_f
    assert icon_f == ""
    assert alert_f == "error"

    # None and unknown status handling
    label_none, icon_none, alert_none = get_lifecycle_status_badge(None)
    assert "UNKNOWN" in label_none
    assert icon_none == ""
    assert alert_none == "info"

    label_u, icon_u, alert_u = get_lifecycle_status_badge("unknown_status")
    assert "UNKNOWN_STATUS" in label_u
    assert icon_u == ""
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
    assert r["Schema Match"] == "Matched"
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
    assert rows_inc[0]["Schema Match"] == "Inconclusive (Missing Intent)"
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
    df = pd.DataFrame(
        {
            "region": ["Central", "East"],
            "sales": [np.float64(100.5), np.float64(200.75)],
            "int_col": [np.int64(10), np.int64(20)],
            "timestamp": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
        }
    )
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


def test_format_numerical_evidence_table_success():
    """Verify formatting raw numerical comparisons into display rows."""
    details = {
        "comparisons": [
            {
                "group_key": ("East",),
                "metric": "sales",
                "duckdb_value": 100.0,
                "independent_value": 100.0,
                "absolute_difference": 0.0,
                "relative_difference": 0.0,
                "passed": True,
            },
            {
                "group_key": ("West",),
                "metric": "sales",
                "duckdb_value": 200.0,
                "independent_value": 200.0,
                "absolute_difference": 0.0,
                "relative_difference": 0.0,
                "passed": True,
            },
        ]
    }
    rows = format_numerical_evidence_table(details)
    assert len(rows) == 2
    assert rows[0]["Group Key"] == "East"
    assert rows[0]["Metric"] == "sales"
    assert rows[0]["DuckDB Value"] == "100.0"
    assert rows[0]["Independent Value"] == "100.0"
    assert rows[0]["Status"] == "PASSED"

    # None and empty inputs
    assert format_numerical_evidence_table(None) == []
    assert format_numerical_evidence_table({}) == []
    assert format_numerical_evidence_table({"comparisons": []}) == []


def test_format_numerical_evidence_table_zero_denominator():
    """Verify relative difference zero denominator formatting."""
    details = {
        "comparisons": [
            {
                "group_key": ("Central",),
                "metric": "profit",
                "duckdb_value": 0.0,
                "independent_value": 0.0,
                "absolute_difference": 0.0,
                "relative_difference": None,
                "passed": True,
            }
        ]
    }
    rows = format_numerical_evidence_table(details)
    assert len(rows) == 1
    assert rows[0]["Relative Difference"] == "N/A (zero denominator)"


def test_format_numerical_validation_display_successful_grouped():
    """Verify UI parsing of successful grouped query evidence."""
    info = {
        "numerical_status": "VERIFIED",
        "numerical_passed": True,
        "numerical_message": "All 2 grouped categories and metrics independently verified.",
        "numerical_details": {
            "expected_groups": 2,
            "actual_groups": 2,
            "matched_groups": 2,
            "compared_metrics": 2,
            "passed_comparisons": 2,
            "comparisons": [
                {
                    "group_key": ("East",),
                    "metric": "sales",
                    "duckdb_value": 100.0,
                    "independent_value": 100.0,
                    "absolute_difference": 0.0,
                    "relative_difference": 0.0,
                    "passed": True,
                }
            ],
        },
    }
    disp = format_numerical_validation_display(info)
    assert disp["mode"] == "grouped"
    assert disp["status"] == "VERIFIED"
    assert disp["passed"] is True
    assert disp["expected_groups"] == 2
    assert disp["matched_groups"] == 2
    assert len(disp["comparison_rows"]) == 1


def test_format_numerical_validation_display_failed_metric_mismatch():
    """Verify UI parsing of metric value mismatch failure."""
    info = {
        "numerical_status": "FAILED",
        "numerical_passed": False,
        "numerical_message": "Group metric mismatch",
        "numerical_duckdb_value": 999.0,
        "numerical_independent_value": 200.0,
        "numerical_absolute_difference": 799.0,
        "numerical_details": {
            "failure_type": "metric_mismatch",
            "failing_group_key": ("West",),
            "failing_metric": "sales",
            "expected_groups": 2,
            "actual_groups": 2,
            "matched_groups": 2,
            "compared_metrics": 2,
            "passed_comparisons": 1,
            "comparisons": [
                {
                    "group_key": ("East",),
                    "metric": "sales",
                    "duckdb_value": 100.0,
                    "independent_value": 100.0,
                    "absolute_difference": 0.0,
                    "relative_difference": 0.0,
                    "passed": True,
                },
                {
                    "group_key": ("West",),
                    "metric": "sales",
                    "duckdb_value": 999.0,
                    "independent_value": 200.0,
                    "absolute_difference": 799.0,
                    "relative_difference": 3.995,
                    "passed": False,
                },
            ],
        },
    }
    disp = format_numerical_validation_display(info)
    assert disp["mode"] == "grouped"
    assert disp["status"] == "FAILED"
    assert disp["failure_type"] == "metric_mismatch"
    assert disp["failing_group_key"] == ("West",)
    assert disp["failing_metric"] == "sales"
    assert len(disp["comparison_rows"]) == 2
    assert disp["comparison_rows"][1]["Status"] == "FAILED"


def test_format_numerical_validation_display_failed_group_structure():
    """Verify UI parsing of group structure failure (missing groups)."""
    info = {
        "numerical_status": "FAILED",
        "numerical_passed": False,
        "numerical_message": "Group count mismatch",
        "numerical_details": {
            "failure_type": "group_structure",
            "failure_reason": "group_count_mismatch",
            "expected_groups": 3,
            "actual_groups": 2,
            "matched_groups": 2,
            "missing_group_keys": (("Central",),),
            "unexpected_group_keys": (),
            "compared_metrics": 2,
            "passed_comparisons": 2,
            "comparisons": [
                {
                    "group_key": ("East",),
                    "metric": "sales",
                    "duckdb_value": 100.0,
                    "independent_value": 100.0,
                    "absolute_difference": 0.0,
                    "relative_difference": 0.0,
                    "passed": True,
                }
            ],
        },
    }
    disp = format_numerical_validation_display(info)
    assert disp["mode"] == "grouped"
    assert disp["failure_type"] == "group_structure"
    assert disp["failure_reason"] == "group_count_mismatch"
    assert disp["missing_group_keys"] == (("Central",),)
    assert len(disp["comparison_rows"]) == 1


def test_format_numerical_validation_display_scalar_evidence():
    """Verify UI parsing of scalar numerical evidence."""
    info = {
        "numerical_status": "VERIFIED",
        "numerical_passed": True,
        "is_scalar": True,
        "numerical_duckdb_value": 5000.0,
        "numerical_independent_value": 5000.0,
        "numerical_absolute_difference": 0.0,
        "numerical_relative_difference": 0.0,
        "numerical_atol": 1e-4,
        "numerical_rtol": 1e-4,
        "numerical_details": {},
    }
    disp = format_numerical_validation_display(info)
    assert disp["mode"] == "scalar"
    assert disp["status"] == "VERIFIED"
    assert disp["duckdb_value"] == 5000.0
    assert disp["independent_value"] == 5000.0
    assert disp["absolute_difference"] == 0.0


def test_format_numerical_validation_display_inconclusive_and_none():
    """Verify that inconclusive and missing numerical details do not manufacture evidence."""
    info = {
        "numerical_status": "INCONCLUSIVE",
        "numerical_passed": False,
        "numerical_message": "QuerySpecification unavailable",
        "numerical_inconclusive_reason": "intent_missing",
        "numerical_details": {},
    }
    disp = format_numerical_validation_display(info)
    assert disp["mode"] == "none"
    assert disp["status"] == "INCONCLUSIVE"
    assert disp["inconclusive_reason"] == "intent_missing"

    # None input
    assert format_numerical_validation_display(None) == {"mode": "none", "status": "UNVERIFIED"}


def test_format_step_validation_summary_extracts_numerical_fields():
    """Verify format_step_validation_summary extracts numerical details from report."""
    exec_res = ExecutionResult(success=True, query="SELECT 1", raw_rows=[(100.0,)])
    num_res = NumericalValidationResult(
        layer_name="Layer 2: Numerical",
        passed=True,
        status=ValidationStatus.VERIFIED,
        message="All 2 grouped categories and metrics independently verified.",
        compared_rows=2,
        details={
            "expected_groups": 2,
            "actual_groups": 2,
            "matched_groups": 2,
            "compared_metrics": 2,
            "passed_comparisons": 2,
            "comparisons": [
                {
                    "group_key": ("East",),
                    "metric": "sales",
                    "duckdb_value": 100.0,
                    "independent_value": 100.0,
                    "absolute_difference": 0.0,
                    "relative_difference": 0.0,
                    "passed": True,
                }
            ],
        },
    )
    step_val = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=True, status=ValidationStatus.VERIFIED, message="ok"
        ),
        numerical_res=num_res,
        semantic_res=SemanticValidationResult(
            layer_name="sem", passed=False, status=ValidationStatus.UNVERIFIED, message="unv"
        ),
        raw_execution_result=exec_res,
    )
    report = WorkflowValidationReport(total_steps=1)
    report.add_step_attempt(step_val)

    summary = format_step_validation_summary(report, 1)
    assert summary is not None
    assert summary["numerical_status"] == "VERIFIED"
    assert summary["numerical_passed"] is True
    assert summary["numerical_details"]["expected_groups"] == 2
    assert len(summary["numerical_details"]["comparisons"]) == 1

    # Ensure clean serialization through serialize_validation_report_for_json
    serialized = serialize_validation_report_for_json(summary)
    json_str = json.dumps(serialized)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["numerical_details"]["expected_groups"] == 2


def test_format_step_validation_summary_extracts_semantic_results():
    """Verify format_step_validation_summary extracts Layer 3 semantic details from report."""
    exec_res = ExecutionResult(success=True, query="SELECT 1", raw_rows=[(100.0,)])
    num_res = NumericalValidationResult(
        layer_name="Layer 2: Numerical",
        passed=True,
        status=ValidationStatus.VERIFIED,
        message="Numerical pass",
    )
    checks = tuple(
        SemanticCheckDetail(
            check_name=name,
            status=ValidationStatus.VERIFIED,
            passed=True,
            diagnostic_code=None,
            explanation=f"{name} verified successfully",
            intent_evidence={"test_key": "intent_val"},
            ast_evidence={"test_key": "ast_val"},
        )
        for name in CANONICAL_SEMANTIC_CHECKS
    )
    sem_res = SemanticValidationResult(
        layer_name="Layer 3: Semantic",
        passed=True,
        status=ValidationStatus.VERIFIED,
        message="Semantic verified",
        checks=checks,
    )
    step_val = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=True, status=ValidationStatus.VERIFIED, message="Execution verified"
        ),
        numerical_res=num_res,
        semantic_res=sem_res,
        raw_execution_result=exec_res,
    )
    report = WorkflowValidationReport(total_steps=1)
    report.add_step_attempt(step_val)

    summary = format_step_validation_summary(report, 1)
    assert summary is not None
    # Verify Layer 1 & 2 unchanged
    assert summary["execution_status"] == "VERIFIED"
    assert summary["numerical_status"] == "VERIFIED"
    # Verify Layer 3 captured
    assert summary["semantic_status"] == "VERIFIED"
    assert summary["semantic_passed"] is True
    assert len(summary["semantic_checks"]) == 8
    first_check = summary["semantic_checks"][0]
    assert first_check["check_name"] in CANONICAL_SEMANTIC_CHECKS
    assert first_check["status"] == "VERIFIED"
    assert first_check["intent_evidence"] == {"test_key": "intent_val"}
    assert first_check["ast_evidence"] == {"test_key": "ast_val"}


def test_format_semantic_validation_display_all_four_statuses():
    """Verify format_semantic_validation_display renders all four status states."""
    # 1. VERIFIED state
    info_verified = {
        "semantic_status": "VERIFIED",
        "semantic_message": "All 8 verified",
        "semantic_checks": [
            {
                "check_name": "check_schema_grounding",
                "status": "VERIFIED",
                "diagnostic_code": "None",
                "explanation": "Grounding ok",
                "intent_evidence": {"t": "superstore"},
                "ast_evidence": {"tables": ["superstore"]},
            }
        ],
    }
    disp_v = format_semantic_validation_display(info_verified)
    assert disp_v["status"] == "VERIFIED"
    assert disp_v["passed_count"] == 1
    assert disp_v["checks"][0]["Status"] == "VERIFIED"
    assert disp_v["checks"][0]["Diagnostic Code"] == "None"

    # 2. FAILED state
    info_failed = {
        "semantic_status": "FAILED",
        "semantic_message": "Contradiction found",
        "semantic_checks": [
            {
                "check_name": "check_ordering_and_ranking",
                "status": "FAILED",
                "diagnostic_code": "ordering_direction_mismatch",
                "explanation": "Expected DESC observed ASC",
            }
        ],
    }
    disp_f = format_semantic_validation_display(info_failed)
    assert disp_f["status"] == "FAILED"
    assert disp_f["passed_count"] == 0
    assert disp_f["checks"][0]["Status"] == "FAILED"
    assert disp_f["checks"][0]["Diagnostic Code"] == "ordering_direction_mismatch"

    # 3. INCONCLUSIVE state
    info_inconclusive = {
        "semantic_status": "INCONCLUSIVE",
        "semantic_inconclusive_reason": "Ambiguous integer representation",
        "semantic_checks": [
            {
                "check_name": "check_data_type_and_precision",
                "status": "INCONCLUSIVE",
                "diagnostic_code": "ambiguous_integer_representation",
                "explanation": "Unspecified rounding policy",
            }
        ],
    }
    disp_i = format_semantic_validation_display(info_inconclusive)
    assert disp_i["status"] == "INCONCLUSIVE"
    assert disp_i["inconclusive_reason"] == "Ambiguous integer representation"
    assert disp_i["checks"][0]["Status"] == "INCONCLUSIVE"

    # 4. UNVERIFIED state
    info_unverified = {
        "semantic_status": "UNVERIFIED",
        "semantic_message": "Deferred to Stage 4",
        "semantic_checks": [],
    }
    disp_u = format_semantic_validation_display(info_unverified)
    assert disp_u["status"] == "UNVERIFIED"
    assert disp_u["total_count"] == 0

    # 5. None handling
    disp_none = format_semantic_validation_display(None)
    assert disp_none["status"] == "UNVERIFIED"
    assert disp_none["checks"] == []


def test_dashboard_graph_construction_with_layer_3_enabled():
    """Verify build_analytical_graph preserves graph default behavior while supporting app.py."""
    db = SessionDatabase()

    # 1. Default graph construction (omitted semantic_validator) leaves _sem_validator as None
    graph_default = build_analytical_graph(db=db)
    # The default graph compiled should exist and run without Layer 3
    assert graph_default is not None

    # 2. Explicit graph construction as in app.py (passes SemanticValidator)
    graph_with_l3 = build_analytical_graph(db=db, semantic_validator=SemanticValidator())
    assert graph_with_l3 is not None
    db.close()


def test_get_status_badge_html_all_variants():
    """Verify get_status_badge_html returns clean, restrained HTML badges without emojis."""
    b_v = get_status_badge_html("VERIFIED")
    assert "status-verified" in b_v
    assert "VERIFIED" in b_v
    assert "<span" in b_v

    b_f = get_status_badge_html("FAILED")
    assert "status-failed" in b_f

    b_i = get_status_badge_html("INCONCLUSIVE")
    assert "status-inconclusive" in b_i

    b_r = get_status_badge_html("RUNNING", "PROCESSING")
    assert "status-running" in b_r
    assert "PROCESSING" in b_r

    b_u = get_status_badge_html("UNVERIFIED")
    assert "status-unverified" in b_u

    b_none = get_status_badge_html(None)
    assert "status-unverified" in b_none
    assert "UNKNOWN" in b_none


def test_format_data_quality_metrics_calculation(data_dir):
    """Verify format_data_quality_metrics computes completeness and ambiguous columns."""
    db = SessionDatabase()
    db.register_csv(data_dir / "sample_superstore.csv", "superstore")
    profile = profile_table(db, "superstore")

    metrics = format_data_quality_metrics(profile)
    assert metrics["total_cells"] > 0
    assert metrics["completeness_pct"] >= 0.0
    assert metrics["completeness_pct"] <= 100.0
    assert isinstance(metrics["ambiguous_date_columns"], list)
    assert metrics["numeric_columns_count"] >= 1
    assert metrics["text_columns_count"] >= 1

    # Empty profile fallback
    empty_metrics = format_data_quality_metrics(None)
    assert empty_metrics["total_cells"] == 0
    assert empty_metrics["completeness_pct"] == 100.0
    db.close()


def test_format_correction_history_display_multi_attempt():
    """Verify format_correction_history_display extracts full attempt chronology."""
    raw1 = ExecutionResult(success=False, query="SELECT bad", error_message="Column not found")
    step_val1 = build_step_validation_result(
        step_id=1,
        attempt_number=1,
        execution_res=ExecutionValidationResult(
            passed=False,
            status=ValidationStatus.FAILED,
            message="Column not found",
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
        diagnostic=CorrectionDiagnostic(
            error_category=ErrorCategory.SYNTAX_ERROR,
            is_correctable=True,
            failed_layer="Layer 1: Execution",
            error_summary="Column not found",
            remediation_instruction="Fix column name",
            prior_attempt_sql="SELECT bad",
            attempt_number=1,
        ),
        executed_sql="SELECT bad",
        raw_execution_result=raw1,
    )

    raw2 = ExecutionResult(
        success=True, query="SELECT 100 AS sales", is_scalar=True, scalar_value=100.0
    )
    step_val2 = build_step_validation_result(
        step_id=1,
        attempt_number=2,
        execution_res=ExecutionValidationResult(
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="Execution verified",
        ),
        numerical_res=NumericalValidationResult(
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="Numerical verified",
        ),
        semantic_res=SemanticValidationResult(
            passed=True,
            status=ValidationStatus.VERIFIED,
            message="Semantic verified",
        ),
        executed_sql="SELECT 100 AS sales",
        raw_execution_result=raw2,
    )

    report = WorkflowValidationReport(total_steps=1)
    report.add_step_attempt(step_val1)
    report.add_step_attempt(step_val2)

    records = format_correction_history_display(report, step_id=1)
    assert len(records) == 2

    # Attempt 1
    assert records[0]["attempt_number"] == 1
    assert records[0]["is_latest"] is False
    assert records[0]["execution_success"] is False
    assert records[0]["error_category"] == "SYNTAX_ERROR"
    assert records[0]["is_correctable"] is True
    assert "queued for correction" in records[0]["decision_note"]

    # Attempt 2
    assert records[1]["attempt_number"] == 2
    assert records[1]["is_latest"] is True
    assert records[1]["execution_success"] is True
    assert records[1]["step_status"] == "VERIFIED"
    assert "Successfully corrected" in records[1]["decision_note"]

    # None report handling
    assert format_correction_history_display(None) == []


def test_get_workflow_nodes_summary_generation():
    """Verify get_workflow_nodes_summary produces info for all 8 compiled LangGraph nodes."""
    raw_res = ExecutionResult(success=True, query="SELECT 1", execution_time_ms=12.5)
    assembled_real = AssembledResult(
        original_question="SELECT 1",
        is_multi_step=False,
        all_steps_succeeded=True,
        step_results={1: raw_res},
        primary_scalar=1,
    )
    final_state = {
        "status": "validated",
        "retry_count": 1,
        "plan": type(
            "MockPlan", (), {"steps": [1], "is_multi_step": False, "execution_order": [1]}
        )(),
        "step_sql": {1: "SELECT 1"},
        "step_results": {1: raw_res},
        "assembled_result": assembled_real,
        "validation_report": WorkflowValidationReport(workflow_status=ValidationStatus.VERIFIED),
    }

    nodes = get_workflow_nodes_summary(final_state)
    assert len(nodes) == 8
    node_names = [n["name"] for n in nodes]
    assert "Planner" in node_names
    assert "Code Generator" in node_names
    assert "Controlled Executor" in node_names
    assert "Error Analyzer" in node_names
    assert "Corrector" in node_names
    assert "Step Advance" in node_names
    assert "Result Assembler" in node_names
    assert "Multi-Layer Validator" in node_names

    # Check executor summary captured latency
    exec_node = next(n for n in nodes if n["name"] == "Controlled Executor")
    assert "12.50 ms" in exec_node["details"]

    # Check assembled result summary
    asm_node = next(n for n in nodes if n["name"] == "Result Assembler")
    assert asm_node["status"] == "COMPLETED"
    assert "Consolidated 1 step output(s)" in asm_node["details"]

    # None state handling
    assert get_workflow_nodes_summary(None) == []


def test_prepare_chart_data_selection():
    """Verify prepare_chart_data selects bar, line, or table_only analytically."""
    # 1. Categorical + Numeric -> bar
    df_cat = pd.DataFrame({"Region": ["East", "West"], "Sales": [100.0, 200.0]})
    ctype, xcol, ycol = prepare_chart_data(df_cat)
    assert ctype == "bar"
    assert xcol == "Region"
    assert ycol == "Sales"

    # 2. Temporal date + Numeric -> line
    df_date = pd.DataFrame(
        {"Order_Date": pd.to_datetime(["2017-01-01", "2017-02-01"]), "Sales": [100.0, 150.0]}
    )
    ctype_d, xcol_d, ycol_d = prepare_chart_data(df_date)
    assert ctype_d == "line"
    assert xcol_d == "Order_Date"
    assert ycol_d == "Sales"

    # 3. Non-numeric columns -> table_only
    df_str = pd.DataFrame({"Col1": ["A", "B"], "Col2": ["C", "D"]})
    ctype_s, _, _ = prepare_chart_data(df_str)
    assert ctype_s == "table_only"

    # 4. Guard against generic step_id by description tables -> table_only
    df_generic = pd.DataFrame(
        {"step_id": [1, 2], "description": ["2016", "2017"], "value": [10.0, 20.0]}
    )
    ctype_g, _, _ = prepare_chart_data(df_generic)
    assert ctype_g == "table_only"

    # 5. Empty or None -> table_only
    assert prepare_chart_data(None) == ("table_only", None, None)
    assert prepare_chart_data(pd.DataFrame()) == ("table_only", None, None)


# ---------------------------------------------------------------------------
# Stage 6 Regression Tests for Robust Workflow Trace and UI Resilience
# ---------------------------------------------------------------------------
def test_regression_scalar_query_trace():
    """Verify workflow trace handles scalar query with real AssembledResult."""
    res = ExecutionResult(
        success=True,
        query="SELECT SUM(sales) FROM t",
        row_count=1,
        column_count=1,
        is_scalar=True,
        scalar_value=12345.67,
        execution_time_ms=8.2,
    )
    asm = AssembledResult(
        original_question="What is total sales?",
        is_multi_step=False,
        all_steps_succeeded=True,
        step_results={1: res},
        primary_scalar=12345.67,
    )
    state = {
        "status": "validated",
        "plan": type("Plan", (), {"steps": [1], "is_multi_step": False})(),
        "step_sql": {1: "SELECT SUM(sales) FROM t"},
        "step_results": {1: res},
        "assembled_result": asm,
        "validation_report": WorkflowValidationReport(workflow_status=ValidationStatus.VERIFIED),
        "retry_count": 0,
    }
    nodes = get_workflow_nodes_summary(state)
    assert len(nodes) == 8
    asm_node = next(n for n in nodes if n["name"] == "Result Assembler")
    assert asm_node["status"] == "COMPLETED"
    assert "Consolidated 1 step output(s)" in asm_node["details"]
    err_node = next(n for n in nodes if n["name"] == "Error Analyzer")
    assert err_node["status"] == "SKIPPED"
    corr_node = next(n for n in nodes if n["name"] == "Corrector")
    assert corr_node["status"] == "SKIPPED"


def test_regression_grouped_and_top_n_query_trace():
    """Verify workflow trace handles grouped and top-n query states."""
    df_grouped = pd.DataFrame({"region": ["East", "West"], "total_sales": [1000.0, 2000.0]})
    res = ExecutionResult(
        success=True,
        query="SELECT region, SUM(sales) FROM t GROUP BY region",
        row_count=2,
        column_count=2,
        is_scalar=False,
        dataframe=df_grouped,
        execution_time_ms=14.1,
    )
    asm = AssembledResult(
        original_question="What are sales by region?",
        is_multi_step=False,
        all_steps_succeeded=True,
        step_results={1: res},
        primary_dataframe=df_grouped,
    )
    state = {
        "status": "validated",
        "plan": type("Plan", (), {"steps": [1], "is_multi_step": False})(),
        "step_sql": {1: "SELECT region, SUM(sales) FROM t GROUP BY region"},
        "step_results": {1: res},
        "assembled_result": asm,
        "validation_report": WorkflowValidationReport(workflow_status=ValidationStatus.VERIFIED),
        "retry_count": 0,
    }
    nodes = get_workflow_nodes_summary(state)
    assert len(nodes) == 8
    exec_node = next(n for n in nodes if n["name"] == "Controlled Executor")
    assert "14.10 ms" in exec_node["details"]
    assert exec_node["status"] == "COMPLETED"


def test_regression_multi_step_query_trace():
    """Verify workflow trace handles multi-step DAG query states with multiple outputs."""
    res1 = ExecutionResult(
        success=True, query="SELECT 1", row_count=1, column_count=1, execution_time_ms=5.0
    )
    res2 = ExecutionResult(
        success=True, query="SELECT 2", row_count=1, column_count=1, execution_time_ms=7.0
    )
    asm = AssembledResult(
        original_question="Compare step 1 and step 2",
        is_multi_step=True,
        all_steps_succeeded=True,
        step_results={1: res1, 2: res2},
        summary_metrics={"step_1_winner": "A", "step_2_winner": "B"},
    )
    state = {
        "status": "validated",
        "plan": type("Plan", (), {"steps": [1, 2], "is_multi_step": True})(),
        "step_sql": {1: "SELECT 1", 2: "SELECT 2"},
        "step_results": {1: res1, 2: res2},
        "assembled_result": asm,
        "validation_report": WorkflowValidationReport(workflow_status=ValidationStatus.VERIFIED),
        "retry_count": 0,
    }
    nodes = get_workflow_nodes_summary(state)
    assert len(nodes) == 8
    step_adv_node = next(n for n in nodes if n["name"] == "Step Advance")
    assert "Iterated across 2 topological DAG steps" in step_adv_node["details"]
    asm_node = next(n for n in nodes if n["name"] == "Result Assembler")
    assert "Consolidated 2 step output(s)" in asm_node["details"]


def test_regression_execution_failure_trace():
    """Verify workflow trace handles query execution failure gracefully."""
    res_fail = ExecutionResult(
        success=False,
        query="SELECT * FROM missing_table",
        error_type="TableNotFound",
        error_message="Table 'missing_table' does not exist",
        execution_time_ms=2.0,
    )
    state = {
        "status": "execution_failed",
        "plan": type("Plan", (), {"steps": [1], "is_multi_step": False})(),
        "step_sql": {1: "SELECT * FROM missing_table"},
        "step_results": {1: res_fail},
        "assembled_result": None,
        "validation_report": None,
        "retry_count": 2,
    }
    nodes = get_workflow_nodes_summary(state)
    assert len(nodes) == 8
    exec_node = next(n for n in nodes if n["name"] == "Controlled Executor")
    assert exec_node["status"] == "FAILED"
    asm_node = next(n for n in nodes if n["name"] == "Result Assembler")
    assert asm_node["status"] == "SKIPPED"
    assert "Result assembly not reached" in asm_node["details"]
    err_node = next(n for n in nodes if n["name"] == "Error Analyzer")
    assert err_node["status"] == "TRIGGERED"


def test_regression_inconclusive_validation_trace():
    """Verify workflow trace marks corrector as INELIGIBLE for inconclusive validation."""
    res = ExecutionResult(success=True, query="SELECT 1", execution_time_ms=6.0)
    asm = AssembledResult(
        original_question="Ambiguous query",
        is_multi_step=False,
        all_steps_succeeded=True,
        step_results={1: res},
    )
    state = {
        "status": "inconclusive",
        "plan": type("Plan", (), {"steps": [1], "is_multi_step": False})(),
        "step_sql": {1: "SELECT 1"},
        "step_results": {1: res},
        "assembled_result": asm,
        "validation_report": WorkflowValidationReport(
            workflow_status=ValidationStatus.INCONCLUSIVE
        ),
        "retry_count": 0,
    }
    nodes = get_workflow_nodes_summary(state)
    assert len(nodes) == 8
    err_node = next(n for n in nodes if n["name"] == "Error Analyzer")
    assert err_node["status"] == "INELIGIBLE"
    assert "inconclusive" in err_node["details"]


def test_regression_empty_result_trace():
    """Verify workflow trace handles empty dataframe output without crashing."""
    empty_df = pd.DataFrame()
    res = ExecutionResult(
        success=True,
        query="SELECT * FROM t WHERE 1=0",
        row_count=0,
        column_count=0,
        is_scalar=False,
        dataframe=empty_df,
        execution_time_ms=3.5,
    )
    asm = AssembledResult(
        original_question="No matches",
        is_multi_step=False,
        all_steps_succeeded=True,
        step_results={1: res},
        primary_dataframe=empty_df,
    )
    state = {
        "status": "validated",
        "plan": type("Plan", (), {"steps": [1], "is_multi_step": False})(),
        "step_sql": {1: "SELECT * FROM t WHERE 1=0"},
        "step_results": {1: res},
        "assembled_result": asm,
        "validation_report": WorkflowValidationReport(workflow_status=ValidationStatus.VERIFIED),
        "retry_count": 0,
    }
    nodes = get_workflow_nodes_summary(state)
    assert len(nodes) == 8
    asm_node = next(n for n in nodes if n["name"] == "Result Assembler")
    assert asm_node["status"] == "COMPLETED"


def test_regression_missing_optional_metadata():
    """Verify get_workflow_nodes_summary never crashes with empty or corrupted state."""
    # Completely empty state returns empty list
    nodes1 = get_workflow_nodes_summary({})
    assert nodes1 == []

    # Partial state missing all sub-objects
    nodes2 = get_workflow_nodes_summary({"status": "executing"})
    assert len(nodes2) == 8

    # State with None objects
    nodes3 = get_workflow_nodes_summary(
        {
            "plan": None,
            "step_results": None,
            "step_sql": None,
            "assembled_result": None,
            "validation_report": None,
        }
    )
    assert len(nodes3) == 8

    # State with arbitrary unknown object for assembled_result
    nodes4 = get_workflow_nodes_summary(
        {
            "assembled_result": "not_an_assembled_result",
            "step_results": "not_a_dict",
        }
    )
    assert len(nodes4) == 8


def test_regression_self_correction_multiple_attempts():
    """Verify trace handles self-correction with multiple retry attempts."""
    res = ExecutionResult(success=True, query="SELECT 1", execution_time_ms=10.0)
    asm = AssembledResult(
        original_question="Corrected query",
        is_multi_step=False,
        all_steps_succeeded=True,
        step_results={1: res},
    )
    state = {
        "status": "validated",
        "plan": type("Plan", (), {"steps": [1], "is_multi_step": False})(),
        "step_sql": {1: "SELECT 1"},
        "step_results": {1: res},
        "assembled_result": asm,
        "validation_report": WorkflowValidationReport(
            workflow_status=ValidationStatus.VERIFIED, total_retries=2
        ),
        "retry_count": 2,
    }
    nodes = get_workflow_nodes_summary(state)
    assert len(nodes) == 8
    err_node = next(n for n in nodes if n["name"] == "Error Analyzer")
    assert err_node["status"] == "TRIGGERED"
    corr_node = next(n for n in nodes if n["name"] == "Corrector")
    assert corr_node["status"] == "REMEDIATED"
    assert "Synthesized 2 corrected query version(s)" in corr_node["details"]


def test_prepare_comparison_chart_df_annual_sales():
    """Verify prepare_comparison_chart_df extracts clean Year vs Total Sales chart."""
    res1 = ExecutionResult(
        success=True,
        query="SELECT SUM(sales) FROM t WHERE year=2016",
        is_scalar=True,
        scalar_value=150000.0,
    )
    res2 = ExecutionResult(
        success=True,
        query="SELECT SUM(sales) FROM t WHERE year=2017",
        is_scalar=True,
        scalar_value=220000.0,
    )
    step1 = type("MockStep", (), {"step_id": 1, "question": "Total sales for 2016"})()
    step2 = type("MockStep", (), {"step_id": 2, "question": "Total sales for 2017"})()
    plan = type(
        "MockPlan",
        (),
        {
            "steps": [step1, step2],
            "execution_order": [1, 2],
            "get_step": lambda self, sid: step1 if sid == 1 else step2,
            "original_question": "Compare 2016 and 2017 total sales",
        },
    )()
    asm = AssembledResult(
        original_question="Compare 2016 and 2017 total sales",
        is_multi_step=True,
        all_steps_succeeded=True,
        step_results={1: res1, 2: res2},
    )

    chart_df, x_col, y_col = prepare_comparison_chart_df(asm, plan)
    assert chart_df is not None
    assert x_col == "Year"
    assert y_col == "Total Sales"
    assert list(chart_df.index) == ["2016", "2017"]
    assert chart_df["Total Sales"].iloc[0] == 150000.0
    assert chart_df["Total Sales"].iloc[1] == 220000.0

    # Non-comparative (single step) returns (None, None, None)
    single_asm = AssembledResult(
        original_question="Total sales for 2016",
        is_multi_step=False,
        all_steps_succeeded=True,
        step_results={1: res1},
    )
    c_df, _, _ = prepare_comparison_chart_df(single_asm, plan)
    assert c_df is None


def test_get_workflow_execution_path_tracing():
    """Verify get_workflow_execution_path reflects actual compiled LangGraph transitions."""
    # 1. Single-step direct success
    state_single = {
        "status": "validated",
        "retry_count": 0,
        "plan": type("MockPlan", (), {"is_multi_step": False, "execution_order": [1]})(),
    }
    path_single = get_workflow_execution_path(state_single)
    assert path_single == ["planner", "codegen", "executor", "assembler", "validator"]

    # 2. Multi-step with topological iteration
    state_multi = {
        "status": "validated",
        "retry_count": 0,
        "plan": type("MockPlan", (), {"is_multi_step": True, "execution_order": [1, 2]})(),
    }
    path_multi = get_workflow_execution_path(state_multi)
    assert "step_advance" in path_multi
    assert path_multi == [
        "planner",
        "codegen",
        "executor",
        "step_advance",
        "codegen",
        "executor",
        "assembler",
        "validator",
    ]

    # 3. Corrected query step
    state_corrected = {
        "status": "validated",
        "retry_count": 1,
        "plan": type("MockPlan", (), {"is_multi_step": False, "execution_order": [1]})(),
    }
    path_corr = get_workflow_execution_path(state_corrected)
    assert "error_analyzer" in path_corr
    assert "corrector" in path_corr
    assert path_corr == [
        "planner",
        "codegen",
        "executor",
        "error_analyzer",
        "corrector",
        "executor",
        "assembler",
        "validator",
    ]

    # 4. None state handling
    assert get_workflow_execution_path(None) == []
