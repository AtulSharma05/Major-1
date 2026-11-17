"""Unit tests for UI-independent ingestion and display helpers."""

import io
from pathlib import Path

import pandas as pd
import pytest

from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import profile_table
from src.ui.helpers import (
    get_excel_sheet_names,
    get_table_preview_df,
    ingest_uploaded_file,
    profile_to_display_df,
    resolve_table_name_collision,
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
