"""Unit tests for dataset ingestion and session-level table security.

Validates:
    - Identifier sanitization (snake_case conversion, numeric prefixes, collision avoidance).
    - Dynamic Pandas DataFrame, CSV, and Excel table registration into DuckDB.
    - Security allowlist enforcement rejecting unauthorized table references.
    - SQLGlot AST inspection properly allowing Common Table Expressions (CTEs).
    - Hard rejection of dangerous external filesystem reading functions (e.g. read_csv).
"""

import pandas as pd
import pytest

from src.agent.data.ingestion import (
    SessionDatabase,
    read_csv_with_fallback,
    sanitize_identifier,
)


def test_sanitize_identifier():
    """Verify identifier normalization across spaces, symbols, and leading digits."""
    # Standard multi-word name
    assert sanitize_identifier("Order ID") == "order_id"
    # Hyphenated names
    assert sanitize_identifier("Sub-Category") == "sub_category"
    # Whitespace padding
    assert sanitize_identifier("  Customer Name  ") == "customer_name"
    # Leading digit must be prefixed to maintain valid SQL identifier syntax
    assert sanitize_identifier("123_number") == "col_123_number"
    # Currency symbols and parentheses
    assert sanitize_identifier("Sales ($USD)") == "sales_usd"
    # Leading and trailing underscores collapsed
    assert sanitize_identifier("___Special___") == "special"
    # Accented/umlaut characters normalized to ASCII
    assert sanitize_identifier("Björn's Orders") == "bjorn_s_orders"
    assert sanitize_identifier("Verkäufe_2024") == "verkaufe_2024"


def test_read_csv_with_fallback(tmp_path):
    """Verify reading CSV files encoded in UTF-8, UTF-8-SIG, CP1252, and Latin-1."""
    import io

    # 1. UTF-8
    utf8_b = io.BytesIO("col1,col2\nval1,val2\n".encode("utf-8"))
    df1 = read_csv_with_fallback(utf8_b)
    assert df1["col1"].iloc[0] == "val1"

    # 2. UTF-8-SIG
    sig_b = io.BytesIO("\ufeffname,score\nAtul,98\n".encode("utf-8-sig"))
    df2 = read_csv_with_fallback(sig_b)
    assert "name" in df2.columns
    assert df2["name"].iloc[0] == "Atul"

    # 3. CP1252 with euro symbol and umlaut
    cp_b = io.BytesIO("item,price\nKäse,5€\n".encode("cp1252"))
    df3 = read_csv_with_fallback(cp_b)
    assert df3["item"].iloc[0] == "Käse"
    assert df3["price"].iloc[0] == "5€"

    # 4. Latin-1 with 0xf6
    latin_b = io.BytesIO(b"city,temp\nM\xfcnchen,18\n")
    df4 = read_csv_with_fallback(latin_b)
    assert df4["city"].iloc[0] == "München"


def test_session_database_dataframe_registration():
    """Verify DataFrame registration with automatic column sanitization and collision handling."""
    db = SessionDatabase()
    df = pd.DataFrame({
        "Customer Name": ["Alice", "Bob"],
        "Order-Total": [100.5, 200.0],
    })
    table_name = db.register_dataframe(df, "Customer Orders")
    assert table_name == "customer_orders"
    assert db.is_table_allowed("customer_orders")
    assert "customer_orders" in db.get_allowed_tables()

    # Verify column sanitization mapping preserves original header reference
    mapping = db.get_column_mapping("customer_orders")
    assert mapping["Customer Name"] == "customer_name"
    assert mapping["Order-Total"] == "order_total"

    # Verify DuckDB executes query against the materialized table
    res_df = db.query_df("SELECT customer_name, order_total FROM customer_orders")
    assert len(res_df) == 2
    assert list(res_df["customer_name"]) == ["Alice", "Bob"]
    db.close()


def test_session_database_csv_registration(data_dir):
    """Verify loading and registering a real CSV file into DuckDB."""
    db = SessionDatabase()
    csv_path = data_dir / "sample_superstore.csv"
    table_name = db.register_csv(csv_path, table_name="superstore")
    assert table_name == "superstore"
    assert db.is_table_allowed("superstore")

    # Confirm row count matches the 18 rows in sample_superstore.csv
    res = db.query_df("SELECT COUNT(*) AS cnt FROM superstore")
    assert int(res["cnt"].iloc[0]) == 18
    db.close()


def test_security_allowlist_enforcement():
    """Verify that queries referencing unregistered tables are blocked prior to execution."""
    db = SessionDatabase()
    df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
    db.register_dataframe(df, "allowed_tbl")

    # Positive test: Allowed table query passes
    is_valid, errors = db.validate_query_security("SELECT * FROM allowed_tbl")
    assert is_valid
    assert len(errors) == 0

    # Negative test: Unregistered table query is flagged
    is_valid, errors = db.validate_query_security("SELECT * FROM secret_tbl")
    assert not is_valid
    assert any("secret_tbl" in err for err in errors)

    # Negative test: Direct query execution raises PermissionError
    with pytest.raises(PermissionError, match="Query validation rejected"):
        db.query_df("SELECT * FROM secret_tbl")
    db.close()


def test_security_cte_handling():
    """Verify that Common Table Expression (CTE) aliases are recognized as valid local scopes."""
    db = SessionDatabase()
    df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
    db.register_dataframe(df, "allowed_tbl")

    # Query uses CTE 'summary' based on registered 'allowed_tbl' -> Must be valid
    cte_sql = """
        WITH summary AS (
            SELECT id, val * 2 AS doubled FROM allowed_tbl
        )
        SELECT * FROM summary WHERE doubled > 20
    """
    is_valid, errors = db.validate_query_security(cte_sql)
    assert is_valid, f"CTE query failed validation: {errors}"

    # Query uses CTE 'summary' based on UNREGISTERED 'unregistered_tbl' -> Must fail
    bad_cte_sql = """
        WITH summary AS (
            SELECT id FROM unregistered_tbl
        )
        SELECT * FROM summary
    """
    is_valid, errors = db.validate_query_security(bad_cte_sql)
    assert not is_valid
    assert any("unregistered_tbl" in err for err in errors)
    db.close()


def test_security_forbidden_functions():
    """Verify that external file reader functions are detected and rejected by AST analysis."""
    db = SessionDatabase()
    for forbidden in ["read_csv", "read_parquet", "parquet_scan"]:
        sql = f"SELECT * FROM {forbidden}('data/file.csv')"
        is_valid, errors = db.validate_query_security(sql)
        assert not is_valid
        assert any("Forbidden external data function" in err for err in errors)
    db.close()
