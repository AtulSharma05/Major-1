"""Unit tests for dataset profiling and schema serialization.

Validates:
    - Accurate calculation of column data types, row counts, and null percentages.
    - Extraction of representative distinct sample values for categorical variables.
    - Proper statistical aggregations (min, max, mean) for numeric/temporal columns.
    - Graceful handling of edge cases (all-null columns, empty tables, unregistered tables).
    - Compact, token-budgeted Markdown serialization for prompt injection.
"""

import pandas as pd
import pytest

from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import (
    format_profile_for_prompt,
    profile_table,
)


@pytest.fixture
def populated_db(data_dir):
    """Fixture providing a SessionDatabase pre-loaded with canonical benchmark datasets."""
    db = SessionDatabase()
    db.register_csv(data_dir / "sample_superstore.csv", "superstore")
    db.register_csv(data_dir / "sample_retail.csv", "retail")
    yield db
    db.close()


def test_profile_table_superstore(populated_db):
    """Verify statistical profiling accuracy on the Superstore dataset."""
    profile = profile_table(populated_db, "superstore")
    assert profile.table_name == "superstore"
    assert profile.row_count == 18
    assert profile.column_count >= 10

    # Test numeric column profiling (Sales)
    sales_col = profile.columns["sales"]
    assert "DOUBLE" in sales_col.duckdb_type or "FLOAT" in sales_col.duckdb_type
    assert sales_col.null_count == 0
    assert sales_col.null_percentage == 0.0
    assert sales_col.distinct_count > 10
    assert sales_col.min_val is not None
    assert sales_col.max_val is not None
    assert sales_col.mean_val is not None

    # Test categorical column profiling (Region)
    region_col = profile.columns["region"]
    assert "VARCHAR" in region_col.duckdb_type
    assert region_col.distinct_count == 4
    assert len(region_col.sample_values) > 0


def test_profile_table_retail_with_nulls(populated_db):
    """Verify that null values in realistic datasets are properly tracked."""
    profile = profile_table(populated_db, "retail")
    cust_col = profile.columns["customerid"]

    # In sample_retail.csv, line 9 contains a null CustomerID
    assert cust_col.null_count == 1
    assert cust_col.null_percentage > 0.0


def test_profile_table_all_nulls():
    """Verify column profiling when an entire column consists of NULL values."""
    db = SessionDatabase()
    df = pd.DataFrame({
        "all_nulls": [None, None, None],
        "category": ["A", "B", "C"],
    })
    db.register_dataframe(df, "edge_case")
    profile = profile_table(db, "edge_case")

    col_prof = profile.columns["all_nulls"]
    assert col_prof.null_count == 3
    assert col_prof.null_percentage == 100.0
    assert col_prof.distinct_count == 0
    assert col_prof.sample_values == []
    db.close()


def test_profile_unregistered_table():
    """Verify that attempting to profile an unverified/unregistered table raises ValueError."""
    db = SessionDatabase()
    with pytest.raises(ValueError, match="not registered in session allowlist"):
        profile_table(db, "ghost_table")
    db.close()


def test_format_profile_for_prompt(populated_db):
    """Verify Markdown prompt formatting and token-budget truncation."""
    profile = profile_table(populated_db, "superstore")

    # Positive test: Standard formatting produces Markdown schema table
    output = format_profile_for_prompt(profile, max_token_budget=1000)
    assert "### Table Schema: `superstore`" in output
    assert "| `order_id` |" in output
    assert "| `sales` |" in output
    assert "| Column | Type |" in output

    # Truncation test: Strict token budget truncates cleanly without breaking formatting
    truncated_output = format_profile_for_prompt(profile, max_token_budget=50)
    assert "truncated to fit token budget" in truncated_output
