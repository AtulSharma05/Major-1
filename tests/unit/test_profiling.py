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
    ColumnProfile,
    format_profile_for_prompt,
    profile_table,
)
from src.ui.helpers import profile_to_display_df


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
    df = pd.DataFrame(
        {
            "all_nulls": [None, None, None],
            "category": ["A", "B", "C"],
        }
    )
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


def test_column_profile_backward_compatibility_old_fields():
    """Verify ColumnProfile instantiation using only the old required fields."""
    cp = ColumnProfile(
        name="sales",
        duckdb_type="DOUBLE",
        total_rows=10,
        null_count=0,
        null_percentage=0.0,
        distinct_count=5,
    )
    assert hasattr(cp, "is_temporal")
    assert hasattr(cp, "date_format")
    assert hasattr(cp, "strptime_format")
    assert cp.is_temporal is False
    assert cp.date_format is None
    assert cp.strptime_format is None


def test_column_profile_with_temporal_fields():
    """Verify ColumnProfile instantiation with explicit temporal fields."""
    cp = ColumnProfile(
        name="order_date",
        duckdb_type="VARCHAR",
        total_rows=10,
        null_count=0,
        null_percentage=0.0,
        distinct_count=5,
        is_temporal=True,
        date_format="DD/MM/YYYY",
        strptime_format="%d/%m/%Y",
    )
    assert cp.is_temporal is True
    assert cp.date_format == "DD/MM/YYYY"
    assert cp.strptime_format == "%d/%m/%Y"


def test_profile_temporal_column_metadata():
    """Verify profiling a table with date columns correctly populates date metadata."""
    db = SessionDatabase()
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "dt": [pd.Timestamp("2017-01-01").date(), pd.Timestamp("2017-06-15").date(), None],
            "ts": [pd.Timestamp("2017-01-01 00:00:00"), pd.Timestamp("2017-06-15 12:00:00"), None],
            "dmy": ["15/01/2017", "28/06/2017", "10/02/2018"],
            "region": ["East", "West", "Central"],
        }
    )
    db.register_dataframe(df, "test_dates")
    profile = profile_table(db, "test_dates")

    # Native DATE
    dt_col = profile.columns["dt"]
    assert dt_col.is_temporal is True
    assert dt_col.date_format == "DATE"
    assert dt_col.strptime_format is None

    # Native TIMESTAMP
    ts_col = profile.columns["ts"]
    assert ts_col.is_temporal is True
    assert ts_col.date_format == "TIMESTAMP"
    assert ts_col.strptime_format is None

    # VARCHAR DD/MM/YYYY
    dmy_col = profile.columns["dmy"]
    assert dmy_col.is_temporal is True
    assert dmy_col.date_format == "DD/MM/YYYY"
    assert dmy_col.strptime_format == "%d/%m/%Y"

    db.close()


def test_profile_non_temporal_column_defaults(populated_db):
    """Verify profiling a non-temporal column preserves backward-compatible defaults."""
    profile = profile_table(populated_db, "superstore")
    region_col = profile.columns["region"]

    assert hasattr(region_col, "is_temporal")
    assert hasattr(region_col, "date_format")
    assert hasattr(region_col, "strptime_format")
    assert region_col.is_temporal is False
    assert region_col.date_format is None
    assert region_col.strptime_format is None


def test_ui_facing_profiling_accesses_date_format(populated_db):
    """Verify UI components can access date_format without AttributeError."""
    profile = profile_table(populated_db, "superstore")

    # Verify UI comprehension in src/ui/app.py:482 executes safely
    ambig_cols = [
        c.name for c in profile.columns.values() if c.date_format and "AMBIGUOUS" in c.date_format
    ]
    assert isinstance(ambig_cols, list)

    # Verify profile_to_display_df produces 'Date Format' column
    display_df = profile_to_display_df(profile)
    assert "Date Format" in display_df.columns
    assert len(display_df) == len(profile.columns)
