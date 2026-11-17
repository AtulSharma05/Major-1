"""Regression tests for date/time query handling across external datasets.

Validates:
    1. Robust date format identification across:
       - Native DATE
       - Native TIMESTAMP
       - VARCHAR ISO dates ('YYYY-MM-DD')
       - VARCHAR DD/MM/YYYY dates ('DD/MM/YYYY')
       - Ambiguous VARCHAR dates ('01/02/2020') where DD and MM are both <= 12
    2. Preservation of TimeConstraint in QuerySpecification and prompt serialization.
    3. Safe SQL query execution on native DATE columns with nulls and inside/outside records.
    4. Safe SQL execution on TIMESTAMP columns with day/year boundary precision
       (guaranteeing transactions at 23:59:59 are preserved by half-open intervals).
    5. Safe SQL execution on VARCHAR DD/MM/YYYY columns with `try_strptime` and ISO literals,
       preventing ASCII comparison failure.
    6. Explicit disclosure and warning on ambiguous date formats to prevent silent corruption.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.agent.codegen.prompts import build_sql_prompt
from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import (
    detect_date_format_and_pattern,
    format_profile_for_prompt,
    profile_table,
)
from src.agent.execution.engine import ControlledExecutionEngine
from src.agent.planning.intent import (
    AggregationType,
    MetricSpec,
    QuerySpecification,
    TimeConstraint,
)


def test_detect_date_format_and_pattern_variations():
    """Verify format pattern recognition on diverse string date samples."""
    # DD/MM/YYYY with day > 12
    is_t, fmt, strp = detect_date_format_and_pattern(["15/06/2017", "28/11/2016", "04/05/2017"])
    assert is_t is True
    assert fmt == "DD/MM/YYYY"
    assert strp == "%d/%m/%Y"

    # MM/DD/YYYY with month <= 12 and day > 12
    is_t, fmt, strp = detect_date_format_and_pattern(["06/15/2017", "11/28/2016", "05/04/2017"])
    assert is_t is True
    assert fmt == "MM/DD/YYYY"
    assert strp == "%m/%d/%Y"

    # ISO dates ('YYYY-MM-DD')
    is_t, fmt, strp = detect_date_format_and_pattern(["2017-06-15", "2016-11-28"])
    assert is_t is True
    assert fmt == "ISO"
    assert strp == "%Y-%m-%d"

    # Ambiguous dates (all days and months <= 12)
    is_t, fmt, strp = detect_date_format_and_pattern(["01/02/2017", "04/05/2018", "06/07/2019"])
    assert is_t is True
    assert "AMBIGUOUS" in fmt
    assert strp is None

    # Non-temporal categorical strings
    is_t, fmt, strp = detect_date_format_and_pattern(["Furniture", "Office Supplies", "Technology"])
    assert is_t is False
    assert fmt is None
    assert strp is None


def test_date_column_profiling_and_identification():
    """Verify profiling identifies column types and formats across an external dataset."""
    db = SessionDatabase()
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "date_iso": ["2017-01-15", "2017-06-20", "2018-02-10", None],
        "date_dmy": ["15/01/2017", "28/06/2017", "10/02/2018", None],
        "date_ambig": ["01/02/2017", "03/04/2017", "05/06/2018", None],
        "category": ["A", "B", "C", "D"],
        "sales": [100.0, 200.0, 300.0, 400.0],
    })
    db.register_dataframe(df, "ext_sales")
    profile = profile_table(db, "ext_sales")

    # ISO date column
    iso_col = profile.columns["date_iso"]
    assert iso_col.is_temporal is True
    assert iso_col.date_format == "ISO"
    assert "2017-01-15" in str(iso_col.min_val)
    assert "2018-02-10" in str(iso_col.max_val)

    # DD/MM/YYYY date column
    dmy_col = profile.columns["date_dmy"]
    assert dmy_col.is_temporal is True
    assert dmy_col.date_format == "DD/MM/YYYY"
    assert dmy_col.strptime_format == "%d/%m/%Y"
    # Min/max must reflect true chronological order (not string '10/02/2018' < '15/01/2017')
    assert "2017-01-15" in str(dmy_col.min_val)
    assert "2018-02-10" in str(dmy_col.max_val)

    # Ambiguous column
    ambig_col = profile.columns["date_ambig"]
    assert ambig_col.is_temporal is True
    assert "AMBIGUOUS" in ambig_col.date_format

    # Categorical non-date column
    cat_col = profile.columns["category"]
    assert cat_col.is_temporal is False

    db.close()


def test_native_date_column_query_execution():
    """Verify execution on genuine DuckDB DATE column with half-open intervals and nulls."""
    db = SessionDatabase()
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "event_date": [
            pd.Timestamp("2016-12-31").date(),
            pd.Timestamp("2017-01-01").date(),
            pd.Timestamp("2017-06-15").date(),
            pd.Timestamp("2017-12-31").date(),
            pd.Timestamp("2018-01-01").date(),
            None,
        ],
        "revenue": [50.0, 100.0, 250.0, 150.0, 300.0, 500.0],
    })
    db.register_dataframe(df, "sales_native_date")

    engine = ControlledExecutionEngine(db=db)

    # Query for calendar year 2017 using half-open interval
    sql = """
        SELECT SUM(revenue) AS total_sales, COUNT(*) AS row_count
        FROM sales_native_date
        WHERE event_date >= '2017-01-01' AND event_date < '2018-01-01'
    """
    res = engine.execute(sql)
    assert res.success is True
    assert res.dataframe is not None

    res_df = res.dataframe
    assert int(res_df["row_count"].iloc[0]) == 3
    # Rows 2 (100.0) + 3 (250.0) + 4 (150.0) = 500.0
    assert float(res_df["total_sales"].iloc[0]) == pytest.approx(500.0)

    db.close()


def test_timestamp_boundary_handling():
    """Verify half-open interval captures 23:59:59 timestamps while BETWEEN loses them."""
    db = SessionDatabase()
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "order_time": [
            pd.Timestamp("2017-01-01 00:00:00"),
            pd.Timestamp("2017-06-15 14:30:00"),
            pd.Timestamp("2017-12-31 23:59:59"),
            pd.Timestamp("2018-01-01 00:00:00"),
        ],
        "amount": [100.0, 200.0, 300.0, 400.0],
    })
    db.register_dataframe(df, "sales_timestamp")

    engine = ControlledExecutionEngine(db=db)

    # Standard half-open interval preserves the entire year including 23:59:59
    half_open_sql = """
        SELECT COUNT(*) AS cnt, SUM(amount) AS total
        FROM sales_timestamp
        WHERE order_time >= '2017-01-01' AND order_time < '2018-01-01'
    """
    res_half_open = engine.execute(half_open_sql)
    assert res_half_open.success is True
    df_half = res_half_open.dataframe
    assert int(df_half["cnt"].iloc[0]) == 3
    assert float(df_half["total"].iloc[0]) == pytest.approx(600.0)

    # BETWEEN '2017-01-01' AND '2017-12-31' fails to capture 23:59:59
    between_sql = """
        SELECT COUNT(*) AS cnt, SUM(amount) AS total
        FROM sales_timestamp
        WHERE order_time BETWEEN '2017-01-01' AND '2017-12-31'
    """
    res_between = engine.execute(between_sql)
    df_between = res_between.dataframe
    # Row 3 (23:59:59) is dropped by BETWEEN because '2017-12-31' casts to 00:00:00
    assert int(df_between["cnt"].iloc[0]) == 2

    db.close()


def test_varchar_dmy_dates_query_execution():
    """Verify VARCHAR DD/MM/YYYY columns work correctly with try_strptime and fail with ASCII."""
    db = SessionDatabase()
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "trans_date": ["15/06/2017", "28/11/2016", "05/01/2018", None],
        "profit": [250.0, 150.0, 300.0, 50.0],
    })
    db.register_dataframe(df, "sales_dmy")

    engine = ControlledExecutionEngine(db=db)

    # Correct type-aware query with try_strptime
    safe_sql = """
        SELECT SUM(profit) AS total_profit, COUNT(*) AS row_count
        FROM sales_dmy
        WHERE try_strptime(trans_date, '%d/%m/%Y') >= '2017-01-01'
          AND try_strptime(trans_date, '%d/%m/%Y') < '2018-01-01'
    """
    res = engine.execute(safe_sql)
    assert res.success is True
    df_safe = res.dataframe
    assert int(df_safe["row_count"].iloc[0]) == 1
    assert float(df_safe["total_profit"].iloc[0]) == pytest.approx(250.0)

    # Naive string comparison reproduces the exact bug: '15/06/2017' < '2017-01-01' in ASCII
    naive_sql = """
        SELECT COUNT(*) AS row_count
        FROM sales_dmy
        WHERE trans_date >= '2017-01-01' AND trans_date < '2018-01-01'
    """
    res_naive = engine.execute(naive_sql)
    # Returns 0 rows because ASCII comparison fails
    assert int(res_naive.dataframe["row_count"].iloc[0]) == 0

    db.close()

    db.close()


def test_ambiguous_date_format_surfacing():
    """Verify that ambiguous date formats are flagged and not silently assumed."""
    db = SessionDatabase()
    df = pd.DataFrame({
        "order_date": ["01/02/2017", "03/04/2017", "05/06/2017"],
        "sales": [10.0, 20.0, 30.0],
    })
    db.register_dataframe(df, "ambig_data")

    profile = profile_table(db, "ambig_data")
    date_col = profile.columns["order_date"]
    assert date_col.is_temporal is True
    assert "AMBIGUOUS" in date_col.date_format
    assert date_col.strptime_format is None

    # Check that format_profile_for_prompt explicitly outputs a warning
    prompt_schema = format_profile_for_prompt(profile)
    assert "WARNING: Date column(s)" in prompt_schema
    assert "order_date" in prompt_schema

    # Check that build_sql_prompt provides disclosure guidance
    prompt = build_sql_prompt("Total sales in 2017?", profile=profile)
    assert "ambiguous format" in prompt.lower()

    db.close()


def test_build_sql_prompt_preserves_time_constraints():
    """Verify that QuerySpecification time_constraints are serialized into the LLM prompt."""
    db = SessionDatabase()
    df = pd.DataFrame({
        "order_date": ["2017-01-01", "2017-06-01"],
        "sales": [100.0, 200.0],
    })
    db.register_dataframe(df, "orders")
    profile = profile_table(db, "orders")

    intent = QuerySpecification(
        raw_question="What were the total sales in 2017?",
        table_name="orders",
        metrics=[MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales")],
        time_constraints=[
            TimeConstraint(
                column="order_date",
                start_time="2017-01-01",
                end_time="2018-01-01",
                is_half_open=True,
            )
        ],
    )

    prompt = build_sql_prompt(
        question="What were the total sales in 2017?",
        profile=profile,
        intent=intent,
    )

    assert "### Structured Intent" in prompt
    assert "Time Constraints:" in prompt
    assert "order_date" in prompt
    assert "2017-01-01" in prompt
    assert "2018-01-01" in prompt

    db.close()
