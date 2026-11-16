"""Data profiling and token-budgeted schema serialization for LLM prompts.

Architectural Context:
    For an autonomous LLM to generate precise, valid analytical SQL, it requires
    deep awareness of the dataset schema, column data types, null rates, and actual
    representative values. Providing only column names often causes LLMs to hallucinate
    non-existent categorical labels or misuse date formatting.

Key Responsibilities of this Module:
    1. Fast Statistical Profiling: Run targeted, vectorized DuckDB queries to compute
       row counts, null counts, null percentages, distinct value counts, min/max ranges,
       and distinct sample values without copying the entire dataset into Python memory.
    2. Strict Schema Pydantic Models: Structure the profiling output into immutable,
       type-safe `ColumnProfile` and `TableProfile` objects.
    3. Token-Budgeted Serialization: Convert rich schema metadata into a concise,
       Markdown-formatted prompt representation that guarantees context window limits
       are never breached (preventing LLM attention degradation and context overflow).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from src.agent.data.ingestion import SessionDatabase


class ColumnProfile(BaseModel):
    """Detailed profile of a single column within a registered table.

    Attributes:
        name: Sanitized column identifier used in SQL queries.
        duckdb_type: Native DuckDB data type (e.g. VARCHAR, DOUBLE, DATE, BIGINT).
        total_rows: Total row count in the table.
        null_count: Number of NULL / missing entries in this column.
        null_percentage: Missing value percentage (0.0% - 100.0%).
        distinct_count: Number of unique non-null values.
        sample_values: Up to 3 representative non-null values for categorical grounding.
        min_val: Minimum observed value for numeric/temporal columns.
        max_val: Maximum observed value for numeric/temporal columns.
        mean_val: Arithmetic mean for numeric columns.
    """

    name: str
    duckdb_type: str
    total_rows: int
    null_count: int
    null_percentage: float = Field(..., ge=0.0, le=100.0)
    distinct_count: int
    sample_values: list[Any] = Field(default_factory=list)
    min_val: Any = None
    max_val: Any = None
    mean_val: float | None = None


class TableProfile(BaseModel):
    """Detailed profile of an entire registered dataset table.

    Aggregates column profiles and table-level cardinality metrics to supply
    complete context to both the planning and code generation nodes.
    """

    table_name: str
    row_count: int
    column_count: int
    columns: dict[str, ColumnProfile]


def profile_table(db: SessionDatabase, table_name: str) -> TableProfile:
    """Inspect and profile a registered DuckDB table using vectorized in-engine queries.

    Methodology:
        Rather than loading raw data into Python and running heavy Pandas operations,
        we execute targeted SQL queries directly inside DuckDB. This takes advantage
        of DuckDB's vectorized columnar engine, computing null counts, distinct counts,
        and min/max/mean metrics in milliseconds even for large datasets.

    Steps:
        1. Query `PRAGMA table_info` to retrieve column names and DuckDB data types.
        2. Query `COUNT(*)` to obtain the base table cardinality.
        3. For each column:
           - Compute `COUNT(*) - COUNT(col)` for exact null counts and percentages.
           - Compute `COUNT(DISTINCT col)` for cardinality awareness.
           - If numeric or date/timestamp, compute `MIN(col)`, `MAX(col)`, and `AVG(col)`.
           - Select up to 3 distinct non-null values to show the LLM representative data.
        4. Assemble into a `TableProfile` object.

    Args:
        db: Active `SessionDatabase` containing registered tables.
        table_name: Identifier of the table to inspect.

    Returns:
        Populated `TableProfile` instance.
    """
    clean_table = table_name.lower()
    if not db.is_table_allowed(clean_table):
        raise ValueError(f"Table '{table_name}' is not registered in session allowlist")

    # Step 1: Extract column names and native DuckDB types via PRAGMA
    info_df = db.query_df(f"PRAGMA table_info('{clean_table}')")
    cols_meta = [
        {"name": row["name"], "type": row["type"]}
        for _, row in info_df.iterrows()
    ]

    # Step 2: Retrieve total row count
    count_df = db.query_df(f"SELECT COUNT(*) AS total_rows FROM {clean_table}")
    total_rows = int(count_df["total_rows"].iloc[0])

    profiles: dict[str, ColumnProfile] = {}

    # Step 3: Profile each column individually
    for meta in cols_meta:
        col_name = meta["name"]
        dtype = str(meta["type"]).upper()

        # Handle empty dataset edge case gracefully
        if total_rows == 0:
            profiles[col_name] = ColumnProfile(
                name=col_name,
                duckdb_type=dtype,
                total_rows=0,
                null_count=0,
                null_percentage=0.0,
                distinct_count=0,
                sample_values=[],
            )
            continue

        # Sub-step 3a: Null count and distinct value count
        # Column name is wrapped in quotes to prevent collision with SQL keywords
        quoted_col = f'"{col_name}"'
        stats_sql = f"""
            SELECT
                COUNT(*) - COUNT({quoted_col}) AS null_count,
                COUNT(DISTINCT {quoted_col}) AS distinct_count
            FROM {clean_table}
        """
        stats_df = db.query_df(stats_sql)
        null_count = int(stats_df["null_count"].iloc[0])
        distinct_count = int(stats_df["distinct_count"].iloc[0])
        null_pct = round((null_count / total_rows) * 100.0, 2)

        # Sub-step 3b: Min, max, and mean for numeric or temporal columns
        min_val = None
        max_val = None
        mean_val = None

        numeric_types = ["INT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "HUGEINT"]
        is_numeric = any(t in dtype for t in numeric_types)
        is_temporal = any(t in dtype for t in ["DATE", "TIME", "TIMESTAMP"])

        if is_numeric or is_temporal:
            minmax_sql = f"""
                SELECT
                    MIN({quoted_col}) AS min_v,
                    MAX({quoted_col}) AS max_v
                    {', AVG(' + quoted_col + ') AS avg_v' if is_numeric else ''}
                FROM {clean_table}
                WHERE {quoted_col} IS NOT NULL
            """
            try:
                mm_df = db.query_df(minmax_sql)
                if not mm_df.empty:
                    min_val = mm_df["min_v"].iloc[0]
                    max_val = mm_df["max_v"].iloc[0]
                    if is_numeric and "avg_v" in mm_df.columns:
                        val = mm_df["avg_v"].iloc[0]
                        if pd.notna(val):
                            mean_val = round(float(val), 2)
            except Exception:
                # If aggregation fails due to type edge cases, leave stats as None
                pass

        # Sub-step 3c: Sample non-null distinct values (up to 3 values)
        # Providing sample values grounds the LLM in valid categorical values
        # (e.g. knowing 'Furniture' is a Category, rather than guessing 'Office Furniture')
        sample_sql = f"""
            SELECT DISTINCT {quoted_col} AS val
            FROM {clean_table}
            WHERE {quoted_col} IS NOT NULL
            LIMIT 3
        """
        try:
            samples_df = db.query_df(sample_sql)
            raw_samples = samples_df["val"].tolist()
            sample_values = [
                str(v) if not isinstance(v, (int, float, bool)) else v
                for v in raw_samples
            ]
        except Exception:
            sample_values = []

        # Sub-step 3d: Store completed column profile
        profiles[col_name] = ColumnProfile(
            name=col_name,
            duckdb_type=dtype,
            total_rows=total_rows,
            null_count=null_count,
            null_percentage=null_pct,
            distinct_count=distinct_count,
            sample_values=sample_values,
            min_val=str(min_val) if min_val is not None else None,
            max_val=str(max_val) if max_val is not None else None,
            mean_val=mean_val,
        )

    # Step 4: Return comprehensive TableProfile
    return TableProfile(
        table_name=clean_table,
        row_count=total_rows,
        column_count=len(cols_meta),
        columns=profiles,
    )


def format_profile_for_prompt(profile: TableProfile, max_token_budget: int = 1200) -> str:
    """Format table profile into a concise, token-budgeted Markdown snippet for LLMs.

    Design Justification:
        Large datasets often feature dozens or hundreds of columns. Blindly dumping
        an entire schema into an LLM prompt inflates token consumption and degrades
        model attention. This function formats the schema into a compact Markdown
        table and strictly enforces a token budget (estimated at 4 chars per token).
        If the budget is reached, it truncates gracefully and informs the LLM.

    Args:
        profile: The populated `TableProfile` to format.
        max_token_budget: Maximum estimated token budget (default 1200 tokens ≈ 4800 chars).

    Returns:
        Structured Markdown string suitable for system prompt injection.
    """
    max_char_budget = max_token_budget * 4

    header = (
        f"### Table Schema: `{profile.table_name}` "
        f"({profile.row_count:,} rows, {profile.column_count} columns)\n\n"
        "| Column | Type | Null % | Distinct | Summary / Samples |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n"
    )

    rows: list[str] = []
    for col in profile.columns.values():
        summary_parts = []
        # If numeric/temporal range is available, present range and average
        if col.min_val is not None and col.max_val is not None:
            if col.mean_val is not None:
                summary_parts.append(
                    f"range: [{col.min_val} .. {col.max_val}], avg: {col.mean_val}"
                )
            else:
                summary_parts.append(f"range: [{col.min_val} .. {col.max_val}]")
        # Otherwise show sample representative values
        elif col.sample_values:
            samples_repr = ", ".join(repr(s) for s in col.sample_values[:3])
            summary_parts.append(f"samples: [{samples_repr}]")

        summary_str = "; ".join(summary_parts) if summary_parts else "N/A"
        row_str = (
            f"| `{col.name}` | {col.duckdb_type} | {col.null_percentage}% | "
            f"{col.distinct_count:,} | {summary_str} |\n"
        )
        rows.append(row_str)

    # Step: Assemble rows while respecting character budget
    output = header
    for r in rows:
        truncation_note = (
            "| ... | ... | ... | ... | *(remaining columns truncated to fit token budget)* |\n"
        )
        if len(output) + len(r) + len(truncation_note) > max_char_budget:
            output += truncation_note
            break
        output += r

    return output
