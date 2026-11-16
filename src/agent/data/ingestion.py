"""Dynamic dataset ingestion and session-level table registration in DuckDB.

Architectural Context:
    In our autonomous data analytics system, users upload arbitrary CSV or Excel
    datasets. To ensure both execution safety and deterministic query generation,
    we utilize DuckDB as an in-process, high-performance analytical SQL engine
    (ADR-09, ADR-10).

Key Responsibilities of this Module:
    1. Dynamic Table Registration: Ingest CSVs, Excel files, and DataFrames into
       in-memory DuckDB tables without persistent disk side-effects.
    2. Identifier Sanitization: Standardize table and column names to lowercase snake_case
       to prevent SQL syntax errors across different database dialects (e.g. eliminating
       fragile bracket/quote handling), while preserving a bidirectional mapping to
       the user's original header names.
    3. Session Table Allowlist: Track every registered table within the active session.
    4. Deterministic AST Security Validation: Parse incoming SQL queries using SQLGlot's
       Abstract Syntax Tree (AST) to enforce that:
       - No unauthorized physical tables are accessed.
       - Common Table Expressions (CTEs) are recognized as valid temporary scopes.
       - Dangerous external filesystem access functions (such as `read_csv`,
         `read_parquet`, `sqlite_scan`) are blocked prior to execution.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp


def sanitize_identifier(name: str) -> str:
    """Normalize a column or table name to a clean lowercase snake_case identifier.

    Design Justification:
        LLMs frequently struggle with column names containing special characters,
        spaces, hyphens, or currency symbols (e.g., 'Sales ($USD)' or 'Order-ID').
        Different SQL dialects quote identifiers differently (e.g., "Order ID" in
        Postgres/DuckDB vs `Order ID` in MySQL vs [Order ID] in T-SQL).
        Sanitizing all identifiers to clean snake_case upon ingestion completely
        eliminates quoting discrepancies while keeping SQL queries readable.

    Transformation Steps:
        1. Replace any non-alphanumeric character with an underscore.
        2. Collapse consecutive underscores to a single underscore.
        3. Strip leading and trailing underscores, and convert to lowercase.
        4. If the identifier starts with a number, prepend 'col_' (valid SQL syntax).

    Args:
        name: Raw identifier string from user file header.

    Returns:
        Clean, SQL-safe snake_case identifier string.
    """
    # Step 1: Replace any character that is not a letter, digit, or underscore
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())

    # Step 2: Collapse multiple underscores into one
    cleaned = re.sub(r"_+", "_", cleaned)

    # Step 3: Trim surrounding underscores and enforce lowercase
    cleaned = cleaned.strip("_").lower()

    # Step 4: Guard against leading numeric digits (e.g., '2024_sales' -> 'col_2024_sales')
    if cleaned and cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"

    # Fallback if the input consisted purely of stripped special characters
    return cleaned or "col"


class SessionDatabase:
    """Thread-safe session-level DuckDB wrapper with strict table allowlisting.

    Each user session operates within its own `SessionDatabase` instance. This class
    maintains an explicit allowlist of authorized table names. Any generated SQL
    query is inspected via AST validation before execution; queries attempting to
    escape the allowlist or execute filesystem read functions are rejected.
    """

    # Comprehensive set of DuckDB functions capable of bypassing the catalog
    # and reading arbitrary filesystem or external network targets.
    FORBIDDEN_FUNCTIONS = {
        "read_csv",
        "read_csv_auto",
        "read_parquet",
        "read_json",
        "read_json_auto",
        "parquet_scan",
        "scan_parquet",
        "postgres_scan",
        "sqlite_scan",
        "glob",
    }

    def __init__(self, db_path: str = ":memory:"):
        """Initialize an in-process DuckDB session.

        Args:
            db_path: DuckDB connection string. Defaults to ':memory:' for transient,
                     zero-disk-footprint analytical workloads.
        """
        # Connect to DuckDB in-memory instance
        self._con = duckdb.connect(database=db_path)

        # Allowlist registry of table names registered during this session
        self._registered_tables: set[str] = set()

        # Metadata repository storing column mappings and original schema profiles
        self._table_metadata: dict[str, dict[str, Any]] = {}

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Access the underlying DuckDB connection object."""
        return self._con

    def get_allowed_tables(self) -> list[str]:
        """Return a sorted list of table names currently registered in this session."""
        return sorted(list(self._registered_tables))

    def is_table_allowed(self, table_name: str) -> bool:
        """Check whether a given table name is authorized in the session allowlist."""
        return table_name.lower() in self._registered_tables

    def get_column_mapping(self, table_name: str) -> dict[str, str]:
        """Return the dictionary mapping original header names to sanitized column names.

        This mapping allows the agent to translate generated analytical findings
        back into the exact column nomenclature used in the original user dataset.
        """
        meta = self._table_metadata.get(table_name.lower(), {})
        return meta.get("column_mapping", {})

    def register_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        sanitize_columns: bool = True,
    ) -> str:
        """Register an in-memory Pandas DataFrame as a physical table in DuckDB.

        Process:
            1. Sanitize the target table name.
            2. If `sanitize_columns` is enabled, normalize all column names to snake_case,
               resolving any deduplication collisions (e.g. 'Date' and 'Date' -> 'date', 'date_1').
            3. Register the DataFrame as a temporary virtual table in DuckDB, then materialize
               it via `CREATE OR REPLACE TABLE ... AS SELECT * FROM ...` to ensure optimal
               columnar query performance.
            4. Record the table in the session allowlist and store metadata.

        Args:
            df: Source pandas DataFrame.
            table_name: Requested name for the table.
            sanitize_columns: Whether to normalize column headers to snake_case.

        Returns:
            The sanitized table name registered in DuckDB.
        """
        clean_table_name = sanitize_identifier(table_name)
        if not clean_table_name:
            raise ValueError(f"Invalid table name: '{table_name}'")

        df_to_load = df.copy()
        column_mapping: dict[str, str] = {}

        if sanitize_columns:
            seen_cols: set[str] = set()
            new_cols: list[str] = []
            for col in df_to_load.columns:
                clean_col = sanitize_identifier(str(col))
                # Resolve potential column name collisions
                candidate = clean_col
                counter = 1
                while candidate in seen_cols:
                    candidate = f"{clean_col}_{counter}"
                    counter += 1
                seen_cols.add(candidate)
                new_cols.append(candidate)
                # Save bidirectional mapping: original -> sanitized
                column_mapping[str(col)] = candidate
            df_to_load.columns = new_cols
        else:
            for col in df_to_load.columns:
                column_mapping[str(col)] = str(col)

        # Step: Materialize into DuckDB native storage
        # We register df_to_load temporarily and immediately create a physical DuckDB table.
        temp_view = f"_temp_{clean_table_name}"
        self._con.register(temp_view, df_to_load)
        create_sql = (
            f"CREATE OR REPLACE TABLE {clean_table_name} AS "
            f"SELECT * FROM {temp_view}"
        )
        self._con.execute(create_sql)
        self._con.unregister(temp_view)

        # Step: Update session allowlist and metadata
        self._registered_tables.add(clean_table_name)
        self._table_metadata[clean_table_name] = {
            "source": "dataframe",
            "row_count": len(df_to_load),
            "column_mapping": column_mapping,
            "columns": list(df_to_load.columns),
        }
        return clean_table_name

    def register_csv(
        self,
        file_path: Path | str,
        table_name: str | None = None,
        sanitize_columns: bool = True,
    ) -> str:
        """Load and register a CSV file into the DuckDB session.

        Reads the CSV file using Pandas to ensure robust delimiter and encoding
        detection before registering the resulting DataFrame into DuckDB.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        target_name = table_name or path.stem
        # low_memory=False ensures consistent pandas dtype inference across large chunks
        df = pd.read_csv(path, low_memory=False)
        return self.register_dataframe(df, target_name, sanitize_columns=sanitize_columns)

    def register_excel(
        self,
        file_path: Path | str,
        table_name: str | None = None,
        sheet_name: str | int = 0,
        sanitize_columns: bool = True,
    ) -> str:
        """Load and register an Excel spreadsheet sheet into the DuckDB session.

        Utilizes `openpyxl` via Pandas to read structured tabular data from .xlsx files.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")

        target_name = table_name or path.stem
        df = pd.read_excel(path, sheet_name=sheet_name)
        return self.register_dataframe(df, target_name, sanitize_columns=sanitize_columns)

    def validate_query_security(self, sql_query: str) -> tuple[bool, list[str]]:
        """Validate query against registered table allowlist and forbidden external functions.

        Design Justification (Why SQLGlot AST instead of Regex):
            Regular expressions are notoriously vulnerable to evasion when analyzing SQL.
            Table names can be hidden inside comments, subqueries, complex JOINs, or string
            literals. SQLGlot constructs a full Abstract Syntax Tree (AST), allowing us to
            unambiguously identify all table identifiers, CTE aliases, and function calls.

        Validation Steps:
            1. Parse the SQL query into an AST for the DuckDB dialect.
            2. Detect any function calls in `FORBIDDEN_FUNCTIONS` (e.g. `read_csv`).
            3. Extract all defined Common Table Expression (CTE) names to exclude them
               from physical table verification.
            4. Extract all `exp.Table` references and verify that each refers to an
               allowlisted table or a valid CTE alias.

        Args:
            sql_query: The SQL query string generated by the agent or user.

        Returns:
            Tuple of (is_valid: bool, error_messages: list[str]).
        """
        errors: list[str] = []

        # Step 1: Parse AST with DuckDB dialect rules
        try:
            parsed = sqlglot.parse_one(sql_query, read="duckdb")
        except Exception as exc:
            return False, [f"SQL syntax parse error: {exc}"]

        # Step 2: Reject forbidden external file access functions
        # We inspect every function invocation in the syntax tree.
        for func in parsed.find_all(exp.Func):
            if isinstance(func, exp.Anonymous):
                func_name = func.name.lower()
            elif hasattr(func, "sql_name"):
                func_name = func.sql_name().lower()
            else:
                func_name = func.key.lower()

            if func_name in self.FORBIDDEN_FUNCTIONS:
                errors.append(f"Forbidden external data function '{func_name}' is not permitted")

        # Step 3: Extract defined CTE names
        # Queries of the form `WITH regional_sales AS (SELECT ...) SELECT * FROM regional_sales`
        # create local CTE aliases that are valid in FROM clauses, even though they are not
        # physical tables registered in DuckDB.
        cte_names = set()
        for with_expr in parsed.find_all(exp.With):
            for cte in with_expr.expressions:
                if isinstance(cte, exp.CTE):
                    cte_names.add(cte.alias_or_name.lower())

        # Step 4: Extract and verify physical table references
        for table in parsed.find_all(exp.Table):
            table_name = table.name.lower()
            if not table_name:
                continue
            # If the identifier matches a known CTE defined in this query, it is allowed
            if table_name in cte_names:
                continue
            # Otherwise, it must exist in our registered session allowlist
            if not self.is_table_allowed(table_name):
                errors.append(
                    f"Table '{table_name}' is not registered in session allowlist. "
                    f"Allowed tables: {self.get_allowed_tables()}"
                )

        return (len(errors) == 0), errors

    def query(self, sql_query: str, params: Any = None) -> duckdb.DuckDBPyRelation:
        """Validate query against security allowlist and execute in DuckDB.

        Raises:
            PermissionError: If query accesses unregistered tables or forbidden functions.
        """
        is_valid, errors = self.validate_query_security(sql_query)
        if not is_valid:
            raise PermissionError(f"Query validation rejected: {'; '.join(errors)}")

        if params is not None:
            return self._con.execute(sql_query, params)
        return self._con.execute(sql_query)

    def query_df(self, sql_query: str, params: Any = None) -> pd.DataFrame:
        """Execute validated SQL query and return results as a Pandas DataFrame."""
        rel = self.query(sql_query, params=params)
        return rel.df()

    def close(self) -> None:
        """Gracefully close the in-process DuckDB session."""
        self._con.close()
