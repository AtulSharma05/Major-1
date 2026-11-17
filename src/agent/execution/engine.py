"""Controlled DuckDB query execution engine with AST validation and timeout enforcement.

Architectural Context (ADR-04, ADR-09, Master Execution Plan):
    Analytical SQL execution must be deterministic, resource-bounded, and safe.
    This module provides:
    1. Pre-execution AST validation via SQLGlot.
    2. Session allowlist verification (preventing table escape or file system calls).
    3. Thread-isolated timeout enforcement with connection interruption.
    4. Explicit classification of errors (Syntax, Column, Table, Timeout, Runtime).
    5. Clean separation of valid empty results from execution failures.
    6. Automatic extraction of scalar values for single-metric queries.
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from src.agent.data.ingestion import SessionDatabase


class ExecutionResult(BaseModel):
    """Structured result of analytical query execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool
    query: str
    dataframe: pd.DataFrame | None = None
    scalar_value: Any = None
    is_scalar: bool = False
    row_count: int = 0
    column_count: int = 0
    execution_time_ms: float = 0.0
    error_type: str | None = None
    error_message: str | None = None

    def is_empty(self) -> bool:
        """Return True if execution succeeded but returned zero rows."""
        return self.success and self.row_count == 0


class ControlledExecutionEngine:
    """Executes validated SQL queries against an active SessionDatabase."""

    def __init__(self, db: SessionDatabase, default_timeout_seconds: float = 5.0):
        self.db = db
        self.timeout_seconds = default_timeout_seconds

    def execute(self, sql_query: str, timeout_seconds: float | None = None) -> ExecutionResult:
        """Validate and execute a SQL query with timeout protection.

        Steps:
            1. Validate query security and table allowlist using SQLGlot.
            2. Run query in an isolated thread worker with timeout.
            3. Measure execution duration in milliseconds.
            4. Inspect result structure: scalar vs tabular.
            5. Return structured ExecutionResult.
        """
        clean_query = sql_query.strip()
        timeout = timeout_seconds or self.timeout_seconds
        start_time = time.perf_counter()

        # Step 1: Pre-execution AST security and allowlist validation
        is_valid, security_errors = self.db.validate_query_security(clean_query)
        if not is_valid:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            error_msg = "; ".join(security_errors)
            error_type = "TableError" if "not registered" in error_msg else "SyntaxError"
            return ExecutionResult(
                success=False,
                query=clean_query,
                execution_time_ms=round(duration_ms, 2),
                error_type=error_type,
                error_message=error_msg,
            )

        # Step 2: Thread-isolated execution with cursor and timeout protection
        # We spawn a dedicated DuckDB cursor for thread-safety and reliable interruption
        cursor = self.db.connection.cursor()

        def _run_query():
            return cursor.execute(clean_query).df()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_run_query)

        try:
            df = future.result(timeout=timeout)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
        except concurrent.futures.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            # Explicitly interrupt the active cursor and await thread termination
            try:
                cursor.interrupt()
            except Exception:
                pass
            executor.shutdown(wait=True)
            return ExecutionResult(
                success=False,
                query=clean_query,
                execution_time_ms=round(duration_ms, 2),
                error_type="TimeoutError",
                error_message=f"Query execution timed out after {timeout:.1f} seconds.",
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            err_str = str(exc)
            # Classify error type
            if "Binder Error" in err_str or "Referenced column" in err_str:
                error_type = "ColumnError"
            elif "Parser Error" in err_str or "syntax error" in err_str.lower():
                error_type = "SyntaxError"
            else:
                error_type = "RuntimeError"

            return ExecutionResult(
                success=False,
                query=clean_query,
                execution_time_ms=round(duration_ms, 2),
                error_type=error_type,
                error_message=err_str,
            )
        finally:
            executor.shutdown(wait=False)
            try:
                cursor.close()
            except Exception:
                pass

        # Step 3: Result structure inspection (scalar vs tabular)
        row_count = len(df)
        column_count = len(df.columns)
        is_scalar = False
        scalar_val = None

        if row_count == 1 and column_count == 1:
            is_scalar = True
            raw_val = df.iloc[0, 0]
            # Convert numpy/pandas scalars to native Python types where appropriate
            if pd.isna(raw_val):
                scalar_val = None
            elif hasattr(raw_val, "item"):
                scalar_val = raw_val.item()
            else:
                scalar_val = raw_val

        return ExecutionResult(
            success=True,
            query=clean_query,
            dataframe=df,
            scalar_value=scalar_val,
            is_scalar=is_scalar,
            row_count=row_count,
            column_count=column_count,
            execution_time_ms=round(duration_ms, 2),
        )
