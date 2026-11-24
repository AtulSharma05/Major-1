"""Controlled DuckDB query execution engine with AST validation and timeout enforcement.

Architectural Context (ADR-04, ADR-09, Master Execution Plan Phase 5 Stage 3):
    Analytical SQL execution must be deterministic, resource-bounded, and safe.
    This module provides:
    1. Pre-execution AST validation via SQLGlot.
    2. Session allowlist verification (preventing table escape or file system calls).
    3. Thread-isolated timeout enforcement with connection interruption.
    4. Bounded timeout lifecycle: cooperative cursor.interrupt() with bounded grace-period
       join (max 0.2s) and non-blocking executor.shutdown(wait=False), preventing indefinite hangs.
    5. Single analytical query execution materialized eagerly into Arrow table.
    6. Explicit classification of errors (Syntax, Column, Table, Timeout, Runtime).
    7. Clean separation of valid empty results from execution failures.
    8. Automatic extraction of scalar values for single-metric queries preserving NULL vs NaN.
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
    raw_rows: list[tuple[Any, ...]] | None = None
    columns: tuple[str, ...] = ()
    scalar_value: Any = None
    is_scalar: bool = False
    row_count: int = 0
    column_count: int = 0
    execution_time_ms: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    timed_out: bool = False

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
        # We spawn a dedicated DuckDB cursor for thread-safety and reliable interruption.
        #
        # Thread Limitation & Connection Safety Invariants:
        # 1. CPython threads cannot be forcibly killed or canceled from outside.
        # 2. When a query exceeds its timeout budget, cursor.interrupt() attempts cooperative
        #    interruption of the DuckDB C++ query execution engine.
        # 3. If interruption fails (e.g. Python UDF or non-interruptible loop), the worker
        #    thread continues running until completion in the background.
        # 4. Attempting cursor.close() on a cursor actively used by a running worker raises
        #    ResourceInUseException. We handle this safely with a guarded try-except block so
        #    the cursor is safely abandoned without raising or hanging.
        # 5. Subsequent queries call execute(), which creates a *fresh* cursor on
        #    self.db.connection.cursor(). DuckDB connections isolate concurrent cursors, so
        #    future queries proceed immediately and safely without deadlock or corruption.
        # 6. We join the worker thread with a bounded grace period (0.2s) and call
        #    executor.shutdown(wait=False, cancel_futures=True), strictly guaranteeing
        #    bounded caller return within timeout + 0.2s.
        cursor = self.db.cursor() if hasattr(self.db, "cursor") else self.db.connection.cursor()

        def _run_query():
            rel = cursor.sql(clean_query)
            tbl = rel.to_arrow_table()
            col_names = list(tbl.column_names)
            # Positional column extraction: preserves distinct values even if column names
            # are duplicated
            cols = [c.to_pylist() for c in tbl.columns]
            raw_rows = [tuple(r) for r in zip(*cols)] if cols else []

            view_name = f"_step_view_{time.time_ns()}"
            cursor.register(view_name, tbl)
            try:
                df = cursor.sql(f"SELECT * FROM {view_name}").df()
                # Restore original projected column names (including any duplicates for
                # Layer 1 detection)
                if len(col_names) == len(df.columns):
                    df.columns = col_names
            finally:
                try:
                    cursor.unregister(view_name)
                except Exception:
                    pass
            return raw_rows, df, col_names

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_run_query)

        try:
            raw_rows, df, col_names = future.result(timeout=timeout)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
        except concurrent.futures.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            # Cooperative cursor interruption
            try:
                cursor.interrupt()
            except Exception:
                pass

            # Bounded thread termination: wait up to a short grace period (0.2s)
            # to allow cooperative exit, but NEVER wait indefinitely.
            for t in list(getattr(executor, "_threads", [])):
                t.join(timeout=0.2)

            # Shutdown executor non-blockingly to guarantee bounded execution return
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            return ExecutionResult(
                success=False,
                query=clean_query,
                timed_out=True,
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
            try:
                cursor.close()
            except Exception:
                pass
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        # Step 3: Result structure inspection (scalar vs tabular)
        row_count = len(df)
        column_count = len(df.columns)
        is_scalar = False
        scalar_val = None

        if row_count == 1 and column_count == 1:
            is_scalar = True
            # Extract scalar directly from raw_rows to preserve None vs float('nan')
            if raw_rows:
                scalar_val = raw_rows[0][0]
            else:
                raw_val = df.iloc[0, 0]
                scalar_val = (
                    None
                    if pd.isna(raw_val)
                    else (raw_val.item() if hasattr(raw_val, "item") else raw_val)
                )

        return ExecutionResult(
            success=True,
            query=clean_query,
            dataframe=df,
            raw_rows=raw_rows,
            columns=tuple(col_names),
            scalar_value=scalar_val,
            is_scalar=is_scalar,
            row_count=row_count,
            column_count=column_count,
            execution_time_ms=round(duration_ms, 2),
        )
