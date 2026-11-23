"""Unit tests for controlled SQL execution engine."""

import pytest

from src.agent.data.ingestion import SessionDatabase
from src.agent.execution.engine import ControlledExecutionEngine


@pytest.fixture
def execution_engine(data_dir):
    """Fixture providing ControlledExecutionEngine with sample datasets."""
    db = SessionDatabase()
    db.register_csv(data_dir / "sample_superstore.csv", "superstore")
    engine = ControlledExecutionEngine(db, default_timeout_seconds=3.0)
    yield engine
    db.close()


def test_execution_scalar_query(execution_engine):
    """Verify execution of a scalar query returning 1 row and 1 column."""
    sql = "SELECT SUM(sales) AS total_sales FROM superstore"
    result = execution_engine.execute(sql)

    assert result.success is True
    assert result.is_scalar is True
    assert result.row_count == 1
    assert result.column_count == 1
    assert result.scalar_value is not None
    assert result.scalar_value > 0
    assert result.execution_time_ms > 0


def test_execution_tabular_query(execution_engine):
    """Verify execution of a grouped tabular query."""
    sql = (
        "SELECT region, SUM(sales) AS reg_sales FROM superstore "
        "GROUP BY region ORDER BY reg_sales DESC"
    )
    result = execution_engine.execute(sql)

    assert result.success is True
    assert result.is_scalar is False
    assert result.row_count == 4
    assert result.column_count == 2
    assert "region" in result.dataframe.columns
    assert "reg_sales" in result.dataframe.columns


def test_execution_empty_result(execution_engine):
    """Verify valid empty result set is not marked as an execution error."""
    # Filter matches zero rows
    sql = "SELECT * FROM superstore WHERE sales > 999999"
    result = execution_engine.execute(sql)

    assert result.success is True
    assert result.is_empty() is True
    assert result.row_count == 0
    assert result.error_type is None


def test_execution_syntax_error(execution_engine):
    """Verify syntax error capture and classification."""
    sql = "SELECT FROM WHERE superstore"
    result = execution_engine.execute(sql)

    assert result.success is False
    assert result.error_type == "SyntaxError"
    assert result.error_message is not None


def test_execution_column_error(execution_engine):
    """Verify invalid column error capture and classification."""
    sql = "SELECT nonexistent_column FROM superstore"
    result = execution_engine.execute(sql)

    assert result.success is False
    assert result.error_type == "ColumnError"
    assert result.error_message is not None


def test_execution_unregistered_table_error(execution_engine):
    """Verify unregistered table access is rejected as TableError."""
    sql = "SELECT * FROM unauthorized_table"
    result = execution_engine.execute(sql)

    assert result.success is False
    assert result.error_type == "TableError"
    assert "not registered" in result.error_message


def test_execution_timeout(execution_engine):
    """Verify timeout interruption on slow/infinite queries."""
    # Simulating long running query or tight timeout
    sql = "SELECT COUNT(*) FROM superstore, superstore AS s2, superstore AS s3"
    result = execution_engine.execute(sql, timeout_seconds=0.0001)

    assert result.success is False
    assert result.timed_out is True
    assert result.error_type == "TimeoutError"
    assert "timed out" in result.error_message.lower()


def test_execution_timeout_worker_cleanup_and_recovery(execution_engine, monkeypatch):
    """Verify timeout interruption, worker termination, and recovery even if interrupt fails."""
    import threading

    sql = "SELECT COUNT(*) FROM superstore, superstore AS s2, superstore AS s3"

    # Part 1: Verify standard timeout, cursor interrupt, and worker thread exit
    before_threads = set(threading.enumerate())
    result = execution_engine.execute(sql, timeout_seconds=0.0001)

    assert result.success is False
    assert result.timed_out is True
    assert result.error_type == "TimeoutError"
    assert "timed out" in result.error_message.lower()

    # Verify that any newly spawned worker thread has cleanly exited
    after_threads = set(threading.enumerate()) - before_threads
    for t in after_threads:
        t.join(timeout=1.0)
        assert not t.is_alive(), "Worker thread leaked or failed to terminate after timeout"

    # Part 2: Verify cleanup succeeds even if cursor.interrupt() raises an exception
    orig_con = execution_engine.db.connection

    class FailingCursorWrapper:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def interrupt(self):
            raise RuntimeError("Simulated cursor interrupt failure")

    class FailingConnectionWrapper:
        def __getattr__(self, name):
            return getattr(orig_con, name)

        def cursor(self):
            return FailingCursorWrapper(orig_con.cursor())

    monkeypatch.setattr(execution_engine.db, "_con", FailingConnectionWrapper())
    res_failing_interrupt = execution_engine.execute(sql, timeout_seconds=0.0001)
    assert res_failing_interrupt.success is False
    assert res_failing_interrupt.timed_out is True
    assert res_failing_interrupt.error_type == "TimeoutError"

    # Part 3: Recovery verification: subsequent queries succeed without connection corruption
    monkeypatch.undo()
    recov_res = execution_engine.execute("SELECT 42 AS val")
    assert recov_res.success is True
    assert recov_res.scalar_value == 42
    assert recov_res.timed_out is False


def test_timeout_bounded_behavior_when_interruption_fails(execution_engine, monkeypatch):
    """Verify bounded timeout return when worker is still running and interruption fails.

    Guarantees:
        1. When cursor.interrupt() fails and the worker thread is still actively running,
           the engine does NOT block indefinitely in executor.shutdown(wait=True).
        2. Execution returns within bounded time (timeout + max 0.2s grace period).
        3. Result is timed_out=True with TimeoutError.
        4. The test itself cannot hang indefinitely (bounded by short assertion).
    """
    import time

    def stubborn_evaluator(val):
        time.sleep(0.4)
        return val

    execution_engine.db.connection.create_function(
        "stubborn_evaluator", stubborn_evaluator, ["DOUBLE"], "DOUBLE"
    )

    orig_con = execution_engine.db.connection

    class StubbornCursorWrapper:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def interrupt(self):
            # Interruption fails and raises an exception; worker continues running
            raise RuntimeError("Simulated interruption failure")

    class StubbornConnectionWrapper:
        def __getattr__(self, name):
            return getattr(orig_con, name)

        def cursor(self):
            return StubbornCursorWrapper(orig_con.cursor())

    monkeypatch.setattr(execution_engine.db, "_con", StubbornConnectionWrapper())

    import threading
    before_threads = set(threading.enumerate())

    t0 = time.perf_counter()
    res = execution_engine.execute(
        "SELECT stubborn_evaluator(sales) FROM superstore LIMIT 1",
        timeout_seconds=0.03,
    )
    t1 = time.perf_counter()
    elapsed = t1 - t0

    assert res.success is False
    assert res.timed_out is True
    assert res.error_type == "TimeoutError"
    # Worker would take 0.4s, but engine MUST return within bounded grace period (< 0.32s)
    assert elapsed < 0.32, f"Execution was unbounded! Elapsed time: {elapsed:.3f}s"

    # Explicit Verification: Prove that the worker thread is STILL ALIVE when the caller returns!
    new_threads = [t for t in (set(threading.enumerate()) - before_threads) if t.is_alive()]
    assert len(new_threads) > 0, "Expected worker thread to still be active in background"
    assert any(t.is_alive() for t in new_threads)

    # Connection safety & non-interference: while the stubborn worker is still running
    # in the background on its abandoned cursor, a subsequent query on a fresh cursor
    # must execute and succeed immediately on the same connection!
    monkeypatch.undo()
    concurrent_res = execution_engine.execute("SELECT 777 AS recovery_val")
    assert concurrent_res.success is True
    assert concurrent_res.scalar_value == 777
    assert concurrent_res.timed_out is False

    # Allow stubborn worker thread to finish sleeping before fixture teardown
    for t in new_threads:
        t.join(timeout=1.0)
        assert not t.is_alive()


def test_arrow_duplicate_column_positional_extraction_and_layer_1_rejection(
    execution_engine,
):
    """Verify positional raw-row extraction preserves distinct values and Layer 1 fails."""
    from src.agent.validation.contracts import ErrorCategory, ValidationStatus
    from src.agent.validation.execution import ExecutionValidator

    sql = "SELECT sales AS value, profit AS value FROM superstore LIMIT 5"
    res = execution_engine.execute(sql)

    assert res.success is True
    assert res.columns == ("value", "value")
    assert len(res.raw_rows) == 5

    # Verify positional values: sales and profit have different values in superstore
    # (Dictionary-based extraction would overwrite the first column with the second)
    for r in res.raw_rows:
        assert len(r) == 2
        assert r[0] != r[1], f"Positional columns overwritten! {r[0]} == {r[1]}"

    # Verify Layer 1 detects duplicate projected column names and flags SCHEMA_ERROR
    validator = ExecutionValidator()
    step_val = validator.validate_step(raw_execution_result=res)
    assert step_val.execution_result.passed is False
    assert step_val.execution_result.status == ValidationStatus.FAILED
    assert "Duplicate column name(s)" in step_val.execution_result.message
    assert step_val.diagnostic is not None
    assert step_val.diagnostic.error_category == ErrorCategory.SCHEMA_ERROR


def test_execution_single_query_evaluation_guarantee(execution_engine):
    """Verify that ControlledExecutionEngine executes the analytical query exactly once.

    Uses an instrumented Python scalar UDF to count actual row evaluations.
    Asserts that:
        1. Execution count equals the exact row limit (1 query pass).
        2. If the query had been executed twice, the count would be 2x.
        3. raw_rows and dataframe represent the exact same materialized values.
    """
    call_counter = {"count": 0}

    def spy_evaluator(val):
        call_counter["count"] += 1
        return val

    execution_engine.db.connection.create_function(
        "spy_evaluator", spy_evaluator, ["DOUBLE"], "DOUBLE"
    )

    sql = "SELECT spy_evaluator(sales) AS obs_sales FROM superstore LIMIT 10"
    result = execution_engine.execute(sql)

    assert result.success is True
    assert call_counter["count"] == 10, (
        f"Query executed multiple times! Count: {call_counter['count']}"
    )
    assert result.raw_rows is not None
    assert len(result.raw_rows) == 10
    assert len(result.dataframe) == 10
    raw_vals = [r[0] for r in result.raw_rows]
    df_vals = list(result.dataframe["obs_sales"])
    assert raw_vals == df_vals


def test_execution_null_vs_ieee_nan_scalar_distinction(execution_engine):
    """Verify that scalar extraction preserves SQL NULL as None and IEEE NaN as float('nan')."""
    import math

    # SQL NULL
    res_null = execution_engine.execute("SELECT 1.0 / NULL AS val")
    assert res_null.success is True
    assert res_null.is_scalar is True
    assert res_null.scalar_value is None
    assert res_null.raw_rows == [(None,)]

    # IEEE Floating NaN
    res_nan = execution_engine.execute("SELECT 0.0 / 0.0 AS val")
    assert res_nan.success is True
    assert res_nan.is_scalar is True
    assert res_nan.scalar_value is not None
    assert isinstance(res_nan.scalar_value, float)
    assert math.isnan(res_nan.scalar_value)
    assert len(res_nan.raw_rows) == 1
    assert math.isnan(res_nan.raw_rows[0][0])


def test_execution_arrow_column_ordering_and_duplicates(execution_engine):
    """Verify explicit Arrow column ordering and column resolution."""
    sql = "SELECT region, sales, quantity FROM superstore LIMIT 5"
    res = execution_engine.execute(sql)

    assert res.success is True
    assert list(res.dataframe.columns) == ["region", "sales", "quantity"]
    assert len(res.raw_rows) == 5
    for r in res.raw_rows:
        assert isinstance(r[0], str)  # region
        assert isinstance(r[1], float)  # sales
        assert isinstance(r[2], (int, float))  # quantity
