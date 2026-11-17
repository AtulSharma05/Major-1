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
    assert result.error_type == "TimeoutError"
    assert "timed out" in result.error_message.lower()
