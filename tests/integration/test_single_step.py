"""Live integration tests for Phase 3: Single-step question-to-SQL-to-result execution.

Runs canonical single-step queries on local Ollama with Qwen2.5-Coder 7B:
1. Scalar aggregation: "What is the total sales?"
2. Grouped aggregation: "What are the total sales by region?"
3. Filtered calculation: "What is total sales for Furniture category?"
4. Distinct count: "How many unique customers made purchases?"
5. Top-N query: "What are the top 3 sub-categories by sales?"
"""

import httpx
import pytest

from src.agent.codegen.generator import CodeGenerator, OllamaClient
from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import profile_table
from src.agent.execution.engine import ControlledExecutionEngine


@pytest.fixture(scope="module")
def ollama_available():
    """Check if local Ollama daemon is reachable on localhost:11434."""
    try:
        res = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        if res.status_code != 200:
            pytest.skip("Ollama returned non-200 status")
        models = [m.get("name", "") for m in res.json().get("models", [])]
        if not any("qwen2.5-coder" in m for m in models):
            pytest.skip("Model qwen2.5-coder:7b not pulled in Ollama")
    except Exception as exc:
        pytest.skip(f"Ollama server not reachable: {exc}")


@pytest.fixture
def setup_engine(data_dir):
    """Setup SessionDatabase with canonical Superstore dataset."""
    db = SessionDatabase()
    db.register_csv(data_dir / "sample_superstore.csv", "superstore")
    engine = ControlledExecutionEngine(db, default_timeout_seconds=10.0)
    profile = profile_table(db, "superstore")
    client = OllamaClient(model="qwen2.5-coder:7b", timeout=30.0)
    generator = CodeGenerator(client=client)
    yield db, engine, profile, generator
    db.close()


def test_single_step_scalar_sales(ollama_available, setup_engine):
    """Benchmark Query 1: Total sales aggregation."""
    db, engine, profile, generator = setup_engine
    question = "What is the total sales?"

    sql = generator.generate_sql(question, profile)
    assert sql, "Generated SQL is empty"

    result = engine.execute(sql)
    assert result.success is True, f"Execution failed: {result.error_message} (SQL: {sql})"
    assert result.is_scalar is True
    assert result.scalar_value > 0


def test_single_step_grouped_region(ollama_available, setup_engine):
    """Benchmark Query 2: Total sales grouped by region."""
    db, engine, profile, generator = setup_engine
    question = "What are the total sales by region?"

    sql = generator.generate_sql(question, profile)
    assert sql, "Generated SQL is empty"

    result = engine.execute(sql)
    assert result.success is True, f"Execution failed: {result.error_message} (SQL: {sql})"
    assert result.row_count > 1
    assert "region" in [c.lower() for c in result.dataframe.columns]


def test_single_step_filtered_category(ollama_available, setup_engine):
    """Benchmark Query 3: Filtered sales for Furniture."""
    db, engine, profile, generator = setup_engine
    question = "What is the total sales for Furniture category?"

    sql = generator.generate_sql(question, profile)
    assert sql, "Generated SQL is empty"

    result = engine.execute(sql)
    assert result.success is True, f"Execution failed: {result.error_message} (SQL: {sql})"
    assert result.row_count >= 1


def test_single_step_distinct_customers(ollama_available, setup_engine):
    """Benchmark Query 4: Distinct count of customer IDs."""
    db, engine, profile, generator = setup_engine
    question = "How many distinct customers made purchases?"

    sql = generator.generate_sql(question, profile)
    assert sql, "Generated SQL is empty"

    result = engine.execute(sql)
    assert result.success is True, f"Execution failed: {result.error_message} (SQL: {sql})"
    assert result.is_scalar is True
    assert result.scalar_value > 0


def test_single_step_top_subcategories(ollama_available, setup_engine):
    """Benchmark Query 5: Top 3 sub-categories by sales."""
    db, engine, profile, generator = setup_engine
    question = "What are the top 3 sub-categories by total sales?"

    sql = generator.generate_sql(question, profile)
    assert sql, "Generated SQL is empty"

    result = engine.execute(sql)
    assert result.success is True, f"Execution failed: {result.error_message} (SQL: {sql})"
    assert result.row_count <= 3
