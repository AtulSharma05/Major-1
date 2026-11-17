"""Unit tests for prompt formatting and SQL extraction."""

from src.agent.codegen.generator import extract_sql_from_response
from src.agent.codegen.prompts import build_sql_prompt
from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import profile_table
from src.agent.planning.intent import AggregationType, MetricSpec, QuerySpecification


def test_extract_sql_from_response():
    """Verify stripping markdown code fences and extraneous text."""
    # Fenced with ```sql
    res1 = "```sql\nSELECT region, SUM(sales) FROM superstore GROUP BY region;\n```"
    expected1 = "SELECT region, SUM(sales) FROM superstore GROUP BY region"
    assert extract_sql_from_response(res1) == expected1

    # Fenced without sql language tag
    res2 = "```\nSELECT COUNT(*) FROM superstore;\n```"
    assert extract_sql_from_response(res2) == "SELECT COUNT(*) FROM superstore"

    # Raw SQL without fences
    res3 = "SELECT AVG(profit) FROM superstore"
    assert extract_sql_from_response(res3) == "SELECT AVG(profit) FROM superstore"


def test_build_sql_prompt(data_dir):
    """Verify prompt formatting with injected schema and intent."""
    db = SessionDatabase()
    db.register_csv(data_dir / "sample_superstore.csv", "superstore")
    profile = profile_table(db, "superstore")

    intent = QuerySpecification(
        raw_question="Total sales by region?",
        table_name="superstore",
        metrics=[MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales")],
        dimensions=["region"],
    )

    prompt = build_sql_prompt(
        question="Total sales by region?",
        profile=profile,
        intent=intent,
    )

    assert "### Dataset Context" in prompt
    assert "superstore" in prompt
    assert "### Structured Intent" in prompt
    assert "SUM(sales) AS total_sales" in prompt
    assert "['region']" in prompt
    db.close()
