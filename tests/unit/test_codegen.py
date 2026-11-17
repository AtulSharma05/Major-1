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


def test_schema_grounding_prompt_for_both_datasets(data_dir):
    """Verify that build_sql_prompt grounds available columns dynamically for both datasets."""
    db = SessionDatabase()
    db.register_csv(data_dir / "sample_superstore.csv", "sample_superstore")
    db.register_csv(data_dir / "sample_superstore_2.csv", "sample_superstore_2")

    p1 = profile_table(db, "sample_superstore")
    p2 = profile_table(db, "sample_superstore_2")

    prompt1 = build_sql_prompt("Find product with highest sales", profile=p1)
    prompt2 = build_sql_prompt("Find product with highest sales", profile=p2)

    # Dataset 1: sample_superstore contains sub_category, NOT product_name
    assert "product_name" not in p1.columns
    assert "sub_category" in p1.columns
    assert "`sub_category`" in prompt1

    # Dataset 2: sample_superstore_2 contains product_name
    assert "product_name" in p2.columns
    assert "`product_name`" in prompt2

    db.close()


def test_separate_sales_and_profit_sql_generation():
    """Verify separate SQL generation for sales and profit queries without cross-sorting."""
    from src.agent.codegen.generator import CodeGenerator
    from src.agent.data.profiling import ColumnProfile, TableProfile

    class DeterministicClient:
        def generate(self, prompt: str, **kwargs):
            if "highest profit" in prompt.lower():
                return (
                    "```sql\n"
                    "SELECT sub_category, SUM(profit) AS total_profit\n"
                    "FROM superstore\n"
                    "GROUP BY sub_category\n"
                    "ORDER BY total_profit DESC LIMIT 1\n"
                    "```"
                )
            return (
                "```sql\n"
                "SELECT sub_category, SUM(sales) AS total_sales\n"
                "FROM superstore\n"
                "GROUP BY sub_category\n"
                "ORDER BY total_sales DESC LIMIT 1\n"
                "```"
            )

    gen = CodeGenerator(client=DeterministicClient())

    col_sub = ColumnProfile(
        name="sub_category",
        duckdb_type="VARCHAR",
        total_rows=10,
        null_count=0,
        null_percentage=0.0,
        distinct_count=5,
    )
    col_sales = ColumnProfile(
        name="sales",
        duckdb_type="DOUBLE",
        total_rows=10,
        null_count=0,
        null_percentage=0.0,
        distinct_count=10,
    )
    col_profit = ColumnProfile(
        name="profit",
        duckdb_type="DOUBLE",
        total_rows=10,
        null_count=0,
        null_percentage=0.0,
        distinct_count=10,
    )

    profile = TableProfile(
        table_name="superstore",
        row_count=10,
        column_count=3,
        columns={"sub_category": col_sub, "sales": col_sales, "profit": col_profit},
    )

    sql_sales = gen.generate_sql("Find sub-category with highest sales", profile=profile)
    sql_profit = gen.generate_sql("Find sub-category with highest profit", profile=profile)

    assert "SUM(sales)" in sql_sales
    assert "ORDER BY total_sales DESC" in sql_sales
    assert "profit" not in sql_sales.lower()

    assert "SUM(profit)" in sql_profit
    assert "ORDER BY total_profit DESC" in sql_profit
    assert "sales" not in sql_profit.lower()
