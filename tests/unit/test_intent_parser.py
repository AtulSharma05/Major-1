"""Unit tests for runtime IntentParser (Phase 5 Stage 3.1).

Verifies:
    1. Total sales in 2017 -> Valid SUM + 2017 date filter.
    2. Average profit by region -> AVG + region dimension.
    3. Count of orders -> Correct COUNT/DISTINCT_COUNT semantics on order_id.
    4. Sales by nonexistent column -> Intent rejected (None).
    5. Ambiguous date filter -> Intent rejected (None).
    6. Unsupported aggregation -> Intent rejected (None).
    7. Compare 2016 vs 2017 sales -> Two independent atomic specifications.
    8. Ambiguous analytical question -> Safe INCONCLUSIVE (None).
    9. AOV contracts: grounded numerator/denominator, missing denominator rejection.
    10. Heuristic confidence: queries without explicit aggregation keywords (e.g. "sales in 2017")
        are rejected by the heuristic fast path and fall through.
    11. CUSTOM formulas: verified against SafeFormulaEvaluator.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import TableProfile, profile_table
from src.agent.planning.intent import (
    AggregationType,
    MetricSpec,
    QuerySpecification,
)
from src.agent.planning.intent_parser import IntentParser
from src.agent.planning.planner import QueryPlanner


@pytest.fixture
def superstore_profile() -> TableProfile:
    """Create a realistic TableProfile for testing intent parsing."""
    db = SessionDatabase()
    df = pd.DataFrame(
        {
            "order_id": ["CA-1", "CA-2", "CA-3", "CA-4"],
            "order_date": ["2016-05-10", "2017-06-12", "2017-11-20", "2018-01-05"],
            "customer_id": ["C1", "C2", "C3", "C4"],
            "region": ["East", "West", "East", "Central"],
            "category": ["Technology", "Furniture", "Technology", "Office Supplies"],
            "sales": [100.50, 250.00, 300.25, 45.00],
            "profit": [20.10, -15.00, 80.50, 5.00],
            "quantity": [2, 5, 3, 1],
        }
    )
    tbl = db.register_dataframe(df, "superstore")
    return profile_table(db, tbl)


@pytest.fixture
def ambiguous_date_profile() -> TableProfile:
    """Create a TableProfile with ambiguous date formatting."""
    db = SessionDatabase()
    df = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "order_date": ["01/02/2020", "03/04/2020"],  # Day and month both <= 12 -> AMBIGUOUS
            "sales": [100.0, 200.0],
            "region": ["East", "West"],
        }
    )
    tbl = db.register_dataframe(df, "ambig_table")
    return profile_table(db, tbl)


# --- 1. Total sales in 2017 ---


def test_parse_total_sales_in_2017(superstore_profile: TableProfile):
    """Requirement 1: 'What were the total sales in 2017?' -> SUM(sales) with 2017 date filter."""
    parser = IntentParser()
    spec = parser.parse("What were the total sales in 2017?", "superstore", superstore_profile)

    assert spec is not None
    assert spec.table_name == "superstore"
    assert len(spec.metrics) == 1
    m = spec.metrics[0]
    assert m.aggregation == AggregationType.SUM
    assert m.column == "sales"
    assert spec.dimensions == []
    assert len(spec.time_constraints) == 1
    tc = spec.time_constraints[0]
    assert tc.column == "order_date"
    assert tc.start_time == "2017-01-01"
    assert tc.end_time == "2018-01-01"
    assert tc.is_half_open is True


# --- 2. Average profit by region ---


def test_parse_average_profit_by_region(superstore_profile: TableProfile):
    """Requirement 2: 'Average profit by region' -> AVG(profit) with region dimension."""
    parser = IntentParser()
    spec = parser.parse("Average profit by region", "superstore", superstore_profile)

    assert spec is not None
    assert len(spec.metrics) == 1
    m = spec.metrics[0]
    assert m.aggregation == AggregationType.AVG
    assert m.column == "profit"
    assert spec.dimensions == ["region"]
    assert spec.time_constraints == []


# --- 3. Count of orders ---


def test_parse_count_of_orders(superstore_profile: TableProfile):
    """Requirement 3: 'Count of orders' -> DISTINCT_COUNT on order_id."""
    parser = IntentParser()
    spec = parser.parse("What is the count of orders?", "superstore", superstore_profile)

    assert spec is not None
    assert len(spec.metrics) == 1
    m = spec.metrics[0]
    assert m.aggregation == AggregationType.DISTINCT_COUNT
    assert m.column == "order_id"


def test_parse_count_of_rows(superstore_profile: TableProfile):
    """Requirement 3b: Generic 'count of records' -> COUNT(*)."""
    parser = IntentParser()
    spec = parser.parse("What is the total count of records?", "superstore", superstore_profile)

    assert spec is not None
    assert len(spec.metrics) == 1
    m = spec.metrics[0]
    assert m.aggregation == AggregationType.COUNT
    assert m.column is None  # Row count


# --- 4. Sales by nonexistent column ---


def test_reject_nonexistent_column(superstore_profile: TableProfile):
    """Requirement 4: Nonexistent columns must be rejected without hallucination."""
    parser = IntentParser()
    # 'nonexistent_dim' is not in superstore schema
    spec = parser.parse("Total sales by nonexistent_dim", "superstore", superstore_profile)
    assert spec is None

    # 'fake_metric' is not in superstore schema
    spec2 = parser.parse("Total fake_metric by region", "superstore", superstore_profile)
    assert spec2 is None


# --- 5. Ambiguous date filter ---


def test_reject_ambiguous_date_filter(ambiguous_date_profile: TableProfile):
    """Requirement 5: Temporal filter on ambiguous date format must be rejected."""
    parser = IntentParser()
    spec = parser.parse("Total sales in 2020", "ambig_table", ambiguous_date_profile)
    assert spec is None


# --- 6. Unsupported aggregation ---


def test_reject_unsupported_aggregation(superstore_profile: TableProfile):
    """Requirement 6: Unsupported aggregations (median, mode) must be rejected."""
    parser = IntentParser()
    # 'median' is not an AggregationType
    spec = parser.parse("What is the median sales by region?", "superstore", superstore_profile)
    assert spec is None


# --- 7. Compare 2016 vs 2017 sales (multi-step decomposition) ---


def test_multi_step_decomposition_independent_specs(superstore_profile: TableProfile):
    """Requirement 7: Multi-step comparison produces independent atomic specifications."""
    class MockClient:
        def generate(self, *args, **kwargs):
            return json.dumps({
                "steps": [
                    {
                        "step_id": 1,
                        "question": "What is the total sales in 2016?",
                        "target_table": "superstore",
                        "depends_on": [],
                    },
                    {
                        "step_id": 2,
                        "question": "What is the total sales in 2017?",
                        "target_table": "superstore",
                        "depends_on": [],
                    },
                ]
            })

    planner = QueryPlanner(client=MockClient())
    plan = planner.plan(
        "Compare 2016 vs 2017 sales",
        "superstore",
        profile=superstore_profile,
    )

    # Question is decomposed into distinct atomic steps
    assert plan.is_multi_step is True
    assert len(plan.steps) == 2

    # Step 1: 2016 sales
    s1 = plan.steps[0]
    assert "2016" in s1.question
    assert s1.structured_intent is not None
    assert s1.structured_intent.metrics[0].aggregation == AggregationType.SUM
    assert s1.structured_intent.metrics[0].column == "sales"
    assert len(s1.structured_intent.time_constraints) == 1
    assert s1.structured_intent.time_constraints[0].start_time == "2016-01-01"

    # Step 2: 2017 sales
    s2 = plan.steps[1]
    assert "2017" in s2.question
    assert s2.structured_intent is not None
    assert s2.structured_intent.metrics[0].aggregation == AggregationType.SUM
    assert s2.structured_intent.metrics[0].column == "sales"
    assert len(s2.structured_intent.time_constraints) == 1
    assert s2.structured_intent.time_constraints[0].start_time == "2017-01-01"


# --- 8. Ambiguous analytical question ---


def test_ambiguous_question_returns_none(superstore_profile: TableProfile):
    """Requirement 8: Ambiguous non-analytical questions return None (downstream INCONCLUSIVE)."""
    parser = IntentParser()
    assert parser.parse("Why did the store close?", "superstore", superstore_profile) is None
    assert parser.parse("Tell me a story about customers", "superstore", superstore_profile) is None


# --- 9. AOV calculation contracts ---


def test_aov_intent_contracts(superstore_profile: TableProfile):
    """Requirement 9: AOV binds numerator (sales) and denominator (order_id)."""
    parser = IntentParser()
    spec = parser.parse("What is the average order value?", "superstore", superstore_profile)

    assert spec is not None
    assert len(spec.metrics) == 1
    m = spec.metrics[0]
    assert m.aggregation == AggregationType.AOV
    assert m.column == "sales"
    assert m.denominator_column == "order_id"


def test_aov_invalid_customer_id_denominator_rejected(superstore_profile: TableProfile):
    """AOV with customer_id as denominator is invalid and must be rejected."""
    parser = IntentParser()
    spec = QuerySpecification(
        raw_question="Average order value per customer",
        table_name="superstore",
        metrics=[
            MetricSpec(
                name="aov",
                aggregation=AggregationType.AOV,
                column="sales",
                denominator_column="customer_id",
            )
        ],
    )
    # Validation gate must reject customer_id for AOV
    assert parser._validate_intent(spec, superstore_profile) is None


def test_aov_missing_denominator_rejected():
    """AOV on a table without an order/transaction ID must be rejected."""
    db = SessionDatabase()
    # Table with sales but no ID column
    df = pd.DataFrame({"sales": [10.0, 20.0], "category": ["A", "B"]})
    tbl = db.register_dataframe(df, "no_id_table")
    prof = profile_table(db, tbl)

    parser = IntentParser()
    spec = parser.parse("What is the average order value?", "no_id_table", prof)
    assert spec is None

    # Also directly verify _validate_intent rejects when no order ID exists
    raw_spec = QuerySpecification(
        raw_question="Average order value",
        table_name="no_id_table",
        metrics=[
            MetricSpec(
                name="aov",
                aggregation=AggregationType.AOV,
                column="sales",
            )
        ],
    )
    assert parser._validate_intent(raw_spec, prof) is None


def test_aov_nonexistent_denominator_rejected(superstore_profile: TableProfile):
    """AOV with nonexistent denominator column must be rejected."""
    parser = IntentParser()
    raw_spec = QuerySpecification(
        raw_question="AOV",
        table_name="superstore",
        metrics=[
            MetricSpec(
                name="aov",
                aggregation=AggregationType.AOV,
                column="sales",
                denominator_column="nonexistent_id",
            )
        ],
    )
    assert parser._validate_intent(raw_spec, superstore_profile) is None


# --- 10. Aggregation Ambiguity Tests ---


def test_ambiguity_how_many_orders_were_placed(superstore_profile: TableProfile):
    """'How many orders were placed?' -> DISTINCT_COUNT on order_id."""
    parser = IntentParser()
    spec = parser.parse("How many orders were placed?", "superstore", superstore_profile)
    assert spec is not None
    assert len(spec.metrics) == 1
    m = spec.metrics[0]
    assert m.aggregation == AggregationType.DISTINCT_COUNT
    assert m.column == "order_id"


def test_ambiguity_how_many_rows_are_in_dataset(superstore_profile: TableProfile):
    """'How many rows are in the dataset?' -> COUNT(*) row count."""
    parser = IntentParser()
    spec = parser.parse("How many rows are in the dataset?", "superstore", superstore_profile)
    assert spec is not None
    assert len(spec.metrics) == 1
    m = spec.metrics[0]
    assert m.aggregation == AggregationType.COUNT
    assert m.column is None


def test_ambiguity_how_many_customers_placed_orders(superstore_profile: TableProfile):
    """'How many customers placed orders?' -> DISTINCT_COUNT on customer_id, not order_id."""
    parser = IntentParser()
    spec = parser.parse("How many customers placed orders?", "superstore", superstore_profile)
    assert spec is not None
    assert len(spec.metrics) == 1
    m = spec.metrics[0]
    assert m.aggregation == AggregationType.DISTINCT_COUNT
    assert m.column == "customer_id"


def test_ambiguity_total_number_of_sales_returns_none(superstore_profile: TableProfile):
    """'What is the total number of sales?' mixes total and number of -> returns None."""
    parser = IntentParser()
    spec = parser.parse("What is the total number of sales?", "superstore", superstore_profile)
    assert spec is None, "Ambiguous query must not silently guess"


# --- 11. Heuristic confidence: No false aggregation inference ---


def test_heuristic_gate_rejects_missing_aggregation_keyword(superstore_profile: TableProfile):
    """Requirement 11: Merely mentioning 'sales' without aggregation keyword must NOT infer SUM."""
    parser = IntentParser()
    # No aggregation keyword like 'total', 'sum', 'average'
    spec = parser.parse("sales in 2017", "superstore", superstore_profile)
    # Fast path rejects; without LLM client, returns None
    assert spec is None


# --- 12. CUSTOM formula validation ---


def test_custom_formula_safety_gate(superstore_profile: TableProfile):
    """CUSTOM formula must be a verified safe AST over existing columns."""
    parser = IntentParser()

    # Valid formula
    raw_spec = QuerySpecification(
        raw_question="Profit margin",
        table_name="superstore",
        metrics=[
            MetricSpec(
                name="profit_margin",
                aggregation=AggregationType.CUSTOM,
                formula="SUM(profit) / SUM(sales)",
            )
        ],
    )
    validated = parser._validate_intent(raw_spec, superstore_profile)
    assert validated is not None

    # Invalid formula: unsafe function call
    bad_spec = QuerySpecification(
        raw_question="Bad formula",
        table_name="superstore",
        metrics=[
            MetricSpec(
                name="bad",
                aggregation=AggregationType.CUSTOM,
                formula="SUM(profit) + __import__('os').system('ls')",
            )
        ],
    )
    assert parser._validate_intent(bad_spec, superstore_profile) is None

    # Invalid formula: nonexistent column
    fake_col_spec = QuerySpecification(
        raw_question="Bad col",
        table_name="superstore",
        metrics=[
            MetricSpec(
                name="bad",
                aggregation=AggregationType.CUSTOM,
                formula="SUM(fake_col) / SUM(sales)",
            )
        ],
    )
    assert parser._validate_intent(fake_col_spec, superstore_profile) is None


# --- 13. Direct LLM Validation-Gate Tests ---


class MockLLMClient:
    """Mock client returning specific JSON responses for testing the LLM extraction path."""

    def __init__(self, response_json: str):
        self.response_json = response_json

    def generate(self, *args, **kwargs) -> str:
        return self.response_json


def test_llm_path_rejects_nonexistent_column(superstore_profile: TableProfile):
    """LLM extraction returning a nonexistent column is rejected by the validation gate."""
    mock_response = json.dumps({
        "table_name": "superstore",
        "metrics": [{"name": "fake", "aggregation": "SUM", "column": "nonexistent_col"}],
    })
    parser = IntentParser(client=MockLLMClient(mock_response))
    # Direct question that skips heuristics
    assert parser.parse("Analyze nonexistent metrics", "superstore", superstore_profile) is None


def test_llm_path_rejects_unsupported_aggregation(superstore_profile: TableProfile):
    """LLM extraction returning an unsupported aggregation (e.g. MEDIAN) is rejected."""
    mock_response = json.dumps({
        "table_name": "superstore",
        "metrics": [{"name": "med", "aggregation": "MEDIAN", "column": "sales"}],
    })
    parser = IntentParser(client=MockLLMClient(mock_response))
    assert parser.parse("Analyze median sales", "superstore", superstore_profile) is None


def test_llm_path_rejects_ambiguous_date_constraints(ambiguous_date_profile: TableProfile):
    """LLM extraction targeting an ambiguous date column is rejected."""
    mock_response = json.dumps({
        "table_name": "ambig_table",
        "metrics": [{"name": "total_sales", "aggregation": "SUM", "column": "sales"}],
        "time_constraints": [{
            "column": "order_date",
            "start_time": "2020-01-01",
            "end_time": "2021-01-01",
            "is_half_open": True,
        }],
    })
    parser = IntentParser(client=MockLLMClient(mock_response))
    res = parser.parse("Sales during fiscal year 2020", "ambig_table", ambiguous_date_profile)
    assert res is None


def test_llm_path_rejects_invalid_filter_operators(superstore_profile: TableProfile):
    """LLM extraction with invalid/injection filter operators is rejected."""
    mock_response = json.dumps({
        "table_name": "superstore",
        "metrics": [{"name": "total_sales", "aggregation": "SUM", "column": "sales"}],
        "filters": [{"column": "category", "operator": "DROP TABLE", "value": "Furniture"}],
    })
    parser = IntentParser(client=MockLLMClient(mock_response))
    assert parser.parse("Sales for furniture category", "superstore", superstore_profile) is None


def test_llm_path_rejects_unsafe_custom_formulas(superstore_profile: TableProfile):
    """LLM extraction with unsafe custom formula is rejected."""
    mock_response = json.dumps({
        "table_name": "superstore",
        "metrics": [{
            "name": "custom_metric",
            "aggregation": "CUSTOM",
            "formula": "SUM(profit) + eval('__import__(\"os\").system(\"ls\")')",
        }],
    })
    parser = IntentParser(client=MockLLMClient(mock_response))
    assert parser.parse("Evaluate custom formula", "superstore", superstore_profile) is None


def test_llm_path_valid_spec_passes_gate(superstore_profile: TableProfile):
    """Valid LLM extraction passes the common validation gate and returns typed spec."""
    mock_response = json.dumps({
        "table_name": "superstore",
        "metrics": [{"name": "avg_profit", "aggregation": "AVG", "column": "profit"}],
        "dimensions": ["category"],
        "filters": [{"column": "region", "operator": "=", "value": "West"}],
        "time_constraints": [{
            "column": "order_date",
            "start_time": "2017-01-01",
            "end_time": "2018-01-01",
            "is_half_open": True,
        }],
    })
    parser = IntentParser(client=MockLLMClient(mock_response))
    q = "Filter and analyze: profit by category for West in 2017"
    spec = parser.parse(q, "superstore", superstore_profile)
    assert spec is not None
    assert spec.table_name == "superstore"
    assert spec.metrics[0].aggregation == AggregationType.AVG
    assert spec.metrics[0].column == "profit"
    assert spec.dimensions == ["category"]
    assert len(spec.filters) == 1
    assert spec.filters[0].column == "region"
    assert spec.filters[0].value == "West"
    assert len(spec.time_constraints) == 1
    assert spec.time_constraints[0].start_time == "2017-01-01"
