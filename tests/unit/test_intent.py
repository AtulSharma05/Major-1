"""Unit tests for structured query intent modeling."""

from src.agent.planning.intent import (
    AggregationType,
    FilterSpec,
    MetricSpec,
    QuerySpecification,
    TimeConstraint,
)


def test_query_specification_scalar():
    """Verify scalar query detection (single metric, no dimensions)."""
    spec = QuerySpecification(
        raw_question="What is the total sales?",
        table_name="superstore",
        metrics=[
            MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales"),
        ],
    )
    assert spec.is_scalar() is True
    assert spec.is_grouped() is False


def test_query_specification_grouped():
    """Verify grouped query detection (dimensions present)."""
    spec = QuerySpecification(
        raw_question="What is total sales by region?",
        table_name="superstore",
        metrics=[
            MetricSpec(name="total_sales", aggregation=AggregationType.SUM, column="sales"),
        ],
        dimensions=["region"],
    )
    assert spec.is_scalar() is False
    assert spec.is_grouped() is True


def test_query_specification_filters_and_time():
    """Verify serialization of filters and half-open time constraints."""
    time_c = TimeConstraint(
        column="order_date",
        start_time="2017-01-01",
        end_time="2018-01-01",
        is_half_open=True,
    )
    filter_s = FilterSpec(column="region", operator="=", value="West")

    spec = QuerySpecification(
        raw_question="What was West region sales in 2017?",
        table_name="superstore",
        metrics=[MetricSpec(name="sales", aggregation=AggregationType.SUM, column="sales")],
        filters=[filter_s],
        time_constraints=[time_c],
    )

    json_str = spec.model_dump_json()
    assert "superstore" in json_str
    assert "2017-01-01" in json_str

    deserialized = QuerySpecification.model_validate_json(json_str)
    assert deserialized.table_name == "superstore"
    assert deserialized.time_constraints[0].is_half_open is True

