"""Unit tests for analytical planning, multi-step decomposition, and topological ordering."""

import pytest

from src.agent.planning.planner import (
    AnalyticalPlan,
    PlanStep,
    QueryPlanner,
    compute_execution_order,
)


def test_empty_question_handling():
    """Verify that empty or whitespace questions are rejected."""
    planner = QueryPlanner()
    with pytest.raises(ValueError, match="Question cannot be empty"):
        planner.plan("", active_table="superstore")

    with pytest.raises(ValueError, match="Question cannot be empty"):
        planner.plan("   ", active_table="superstore")


def test_single_step_planning():
    """Verify fast-tracking of straightforward single-step queries."""
    planner = QueryPlanner()
    question = "What is the total sales for West region?"
    plan = planner.plan(question, active_table="superstore")

    assert plan.original_question == question
    assert plan.is_multi_step is False
    assert len(plan.steps) == 1
    assert plan.execution_order == [1]
    assert plan.steps[0].step_id == 1
    assert plan.steps[0].target_table == "superstore"
    assert plan.steps[0].depends_on == []


def test_compute_execution_order_two_independent_steps():
    """Verify topological ordering of two independent analytical steps."""
    step1 = PlanStep(
        step_id=1,
        question="Total sales in 2016",
        target_table="superstore",
        depends_on=[],
    )
    step2 = PlanStep(
        step_id=2,
        question="Total sales in 2017",
        target_table="superstore",
        depends_on=[],
    )

    order = compute_execution_order([step1, step2])
    assert order == [1, 2]

    plan = AnalyticalPlan(
        original_question="Compare 2016 and 2017 sales",
        steps=[step1, step2],
        execution_order=order,
        is_multi_step=True,
    )
    assert plan.is_multi_step is True
    assert plan.get_step(1) == step1
    assert plan.get_step(2) == step2


def test_compute_execution_order_sequential_dependent_steps():
    """Verify topological ordering of sequentially dependent steps (Step 2 depends on Step 1)."""
    step1 = PlanStep(
        step_id=1,
        question="Calculate regional sales",
        target_table="superstore",
        depends_on=[],
    )
    step2 = PlanStep(
        step_id=2,
        question="Find region with highest growth",
        target_table="superstore",
        depends_on=[1],
    )

    order = compute_execution_order([step2, step1])  # Given out of order
    assert order == [1, 2]

    plan = AnalyticalPlan(
        original_question="Regional sales and highest growth",
        steps=[step1, step2],
        execution_order=order,
        is_multi_step=True,
    )
    deps = plan.get_dependencies(2)
    assert len(deps) == 1
    assert deps[0].step_id == 1


def test_compute_execution_order_multiple_dependencies():
    """Verify topological ordering when a step depends on multiple preceding steps (Diamond DAG)."""
    step1 = PlanStep(
        step_id=1,
        question="Compute 2016 sales",
        target_table="superstore",
        depends_on=[],
    )
    step2 = PlanStep(
        step_id=2,
        question="Compute 2017 sales",
        target_table="superstore",
        depends_on=[],
    )
    step3 = PlanStep(
        step_id=3,
        question="Compute growth percentage",
        target_table="superstore",
        depends_on=[1, 2],
    )

    order = compute_execution_order([step3, step2, step1])
    assert order.index(1) < order.index(3)
    assert order.index(2) < order.index(3)
    assert order[-1] == 3


def test_compute_execution_order_invalid_dependency_reference():
    """Verify rejection of dependency referencing non-existent step_id."""
    step1 = PlanStep(
        step_id=1,
        question="Compute sales",
        target_table="superstore",
        depends_on=[99],
    )

    with pytest.raises(ValueError, match="non-existent dependency step 99"):
        compute_execution_order([step1])


def test_compute_execution_order_self_dependency():
    """Verify rejection when a step depends on itself."""
    step1 = PlanStep(
        step_id=1,
        question="Compute sales",
        target_table="superstore",
        depends_on=[1],
    )

    with pytest.raises(ValueError, match="cannot depend on itself"):
        compute_execution_order([step1])


def test_compute_execution_order_circular_dependency():
    """Verify cycle detection (Step 1 -> Step 2 -> Step 1)."""
    step1 = PlanStep(step_id=1, question="Part A", target_table="superstore", depends_on=[2])
    step2 = PlanStep(step_id=2, question="Part B", target_table="superstore", depends_on=[1])

    with pytest.raises(ValueError, match="Circular dependency detected"):
        compute_execution_order([step1, step2])


def test_compute_execution_order_duplicate_step_ids():
    """Verify rejection when duplicate step_ids exist."""
    step1a = PlanStep(step_id=1, question="Step 1A", target_table="superstore", depends_on=[])
    step1b = PlanStep(step_id=1, question="Step 1B", target_table="superstore", depends_on=[])

    with pytest.raises(ValueError, match="duplicate step IDs"):
        compute_execution_order([step1a, step1b])


def test_malformed_llm_planner_output_fallback():
    """Verify that malformed LLM responses safely fall back to single-step plan."""
    class MockOllamaClient:
        def generate(self, *args, **kwargs):
            return "This is not valid JSON at all! Just random commentary."

    planner = QueryPlanner(client=MockOllamaClient())
    question = "Compare 2016 vs 2017 sales"
    plan = planner.plan(question, active_table="superstore")

    # Should gracefully fall back to a 1-step plan rather than crashing
    assert plan.is_multi_step is False
    assert len(plan.steps) == 1
    assert plan.original_question == question


def test_plan_independent_ranking_objectives(data_dir):
    """Verify that independent ranking questions decompose into two independent steps."""
    from src.agent.data.ingestion import SessionDatabase
    from src.agent.data.profiling import profile_table

    db = SessionDatabase()
    db.register_csv(data_dir / "sample_superstore.csv", "sample_superstore")
    db.register_csv(data_dir / "sample_superstore_2.csv", "sample_superstore_2")

    p1 = profile_table(db, "sample_superstore")
    p2 = profile_table(db, "sample_superstore_2")

    planner = QueryPlanner()
    question = (
        "Which product has the highest sales and which product has the highest profit? "
        "They can be the same or different."
    )

    # Test 1: sample_superstore (no product_name -> maps to sub_category)
    plan1 = planner.plan(question, active_table="sample_superstore", profile=p1)
    assert plan1.is_multi_step is True
    assert len(plan1.steps) == 2
    assert plan1.steps[0].depends_on == []
    assert plan1.steps[1].depends_on == []
    assert "sub_category" in plan1.steps[0].question
    assert "sub_category" in plan1.steps[1].question
    assert "sales" in plan1.steps[0].question.lower()
    assert "profit" in plan1.steps[1].question.lower()

    # Test 2: sample_superstore_2 (has product_name -> maps to product_name)
    plan2 = planner.plan(question, active_table="sample_superstore_2", profile=p2)
    assert plan2.is_multi_step is True
    assert len(plan2.steps) == 2
    assert plan2.steps[0].depends_on == []
    assert plan2.steps[1].depends_on == []
    assert "product_name" in plan2.steps[0].question
    assert "product_name" in plan2.steps[1].question
    assert "sales" in plan2.steps[0].question.lower()
    assert "profit" in plan2.steps[1].question.lower()

    db.close()
