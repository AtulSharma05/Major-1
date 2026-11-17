"""Unit tests for LangGraph state machine, 7-node workflow, and execution routing."""

import pytest

from src.agent.codegen.generator import CodeGenerator
from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import profile_table
from src.agent.planning.planner import AnalyticalPlan, PlanStep, QueryPlanner
from src.agent.workflow.graph import build_analytical_graph
from src.agent.workflow.state import AgentState


class MockCodeGenerator(CodeGenerator):
    """Deterministic generator for unit testing workflow routing."""

    def __init__(self, sql_map: dict[str, str] | None = None):
        self.sql_map = sql_map or {}
        self.call_count = 0

    def generate_sql(
        self,
        question: str,
        profile=None,
        intent=None,
        error_feedback=None,
        prior_step_context=None,
        **kwargs,
    ) -> str:
        self.call_count += 1
        if question in self.sql_map:
            return self.sql_map[question]
        return "SELECT SUM(sales) AS total_sales FROM superstore"


@pytest.fixture
def workflow_setup(data_dir):
    """Fixture providing initialized database and profile."""
    db = SessionDatabase()
    db.register_csv(data_dir / "sample_superstore.csv", "superstore")
    profile = profile_table(db, "superstore")
    yield db, profile
    db.close()


def test_workflow_compilation(workflow_setup):
    """Verify that the 7-node LangGraph workflow compiles without schema errors."""
    db, _ = workflow_setup
    graph = build_analytical_graph(db=db)
    assert graph is not None


def test_workflow_single_step_execution(workflow_setup):
    """Verify single-step flow: Planner -> CodeGen -> Executor -> Assembler -> Validator."""
    db, profile = workflow_setup
    mock_gen = MockCodeGenerator({
        "What is total sales?": "SELECT SUM(sales) AS total_sales FROM superstore",
    })
    graph = build_analytical_graph(db=db, generator=mock_gen)

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_session_1",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "validated"
    assert final_state["plan"] is not None
    assert final_state["plan"].is_multi_step is False
    assert final_state["assembled_result"] is not None
    assert final_state["assembled_result"].all_steps_succeeded is True
    assert final_state["assembled_result"].primary_scalar is not None
    assert final_state["assembled_result"].primary_scalar > 0


def test_workflow_multi_step_loop_execution(workflow_setup):
    """Verify multi-step planning loop through multiple steps before reaching Assembler."""
    db, profile = workflow_setup

    step1 = PlanStep(step_id=1, question="Sales 2016", target_table="superstore", depends_on=[])
    step2 = PlanStep(step_id=2, question="Sales 2017", target_table="superstore", depends_on=[])
    mock_plan = AnalyticalPlan(
        original_question="Compare 2016 vs 2017 sales",
        steps=[step1, step2],
        execution_order=[1, 2],
        is_multi_step=True,
    )

    class MockPlanner(QueryPlanner):
        def plan(self, *args, **kwargs):
            return mock_plan

    mock_gen = MockCodeGenerator({
        "Sales 2016": "SELECT SUM(sales) FROM superstore WHERE order_date < '2017-01-01'",
        "Sales 2017": "SELECT SUM(sales) FROM superstore WHERE order_date >= '2017-01-01'",
    })

    graph = build_analytical_graph(db=db, generator=mock_gen, planner=MockPlanner())

    initial_state: AgentState = {
        "question": "Compare 2016 vs 2017 sales",
        "session_id": "test_multi_1",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "validated"
    assert len(final_state["step_results"]) == 2
    assert final_state["assembled_result"] is not None
    assert final_state["assembled_result"].is_multi_step is True
    assert final_state["assembled_result"].all_steps_succeeded is True
    assert len(final_state["assembled_result"].primary_dataframe) == 2


def test_workflow_error_routing_structural_boundary(workflow_setup):
    """Verify routing to Error Analyzer and Corrector structural boundary in Phase 4."""
    db, profile = workflow_setup

    class FailingGenerator(CodeGenerator):
        def generate_sql(self, *args, **kwargs):
            return "SELECT invalid_column FROM superstore"

    graph = build_analytical_graph(db=db, generator=FailingGenerator())

    initial_state: AgentState = {
        "question": "What is total sales?",
        "session_id": "test_fail_1",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    # In Phase 4, error Analyzer & Corrector act as structural placeholders
    assert final_state["status"] == "correction_deferred_to_phase_5"
    assert final_state["execution_result"].success is False
    assert final_state["execution_result"].error_type == "ColumnError"
    assert final_state["error_feedback"] is not None
