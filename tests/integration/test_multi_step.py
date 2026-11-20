"""Live integration tests for Phase 4: Multi-step query planning & LangGraph.

Verifies end-to-end multi-step planning, iterative execution loops, and result assembly
using local Ollama with Qwen2.5-Coder 7B on canonical multi-part questions.
"""

from __future__ import annotations

import httpx
import pytest

from src.agent.codegen.generator import CodeGenerator, OllamaClient
from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import profile_table
from src.agent.planning.planner import QueryPlanner
from src.agent.workflow.graph import build_analytical_graph
from src.agent.workflow.state import AgentState


@pytest.fixture(scope="module")
def ollama_available():
    """Verify local Ollama daemon is reachable on localhost:11434 with qwen2.5-coder."""
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
def multi_step_env(data_dir):
    """Setup SessionDatabase with canonical Superstore dataset and Ollama components."""
    db = SessionDatabase()
    db.register_csv(data_dir / "sample_superstore.csv", "superstore")
    profile = profile_table(db, "superstore")
    client = OllamaClient(model="qwen2.5-coder:7b", timeout=30.0)
    generator = CodeGenerator(client=client)
    planner = QueryPlanner(client=client)
    graph = build_analytical_graph(db=db, generator=generator, planner=planner)
    yield db, profile, planner, graph
    db.close()


def test_multi_step_planner_decomposition(ollama_available, multi_step_env):
    """Verify that QueryPlanner decomposes a multi-part comparison question into sub-steps."""
    _, profile, planner, _ = multi_step_env
    question = "Compare total sales in 2016 versus 2017"

    plan = planner.plan(question=question, active_table="superstore", profile=profile)

    assert plan.original_question == question
    assert plan.is_multi_step is True
    assert len(plan.steps) >= 2
    assert len(plan.execution_order) == len(plan.steps)
    # Ensure step IDs are valid
    for s_id in plan.execution_order:
        assert plan.get_step(s_id) is not None


def test_multi_step_workflow_execution(ollama_available, multi_step_env):
    """Verify end-to-end multi-step execution loop through LangGraph with result assembly."""
    _, profile, _, graph = multi_step_env
    question = "Compare total sales in 2016 versus 2017"

    initial_state: AgentState = {
        "question": question,
        "session_id": "test_multi_live_1",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] in ("inconclusive", "execution_verified")
    assert final_state["plan"] is not None
    assert final_state["plan"].is_multi_step is True
    assert len(final_state["step_results"]) >= 2

    # Verify each step executed successfully
    for step_id, res in final_state["step_results"].items():
        assert res.success is True, f"Step {step_id} failed: {res.error_message}"
        assert res.dataframe is not None

    # Verify assembled result
    assembled = final_state["assembled_result"]
    assert assembled is not None
    assert assembled.all_steps_succeeded is True
    assert assembled.is_multi_step is True
    assert assembled.primary_dataframe is not None


def test_single_step_through_langgraph_workflow(ollama_available, multi_step_env):
    """Verify that single-step questions follow the exact same LangGraph workflow."""
    _, profile, _, graph = multi_step_env
    question = "What is the total sales for each region?"

    initial_state: AgentState = {
        "question": question,
        "session_id": "test_single_live_1",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] in ("inconclusive", "execution_verified")
    assert final_state["plan"].is_multi_step is False
    assert len(final_state["step_results"]) == 1

    assembled = final_state["assembled_result"]
    assert assembled is not None
    assert assembled.all_steps_succeeded is True
    assert assembled.primary_dataframe is not None
    assert "region" in [c.lower() for c in assembled.primary_dataframe.columns]


def test_independent_ranking_live_decomposition_and_workflow(ollama_available, multi_step_env):
    """Verify live Ollama decomposition and LangGraph execution of independent ranking question."""
    _, profile, _, graph = multi_step_env
    question = (
        "Which product has the highest sales and which product has the highest profit? "
        "They can be the same or different."
    )

    initial_state: AgentState = {
        "question": question,
        "session_id": "test_ranking_live_1",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] in ("inconclusive", "execution_verified")
    plan = final_state["plan"]
    assert plan is not None
    assert plan.is_multi_step is True
    assert len(plan.steps) == 2

    # Both steps must be independent
    assert plan.steps[0].depends_on == []
    assert plan.steps[1].depends_on == []

    # Both steps must execute successfully
    step_results = final_state["step_results"]
    assert len(step_results) == 2
    for s_id, res in step_results.items():
        assert res.success is True, f"Step {s_id} failed: {res.error_message}"

    # Assembled result must retain answers
    assembled = final_state["assembled_result"]
    assert assembled is not None
    assert assembled.all_steps_succeeded is True
    assert assembled.primary_dataframe is not None
    assert "step_1_winner" in assembled.summary_metrics
    assert "step_2_winner" in assembled.summary_metrics
