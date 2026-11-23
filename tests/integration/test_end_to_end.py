"""End-to-end integration tests for Phase 5 Stage 3.1: Runtime Intent Extraction Integration.

Verifies that:
1. "What were the total sales in 2017?" extracts a grounded QuerySpecification.
2. The structured intent propagates to Layer 1 (execution) and Layer 2 (numerical validation).
3. Both Layer 1 and Layer 2 report VERIFIED with exact/matching numerical values.
4. Ambiguous/unresolvable intent safely returns None, yielding INCONCLUSIVE without crashing.
5. Live Ollama code generation (when available) correctly synthesizes and validates the query.
"""

import httpx
import pytest

from src.agent.codegen.generator import CodeGenerator, OllamaClient
from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import profile_table
from src.agent.planning.intent import AggregationType
from src.agent.planning.planner import QueryPlanner
from src.agent.validation.contracts import ValidationStatus
from src.agent.workflow.graph import build_analytical_graph
from tests.unit.test_workflow import MockCodeGenerator


@pytest.fixture(scope="module")
def ollama_available():
    """Check if local Ollama daemon is reachable on localhost:11434 with qwen2.5-coder."""
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
def superstore_context(data_dir):
    """Setup SessionDatabase with canonical 18-row Superstore dataset.

    Fixture:
        `data_dir` from `tests/conftest.py` points to `<repo_root>/data`.
        Loads `data/sample_superstore.csv` (18 rows) into DuckDB as table 'superstore'.

    Ground Truth for 2017 Sales (Expected Value = 2162.41):
        In `data/sample_superstore.csv`, exactly 12 records have `order_date` in 2017:
        Rows 4-13, 16, 17:
        48.86 + 907.15 + 71.37 + 15.55 + 224.90 + 2.78 + 25.25 + 399.98 + 160.00
        + 22.38 + 272.94 + 11.25 = 2162.41.
    """
    db = SessionDatabase()
    db.register_csv(data_dir / "sample_superstore.csv", "superstore")
    profile = profile_table(db, "superstore")
    yield db, profile
    db.close()


def test_end_to_end_motivating_query_intent_and_validation(superstore_context):
    """Verify motivating query 'What were the total sales in 2017?' achieves Layer 1 & 2 VERIFIED.

    Addresses user request:
    - QuerySpecification was previously unavailable (None), causing Layer 2 INCONCLUSIVE.
    - With Stage 3.1 IntentParser, QueryPlanner produces grounded QuerySpecification.
    - Numerical validator independently calculates 2017 sales in Pandas and verifies DuckDB output.
    """
    db, profile = superstore_context

    question = "What were the total sales in 2017?"

    # 1. Verify planner directly extracts structured_intent
    planner = QueryPlanner()
    plan = planner.plan(question, "superstore", profile)
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.structured_intent is not None, "QuerySpecification must not be None"
    intent = step.structured_intent
    assert intent.table_name == "superstore"
    assert len(intent.metrics) == 1
    assert intent.metrics[0].column == "sales"
    assert intent.metrics[0].aggregation == AggregationType.SUM
    assert len(intent.time_constraints) == 1
    assert intent.time_constraints[0].column == "order_date"
    assert intent.time_constraints[0].start_time == "2017-01-01"
    assert intent.time_constraints[0].end_time == "2018-01-01"

    # 2. Run through full analytical graph
    sql_2017 = (
        "SELECT SUM(sales) AS total_sales "
        "FROM superstore "
        "WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'"
    )
    mock_gen = MockCodeGenerator({question: sql_2017})
    graph = build_analytical_graph(db=db, generator=mock_gen, planner=planner)

    initial_state = {
        "question": question,
        "session_id": "test_e2e_2017_sales",
        "active_table": "superstore",
        "schema_profile": profile,
    }

    final_state = graph.invoke(initial_state)

    # 3. Assert workflow execution and validation status
    assert final_state["status"] == "execution_verified"
    val_report = final_state["validation_report"]
    assert val_report is not None
    assert 1 in val_report.latest_step_results

    step_val = val_report.latest_step_results[1]
    # Layer 1 must be VERIFIED
    assert step_val.execution_result.status == ValidationStatus.VERIFIED
    assert step_val.execution_result.passed is True

    # Layer 2 must be VERIFIED (not INCONCLUSIVE)
    assert step_val.numerical_result.status == ValidationStatus.VERIFIED
    assert step_val.numerical_result.passed is True
    assert step_val.numerical_result.inconclusive_reason is None
    assert step_val.numerical_result.duckdb_value is not None
    assert step_val.numerical_result.independent_value is not None
    assert abs(step_val.numerical_result.duckdb_value - 2162.41) < 0.01
    assert abs(step_val.numerical_result.independent_value - 2162.41) < 0.01


def test_end_to_end_unresolvable_intent_safe_inconclusive(superstore_context):
    """Verify that unresolvable questions fail closed to None and yield INCONCLUSIVE."""
    db, profile = superstore_context
    planner = QueryPlanner()
    unresolvable_question = "What is the vibe of sales?"

    plan = planner.plan(unresolvable_question, "superstore", profile)
    assert len(plan.steps) == 1
    assert plan.steps[0].structured_intent is None, "Ambiguous question must return None"

    mock_sql = "SELECT SUM(sales) AS total_sales FROM superstore"
    mock_gen = MockCodeGenerator({unresolvable_question: mock_sql})
    graph = build_analytical_graph(db=db, generator=mock_gen, planner=planner)

    final_state = graph.invoke({
        "question": unresolvable_question,
        "session_id": "test_e2e_ambiguous",
        "active_table": "superstore",
        "schema_profile": profile,
    })

    assert final_state["status"] == "inconclusive"
    val_report = final_state["validation_report"]
    assert val_report is not None
    step_val = val_report.latest_step_results[1]
    assert step_val.numerical_result.status == ValidationStatus.INCONCLUSIVE
    assert "QuerySpecification unavailable" in (step_val.numerical_result.inconclusive_reason or "")


def test_end_to_end_live_ollama_qwen_execution(ollama_available, superstore_context):
    """Live Ollama test: synthesizes SQL with structured intent and verifies Layers 1 & 2."""
    db, profile = superstore_context
    planner = QueryPlanner()
    question = "What is the total sales?"

    client = OllamaClient(model="qwen2.5-coder:7b", timeout=30.0)
    generator = CodeGenerator(client=client)
    graph = build_analytical_graph(db=db, generator=generator, planner=planner)

    final_state = graph.invoke({
        "question": question,
        "session_id": "test_e2e_ollama_live",
        "active_table": "superstore",
        "schema_profile": profile,
    })

    val_report = final_state.get("validation_report")
    assert val_report is not None
    assert 1 in val_report.latest_step_results
    step_val = val_report.latest_step_results[1]
    assert step_val.execution_result.passed is True
    assert step_val.numerical_result.passed is True
    assert step_val.numerical_result.status == ValidationStatus.VERIFIED
