"""LangGraph 7-node state machine and workflow graph.

Architectural Context (ADR-01, ADR-03, Master Execution Plan Phase 4):
    Orchestrates the analytical lifecycle across 7 architectural nodes:
    1. Planner: Decomposes question into single or multi-step AnalyticalPlan.
    2. Code Generator: Synthesizes DuckDB SQL for the active plan step with prior step context.
    3. Executor: Executes SQL with thread-isolated timeout and allowlist security.
    4. Result Assembler: Consolidates multi-step intermediate outputs.
    5. Validator: Pass-through structural placeholder in Phase 4 (Layer 1-3 validation in Phase 5).
    6. Error Analyzer: Structural placeholder diagnosing execution failure (Phase 5).
    7. Corrector: Structural placeholder for iterative self-correction (Phase 5).
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agent.codegen.generator import CodeGenerator
from src.agent.data.ingestion import SessionDatabase
from src.agent.execution.engine import ControlledExecutionEngine
from src.agent.planning.assembler import ResultAssembler
from src.agent.planning.planner import QueryPlanner
from src.agent.workflow.state import AgentState


def build_analytical_graph(
    db: SessionDatabase,
    generator: CodeGenerator | None = None,
    planner: QueryPlanner | None = None,
    assembler: ResultAssembler | None = None,
    default_timeout_seconds: float = 5.0,
):
    """Compile and return the 7-node LangGraph analytical workflow.

    Args:
        db: Active SessionDatabase instance.
        generator: Optional CodeGenerator instance.
        planner: Optional QueryPlanner instance.
        assembler: Optional ResultAssembler instance.
        default_timeout_seconds: Query execution timeout limit.

    Returns:
        Compiled LangGraph Pregel executable graph.
    """
    _planner = planner or QueryPlanner()
    _generator = generator or CodeGenerator()
    _assembler = assembler or ResultAssembler()
    _engine = ControlledExecutionEngine(db, default_timeout_seconds=default_timeout_seconds)

    # Node 1: Planner
    def planner_node(state: AgentState) -> dict[str, Any]:
        question = state["question"]
        active_table = state["active_table"]
        profile = state.get("schema_profile")
        plan = _planner.plan(question=question, active_table=active_table, profile=profile)
        return {
            "plan": plan,
            "current_step_index": 0,
            "step_results": {},
            "step_sql": {},
            "retry_count": state.get("retry_count", 0),
            "max_retries": state.get("max_retries", 2),
            "status": "executing",
        }

    # Node 2: Code Generator
    def codegen_node(state: AgentState) -> dict[str, Any]:
        plan = state["plan"]
        if not plan or not plan.execution_order:
            raise ValueError("Cannot generate code: analytical plan is missing or empty.")

        step_idx = state.get("current_step_index", 0)
        step_id = plan.execution_order[step_idx]
        current_step = plan.get_step(step_id)

        step_question = current_step.question if current_step else state["question"]
        profile = state.get("schema_profile")
        intent = current_step.structured_intent if current_step else None
        feedback = state.get("error_feedback")

        # Contextual awareness: Provide summary of prior step outputs to subsequent steps
        step_results = state.get("step_results", {})
        prior_context_parts = []
        for prev_id, prev_res in step_results.items():
            prev_step = plan.get_step(prev_id)
            desc = prev_step.question if prev_step else f"Step {prev_id}"
            if prev_res.is_scalar:
                prior_context_parts.append(
                    f"- Step {prev_id} ('{desc}'): result = {prev_res.scalar_value}"
                )
            elif prev_res.dataframe is not None:
                prior_context_parts.append(
                    f"- Step {prev_id} ('{desc}'): returned {prev_res.row_count} rows"
                )
        prior_context = "\n".join(prior_context_parts) if prior_context_parts else None

        sql = _generator.generate_sql(
            question=step_question,
            profile=profile,
            intent=intent,
            error_feedback=feedback,
            prior_step_context=prior_context,
        )

        step_sql = dict(state.get("step_sql", {}))
        step_sql[step_id] = sql

        return {
            "generated_code": sql,
            "step_sql": step_sql,
        }

    # Node 3: Executor
    def executor_node(state: AgentState) -> dict[str, Any]:
        sql = state.get("generated_code", "")
        res = _engine.execute(sql)

        plan = state["plan"]
        step_idx = state.get("current_step_index", 0)
        step_id = plan.execution_order[step_idx] if plan and plan.execution_order else 1

        new_step_results = dict(state.get("step_results", {}))
        new_step_results[step_id] = res

        return {
            "execution_result": res,
            "step_results": new_step_results,
        }

    # Multi-step iterator node
    def step_advance_node(state: AgentState) -> dict[str, Any]:
        return {
            "current_step_index": state.get("current_step_index", 0) + 1,
            "error_feedback": None,
        }

    # Node 4: Result Assembler
    def assembler_node(state: AgentState) -> dict[str, Any]:
        plan = state["plan"]
        step_results = state.get("step_results", {})
        assembled = _assembler.assemble(plan=plan, step_results=step_results)
        return {
            "assembled_result": assembled,
            "status": "assembled",
        }

    # Node 5: Validator (Phase 4 structural placeholder; 3-layer validation in Phase 5)
    def validator_node(state: AgentState) -> dict[str, Any]:
        assembled = state.get("assembled_result")
        all_passed = assembled.all_steps_succeeded if assembled else False
        return {
            "status": "validated" if all_passed else "validation_failed",
            "validation_report": {"execution_passed": all_passed},
        }

    # Node 6: Error Analyzer (Phase 4 structural placeholder; error taxonomy in Phase 5)
    def error_analyzer_node(state: AgentState) -> dict[str, Any]:
        res = state.get("execution_result")
        err_msg = res.error_message if res else "Execution failure"
        return {
            "error_feedback": err_msg,
            "status": "execution_failed",
        }

    # Node 7: Corrector (Phase 4 structural placeholder; active self-correction loop in Phase 5)
    def corrector_node(state: AgentState) -> dict[str, Any]:
        return {
            "status": "correction_deferred_to_phase_5",
        }

    # Conditional Routing Functions
    def route_after_executor(state: AgentState) -> str:
        res = state.get("execution_result")
        if not res or not res.success:
            return "error_analyzer"

        plan = state.get("plan")
        step_idx = state.get("current_step_index", 0)
        # Advance through all steps in topological execution order
        if plan and step_idx + 1 < len(plan.execution_order):
            return "next_step"

        return "assembler"

    # Construct StateGraph
    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("codegen", codegen_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("step_advance", step_advance_node)
    workflow.add_node("assembler", assembler_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("error_analyzer", error_analyzer_node)
    workflow.add_node("corrector", corrector_node)

    # Define Edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "codegen")
    workflow.add_edge("codegen", "executor")

    # Routing from Executor
    workflow.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "next_step": "step_advance",
            "assembler": "assembler",
            "error_analyzer": "error_analyzer",
        },
    )

    # Multi-step loop back to codegen
    workflow.add_edge("step_advance", "codegen")

    # Assembler to Validator
    workflow.add_edge("assembler", "validator")
    workflow.add_edge("validator", END)

    # Error handling boundary for Phase 4: Structural placeholders terminate gracefully
    workflow.add_edge("error_analyzer", "corrector")
    workflow.add_edge("corrector", END)

    return workflow.compile()
