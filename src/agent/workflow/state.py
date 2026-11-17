"""State definition for the 7-node LangGraph analytical state machine.

Architectural Context (ADR-01, Master Execution Plan Phase 4):
    Maintains the complete lifecycle of a conversational analytical request:
    from raw natural language query to planning, code generation, controlled execution,
    result assembly, validation, and error correction.
"""

from __future__ import annotations

from typing import Any, TypedDict

from src.agent.data.profiling import TableProfile
from src.agent.execution.engine import ExecutionResult
from src.agent.planning.assembler import AssembledResult
from src.agent.planning.planner import AnalyticalPlan


class AgentState(TypedDict, total=False):
    """Complete shared state propagated across the LangGraph analytical workflow.

    Fields:
        question: Original user natural language question.
        session_id: Active session identifier.
        active_table: Active registered DuckDB table name.
        schema_profile: Data profile and statistics of the active table.
        plan: Decomposed analytical plan with execution order.
        current_step_index: Index in plan.execution_order currently being executed.
        generated_code: Most recently generated DuckDB SQL string.
        step_sql: Mapping of step_id -> generated SQL string across all plan steps.
        execution_result: ExecutionResult of the most recently executed step.
        step_results: Consolidated dictionary mapping step_id -> ExecutionResult.
        assembled_result: Consolidated multi-step output from ResultAssembler.
        validation_report: Three-layer validation results (Phase 5).
        error_feedback: Structured error diagnostic message for code correction.
        retry_count: Current correction retry iteration count.
        max_retries: Maximum permitted correction retries (default 2).
        status: Current workflow status string ('planning', 'executing', 'assembled',
                'validated', 'failed').
    """

    question: str
    session_id: str
    active_table: str
    schema_profile: TableProfile | None
    plan: AnalyticalPlan | None
    current_step_index: int
    generated_code: str | None
    step_sql: dict[int, str]
    execution_result: ExecutionResult | None
    step_results: dict[int, ExecutionResult]
    assembled_result: AssembledResult | None
    validation_report: Any | None
    error_feedback: str | None
    retry_count: int
    max_retries: int
    status: str

