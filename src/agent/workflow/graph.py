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

from collections.abc import Sequence
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agent.codegen.generator import CodeGenerator
from src.agent.data.ingestion import SessionDatabase
from src.agent.execution.engine import ControlledExecutionEngine
from src.agent.planning.assembler import ResultAssembler
from src.agent.planning.planner import QueryPlanner
from src.agent.validation.contracts import (
    ConfidenceLevel,
    StepValidationResult,
    ValidationStatus,
    WorkflowValidationReport,
    build_step_validation_result,
)
from src.agent.validation.execution import ExecutionValidator
from src.agent.validation.numerical import NumericalValidator
from src.agent.validation.semantic import SemanticValidator
from src.agent.workflow.state import AgentState


def format_correction_feedback(
    step_val: StepValidationResult,
    prior_attempts: Sequence[StepValidationResult],
    retry_count: int,
    max_retries: int,
) -> str:
    """Format structured diagnostic feedback for code generator self-correction."""
    attempt_num = len(prior_attempts)
    diag = step_val.diagnostic
    err_cat = diag.error_category.value if diag else "EXECUTION_OR_VALIDATION_ERROR"
    summary = (
        diag.error_summary if diag else (step_val.execution_result.message or "Validation failed")
    )
    remediation = (
        diag.remediation_instruction
        if diag
        else "Correct the query to satisfy dialect, schema, numerical, and semantic requirements."
    )

    failed_checks = []
    if step_val.semantic_result and step_val.semantic_result.checks:
        for c in step_val.semantic_result.checks:
            if not c.passed:
                failed_checks.append(f"{c.check_name}: {c.explanation}")

    failed_checks_block = ""
    if failed_checks:
        failed_checks_block = "\nFailed Semantic Checks:\n" + "\n".join(
            f"- {fc}" for fc in failed_checks
        )

    prior_attempts_block = ""
    if prior_attempts:
        prior_summary_lines = []
        for att in prior_attempts:
            att_err = (
                att.diagnostic.error_summary
                if att.diagnostic
                else (att.execution_result.message or f"Status: {att.status.value}")
            )
            prior_summary_lines.append(
                f"- Attempt {att.attempt_number}: status={att.status.value}, error={att_err}"
            )
        prior_attempts_block = "\nPrevious Attempt History:\n" + "\n".join(prior_summary_lines)

    remaining_budget = max(0, max_retries - retry_count)

    l1_stat = f"{step_val.execution_result.status.value}: {step_val.execution_result.message}"
    l2_stat = f"{step_val.numerical_result.status.value}: {step_val.numerical_result.message}"
    l3_stat = f"{step_val.semantic_result.status.value}: {step_val.semantic_result.message}"

    feedback = f"""=== CORRECTION DIAGNOSTIC (Attempt {attempt_num}) ===
Executed SQL:
{step_val.executed_sql or "N/A"}

Validation Statuses:
- Layer 1 (Execution): {l1_stat}
- Layer 2 (Numerical): {l2_stat}
- Layer 3 (Semantic): {l3_stat}

Primary Diagnosis:
- Error Category: {err_cat}
- Summary: {summary}{failed_checks_block}{prior_attempts_block}

Remediation Instruction:
{remediation}

Retry Budget: {remaining_budget} retries remaining.
Constraint: Preserve original intent; fix only the diagnosed failure."""
    return feedback.strip()


def build_analytical_graph(
    db: SessionDatabase,
    generator: CodeGenerator | None = None,
    planner: QueryPlanner | None = None,
    assembler: ResultAssembler | None = None,
    validator: ExecutionValidator | None = None,
    numerical_validator: NumericalValidator | None = None,
    semantic_validator: SemanticValidator | None = None,
    default_timeout_seconds: float = 5.0,
):
    """Compile and return the 7-node LangGraph analytical workflow.

    Args:
        db: Active SessionDatabase instance.
        generator: Optional CodeGenerator instance.
        planner: Optional QueryPlanner instance.
        assembler: Optional ResultAssembler instance.
        validator: Optional ExecutionValidator instance (Phase 5 Layer 1).
        numerical_validator: Optional NumericalValidator instance (Phase 5 Layer 2).
        semantic_validator: Optional SemanticValidator instance (Phase 5 Layer 3).
        default_timeout_seconds: Query execution timeout limit.

    Returns:
        Compiled LangGraph Pregel executable graph.
    """
    _planner = planner or QueryPlanner()
    _generator = generator or CodeGenerator()
    _assembler = assembler or ResultAssembler()
    _validator = validator or ExecutionValidator()
    _num_validator = numerical_validator or NumericalValidator()
    _sem_validator = semantic_validator
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
        current_step = plan.get_step(step_id) if plan else None

        new_step_results = dict(state.get("step_results", {}))
        new_step_results[step_id] = res

        # Phase 5 Stage 2: Layer 1 Execution Validation
        existing_report = state.get("validation_report")
        if isinstance(existing_report, WorkflowValidationReport):
            val_report = existing_report
        else:
            val_report = WorkflowValidationReport()

        prior_attempts = val_report.step_history.get(step_id, [])
        attempt_number = len(prior_attempts) + 1

        structured_intent = current_step.structured_intent if current_step else None

        # Phase 5 Layer 1: Physical Execution & Shape/Schema Verification
        step_val = _validator.validate_step(
            raw_execution_result=res,
            structured_intent=structured_intent,
            step=current_step,
            step_id=step_id,
            attempt_number=attempt_number,
            executed_sql=sql,
        )

        # Phase 5 Layer 2: Independent Numerical Validation (Stage 3)
        if (
            step_val.numerical_result.status != ValidationStatus.UNVERIFIED
            and "deferred to Stage 3" not in (step_val.numerical_result.message or "")
        ):
            num_res = step_val.numerical_result
        else:
            target_table = current_step.target_table if current_step else state.get("active_table")
            source_df = db.get_table_dataframe(target_table) if target_table else None

            num_res = _num_validator.validate_step(
                step=current_step,
                execution_result=res,
                source_df=source_df,
                table_profile=state.get("schema_profile"),
            )

        # Phase 5 Layer 3: Scope-Aware Semantic Validation (Stage 4)
        if _sem_validator is not None:
            if (
                step_val.semantic_result.status != ValidationStatus.UNVERIFIED
                and "deferred to Stage 4" not in (step_val.semantic_result.message or "")
            ):
                sem_res = step_val.semantic_result
            else:
                sem_res = _sem_validator.validate_step(
                    step=current_step,
                    executed_sql=sql,
                    execution_result=res,
                    table_profile=state.get("schema_profile"),
                    available_tables=db.get_allowed_tables(),
                )
        else:
            sem_res = step_val.semantic_result

        # Combine Layer 1, 2, and 3 validation evidence into unified StepValidationResult
        meaningful = (
            step_val.confidence != ConfidenceLevel.LOW if step_val.confidence is not None else None
        )
        step_val = build_step_validation_result(
            step_id=step_id,
            attempt_number=attempt_number,
            execution_res=step_val.execution_result,
            numerical_res=num_res,
            semantic_res=sem_res,
            diagnostic=step_val.diagnostic,
            executed_sql=sql,
            raw_execution_result=res,
            has_meaningful_output=meaningful,
        )

        val_report.add_step_attempt(step_val)

        return {
            "execution_result": res,
            "step_results": new_step_results,
            "validation_report": val_report,
        }

    # Multi-step iterator node
    def step_advance_node(state: AgentState) -> dict[str, Any]:
        return {
            "current_step_index": state.get("current_step_index", 0) + 1,
            "error_feedback": None,
            "retry_count": 0,
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

    # Node 5: Validator (Consolidates validation report; Layer 1-3 in Phase 5)
    def validator_node(state: AgentState) -> dict[str, Any]:
        assembled = state.get("assembled_result")
        val_report = state.get("validation_report")

        # Option B: Differentiated Stage 2 Execution Status
        # Priority: validation_failed -> inconclusive -> execution_verified -> validated
        assembly_failed = assembled is None or not assembled.all_steps_succeeded
        report_failed = isinstance(val_report, WorkflowValidationReport) and (
            val_report.workflow_status == ValidationStatus.FAILED
            or any(
                s.status == ValidationStatus.FAILED
                or s.execution_result.status == ValidationStatus.FAILED
                or not s.execution_result.passed
                for s in val_report.latest_step_results.values()
            )
        )

        if assembly_failed or report_failed:
            status = "validation_failed"
        elif isinstance(val_report, WorkflowValidationReport) and (
            val_report.workflow_status == ValidationStatus.INCONCLUSIVE
            or any(
                s.status == ValidationStatus.INCONCLUSIVE
                or s.execution_result.status == ValidationStatus.INCONCLUSIVE
                for s in val_report.latest_step_results.values()
            )
        ):
            status = "inconclusive"
        elif (
            isinstance(val_report, WorkflowValidationReport)
            and not val_report.all_steps_verified
            and assembled is not None
            and assembled.all_steps_succeeded
        ):
            status = "execution_verified"
        elif isinstance(val_report, WorkflowValidationReport) and val_report.all_steps_verified:
            status = "validated"
        elif assembled is not None and assembled.all_steps_succeeded:
            status = "execution_verified"
        else:
            status = "validation_failed"

        return {
            "status": status,
            "validation_report": (
                val_report if val_report is not None else {"execution_passed": not assembly_failed}
            ),
        }

    # Node 6: Error Analyzer (Evaluates correction eligibility and synthesizes diagnostic feedback)
    def error_analyzer_node(state: AgentState) -> dict[str, Any]:
        val_report = state.get("validation_report")
        plan = state.get("plan")
        step_idx = state.get("current_step_index", 0)
        step_id = plan.execution_order[step_idx] if plan and plan.execution_order else 1

        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)

        latest_step_val: StepValidationResult | None = None
        if isinstance(val_report, WorkflowValidationReport):
            latest_step_val = val_report.latest_step_results.get(step_id)

        # Evaluate correction eligibility
        is_eligible = False
        if latest_step_val is not None:
            # 1. Inconclusive validation is strictly non-correctable
            if latest_step_val.status == ValidationStatus.INCONCLUSIVE:
                is_eligible = False
            # 2. Strict retry budget enforcement
            elif retry_count >= max_retries:
                is_eligible = False
            # 3. Check diagnostic correctability
            elif latest_step_val.diagnostic is not None:
                is_eligible = latest_step_val.diagnostic.is_correctable
            # 4. If status is FAILED and budget remains
            elif latest_step_val.status == ValidationStatus.FAILED:
                is_eligible = True

        if is_eligible and latest_step_val is not None:
            prior_attempts = val_report.step_history.get(step_id, [])
            feedback = format_correction_feedback(
                step_val=latest_step_val,
                prior_attempts=prior_attempts,
                retry_count=retry_count,
                max_retries=max_retries,
            )
            return {
                "error_feedback": feedback,
                "status": "diagnosed_for_correction",
            }
        else:
            if retry_count >= max_retries:
                summary = (
                    f"Correction retry budget exhausted ({retry_count}/{max_retries} retries). "
                    "Halting correction."
                )
            elif latest_step_val and latest_step_val.status == ValidationStatus.INCONCLUSIVE:
                inc_reason = (
                    latest_step_val.diagnostic.error_summary
                    if latest_step_val.diagnostic
                    else "Ambiguous data or unprovable intent"
                )
                summary = f"Validation is INCONCLUSIVE: {inc_reason}. Retries not permitted."
            elif (
                latest_step_val
                and latest_step_val.diagnostic
                and not latest_step_val.diagnostic.is_correctable
            ):
                summary = f"Non-correctable failure: {latest_step_val.diagnostic.error_summary}."
            else:
                res = state.get("execution_result")
                summary = (
                    (res.error_message if res and res.error_message else None)
                    or state.get("error_feedback")
                    or "Execution failure"
                )

            return {
                "error_feedback": summary,
                "status": "correction_ineligible",
            }

    # Node 7: Corrector (Synthesizes corrected query preserving original analytical intent)
    def corrector_node(state: AgentState) -> dict[str, Any]:
        plan = state["plan"]
        step_idx = state.get("current_step_index", 0)
        step_id = plan.execution_order[step_idx] if plan and plan.execution_order else 1
        current_step = plan.get_step(step_id) if plan else None

        step_question = current_step.question if current_step else state["question"]
        profile = state.get("schema_profile")
        # Ground truth analytical intent is preserved completely unmodified
        intent = current_step.structured_intent if current_step else None
        feedback = state.get("error_feedback")

        # Contextual awareness: summary of prior step outputs for multi-step queries
        step_results = state.get("step_results", {})
        prior_context_parts = []
        for prev_id, prev_res in step_results.items():
            if prev_id == step_id:
                continue
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

        current_retry = state.get("retry_count", 0)
        next_retry = current_retry + 1

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
            "retry_count": next_retry,
            "status": "correcting",
        }

    # Conditional Routing Functions
    def route_after_executor(state: AgentState) -> str:
        plan = state.get("plan")
        step_idx = state.get("current_step_index", 0)
        step_id = plan.execution_order[step_idx] if plan and plan.execution_order else 1

        val_report = state.get("validation_report")
        if (
            isinstance(val_report, WorkflowValidationReport)
            and step_id in val_report.latest_step_results
        ):
            step_val = val_report.latest_step_results[step_id]
            route, blocking_reason = _validator.evaluate_routing(
                plan=plan,
                step_id=step_id,
                step_validation=step_val,
                current_step_index=step_idx,
            )
            if blocking_reason:
                state["error_feedback"] = blocking_reason
            return route

        res = state.get("execution_result")
        if not res or not res.success:
            return "error_analyzer"

        # Advance through all steps in topological execution order
        if plan and step_idx + 1 < len(plan.execution_order):
            return "next_step"

        return "assembler"

    def route_after_error_analyzer(state: AgentState) -> str:
        if state.get("status") == "diagnosed_for_correction":
            return "corrector"
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

    # Phase 5 Stage 5 Bounded Self-Correction Loop
    workflow.add_conditional_edges(
        "error_analyzer",
        route_after_error_analyzer,
        {
            "corrector": "corrector",
            "assembler": "assembler",
        },
    )
    workflow.add_edge("corrector", "executor")

    return workflow.compile()
