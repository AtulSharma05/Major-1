"""Lightweight analytical result assembler for multi-step query execution.

Architectural Context (ADR-03, Master Execution Plan Phase 4):
    In a multi-step analytical plan, individual execution steps yield distinct
    `ExecutionResult` objects (scalar numbers or tabular DataFrames).

    The `ResultAssembler`:
    1. For single-step plans: transparently passes through the primary `ExecutionResult`.
    2. For multi-step scalar comparisons (e.g. 2016 sales vs 2017 sales): merges scalar values
       into a unified comparison summary table and calculates variance/growth if applicable.
    3. For multi-step tabular outputs sharing categorical keys (e.g. regional sales):
       performs a key-aligned join to present side-by-side analytical comparisons.
    4. For heterogeneous/unaligned outputs: aggregates step results into an accessible structured
       container without complex custom DAG frameworks.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.agent.execution.engine import ExecutionResult
from src.agent.planning.planner import AnalyticalPlan


class AssembledResult(BaseModel):
    """Unified analytical result combining outputs from all plan steps."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    original_question: str
    is_multi_step: bool
    all_steps_succeeded: bool
    step_results: dict[int, ExecutionResult] = Field(default_factory=dict)
    primary_dataframe: pd.DataFrame | None = None
    primary_scalar: Any = None
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    assembly_notes: list[str] = Field(default_factory=list)

    @property
    def step_outputs(self) -> dict[int, ExecutionResult]:
        """Backward-compatibility property alias returning step_results."""
        return self.step_results


class ResultAssembler:
    """Combines intermediate execution step results into an AssembledResult."""

    def assemble(
        self,
        plan: AnalyticalPlan,
        step_results: dict[int, ExecutionResult],
    ) -> AssembledResult:
        """Assemble step execution results according to the analytical plan.

        Args:
            plan: The active AnalyticalPlan.
            step_results: Mapping of step_id -> ExecutionResult.

        Returns:
            AssembledResult containing consolidated data and summary metrics.
        """
        all_succeeded = all(r.success for r in step_results.values()) and len(step_results) == len(
            plan.steps
        )

        # Case 1: Single-Step Plan (Pass-Through)
        if not plan.is_multi_step:
            step_id = plan.steps[0].step_id if plan.steps else 1
            res = step_results.get(step_id)
            if res is None:
                return AssembledResult(
                    original_question=plan.original_question,
                    is_multi_step=False,
                    all_steps_succeeded=False,
                    step_results=step_results,
                    assembly_notes=["Step result missing from execution."],
                )

            step_obj = plan.steps[0] if plan.steps else None
            intent = step_obj.structured_intent if step_obj else None
            if intent and (not intent.is_grounded or intent.unsupported_entities):
                unsupp_str = (
                    ", ".join(intent.unsupported_entities)
                    if intent.unsupported_entities
                    else "unsupported entity"
                )
                return AssembledResult(
                    original_question=plan.original_question,
                    is_multi_step=False,
                    all_steps_succeeded=False,
                    step_results=step_results,
                    primary_dataframe=None,
                    primary_scalar=None,
                    summary_metrics={},
                    assembly_notes=[
                        f"Calculation inconclusive: requested entity '{unsupp_str}' "
                        "is absent from dataset schema."
                    ],
                )

            return AssembledResult(
                original_question=plan.original_question,
                is_multi_step=False,
                all_steps_succeeded=res.success,
                step_results=step_results,
                primary_dataframe=res.dataframe,
                primary_scalar=res.scalar_value,
                summary_metrics={"result": res.scalar_value} if res.is_scalar else {},
                assembly_notes=["Single-step query pass-through."],
            )

        # Case 2: Multi-Step Plan with Execution Failures
        if not all_succeeded:
            failed_steps = [s_id for s_id, r in step_results.items() if not r.success]
            missing_steps = [s.step_id for s in plan.steps if s.step_id not in step_results]
            notes = []
            if failed_steps:
                notes.append(f"Steps failed: {failed_steps}")
            if missing_steps:
                notes.append(f"Steps not executed: {missing_steps}")

            return AssembledResult(
                original_question=plan.original_question,
                is_multi_step=True,
                all_steps_succeeded=False,
                step_results=step_results,
                assembly_notes=notes,
            )

        # Case 3: All steps are scalar results (Comparison / Summary)
        all_scalar = all(r.is_scalar for r in step_results.values())
        if all_scalar:
            rows = []
            summary_metrics: dict[str, Any] = {}
            for step_id in plan.execution_order:
                step = plan.get_step(step_id)
                res = step_results[step_id]
                desc = step.question if step else f"Step {step_id}"
                val = res.scalar_value
                rows.append({"step_id": step_id, "description": desc, "value": val})
                summary_metrics[f"step_{step_id}_value"] = val

            summary_df = pd.DataFrame(rows)

            # If exactly 2 scalar steps (e.g. 2016 vs 2017), calculate variance
            if len(rows) == 2 and all(isinstance(r["value"], (int, float)) for r in rows):
                v1 = float(rows[0]["value"])
                v2 = float(rows[1]["value"])
                diff = v2 - v1
                summary_metrics["absolute_difference"] = round(diff, 4)
                if v1 != 0.0:
                    summary_metrics["percentage_change"] = round((diff / v1) * 100.0, 2)
                else:
                    summary_metrics["percentage_change"] = None

            return AssembledResult(
                original_question=plan.original_question,
                is_multi_step=True,
                all_steps_succeeded=True,
                step_results=step_results,
                primary_dataframe=summary_df,
                summary_metrics=summary_metrics,
                assembly_notes=["Multi-step scalar comparison assembled."],
            )

        # Case 4: Tabular Steps (Check for shared categorical keys or independent rankings)
        tabular_steps = [
            (s_id, step_results[s_id].dataframe)
            for s_id in plan.execution_order
            if step_results[s_id].dataframe is not None and not step_results[s_id].is_scalar
        ]

        if len(tabular_steps) >= 2:
            # Check for common column keys across tables
            first_cols = set(tabular_steps[0][1].columns)
            common_keys = first_cols.intersection(*(set(df.columns) for _, df in tabular_steps[1:]))

            # Look for non-numeric categorical key (e.g., 'sub_category', 'product_name')
            def _is_categorical(col_name: str, df: pd.DataFrame) -> bool:
                dtype_str = str(df[col_name].dtype)
                return df[col_name].dtype == "object" or "str" in dtype_str

            join_keys = [k for k in common_keys if _is_categorical(k, tabular_steps[0][1])]

            summary_metrics: dict[str, Any] = {}
            for s_id, df in tabular_steps:
                if df is not None and not df.empty:
                    first_row = df.iloc[0]
                    cat_cols = [
                        c
                        for c in df.columns
                        if df[c].dtype == "object" or "str" in str(df[c].dtype)
                    ]
                    num_cols = [c for c in df.columns if c not in cat_cols]
                    winner_val = str(first_row[cat_cols[0]]) if cat_cols else "N/A"
                    summary_metrics[f"step_{s_id}_winner"] = winner_val
                    if num_cols:
                        metric_name = num_cols[0]
                        summary_metrics[f"step_{s_id}_{metric_name}"] = first_row[metric_name]

            notes: list[str] = []
            if join_keys:
                key = join_keys[0]
                merged_df = tabular_steps[0][1].copy()
                step_obj_0 = plan.get_step(tabular_steps[0][0])
                suffix_0 = f"_{step_obj_0.step_id}" if step_obj_0 else "_1"

                # Rename non-key columns with step suffixes
                rename_map_0 = {c: f"{c}{suffix_0}" for c in merged_df.columns if c != key}
                merged_df = merged_df.rename(columns=rename_map_0)

                for s_id, next_df in tabular_steps[1:]:
                    step_obj = plan.get_step(s_id)
                    suffix = f"_{step_obj.step_id}" if step_obj else f"_{s_id}"
                    df_copy = next_df.copy()
                    rename_map = {c: f"{c}{suffix}" for c in df_copy.columns if c != key}
                    df_copy = df_copy.rename(columns=rename_map)
                    merged_df = pd.merge(merged_df, df_copy, on=key, how="outer")

                notes.append(f"Tabular steps merged on shared key '{key}'.")
            else:
                # Retain both answers in a unified independent ranking table
                ranking_rows = []
                for s_id, df in tabular_steps:
                    s_obj = plan.get_step(s_id)
                    s_goal = s_obj.question if s_obj else f"Step {s_id}"
                    if df is not None and not df.empty:
                        first_row = df.iloc[0]
                        cat_cols = [
                            c
                            for c in df.columns
                            if df[c].dtype == "object" or "str" in str(df[c].dtype)
                        ]
                        num_cols = [c for c in df.columns if c not in cat_cols]
                        ent_val = str(first_row[cat_cols[0]]) if cat_cols else "N/A"
                        met_val = first_row[num_cols[0]] if num_cols else "N/A"
                        ranking_rows.append(
                            {
                                "step_id": s_id,
                                "objective": s_goal,
                                "top_entity": ent_val,
                                "value": met_val,
                            }
                        )
                merged_df = pd.DataFrame(ranking_rows)
                notes.append("Independent ranking results assembled.")

            # Granularity disclosure if sub_category was used when product was requested
            for s in plan.steps:
                is_subcat = "sub_category" in s.question.lower()
                is_prod_q = "product" in plan.original_question.lower()
                if is_subcat and is_prod_q:
                    notes.append(
                        "Note: Dataset does not contain 'product_name'; "
                        "analysis was performed at 'sub_category' granularity."
                    )
                    break

            return AssembledResult(
                original_question=plan.original_question,
                is_multi_step=True,
                all_steps_succeeded=True,
                step_results=step_results,
                primary_dataframe=merged_df,
                summary_metrics=summary_metrics,
                assembly_notes=notes,
            )

        # Case 5: Heterogeneous or mixed steps (Last step or primary tabular as default)
        last_step_id = plan.execution_order[-1]
        last_res = step_results[last_step_id]

        return AssembledResult(
            original_question=plan.original_question,
            is_multi_step=True,
            all_steps_succeeded=True,
            step_results=step_results,
            primary_dataframe=last_res.dataframe,
            primary_scalar=last_res.scalar_value,
            assembly_notes=["Multi-step outputs assembled into structured container."],
        )
