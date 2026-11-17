"""Analytical planning, multi-step query decomposition, and topological dependency ordering.

Architectural Context (ADR-03, Master Execution Plan Phase 4):
    Complex analytical inquiries (e.g., "Compare 2016 vs 2017 sales and identify which
    region had the largest growth") cannot be solved reliably in a single monolithic
    SQL query without fragility.

    This module provides:
    1. `PlanStep`: A discrete, atomic analytical operation with clear table targets,
       structured intent, and explicit dependencies on preceding steps.
    2. `AnalyticalPlan`: An end-to-end plan representation containing the original question,
       ordered steps, and verified topological execution sequence.
    3. `QueryPlanner`: Analytical engine that classifies single-step vs multi-part questions,
       decomposes multi-part questions via local LLM or heuristic rules, validates dependency
       integrity, and enforces acyclic dependency graphs (DAGs).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from src.agent.data.profiling import TableProfile, format_profile_for_prompt
from src.agent.planning.intent import QuerySpecification

if TYPE_CHECKING:
    from src.agent.codegen.generator import OllamaClient


class PlanStep(BaseModel):
    """Represents an atomic, discrete step within an analytical execution plan."""

    step_id: int = Field(description="Unique positive integer identifying this step")
    question: str = Field(description="Atomic analytical question/sub-goal for this step")
    target_table: str = Field(description="Registered DuckDB table name to query")
    structured_intent: QuerySpecification | None = Field(
        default=None,
        description="Optional structured intent for this step",
    )
    depends_on: list[int] = Field(
        default_factory=list,
        description="List of step_ids that must complete before this step can execute",
    )


class AnalyticalPlan(BaseModel):
    """Complete analytical plan encompassing atomic steps and topological execution order."""

    original_question: str = Field(description="Original user analytical question")
    steps: list[PlanStep] = Field(description="List of analytical steps")
    execution_order: list[int] = Field(
        description="Topologically sorted step_ids guaranteeing dependency satisfaction",
    )
    is_multi_step: bool = Field(
        description="True if the plan consists of multiple dependent or independent steps",
    )

    def get_step(self, step_id: int) -> PlanStep | None:
        """Retrieve a specific step by its unique ID."""
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    def get_dependencies(self, step_id: int) -> list[PlanStep]:
        """Return the list of PlanStep objects that the specified step depends upon."""
        target = self.get_step(step_id)
        if not target:
            return []
        step_map = {s.step_id: s for s in self.steps}
        return [step_map[dep] for dep in target.depends_on if dep in step_map]


def compute_execution_order(steps: list[PlanStep]) -> list[int]:
    """Compute a valid topological execution order using Kahn's algorithm.

    Validates:
        1. All step IDs are unique.
        2. All `depends_on` IDs reference existing steps.
        3. No circular dependencies (cycles) exist in the step dependency graph.

    Args:
        steps: List of PlanStep instances to order.

    Returns:
        List of step_id integers in valid topological execution order.

    Raises:
        ValueError: If a step references an unknown dependency or a cycle is detected.
    """
    if not steps:
        return []

    step_ids = {s.step_id for s in steps}
    if len(step_ids) != len(steps):
        raise ValueError("Plan contains duplicate step IDs.")

    # Build adjacency list: dependency -> list of dependent step_ids
    # and compute in-degrees: step_id -> number of unfinished prerequisites
    adj: dict[int, list[int]] = {s.step_id: [] for s in steps}
    in_degree: dict[int, int] = {s.step_id: 0 for s in steps}

    for step in steps:
        for dep in step.depends_on:
            if dep not in step_ids:
                raise ValueError(
                    f"Step {step.step_id} references non-existent dependency step {dep}."
                )
            if dep == step.step_id:
                raise ValueError(f"Step {step.step_id} cannot depend on itself.")
            adj[dep].append(step.step_id)
            in_degree[step.step_id] += 1

    # Queue of steps with in-degree 0 (ready to execute)
    # Sort initial items to guarantee deterministic ordering among independent steps
    queue = sorted([s_id for s_id, deg in in_degree.items() if deg == 0])
    order: list[int] = []

    while queue:
        curr = queue.pop(0)
        order.append(curr)

        # For all steps depending on `curr`, decrement their in-degree
        for neighbor in sorted(adj[curr]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If order does not contain all steps, a cycle must exist
    if len(order) != len(steps):
        unresolved = [s_id for s_id, deg in in_degree.items() if deg > 0]
        raise ValueError(
            f"Circular dependency detected in analytical plan involving steps: {unresolved}."
        )

    return order


PLANNER_PROMPT_TEMPLATE = """You are an expert analytical query planner.
Your task is to analyze a natural language data question and decide whether it requires a single
analytical query or must be decomposed into multiple atomic sub-steps.

DATASET SCHEMA:
{schema_markdown}

USER QUESTION:
"{question}"

TARGET TABLE:
`{table_name}`

DECOMPOSITION RULES:
1. SINGLE-STEP: If the user asks for a single aggregate (e.g., "Total sales"), a grouped summary
   (e.g., "Sales by region"), a single-condition filter, or top-N query, keep it as 1 step:
   - "is_multi_step": false, 1 step in "steps".
2. MULTI-STEP COMPARISON: If the user asks to COMPARE two different periods, categories, or metrics
   (e.g., "Compare 2016 vs 2017 sales", "Compare East and West sales"), or asks a multi-stage
   question (e.g. "Find top region, then find its products"), DECOMPOSE into discrete steps:
   - For comparisons between A and B, produce Step 1 for A, Step 2 for B.
   - Set "is_multi_step": true.
   - Steps that can run independently should have "depends_on": [].
   - Steps that require outputs from earlier steps should list their step_id in "depends_on".
3. INDEPENDENT RANKINGS / MULTI-OBJECTIVE:
   If the user asks for multiple independent rankings or metrics
   (e.g., "Which product has highest sales and which has highest profit?"),
   DECOMPOSE into separate independent steps:
   - Step 1: Find the entity with the highest/lowest metric A (e.g., SUM(sales)).
   - Step 2: Find the entity with the highest/lowest metric B (e.g., SUM(profit)).
   - Both steps must be INDEPENDENT: "depends_on": [] for both.
   - Do NOT combine them into a single query using one metric as a secondary sort criterion.

DATASET & GRANULARITY AWARENESS:
- Examine the available column names in the provided schema.
- If the question asks for "product", inspect the available columns:
  - If a column like `product_name` exists, use it.
  - If `product_name` is absent but `sub_category` exists, query `sub_category`.
- NEVER invent column names that do not appear in the schema.

JSON FORMAT SPECIFICATION:
{{
  "is_multi_step": true,
  "steps": [
    {{
      "step_id": 1,
      "question": "First atomic question/sub-goal",
      "target_table": "{table_name}",
      "depends_on": []
    }},
    {{
      "step_id": 2,
      "question": "Second atomic question/sub-goal",
      "target_table": "{table_name}",
      "depends_on": []
    }}
  ]
}}

Output ONLY the JSON object. Do not include markdown preamble, explanations, or fences.
"""


class QueryPlanner:
    """Plans and decomposes analytical questions into executable steps."""

    def __init__(self, client: OllamaClient | None = None):
        """Initialize the query planner with an optional Ollama LLM client."""
        self.client = client

    def plan(
        self,
        question: str,
        active_table: str,
        profile: TableProfile | None = None,
    ) -> AnalyticalPlan:
        """Generate a structured analytical plan for the user's question.

        Args:
            question: User's analytical question.
            active_table: Registered DuckDB table name.
            profile: Optional TableProfile containing table statistics and schema.

        Returns:
            Validated `AnalyticalPlan` with verified topological execution order.

        Raises:
            ValueError: If the question is empty or dependencies are invalid/circular.
        """
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")

        # Step 1: Pre-classification heuristic
        # If question is clearly simple, avoid LLM latency and generate 1-step plan immediately
        if self._is_definitely_single_step(clean_question):
            step = PlanStep(
                step_id=1,
                question=clean_question,
                target_table=active_table,
                structured_intent=None,
                depends_on=[],
            )
            return AnalyticalPlan(
                original_question=clean_question,
                steps=[step],
                execution_order=[1],
                is_multi_step=False,
            )

        # Step 2: If LLM client is available and schema profile is provided, attempt LLM planning
        if self.client and profile:
            try:
                return self._plan_with_llm(clean_question, active_table, profile)
            except Exception:
                # Fall back to heuristic decomposition or single-step plan
                pass

        # Step 3: Heuristic multi-step decomposition for independent rankings / comparisons
        heuristic_plan = self._detect_independent_rankings(clean_question, active_table, profile)
        if heuristic_plan:
            return heuristic_plan

        # Step 4: Default fallback: treat as single-step plan
        step = PlanStep(
            step_id=1,
            question=clean_question,
            target_table=active_table,
            structured_intent=None,
            depends_on=[],
        )
        return AnalyticalPlan(
            original_question=clean_question,
            steps=[step],
            execution_order=[1],
            is_multi_step=False,
        )

    def _is_definitely_single_step(self, question: str) -> bool:
        """Heuristic check to fast-track straightforward single-step queries."""
        q_lower = question.lower()
        multi_step_triggers = [
            "compare",
            " vs ",
            "versus",
            "growth between",
            "difference between",
            "then find",
            "and also find",
            "after calculating",
            "contributing factor",
            "why did",
            " and which ",
            " and what ",
            " and who ",
            "can be the same",
            "same or different",
            "both",
            "as well as which",
        ]
        if any(trigger in q_lower for trigger in multi_step_triggers):
            return False

        # Multiple ranking superlatives with conjunctions (e.g. "highest sales ... highest profit")
        superlatives = ["highest", "lowest", "top", "bottom", "most", "least", "best", "worst"]
        found_sup = [s for s in superlatives if s in q_lower]
        if len(found_sup) >= 2 and (" and " in q_lower or " as well as " in q_lower):
            return False

        return True

    def _detect_independent_rankings(
        self,
        question: str,
        active_table: str,
        profile: TableProfile | None = None,
    ) -> AnalyticalPlan | None:
        """Decompose questions asking for multiple independent rankings into distinct steps."""
        q_lower = question.lower()
        has_sales_rank = any(k in q_lower for k in ["highest sales", "top sales", "most sales"])
        has_profit_rank = any(k in q_lower for k in ["highest profit", "top profit", "most profit"])

        if not (has_sales_rank and has_profit_rank):
            # Check conjunction of sales and profit with independent ranking phrases
            has_sales_profit = "sales" in q_lower and "profit" in q_lower
            has_conjunction = any(
                c in q_lower
                for c in [" and which ", "same or different", "can be the same", "both"]
            )
            if not (has_sales_profit and has_conjunction):
                return None

        # Resolve entity name dynamically based on schema profile
        entity = "product"
        if profile and profile.columns:
            cols = profile.columns.keys()
            if "product_name" in cols:
                entity = "product_name"
            elif "sub_category" in cols:
                entity = "sub_category"
            else:
                for c, p in profile.columns.items():
                    if any(t in p.duckdb_type.upper() for t in ["VARCHAR", "TEXT", "STRING"]):
                        entity = c
                        break

        step1 = PlanStep(
            step_id=1,
            question=f"Find the {entity} with the highest SUM(sales)",
            target_table=active_table,
            depends_on=[],
        )
        step2 = PlanStep(
            step_id=2,
            question=f"Find the {entity} with the highest SUM(profit)",
            target_table=active_table,
            depends_on=[],
        )
        return AnalyticalPlan(
            original_question=question,
            steps=[step1, step2],
            execution_order=[1, 2],
            is_multi_step=True,
        )

    def _plan_with_llm(
        self,
        question: str,
        active_table: str,
        profile: TableProfile,
    ) -> AnalyticalPlan:
        """Use local LLM to decompose question into structured steps."""
        schema_markdown = format_profile_for_prompt(profile, max_token_budget=1000)
        prompt = PLANNER_PROMPT_TEMPLATE.format(
            schema_markdown=schema_markdown,
            question=question,
            table_name=active_table,
        )

        response_text = self.client.generate(prompt=prompt, temperature=0.0)

        # Extract JSON from potential code block
        json_match = re.search(r"(\{[\s\S]*\})", response_text)
        if not json_match:
            raise ValueError("No JSON object found in planner response.")

        parsed_json: dict[str, Any] = json.loads(json_match.group(1))
        raw_steps = parsed_json.get("steps", [])

        if not raw_steps or not isinstance(raw_steps, list):
            raise ValueError("Malformed planner output: 'steps' list missing or empty.")

        steps: list[PlanStep] = []
        for s in raw_steps:
            steps.append(
                PlanStep(
                    step_id=int(s["step_id"]),
                    question=str(s.get("question", question)),
                    target_table=str(s.get("target_table", active_table)),
                    depends_on=[int(d) for d in s.get("depends_on", [])],
                )
            )

        execution_order = compute_execution_order(steps)
        is_multi = len(steps) > 1

        return AnalyticalPlan(
            original_question=question,
            steps=steps,
            execution_order=execution_order,
            is_multi_step=is_multi,
        )
