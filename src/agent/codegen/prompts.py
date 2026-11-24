"""Prompt engineering and template generation for SQL synthesis with Qwen2.5-Coder 7B."""

from __future__ import annotations

from src.agent.data.profiling import TableProfile, format_profile_for_prompt
from src.agent.planning.intent import QuerySpecification

SYSTEM_PROMPT = """You are an expert DuckDB SQL data analytics assistant.
Your task is to generate clean, executable DuckDB SQL queries based strictly on the provided schema.

CRITICAL RULES:
1. DIALECT: Output standard DuckDB SQL syntax only.
2. TABLE & COLUMNS:
   - Query ONLY the specified registered table.
   - Use ONLY the exact column names present in the table schema.
   - NEVER invent, assume, or hallucinate columns (e.g., if there is no `product_name`,
     use `sub_category` or `category`; if there is no generic `date`, use `order_date`).
   - Every column in SELECT, WHERE, GROUP BY, or ORDER BY MUST appear in the schema.
3. IDENTIFIERS: All column names in the schema are lowercase snake_case. Do not wrap them in
   unnecessary brackets or quotes unless required.
4. AGGREGATIONS & RANKINGS:
   - For total sum, use SUM(col).
   - For distinct count, use COUNT(DISTINCT col).
   - For row count, use COUNT(*).
   - For Average Order Value (AOV), use: SUM(sales) / NULLIF(COUNT(DISTINCT order_id), 0).
   - When ranking top/bottom entities on grouped queries (Top N, Bottom N, highest, lowest):
     Compute total per entity: GROUP BY <entity> ORDER BY <metric> <DIR>, <dim> ASC LIMIT N.
     ALWAYS include the primary metric/direction FIRST, and then include ALL grouping
     dimensions in ASC order as deterministic secondary tie-breakers:
     (e.g., `ORDER BY total_sales ASC, sub_category ASC LIMIT 5`).
     Do NOT use another arbitrary metric as a secondary sort criterion.
5. TIMESTAMPS & DATES:
   - For period filters on DATE or TIMESTAMP columns, prefer half-open intervals:
     col >= 'YYYY-01-01' AND col < 'YYYY+1-01-01'
   - Avoid BETWEEN for timestamps as it drops records occurring after 00:00:00 on the final day.
   - For VARCHAR columns containing dates formatted as DD/MM/YYYY:
     Use try_strptime(col, '%d/%m/%Y') and compare against ISO date literals:
     try_strptime(col, '%d/%m/%Y') >= 'YYYY-01-01' AND
     try_strptime(col, '%d/%m/%Y') < 'YYYY+1-01-01'
   - Right-hand side comparison literals MUST always be ISO formatted ('YYYY-MM-DD').
   - For ambiguous date formats, report ambiguity rather than guessing silently.
6. GROUPING:
   - Every column in the SELECT clause that is not an aggregate must appear in the GROUP BY clause.
7. OUTPUT FORMAT:
   - Return ONLY the executable SQL query inside a single ```sql ... ``` code block.
   - Do NOT provide markdown explanations, apologies, conversational remarks, or preamble.
"""


def build_sql_prompt(
    question: str,
    profile: TableProfile,
    intent: QuerySpecification | None = None,
    error_feedback: str | None = None,
    prior_step_context: str | None = None,
) -> str:
    """Construct a schema-injected prompt for the LLM.

    Args:
        question: The user's natural language question.
        profile: TableProfile containing the schema and statistics.
        intent: Optional structured QuerySpecification.
        error_feedback: Optional error message from a previous failed execution attempt.
        prior_step_context: Optional summary of outputs from previous dependent plan steps.

    Returns:
        The formatted user prompt string.
    """
    schema_markdown = format_profile_for_prompt(profile, max_token_budget=1200)
    valid_cols = ", ".join(f"`{c}`" for c in profile.columns.keys())

    cols = profile.columns.keys()
    q_lower = question.lower()
    guidance_notes = []

    # Dynamic entity resolution when question asks for 'product' but column is missing
    if ("product" in q_lower or "item" in q_lower) and "product_name" not in cols:
        if "sub_category" in cols:
            guidance_notes.append(
                "- `product_name` does NOT exist in this table. "
                "Perform product analysis using `sub_category`."
            )
        elif "category" in cols:
            guidance_notes.append(
                "- `product_name` does NOT exist in this table. "
                "Perform product analysis using `category`."
            )

    # Ranking aggregation guidance: ensure SUM is used and deterministic tie-breakers are included
    ranking_triggers = [
        "highest sales",
        "top sales",
        "lowest sales",
        "bottom sales",
        "highest profit",
        "top profit",
        "lowest profit",
        "bottom profit",
        "most sales",
        "least sales",
        "most profit",
        "least profit",
        "top",
        "bottom",
        "highest",
        "lowest",
        "best",
        "worst",
    ]
    if any(k in q_lower for k in ranking_triggers):
        guidance_notes.append(
            "- When ranking or using LIMIT with ORDER BY on grouped queries, ALWAYS include "
            "the primary metric/direction followed by all grouping dimensions in ASC order as "
            "deterministic secondary tie-breakers: `ORDER BY <metric> <DIR>, <dim1> ASC LIMIT N` "
            "(e.g. `ORDER BY total_sales ASC, sub_category ASC LIMIT 5`)."
        )

    # Dynamic temporal column guidance
    for c_name, c_prof in profile.columns.items():
        if c_prof.is_temporal:
            if c_prof.strptime_format:
                guidance_notes.append(
                    f"- Column `{c_name}` stores dates as VARCHAR in `{c_prof.date_format}` "
                    f"format. You MUST use `try_strptime({c_name}, '{c_prof.strptime_format}')` "
                    f"for temporal filtering/ordering, and compare against ISO date literals: "
                    f"`try_strptime({c_name}, '{c_prof.strptime_format}') >= 'YYYY-MM-DD' AND "
                    f"try_strptime({c_name}, '{c_prof.strptime_format}') < 'YYYY-MM-DD'`. "
                    f"Do NOT compare raw string literals or use non-ISO literals!"
                )
            elif c_prof.date_format and "AMBIGUOUS" in c_prof.date_format:
                guidance_notes.append(
                    f"- Column `{c_name}` contains dates in an ambiguous format. "
                    f"Do not guess silently; disclose format ambiguity if filtering on this column."
                )
            else:
                guidance_notes.append(
                    f"- Column `{c_name}` is a {c_prof.duckdb_type} column. "
                    f"For full calendar periods, prefer half-open intervals: "
                    f"`{c_name} >= 'YYYY-01-01' AND {c_name} < 'YYYY+1-01-01'`."
                )

    guidance_block = ""
    if guidance_notes:
        guidance_block = "\nGrounding Guidance:\n" + "\n".join(guidance_notes) + "\n"

    prompt = f"""### Dataset Context
{schema_markdown}

Valid columns in `{profile.table_name}`:
[{valid_cols}]
(You must select only from these valid columns. Do NOT invent columns.){guidance_block}
### Analytical Question
"{question}"
"""

    if prior_step_context:
        prompt += f"""
### Results from Prior Plan Steps
{prior_step_context}
"""

    if intent:
        metric_desc = ", ".join(
            f"{m.aggregation.value}({m.column or '*'}) AS {m.name}" for m in intent.metrics
        )
        time_desc_parts = []
        for tc in intent.time_constraints:
            left_bracket = "["
            right_bracket = ")" if tc.is_half_open else "]"
            time_desc_parts.append(
                f"{tc.column} ({left_bracket}{tc.start_time or '-inf'} .. "
                f"{tc.end_time or '+inf'}{right_bracket})"
            )
        time_desc = ", ".join(time_desc_parts)
        prompt += f"""
### Structured Intent
- Target Table: `{intent.table_name}`
- Metrics: {metric_desc or "Not specified"}
- Group By Dimensions: {intent.dimensions or "None (Scalar metric)"}
- Filters: {[f"{f.column} {f.operator} {f.value}" for f in intent.filters] or "None"}
- Time Constraints: {time_desc or "None"}
- Order By: {intent.order_by or "None"}
- Limit: {intent.limit or "None"}
"""

    if error_feedback:
        prompt += f"""
### Previous Error Feedback (Must Fix)
Your previous attempt failed with the following error:
{error_feedback}

Please carefully correct the query to resolve this error.
"""

    prompt += "\nGenerate the DuckDB SQL query to answer the question:"
    return prompt
