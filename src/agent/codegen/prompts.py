"""Prompt engineering and template generation for SQL synthesis with Qwen2.5-Coder 7B."""

from __future__ import annotations

from src.agent.data.profiling import TableProfile, format_profile_for_prompt
from src.agent.planning.intent import QuerySpecification

SYSTEM_PROMPT = """You are an expert DuckDB SQL data analytics assistant.
Your task is to generate clean, executable DuckDB SQL queries based strictly on the provided schema.

CRITICAL RULES:
1. DIALECT: Output standard DuckDB SQL syntax only.
2. TABLE & COLUMNS: Query ONLY the specified registered table. Use ONLY the exact column names
   present in the schema. Do not hallucinate columns.
3. IDENTIFIERS: All column names in the schema are lowercase snake_case. Do not wrap them in
   unnecessary brackets or quotes unless required.
4. AGGREGATIONS:
   - For total sum, use SUM(col).
   - For distinct count, use COUNT(DISTINCT col).
   - For row count, use COUNT(*).
   - For Average Order Value (AOV), use: SUM(sales) / NULLIF(COUNT(DISTINCT order_id), 0).
5. TIMESTAMPS & DATES:
   - For full calendar year filters on timestamps, use half-open intervals:
     col >= 'YYYY-01-01' AND col < 'YYYY+1-01-01'
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
) -> str:
    """Construct a schema-injected prompt for the LLM.

    Args:
        question: The user's natural language question.
        profile: TableProfile containing the schema and statistics.
        intent: Optional structured QuerySpecification.
        error_feedback: Optional error message from a previous failed execution attempt.

    Returns:
        The formatted user prompt string.
    """
    schema_markdown = format_profile_for_prompt(profile, max_token_budget=1200)

    prompt = f"""### Dataset Context
{schema_markdown}

### Analytical Question
"{question}"
"""

    if intent:
        metric_desc = ", ".join(
            f"{m.aggregation.value}({m.column or '*'}) AS {m.name}"
            for m in intent.metrics
        )
        prompt += f"""
### Structured Intent
- Target Table: `{intent.table_name}`
- Metrics: {metric_desc or 'Not specified'}
- Group By Dimensions: {intent.dimensions or 'None (Scalar metric)'}
- Filters: {[f"{f.column} {f.operator} {f.value}" for f in intent.filters] or 'None'}
- Limit: {intent.limit or 'None'}
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
