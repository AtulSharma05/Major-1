"""Runtime analytical intent parser and schema grounding engine.

Architectural Context (ADR-02, Phase 5 Stage 3.1):
    Converts natural language analytical questions into validated, typed
    `QuerySpecification` instances grounded against the registered DuckDB
    `TableProfile`.

Key Guarantees:
    1. Strict Schema Grounding: Every metric column, denominator column, grouping
       dimension, and filter column MUST exist in `TableProfile.columns`.
       Hallucinated or non-existent columns are rejected (never silently substituted).
    2. Aggregation Whitelist: Only supported `AggregationType` members are accepted.
       `CUSTOM` formulas are validated against `SafeFormulaEvaluator`.
    3. Strict AOV Contracts: Average Order Value requires grounded numerator (numeric)
       and denominator (distinct entity ID).
    4. Deterministic Heuristic Gate: The heuristic fast path only accepts queries with
       explicit aggregation keywords, grounded numeric columns, and clear boundaries.
       Questions without aggregation keywords (e.g., "sales in 2017") are rejected by heuristics.
    5. Clean Fail-Closed Invariant: If intent extraction fails or encounters ambiguity,
       returns `None`. Execution continues and Layer 2 safely reports `INCONCLUSIVE`.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from src.agent.data.profiling import TableProfile, format_profile_for_prompt
from src.agent.planning.intent import (
    AggregationType,
    FilterSpec,
    MetricSpec,
    QuerySpecification,
    TimeConstraint,
)

if TYPE_CHECKING:
    from src.agent.codegen.generator import OllamaClient


INTENT_PARSER_SYSTEM_PROMPT = """You are an expert analytical intent extractor.
Your task is to parse a natural language business question into a strictly structured JSON
conforming to the QuerySpecification schema.

DATASET SCHEMA:
{schema_markdown}

TARGET TABLE:
`{table_name}`

CRITICAL RULES:
1. COLUMN GROUNDING:
   - Use ONLY valid column names that exist in the provided schema.
   - NEVER invent or hallucinate column names.
   - Do NOT guess columns if they do not exist.
2. AGGREGATIONS:
   - Use only: "SUM", "COUNT", "DISTINCT_COUNT", "AVG", "AOV", "MIN", "MAX", "CUSTOM".
   - For total monetary/numeric volume, use "SUM".
   - For order/transaction count, use "DISTINCT_COUNT" on the ID column (e.g., `order_id`).
   - For row count, use "COUNT" with column=null.
   - For average order value, use "AOV" with column="sales" and denominator_column="order_id".
3. TIME CONSTRAINTS:
   - For calendar year filters (e.g. 2017), use half-open intervals:
     start_time="YYYY-01-01", end_time="YYYY+1-01-01", is_half_open=true.
   - Target the temporal date column present in the schema (e.g., `order_date`).
4. OUTPUT FORMAT:
   - Output ONLY a JSON object matching this schema:
{{
  "table_name": "{table_name}",
  "metrics": [
    {{
      "name": "output_alias",
      "aggregation": "SUM",
      "column": "existing_column_name",
      "denominator_column": null,
      "formula": null
    }}
  ],
  "dimensions": ["group_col"],
  "filters": [
    {{
      "column": "filter_col",
      "operator": "=",
      "value": "filter_val"
    }}
  ],
  "time_constraints": [
    {{
      "column": "date_col",
      "start_time": "2017-01-01",
      "end_time": "2018-01-01",
      "is_half_open": true
    }}
  ],
  "limit": null
}}
Do not output markdown fences, commentary, or conversational text. Output raw JSON only.
"""


class IntentParser:
    """Parses natural-language queries into schema-grounded QuerySpecifications."""

    def __init__(self, client: OllamaClient | None = None) -> None:
        self.client = client

    @staticmethod
    def _get_formula_evaluator():
        from src.agent.validation.numerical import SafeFormulaEvaluator

        return SafeFormulaEvaluator()

    @staticmethod
    def _is_numeric(col_prof: Any) -> bool:
        """Check whether a column is numeric according to its DuckDB type or statistics."""
        numeric_types = (
            "INT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC",
            "HUGEINT", "TINYINT", "SMALLINT", "BIGINT", "REAL",
        )
        dtype = str(getattr(col_prof, "duckdb_type", "")).upper()
        return (
            any(t in dtype for t in numeric_types)
            or getattr(col_prof, "mean_val", None) is not None
        )

    def parse(
        self,
        question: str,
        target_table: str,
        profile: TableProfile,
    ) -> QuerySpecification | None:
        """Parse natural language question into a validated QuerySpecification.

        Evaluation Order:
            1. Fast-Path Heuristic Gate (strict deterministic conditions).
            2. LLM Extraction Path (if heuristics do not match and client is available).
            3. Common Deterministic Schema Validation Gate.

        Returns:
            Validated QuerySpecification, or None if intent cannot be unambiguously established.
        """
        clean_q = question.strip()
        if not clean_q or not profile:
            return None

        # Step 1: Attempt deterministic heuristic extraction
        heuristic_spec = self._parse_heuristics(clean_q, target_table, profile)
        if heuristic_spec is not None:
            validated = self._validate_intent(heuristic_spec, profile)
            if validated is not None:
                return validated

        # Step 2: Attempt LLM extraction if client is available
        if self.client:
            llm_spec = self._parse_with_llm(clean_q, target_table, profile)
            if llm_spec is not None:
                return self._validate_intent(llm_spec, profile)

        # Fail-closed: intent cannot be established
        return None

    def _parse_heuristics(
        self,
        question: str,
        target_table: str,
        profile: TableProfile,
    ) -> QuerySpecification | None:
        """Deterministic heuristic fast path adhering strictly to the 5 acceptance conditions."""
        q_lower = question.lower().strip()

        # Condition 5: Rejection of noise, multi-clause, or ranking constructs
        rejection_tokens = [
            " and ", " or ", " vs ", "versus", "growth", "difference",
            "compare", "top ", "bottom ", "best ", "worst ", "highest ", "lowest ",
            "last month", "recent", "why did", "trend", "ratio of",
        ]
        if any(tok in q_lower for tok in rejection_tokens):
            return None

        cols = profile.columns

        # Ambiguity check: "total number of <column>" where column is a numeric metric (e.g. sales)
        # Combines "total" (SUM) and "number of" (COUNT) on a numeric metric -> reject as ambiguous
        m_ambig = re.search(r"\btotal\s+(?:number|count)\s+of\s+([a-zA-Z_]+)\b", q_lower)
        if m_ambig:
            ambig_word = m_ambig.group(1).lower()
            for c, p in cols.items():
                if c.lower() == ambig_word and self._is_numeric(p):
                    return None

        # Condition 1: Explicit aggregation keyword required
        agg_type: AggregationType | None = None

        count_dist_pat = (
            r"\b(count\s+of\s+distinct|number\s+of\s+distinct|"
            r"count\s+of\s+unique|number\s+of\s+unique)\b"
        )
        if any(k in q_lower for k in ["average order value", "aov"]):
            agg_type = AggregationType.AOV
        elif re.search(count_dist_pat, q_lower):
            agg_type = AggregationType.DISTINCT_COUNT
        elif re.search(r"\b(total\s+count|count\s+of|number\s+of|how\s+many|count)\b", q_lower):
            entity_match = re.search(
                r"\b(?:how\s+many|number\s+of|count\s+of|total\s+count\s+of)\s+([a-zA-Z_]+)\b",
                q_lower,
            )
            counted_entity = entity_match.group(1).lower() if entity_match else ""
            if counted_entity in ("customer", "customers", "client", "clients"):
                agg_type = AggregationType.DISTINCT_COUNT
            elif counted_entity in ("order", "orders", "transaction", "transactions"):
                agg_type = AggregationType.DISTINCT_COUNT
            else:
                agg_type = AggregationType.COUNT
        elif re.search(r"\b(average|avg|mean)\b", q_lower):
            agg_type = AggregationType.AVG
        elif re.search(r"\b(total|sum\s+of|aggregate\s+of|combined)\b", q_lower):
            agg_type = AggregationType.SUM
        elif re.search(r"\b(minimum|lowest|min)\b", q_lower):
            agg_type = AggregationType.MIN
        elif re.search(r"\b(maximum|highest|max)\b", q_lower):
            agg_type = AggregationType.MAX

        if agg_type is None:
            # Questions without explicit aggregation keyword (e.g. "sales in 2017")
            # MUST NOT be heuristically inferred as SUM!
            return None

        # Condition 2: Target metric column binding
        target_col: str | None = None
        denominator_col: str | None = None
        valid_order_denominators = ("order_id", "order_number", "order_key", "orders_id")

        if agg_type == AggregationType.AOV:
            # Sales column for numerator, order_id for denominator
            sales_cols = [
                c for c in cols if c.lower() in ("sales", "revenue", "order_amount", "amount")
            ]
            order_cols = [c for c in cols if c.lower() in valid_order_denominators]
            if len(sales_cols) == 1 and len(order_cols) == 1:
                target_col = sales_cols[0]
                denominator_col = order_cols[0]
            else:
                return None
        elif agg_type == AggregationType.DISTINCT_COUNT:
            entity_match = re.search(
                r"\b(?:how\s+many|number\s+of|count\s+of|total\s+count\s+of)\s+([a-zA-Z_]+)\b",
                q_lower,
            )
            counted_entity = entity_match.group(1).lower() if entity_match else ""
            if (
                counted_entity in ("customer", "customers", "client", "clients")
                or "customer" in q_lower
            ):
                cust_cols = [c for c in cols if c.lower() in ("customer_id", "client_id")]
                target_col = cust_cols[0] if cust_cols else None
            elif (
                counted_entity in ("order", "orders", "transaction", "transactions")
                or "order" in q_lower
            ):
                order_cols = [c for c in cols if c.lower() in valid_order_denominators]
                target_col = order_cols[0] if order_cols else None
            else:
                # Find matching column name
                for c in cols:
                    if re.search(rf"\b{re.escape(c.lower())}\b", q_lower):
                        target_col = c
                        break
            if not target_col:
                return None
        elif agg_type == AggregationType.COUNT:
            # Row count COUNT(*)
            target_col = None
        else:
            # SUM, AVG, MIN, MAX: Find referenced numeric column
            matched_num_cols = []
            for c, p in cols.items():
                if self._is_numeric(p) and re.search(rf"\b{re.escape(c.lower())}\b", q_lower):
                    matched_num_cols.append(c)
            if len(matched_num_cols) == 1:
                target_col = matched_num_cols[0]
            else:
                # If zero or multiple numeric columns match, cannot resolve unambiguously
                return None

        # Condition 3: Explicit temporal boundaries
        time_constraints: list[TimeConstraint] = []
        year_match = re.search(r"\b(?:in|for|during)\s+(20\d\d|19\d\d)\b", q_lower)
        if year_match:
            year = int(year_match.group(1))
            # Find temporal column in profile
            temporal_cols = [c for c, p in cols.items() if p.is_temporal]
            # Prefer order_date or date
            date_col = None
            if "order_date" in temporal_cols:
                date_col = "order_date"
            elif temporal_cols:
                date_col = temporal_cols[0]

            if date_col:
                c_prof = cols[date_col]
                if c_prof.date_format and "AMBIGUOUS" in c_prof.date_format:
                    # Ambiguous date format -> reject temporal filter
                    return None
                time_constraints.append(
                    TimeConstraint(
                        column=date_col,
                        start_time=f"{year}-01-01",
                        end_time=f"{year + 1}-01-01",
                        is_half_open=True,
                    )
                )
            else:
                return None

        # Condition 4: Explicit grouping dimension binding
        dimensions: list[str] = []
        dim_match = re.search(r"\b(?:by|per|across|for\s+each)\s+([a-zA-Z0-9_]+)\b", q_lower)
        if dim_match:
            cand_dim = dim_match.group(1).lower()
            resolved_dim = None
            for c in cols:
                if c.lower() == cand_dim:
                    resolved_dim = c
                    break
            if resolved_dim:
                dimensions.append(resolved_dim)
            else:
                # Dimension was requested but column does not exist -> reject
                return None

        if agg_type == AggregationType.AOV:
            metric_name = "aov"
        elif agg_type == AggregationType.SUM and target_col and "total" in question.lower():
            metric_name = f"total_{target_col}"
        elif (
            agg_type == AggregationType.AVG
            and target_col
            and ("avg" in question.lower() or "average" in question.lower())
        ):
            metric_name = f"avg_{target_col}"
        elif target_col:
            metric_name = f"{agg_type.value.lower()}_{target_col}"
        else:
            metric_name = f"{agg_type.value.lower()}_all"

        metric_spec = MetricSpec(
            name=metric_name,
            aggregation=agg_type,
            column=target_col,
            denominator_column=denominator_col,
        )

        return QuerySpecification(
            raw_question=question,
            table_name=target_table,
            metrics=[metric_spec],
            dimensions=dimensions,
            time_constraints=time_constraints,
        )

    def _parse_with_llm(
        self,
        question: str,
        target_table: str,
        profile: TableProfile,
    ) -> QuerySpecification | None:
        """Extract structured intent using local Ollama model."""
        if not self.client:
            return None

        schema_markdown = format_profile_for_prompt(profile, max_token_budget=1000)
        prompt = INTENT_PARSER_SYSTEM_PROMPT.format(
            schema_markdown=schema_markdown,
            table_name=target_table,
        )
        user_msg = f'Extract the QuerySpecification JSON for: "{question}"'

        try:
            response_text = self.client.generate(prompt=user_msg, system=prompt, temperature=0.0)
            json_match = re.search(r"(\{[\s\S]*\})", response_text)
            if not json_match:
                return None

            raw_dict = json.loads(json_match.group(1))
            raw_dict["raw_question"] = question
            raw_dict["table_name"] = target_table

            return QuerySpecification.model_validate(raw_dict)
        except (json.JSONDecodeError, ValidationError, Exception):
            return None

    def _validate_intent(
        self,
        spec: QuerySpecification,
        profile: TableProfile,
    ) -> QuerySpecification | None:
        """Deterministic schema grounding and validation gate.

        Guarantees:
            1. Every metric column, denominator column, and dimension exists in profile.columns.
            2. Aggregations conform strictly to AggregationType.
            3. CUSTOM formulas are valid ASTs over existing columns.
            4. Operators in FilterSpec are valid Layer 2 operators.
            5. TimeConstraints reference valid, non-ambiguous temporal columns.
        """
        valid_cols = profile.columns

        # Gate 0: Rankings, limits, or sort orders are not supported by Stage 3 NumericalValidator
        q_raw = (spec.raw_question or "").lower()
        ranking_triggers = [
            "highest", "lowest", "top ", "bottom ", "best ", "worst ", "most ", "least ",
        ]
        if spec.limit is not None or spec.order_by or any(rt in q_raw for rt in ranking_triggers):
            return None

        # Gate 1: Metrics
        if not spec.metrics:
            return None

        validated_metrics: list[MetricSpec] = []
        for m in spec.metrics:
            agg = m.aggregation
            if not isinstance(agg, AggregationType):
                try:
                    agg = AggregationType(str(agg).upper())
                except ValueError:
                    return None

            # Validate column
            col = m.column
            resolved_col: str | None = None
            if col and col != "*":
                resolved_col = self._resolve_col(col, valid_cols)
                if resolved_col is None:
                    # Non-existent column -> reject
                    return None
            elif col == "*":
                resolved_col = None

            # Aggregation type-specific constraints
            denom_col = m.denominator_column
            resolved_denom: str | None = None
            if agg == AggregationType.AOV:
                if not resolved_col or not self._is_numeric(valid_cols[resolved_col]):
                    return None
                valid_order_denominators = ("order_id", "order_number", "order_key", "orders_id")
                if not denom_col:
                    # Look for distinct order identifier
                    cands = [c for c in valid_cols if c.lower() in valid_order_denominators]
                    if len(cands) == 1:
                        resolved_denom = cands[0]
                    else:
                        return None
                else:
                    resolved_denom = self._resolve_col(denom_col, valid_cols)
                    if not resolved_denom or resolved_denom.lower() not in valid_order_denominators:
                        # Denominator must be a distinct order ID (not customer_id, etc.)
                        return None

            elif agg in (AggregationType.SUM, AggregationType.AVG):
                if not resolved_col or not self._is_numeric(valid_cols[resolved_col]):
                    return None

            elif agg in (AggregationType.MIN, AggregationType.MAX):
                if not resolved_col:
                    return None

            elif agg == AggregationType.DISTINCT_COUNT:
                if not resolved_col:
                    return None

            elif agg == AggregationType.CUSTOM:
                if not m.formula:
                    return None
                evaluator = self._get_formula_evaluator()
                if not evaluator.is_safe(m.formula):
                    return None
                req_aggs = evaluator.extract_aggregates(m.formula)
                if req_aggs is None:
                    return None
                # Validate all columns in formula exist in schema
                for _, f_col in req_aggs:
                    if not self._resolve_col(f_col, valid_cols):
                        return None

            validated_metrics.append(
                MetricSpec(
                    name=m.name or f"{agg.value.lower()}_{resolved_col or 'all'}",
                    aggregation=agg,
                    column=resolved_col,
                    denominator_column=resolved_denom,
                    formula=m.formula,
                )
            )

        # Gate 2: Dimensions
        validated_dims: list[str] = []
        for d in spec.dimensions:
            res_d = self._resolve_col(d, valid_cols)
            if not res_d:
                # Referenced dimension does not exist in schema -> reject
                return None
            validated_dims.append(res_d)

        # Gate 3: Filters
        validated_filters: list[FilterSpec] = []
        allowed_ops = {"=", "!=", ">", "<", ">=", "<=", "IN", "LIKE"}
        for f in spec.filters:
            res_fc = self._resolve_col(f.column, valid_cols)
            if not res_fc:
                return None
            clean_op = f.operator.strip().upper()
            if clean_op not in allowed_ops:
                return None
            if f.value is None or (isinstance(f.value, str) and f.value.lower() == "null"):
                return None
            validated_filters.append(
                FilterSpec(column=res_fc, operator=clean_op, value=f.value)
            )

        # Gate 4: Time Constraints
        validated_tc: list[TimeConstraint] = []
        for tc in spec.time_constraints:
            res_tc = self._resolve_col(tc.column, valid_cols)
            if not res_tc:
                return None
            col_prof = valid_cols[res_tc]
            is_temp = col_prof.is_temporal or any(
                t in col_prof.duckdb_type.upper() for t in ["DATE", "TIME"]
            )
            if not is_temp:
                return None
            if col_prof.date_format and "AMBIGUOUS" in col_prof.date_format:
                # Ambiguous date format -> fail closed
                return None
            validated_tc.append(
                TimeConstraint(
                    column=res_tc,
                    start_time=tc.start_time,
                    end_time=tc.end_time,
                    is_half_open=tc.is_half_open,
                )
            )

        return QuerySpecification(
            raw_question=spec.raw_question,
            table_name=profile.table_name,
            metrics=validated_metrics,
            dimensions=validated_dims,
            filters=validated_filters,
            time_constraints=validated_tc,
            order_by=spec.order_by,
            limit=spec.limit,
        )

    def _resolve_col(self, col_name: str, valid_cols: dict[str, Any]) -> str | None:
        """Resolve column name case-insensitively, rejecting non-existent names."""
        if not col_name:
            return None
        if col_name in valid_cols:
            return col_name
        col_lower = col_name.lower().strip()
        for c in valid_cols:
            if c.lower() == col_lower:
                return c
        return None
