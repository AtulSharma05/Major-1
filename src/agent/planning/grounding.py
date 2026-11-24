"""Semantic entity and schema grounding engine.

Architectural Context (ADR-02, Phase 5):
    Guarantees that generated analytical queries and structured intent remain faithful
    to the original natural-language question.

Core Principles:
    1. Zero Hallucinated Grounding: A column is grounded if and only if there is
       demonstrable semantic evidence in the natural-language question.
    2. Explicit Grounding Evidence: Distinguishes between grounded intent, ungrounded intent
       (requested concepts absent from schema), and ambiguous intent.
    3. No Silent Entity Substitution: An unsupported entity (e.g. 'students', 'people')
       must NEVER be silently mapped to an available entity (e.g. 'customers').
    4. Deterministic Precedence: Syntactic execution and numerical agreement cannot override
       a semantic grounding failure.
"""

from __future__ import annotations

import re

from src.agent.data.profiling import TableProfile

ENTITY_GROUNDING_KEYWORDS: dict[str, set[str]] = {
    "customer": {
        "customer",
        "customers",
        "client",
        "clients",
        "buyer",
        "buyers",
        "account",
        "accounts",
        "patron",
        "patrons",
        "shopper",
        "shoppers",
    },
    "order": {
        "order",
        "orders",
        "transaction",
        "transactions",
        "purchase",
        "purchases",
        "deal",
        "deals",
    },
    "product": {
        "product",
        "products",
        "item",
        "items",
        "goods",
        "merchandise",
        "category",
        "categories",
        "sub_category",
        "sub-category",
        "sub category",
    },
    "sales": {
        "sale",
        "sales",
        "revenue",
        "dollar",
        "dollars",
        "amount",
        "spend",
        "spending",
        "turnover",
        "order value",
        "value",
        "aov",
    },
    "profit": {
        "profit",
        "profits",
        "margin",
        "margins",
        "earnings",
        "gain",
    },
    "quantity": {
        "quantity",
        "quantities",
        "unit",
        "units",
        "volume",
    },
    "discount": {
        "discount",
        "discounts",
        "markdown",
        "rebate",
    },
    "region": {
        "region",
        "regions",
        "area",
        "areas",
        "territory",
        "territories",
    },
    "city": {
        "city",
        "cities",
        "town",
        "towns",
        "metro",
    },
    "state": {
        "state",
        "states",
        "province",
        "provinces",
    },
    "country": {
        "country",
        "countries",
        "nation",
        "nations",
    },
    "date": {
        "date",
        "dates",
        "day",
        "days",
        "month",
        "months",
        "year",
        "years",
        "time",
        "period",
        "quarter",
        "when",
    },
}

ROW_COUNT_KEYWORDS: frozenset[str] = frozenset(
    {
        "row",
        "rows",
        "record",
        "records",
        "entry",
        "entries",
        "line",
        "lines",
        "tuple",
        "tuples",
        "observation",
        "observations",
        "data point",
        "data points",
    }
)


def is_explicit_row_count_query(question: str) -> bool:
    """Return True if question explicitly requests a physical row or record count."""
    q_lower = question.lower()
    pattern = (
        r"\b(?:how\s+many|number\s+of|count\s+of|total\s+count\s+of|count)\s+"
        r"(?:total\s+)?(rows?|records?|entries|lines?|tuples?|observations?|data\s+points?)\b|"
        r"\b(?:row|record)\s+count\b|"
        r"\b(?:rows?|records?|entries)\s+(?:in\s+this|in\s+the|are\s+there)\b"
    )
    if re.search(pattern, q_lower):
        return True
    if re.search(r"^(?:what\s+is\s+the\s+)?(?:total\s+)?count\s*[?.]?$", q_lower.strip()):
        return True
    return False


def has_column_grounding_evidence(
    col_name: str,
    question: str,
    profile: TableProfile | None = None,
) -> bool:
    """Verify whether a referenced column has demonstrable semantic grounding in the question.

    Rules:
        1. Explicit Name Match: Column name or stem appears in natural language question.
        2. Semantic Synonym Match: Column maps to a recognized entity (e.g. customer_id ->
           'customer') and that specific concept is explicitly present in the question.
        3. Temporal Grounding: Date/timestamp columns are grounded if question specifies
           temporal intervals, years (e.g. 2017), or date keywords.
        4. Strict Anti-Substitution: Unsupported entities (e.g. 'students') NEVER provide grounding
           for an unrelated schema column (e.g. 'customer_id').
    """
    if not col_name or not question:
        return False

    q_lower = question.lower()
    c_lower = col_name.lower().strip()

    # Rule 1: Exact column identifier present as a whole word in the question
    if re.search(rf"\b{re.escape(c_lower)}\b", q_lower):
        return True

    # Rule 1b: Stemmed / non-suffix column name in question (e.g. customer_id -> 'customer')
    c_stem = re.sub(r"(_id|_name|_key|_desc|_code|_num|_number)$", "", c_lower)
    if c_stem and re.search(rf"\b{re.escape(c_stem)}\b", q_lower):
        return True

    # Multi-word column stems (e.g. sub_category -> 'sub-category' or 'sub category')
    if "_" in c_stem:
        hyphen_stem = c_stem.replace("_", "-")
        space_stem = c_stem.replace("_", " ")
        if re.search(rf"\b{re.escape(hyphen_stem)}\b", q_lower) or re.search(
            rf"\b{re.escape(space_stem)}\b", q_lower
        ):
            return True

    # Rule 2: Domain semantic entity keywords
    for entity_type, keywords in ENTITY_GROUNDING_KEYWORDS.items():
        # Check if the column belongs to this entity type
        col_matches_entity = (
            entity_type in c_lower
            or any(k in c_lower for k in keywords)
            or (
                profile
                and c_lower in profile.columns
                and entity_type in str(profile.columns[c_lower].duckdb_type).lower()
            )
        )
        if col_matches_entity:
            # Check if any keyword of this entity type is explicitly present in question
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", q_lower):
                    return True

    # Rule 3: Temporal date/timestamp column grounding
    is_temporal = False
    if profile and c_lower in profile.columns:
        is_temporal = profile.columns[c_lower].is_temporal
    elif "date" in c_lower or "time" in c_lower or "year" in c_lower:
        is_temporal = True

    if is_temporal:
        # Check for year literals, date strings, or temporal boundary keywords
        if re.search(r"\b(20\d\d|19\d\d)\b", q_lower) or re.search(
            r"\b(in|during|for|since|before|after|between)\s+(20\d\d|19\d\d)\b", q_lower
        ):
            return True
        for kw in ENTITY_GROUNDING_KEYWORDS["date"]:
            if re.search(rf"\b{re.escape(kw)}\b", q_lower):
                return True

    return False


def extract_requested_concepts(question: str) -> list[str]:
    """Extract candidate entity and attribute concepts requested in natural language."""
    q_lower = question.lower().strip()
    concepts: list[str] = []

    # Count entities: "how many X", "number of X", "count of X"
    m_count = re.search(
        r"\b(?:how\s+many|number\s+of|count\s+of|total\s+count\s+of)\s+"
        r"([a-zA-Z0-9_\-\s]+?)"
        r"(?:\s+(?:are\s+there|is\s+there|were\s+there|was\s+there|are\s+in|is\s+in|were\s+in|"
        r"were\s+placed|were\s+made|did\s+we|do\s+we|have\s+been|exist|exist\s+in|"
        r"in\s+this|in\s+the|from\s+the|per\s+|by\s+)\b|[?,.]|$)",
        q_lower,
    )
    if m_count:
        cand = m_count.group(1).strip()
        cand = re.sub(r"\b(unique|distinct|total|all)\b", "", cand).strip()
        if cand:
            concepts.append(cand)

    # Aggregated attributes: "average student age", "total sales of students"
    m_agg = re.search(
        r"\b(?:average|avg|mean|total|sum|minimum|min|maximum|max)\s+([a-zA-Z0-9_\-\s]+?)(?:\s+(?:by|per|for|across|in)\b|[?,.]|$)",
        q_lower,
    )
    if m_agg:
        cand = m_agg.group(1).strip()
        cand = re.sub(r"\b(of|the)\b", "", cand).strip()
        if cand and cand not in concepts:
            concepts.append(cand)

    return concepts


def detect_ungrounded_entities(question: str, profile: TableProfile) -> list[str]:
    """Identify concepts or entities in question that cannot be grounded in active schema."""
    if is_explicit_row_count_query(question):
        return []

    concepts = extract_requested_concepts(question)
    if not concepts:
        return []

    ungrounded: list[str] = []
    schema_cols = profile.columns if profile else {}

    for concept in concepts:
        # Check if concept is an explicit row/record count
        if concept in ROW_COUNT_KEYWORDS:
            continue

        # Check if concept or its tokens ground to ANY valid schema column
        concept_grounded = False
        concept_tokens = [t for t in concept.split() if t not in ("of", "the", "and", "in")]

        for col in schema_cols:
            if has_column_grounding_evidence(col, concept, profile):
                concept_grounded = True
                break
            for token in concept_tokens:
                if has_column_grounding_evidence(col, token, profile):
                    concept_grounded = True
                    break
            if concept_grounded:
                break

        if not concept_grounded:
            ungrounded.append(concept)

    return ungrounded
