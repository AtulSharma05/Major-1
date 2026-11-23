"""Layer 2 Independent Numerical Validator and Calculator.

Architectural Context (ADR-02, Phase 5 Stage 3):
    Calculates analytical metrics independently from the original registered Pandas
    DataFrame using structured QuerySpecification intent, then verifies DuckDB execution
    results within strict numerical and categorical tolerances.

Key Guarantees:
    1. Zero SQL Reuse: Does not touch DuckDB, execute SQL, or parse query strings.
    2. Categorical Group Keys: Zero-tolerance exact matching; boolean values strictly
       differentiated from integers (avoiding Python bool-is-subclass-of-int leakage).
    3. Lossless NULL vs. IEEE NaN: Distinguishes SQL NULL from floating-point NaN;
       correctly maps Pandas source-data NaN to SQL NULL while preserving DuckDB IEEE NaN.
    4. Safe AST Formula Evaluation: Whitelist-only AST evaluator for derived metrics
       with verified division-by-zero semantics, argument count validation, and no eval() or exec().
    5. Numerical Precision: Preserves exact integer and Decimal calculations beyond 2^53 - 1.
    6. Date Format Safety: Uses reliable format metadata, rejects ambiguous dates,
       and returns INCONCLUSIVE.
    7. Clean Status Semantics: VERIFIED on match, FAILED on numerical or group mismatch,
       INCONCLUSIVE on missing intent, ambiguous dates, or unsupported operators.
"""

from __future__ import annotations

import ast
import datetime
import math
import re
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from src.agent.data.profiling import TableProfile, detect_date_format_and_pattern
from src.agent.execution.engine import ExecutionResult
from src.agent.planning.intent import (
    AggregationType,
    FilterSpec,
    QuerySpecification,
    TimeConstraint,
)
from src.agent.planning.planner import PlanStep
from src.agent.validation.contracts import (
    NumericalValidationResult,
    ValidationStatus,
)

# Sentinels uniquely identifying SQL NULL vs. IEEE 754 Floating NaN
CANONICAL_SQL_NULL = object()
CANONICAL_FLOAT_NAN = object()


def is_scalar_null(val: Any) -> bool:
    """Return True if and only if val is a genuine scalar missing value.

    Guarantees:
        1. Literal strings (e.g. '<NA>', 'null', '') and bytes are NEVER null.
        2. Non-scalar collections (lists, tuples, dicts, ndarrays) never trigger
           ambiguous truth-value errors and evaluate to False.
        3. Only true Python/NumPy/Pandas scalar missing values (None, np.nan,
           pd.NA, pd.NaT) evaluate to True.
    """
    if isinstance(val, (str, bytes)):
        return False
    if isinstance(val, (list, tuple, dict, set, np.ndarray, pd.Series, pd.Index)):
        return False
    if val is None or val is pd.NA or val is pd.NaT:
        return True
    if isinstance(val, (float, np.floating)):
        return math.isnan(val)
    try:
        return bool(np.isscalar(val) and pd.isna(val))
    except Exception:
        return False


def normalize_group_key_element(val: Any, is_source_data: bool = False) -> Any:
    """Normalize a scalar group-key component to its canonical form.

    Args:
        val: Raw scalar value from DuckDB raw_rows or Pandas grouping.
        is_source_data: If True, float NaN represents SQL NULL (Pandas missing representation).
            If False (DuckDB output), None is SQL NULL while float('nan') is IEEE NaN.
    """
    if val is CANONICAL_SQL_NULL or val is CANONICAL_FLOAT_NAN:
        return val

    if val is None or val is pd.NA or val is pd.NaT:
        return CANONICAL_SQL_NULL

    if isinstance(val, (float, np.floating)) and math.isnan(val):
        return CANONICAL_SQL_NULL if is_source_data else CANONICAL_FLOAT_NAN

    if isinstance(val, (str, bytes)):
        return val

    if is_source_data and is_scalar_null(val):
        return CANONICAL_SQL_NULL

    return val


def match_group_key_element(
    k1: Any,
    k2: Any,
    k1_is_source: bool = False,
    k2_is_source: bool = False,
) -> bool:
    """Compare two scalar group-key components enforcing strict type safety and zero tolerance.

    Evaluation Order:
        1. Canonical Sentinels (SQL NULL vs Float NaN)
        2. Boolean Types (Strict bool/np.bool_ matching; reject integer leakage)
        3. String / Bytes Types (Exact categorical identity)
        4. Integer Types (Exact integer equality)
        5. Float Types (Exact bitwise/numerical equality; ZERO tolerance)
        6. Precision-Safe Integer/Float Cross-Comparison
    """
    norm1 = normalize_group_key_element(k1, is_source_data=k1_is_source)
    norm2 = normalize_group_key_element(k2, is_source_data=k2_is_source)

    # 1. Canonical Sentinels
    if norm1 is CANONICAL_SQL_NULL or norm2 is CANONICAL_SQL_NULL:
        return norm1 is CANONICAL_SQL_NULL and norm2 is CANONICAL_SQL_NULL

    if norm1 is CANONICAL_FLOAT_NAN or norm2 is CANONICAL_FLOAT_NAN:
        return norm1 is CANONICAL_FLOAT_NAN and norm2 is CANONICAL_FLOAT_NAN

    # 2. Boolean Types (MUST precede integer checks because bool inherits from int)
    is_bool_1 = isinstance(k1, (bool, np.bool_))
    is_bool_2 = isinstance(k2, (bool, np.bool_))

    if is_bool_1 or is_bool_2:
        if not (is_bool_1 and is_bool_2):
            return False
        return bool(k1) is bool(k2)

    # 3. String / Bytes Types (No implicit numeric conversion)
    if isinstance(k1, (str, bytes)) or isinstance(k2, (str, bytes)):
        if type(k1) is not type(k2):
            return False
        return k1 == k2

    # 4. Integer vs Integer (Exact equality)
    if isinstance(k1, (int, np.integer)) and isinstance(k2, (int, np.integer)):
        return int(k1) == int(k2)

    # 5. Float vs Float (Exact equality, ZERO tolerance)
    if isinstance(k1, (float, np.floating)) and isinstance(k2, (float, np.floating)):
        if math.isnan(k1) and math.isnan(k2):
            return True
        return k1 == k2

    # 6. Safe Integer vs Float Cross-Comparison
    if isinstance(k1, (int, np.integer)) and isinstance(k2, (float, np.floating)):
        int_val, float_val = int(k1), float(k2)
    elif isinstance(k2, (int, np.integer)) and isinstance(k1, (float, np.floating)):
        int_val, float_val = int(k2), float(k1)
    elif isinstance(k1, Decimal) or isinstance(k2, Decimal):
        try:
            return Decimal(str(k1)) == Decimal(str(k2))
        except Exception:
            return False
    else:
        return False

    # If the float has a non-zero fractional component, it cannot equal an integer
    if float_val % 1.0 != 0.0:
        return False

    # Within IEEE 754 safe integer range (|n| <= 2^53 - 1 = 9007199254740991)
    if abs(int_val) <= 9007199254740991:
        return float(int_val) == float_val

    # Beyond safe range: compare with Decimal to prevent precision truncation
    return Decimal(str(int_val)) == Decimal(str(float_val))


class SafeFormulaEvaluator:
    """Restricted AST evaluator for derived analytical expressions without eval() or exec()."""

    ALLOWED_FUNCTIONS = {"SUM", "COUNT", "DISTINCT_COUNT", "AVG", "MIN", "MAX"}
    ALLOWED_OPERATOR_TYPES = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)

    def is_safe(self, formula: str) -> bool:
        """Return True if expression conforms strictly to the allowed AST grammar."""
        if not formula or not isinstance(formula, str):
            return False

        try:
            tree = ast.parse(formula.strip(), mode="eval")
        except Exception:
            return False

        if not isinstance(tree, ast.Expression):
            return False

        for node in ast.walk(tree):
            if isinstance(node, ast.Expression):
                continue
            elif isinstance(node, ast.BinOp):
                if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                    return False
            elif isinstance(node, ast.UnaryOp):
                if not isinstance(node.op, (ast.USub, ast.UAdd)):
                    return False
            elif isinstance(node, ast.Constant):
                # Allow numeric constants only (int or float, not bool, string, etc.)
                if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                    return False
            elif isinstance(node, ast.Name):
                if not isinstance(node.ctx, ast.Load):
                    return False
            elif isinstance(node, ast.Call):
                # Validate function name
                if not isinstance(node.func, ast.Name):
                    return False
                func_name = node.func.id.upper()
                if func_name not in self.ALLOWED_FUNCTIONS:
                    return False
                # Reject keyword arguments
                if node.keywords:
                    return False
                # Require strictly 1 argument that is a simple column name
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Name):
                    return False
            elif isinstance(node, (ast.Load, *self.ALLOWED_OPERATOR_TYPES)):
                continue
            else:
                return False

        return True

    def extract_aggregates(self, formula: str) -> list[tuple[str, str]] | None:
        """Extract atomic aggregate references, e.g. [('SUM', 'profit'), ('SUM', 'sales')].

        Returns:
            List of (func_name, col_name) tuples, or None if formula is malformed/unsafe.
        """
        if not self.is_safe(formula):
            return None

        try:
            tree = ast.parse(formula.strip(), mode="eval")
        except Exception:
            return None

        aggs: list[tuple[str, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                func = node.func.id.upper()
                if len(node.args) == 1 and isinstance(node.args[0], ast.Name):
                    col = node.args[0].id
                    aggs.append((func, col))
                else:
                    return None
        return aggs

    def evaluate(self, formula: str, context: dict[str, Any]) -> Any:
        """Safely evaluate expression with verified DuckDB division-by-zero and null semantics.

        Args:
            formula: Arithmetic formula string.
            context: Mapping of variable / aggregate names to values.

        Returns:
            Computed numerical scalar (or None / float('inf') / float('nan')).
        """
        if not self.is_safe(formula):
            raise ValueError(f"Unsafe or unsupported formula expression: '{formula}'")

        tree = ast.parse(formula.strip(), mode="eval")
        return self._eval_node(tree.body, context)

    def _eval_node(self, node: ast.AST, context: dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            name = node.id
            if name in context:
                return context[name]
            for k, v in context.items():
                if k.lower() == name.lower():
                    return v
            raise KeyError(f"Identifier '{name}' not found in formula context")

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id.upper()
            col_name = node.args[0].id if (node.args and isinstance(node.args[0], ast.Name)) else ""
            key = f"{func_name}({col_name})"
            for k, v in context.items():
                if k.replace(" ", "").upper() == key.replace(" ", "").upper():
                    return v
            raise KeyError(f"Aggregate '{key}' not found in formula context")

        if isinstance(node, ast.UnaryOp):
            val = self._eval_node(node.operand, context)
            if val is None or is_scalar_null(val):
                return None
            if isinstance(node.op, ast.USub):
                return -val
            if isinstance(node.op, ast.UAdd):
                return +val
            raise ValueError(f"Unsupported unary operator: {type(node.op)}")

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, context)
            right = self._eval_node(node.right, context)

            # Null operand propagation (ANSI SQL standard)
            if left is None or is_scalar_null(left) or right is None or is_scalar_null(right):
                return None

            try:
                left_f = float(left)
                right_f = float(right)
            except (ValueError, TypeError):
                return None

            if isinstance(node.op, ast.Add):
                return left_f + right_f
            if isinstance(node.op, ast.Sub):
                return left_f - right_f
            if isinstance(node.op, ast.Mult):
                return left_f * right_f
            if isinstance(node.op, ast.Div):
                # Verified DuckDB floating-point division semantics
                if right_f == 0.0:
                    if left_f > 0.0:
                        return float("inf")
                    elif left_f < 0.0:
                        return float("-inf")
                    else:
                        return float("nan")
                return left_f / right_f

            raise ValueError(f"Unsupported binary operator: {type(node.op)}")

        raise ValueError(f"Unsupported AST node type: {type(node)}")


class IndependentCalculator:
    """Pure Pandas calculation engine driven by structured QuerySpecification."""

    SUPPORTED_OPERATORS = {"=", "!=", ">", "<", ">=", "<=", "IN", "LIKE"}

    def __init__(self) -> None:
        self.formula_evaluator = SafeFormulaEvaluator()

    def apply_filters(
        self,
        df: pd.DataFrame,
        filters: list[FilterSpec],
        time_constraints: list[TimeConstraint],
        table_profile: TableProfile | None = None,
    ) -> tuple[pd.DataFrame | None, str | None]:
        """Apply filters with SQL LIKE wildcard matching and three-valued null logic.

        Returns:
            Tuple of (filtered_df, inconclusive_reason). If an unsupported filter or
            ambiguous date is encountered, returns (None, reason).
        """
        filtered = df.copy()

        # Step 1: Standard FilterSpec conditions
        for f in filters:
            col = f.column
            if col not in filtered.columns:
                match = [c for c in filtered.columns if c.lower() == col.lower()]
                if not match:
                    return None, f"Filter column '{col}' not found in dataset"
                col = match[0]

            op = f.operator.strip().upper()
            if op not in self.SUPPORTED_OPERATORS:
                return None, f"Unsupported filter operator: '{f.operator}'"

            val = f.value

            # Guard: Comparison with literal None requires IS NULL which is deferred
            if val is None or (isinstance(val, str) and val.lower() == "null"):
                return None, f"Null comparison filter on '{col}' requires deferred IS NULL operator"

            # Three-valued logic: rows with null in filter column are excluded by WHERE
            valid_mask = filtered[col].notna()

            try:
                if op == "=":
                    mask = valid_mask & (filtered[col] == val)
                elif op == "!=":
                    mask = valid_mask & (filtered[col] != val)
                elif op == ">":
                    mask = valid_mask & (filtered[col] > val)
                elif op == "<":
                    mask = valid_mask & (filtered[col] < val)
                elif op == ">=":
                    mask = valid_mask & (filtered[col] >= val)
                elif op == "<=":
                    mask = valid_mask & (filtered[col] <= val)
                elif op == "IN":
                    if not isinstance(val, (list, tuple, set)):
                        return None, f"Filter operator IN requires list value, got {type(val)}"
                    mask = valid_mask & (filtered[col].isin(val))
                elif op == "LIKE":
                    pattern = str(val)
                    regex_pat = self._sql_like_to_regex(pattern)
                    str_series = filtered[col].astype(str)
                    mask = valid_mask & str_series.str.contains(regex_pat, regex=True, na=False)
                else:
                    return None, f"Unhandled filter operator: '{op}'"

                filtered = filtered.loc[mask]
            except Exception as exc:
                return None, f"Filter evaluation error on '{col}': {exc}"

        # Step 2: TimeConstraint intervals with reliable date parsing
        for tc in time_constraints:
            col = tc.column
            if col not in filtered.columns:
                match = [c for c in filtered.columns if c.lower() == col.lower()]
                if not match:
                    return None, f"Time constraint column '{col}' not found in dataset"
                col = match[0]

            valid_mask = filtered[col].notna()
            series_non_null = filtered.loc[valid_mask, col]

            # Determine whether column is native datetime or requires format-guided parsing
            is_native_dt = pd.api.types.is_datetime64_any_dtype(filtered[col])
            if not is_native_dt and len(series_non_null) > 0:
                first_elem = series_non_null.iloc[0]
                if isinstance(first_elem, (datetime.date, datetime.datetime, pd.Timestamp)):
                    is_native_dt = True

            try:
                if is_native_dt:
                    col_series = pd.to_datetime(filtered[col])
                else:
                    strp_fmt = None
                    date_fmt = None
                    if table_profile and table_profile.columns:
                        col_prof = table_profile.columns.get(col)
                        if not col_prof:
                            for cp_name, cp in table_profile.columns.items():
                                if cp_name.lower() == col.lower():
                                    col_prof = cp
                                    break
                        if col_prof and col_prof.is_temporal:
                            date_fmt = col_prof.date_format
                            strp_fmt = col_prof.strptime_format

                    if date_fmt is None and len(series_non_null) > 0:
                        samples = series_non_null.head(100).tolist()
                        _, date_fmt, strp_fmt = detect_date_format_and_pattern(samples)

                    if date_fmt and "AMBIGUOUS" in date_fmt:
                        return (
                            None,
                            f"Ambiguous date format in column '{col}'; "
                            "date interpretation cannot be established",
                        )

                    if strp_fmt:
                        col_series = pd.to_datetime(filtered[col], format=strp_fmt, errors="coerce")
                    else:
                        return (
                            None,
                            f"Ambiguous or undetermined date format in column '{col}'; "
                            "date interpretation cannot be established",
                        )

                if tc.start_time:
                    start_dt = pd.to_datetime(tc.start_time)
                    valid_mask &= col_series >= start_dt
                if tc.end_time:
                    end_dt = pd.to_datetime(tc.end_time)
                    if tc.is_half_open:
                        valid_mask &= col_series < end_dt
                    else:
                        valid_mask &= col_series <= end_dt
                filtered = filtered.loc[valid_mask]
            except Exception as exc:
                return None, f"Time constraint evaluation error on '{col}': {exc}"

        return filtered, None

    def _sql_like_to_regex(self, pattern: str) -> str:
        """Convert SQL LIKE pattern with % and _ wildcards into an anchored regex."""
        i = 0
        n = len(pattern)
        res = ["^"]
        while i < n:
            c = pattern[i]
            if c == "\\":
                if i + 1 < n:
                    res.append(re.escape(pattern[i + 1]))
                    i += 2
                else:
                    res.append(r"\\")
                    i += 1
            elif c == "%":
                res.append(".*")
                i += 1
            elif c == "_":
                res.append(".")
                i += 1
            else:
                res.append(re.escape(c))
                i += 1
        res.append("$")
        return "".join(res)

    def compute_atomic_metric(
        self,
        df: pd.DataFrame,
        agg_type: AggregationType,
        column: str | None,
        denominator_col: str | None = None,
    ) -> Any:
        """Compute an atomic SQL-compatible aggregation in Pandas preserving exact types."""
        if agg_type == AggregationType.COUNT:
            if not column or column.strip() == "*":
                return int(len(df))
            if column in df.columns:
                return int(df[column].count())
            match = [c for c in df.columns if c.lower() == column.lower()]
            if match:
                return int(df[match[0]].count())
            return 0

        if agg_type == AggregationType.DISTINCT_COUNT:
            if not column or column not in df.columns:
                match = [c for c in df.columns if c.lower() == (column or "").lower()]
                if match:
                    column = match[0]
                else:
                    return 0
            return int(df[column].nunique(dropna=True))

        # For SUM, AVG, MIN, MAX on empty datasets, SQL returns NULL (None)
        if len(df) == 0:
            return None

        target_col = column
        if target_col not in df.columns:
            match = [c for c in df.columns if c.lower() == (column or "").lower()]
            if match:
                target_col = match[0]
            else:
                return None

        series = df[target_col].dropna()
        if len(series) == 0:
            return None

        if agg_type == AggregationType.SUM:
            # Check for Decimal types
            first_val = series.iloc[0]
            if isinstance(first_val, Decimal) or any(isinstance(x, Decimal) for x in series):
                try:
                    return sum(Decimal(str(x)) for x in series)
                except Exception:
                    pass

            # Check for pure integers (including huge integers > 2^53 - 1)
            if pd.api.types.is_integer_dtype(series) or all(
                isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))
                for x in series
            ):
                return int(sum(int(x) for x in series))

            num_series = pd.to_numeric(series, errors="coerce").dropna()
            if len(num_series) == 0:
                return None
            val = num_series.sum()
            return float(val) if not pd.isna(val) else None

        if agg_type == AggregationType.AVG:
            num_series = pd.to_numeric(series, errors="coerce").dropna()
            if len(num_series) == 0:
                return None
            val = num_series.mean()
            return float(val) if not pd.isna(val) else None

        if agg_type == AggregationType.MIN:
            first_val = series.iloc[0]
            if isinstance(first_val, Decimal) or any(isinstance(x, Decimal) for x in series):
                try:
                    return min(Decimal(str(x)) for x in series)
                except Exception:
                    pass
            if pd.api.types.is_integer_dtype(series) or all(
                isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))
                for x in series
            ):
                return int(series.min())
            val = series.min()
            return val if not pd.isna(val) else None

        if agg_type == AggregationType.MAX:
            first_val = series.iloc[0]
            if isinstance(first_val, Decimal) or any(isinstance(x, Decimal) for x in series):
                try:
                    return max(Decimal(str(x)) for x in series)
                except Exception:
                    pass
            if pd.api.types.is_integer_dtype(series) or all(
                isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))
                for x in series
            ):
                return int(series.max())
            val = series.max()
            return val if not pd.isna(val) else None

        if agg_type == AggregationType.AOV:
            num_series = pd.to_numeric(series, errors="coerce").dropna()
            sum_val = num_series.sum()
            if not denominator_col or denominator_col not in df.columns:
                match = [c for c in df.columns if c.lower() == (denominator_col or "").lower()]
                if match:
                    denominator_col = match[0]
                else:
                    return None
            denom_count = df[denominator_col].nunique(dropna=True)
            if denom_count == 0:
                return (
                    float("inf")
                    if sum_val > 0
                    else (float("-inf") if sum_val < 0 else float("nan"))
                )
            return float(sum_val) / float(denom_count)

        return None

    def calculate(
        self,
        df: pd.DataFrame,
        spec: QuerySpecification,
        table_profile: TableProfile | None = None,
    ) -> tuple[dict[str, Any] | list[dict[str, Any]] | None, str | None]:
        """Perform full independent calculation from QuerySpecification.

        Returns:
            Tuple of (calculated_result, inconclusive_reason).
            For scalar/single-row: dict of {metric_name: value}.
            For grouped: list of row dicts [{dim1: k1, ..., metric1: v1}, ...].
        """
        filtered_df, err = self.apply_filters(
            df, spec.filters, spec.time_constraints, table_profile=table_profile
        )
        if err or filtered_df is None:
            return None, err or "Filter evaluation failed"

        # Case 1: Non-grouped Query (Scalar or Multi-Metric Summary)
        if not spec.is_grouped():
            metrics_out: dict[str, Any] = {}
            for m in spec.metrics:
                if m.formula:
                    continue
                val = self.compute_atomic_metric(
                    filtered_df,
                    m.aggregation,
                    m.column,
                    m.denominator_column,
                )
                metrics_out[m.name] = val

            for m in spec.metrics:
                if m.formula:
                    if not self.formula_evaluator.is_safe(m.formula):
                        return (
                            None,
                            f"Unsupported or malformed formula in metric '{m.name}': {m.formula}",
                        )
                    req_aggs = self.formula_evaluator.extract_aggregates(m.formula)
                    if req_aggs is None:
                        return (
                            None,
                            "Failed to extract aggregates from formula in "
                            f"metric '{m.name}': {m.formula}",
                        )
                    context = dict(metrics_out)
                    for fn_name, col_name in req_aggs:
                        key = f"{fn_name}({col_name})"
                        if key not in context:
                            agg_enum = getattr(AggregationType, fn_name, None)
                            if not agg_enum:
                                return None, f"Unsupported aggregate '{fn_name}' in formula"
                            val = self.compute_atomic_metric(filtered_df, agg_enum, col_name)
                            context[key] = val
                    try:
                        derived_val = self.formula_evaluator.evaluate(m.formula, context)
                        metrics_out[m.name] = derived_val
                    except Exception as exc:
                        return None, f"Formula evaluation error for '{m.name}': {exc}"

            return metrics_out, None

        # Case 2: Grouped Query
        dim_cols: list[str] = []
        for d in spec.dimensions:
            if d in filtered_df.columns:
                dim_cols.append(d)
            else:
                match = [c for c in filtered_df.columns if c.lower() == d.lower()]
                if match:
                    dim_cols.append(match[0])
                else:
                    return None, f"Grouping dimension '{d}' not found in dataset"

        if len(filtered_df) == 0:
            return [], None

        grouped_rows: list[dict[str, Any]] = []
        groups = filtered_df.groupby(dim_cols, dropna=False)

        for group_key, group_df in groups:
            row_dict: dict[str, Any] = {}
            group_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
            for idx, col in enumerate(dim_cols):
                row_dict[col] = group_tuple[idx]

            for m in spec.metrics:
                if m.formula:
                    continue
                val = self.compute_atomic_metric(
                    group_df,
                    m.aggregation,
                    m.column,
                    m.denominator_column,
                )
                row_dict[m.name] = val

            for m in spec.metrics:
                if m.formula:
                    if not self.formula_evaluator.is_safe(m.formula):
                        return (
                            None,
                            f"Unsupported or malformed formula in metric '{m.name}': {m.formula}",
                        )
                    req_aggs = self.formula_evaluator.extract_aggregates(m.formula)
                    if req_aggs is None:
                        return (
                            None,
                            "Failed to extract aggregates from formula in "
                            f"metric '{m.name}': {m.formula}",
                        )
                    context = dict(row_dict)
                    for fn_name, col_name in req_aggs:
                        key = f"{fn_name}({col_name})"
                        if key not in context:
                            agg_enum = getattr(AggregationType, fn_name, None)
                            if not agg_enum:
                                return None, f"Unsupported aggregate '{fn_name}' in formula"
                            val = self.compute_atomic_metric(group_df, agg_enum, col_name)
                            context[key] = val
                    try:
                        derived_val = self.formula_evaluator.evaluate(m.formula, context)
                        row_dict[m.name] = derived_val
                    except Exception as exc:
                        return None, f"Formula evaluation error in group: {exc}"

            grouped_rows.append(row_dict)

        return grouped_rows, None


class NumericalValidator:
    """Layer 2 Numerical Validator comparing DuckDB results against independent calculation."""

    def __init__(self, atol: float = 1e-4, rtol: float = 1e-4) -> None:
        self.atol = atol
        self.rtol = rtol
        self.calculator = IndependentCalculator()

    def compare_numeric_values(
        self, actual: Any, expected: Any
    ) -> tuple[bool, float | None, float | None]:
        """Compare two scalar metric values respecting SQL NULL vs. IEEE NaN and tolerances.

        Returns:
            Tuple of (matched: bool, abs_diff: float | None, rel_diff: float | None).
        """
        actual_norm = normalize_group_key_element(actual, is_source_data=False)
        expected_norm = normalize_group_key_element(expected, is_source_data=False)

        # Case 1: Both SQL NULL
        if actual_norm is CANONICAL_SQL_NULL and expected_norm is CANONICAL_SQL_NULL:
            return True, 0.0, 0.0

        # Case 2: One is NULL, one is not
        if actual_norm is CANONICAL_SQL_NULL or expected_norm is CANONICAL_SQL_NULL:
            return False, None, None

        # Case 3: Both IEEE Float NaN
        if actual_norm is CANONICAL_FLOAT_NAN and expected_norm is CANONICAL_FLOAT_NAN:
            return True, 0.0, 0.0

        if actual_norm is CANONICAL_FLOAT_NAN or expected_norm is CANONICAL_FLOAT_NAN:
            return False, None, None

        # Case 4: Exact integer comparison before float conversion (preserves > 2^53 - 1)
        is_int_act = isinstance(actual, (int, np.integer)) and not isinstance(
            actual, (bool, np.bool_)
        )
        is_int_exp = isinstance(expected, (int, np.integer)) and not isinstance(
            expected, (bool, np.bool_)
        )
        if is_int_act and is_int_exp:
            act_i = int(actual)
            exp_i = int(expected)
            matched = act_i == exp_i
            diff = abs(act_i - exp_i)
            rel_diff = (diff / abs(exp_i)) if exp_i != 0 else 0.0
            return matched, float(diff), float(rel_diff)

        # Case 5: Decimal comparison
        if isinstance(actual, Decimal) or isinstance(expected, Decimal):
            try:
                act_dec = Decimal(str(actual))
                exp_dec = Decimal(str(expected))
                matched = act_dec == exp_dec
                diff = abs(act_dec - exp_dec)
                rel_diff = float(diff / abs(exp_dec)) if exp_dec != Decimal(0) else 0.0
                return matched, float(diff), float(rel_diff)
            except Exception:
                pass

        # Case 6: Infinities
        try:
            act_f = float(actual)
            exp_f = float(expected)
        except (ValueError, TypeError):
            return (actual == expected), None, None

        if math.isinf(act_f) or math.isinf(exp_f):
            if math.isinf(act_f) and math.isinf(exp_f):
                return ((act_f > 0) == (exp_f > 0)), 0.0, 0.0
            return False, None, None

        # Case 7: Standard finite float tolerance check
        abs_diff = abs(act_f - exp_f)
        tol = self.atol + self.rtol * abs(exp_f)
        matched = bool(abs_diff <= tol)
        rel_diff = (abs_diff / abs(exp_f)) if abs(exp_f) > 0 else 0.0
        return matched, abs_diff, rel_diff

    def validate_step(
        self,
        step: PlanStep,
        execution_result: ExecutionResult,
        source_df: pd.DataFrame | None,
        table_profile: TableProfile | None = None,
    ) -> NumericalValidationResult:
        """Validate a step execution result against independent Pandas calculation.

        Invariants enforced:
            - If source_df is missing -> INCONCLUSIVE.
            - If structured_intent is missing -> INCONCLUSIVE.
            - If execution failed -> UNVERIFIED.
            - Group keys matched with zero tolerance, strict boolean separation.
            - Numerical values matched within (atol, rtol).
        """
        if not execution_result.success:
            return NumericalValidationResult(
                layer_name="Layer 2: Numerical",
                passed=False,
                status=ValidationStatus.UNVERIFIED,
                message="Execution failed; numerical validation was not performed.",
                inconclusive_reason="raw_execution_failed",
            )

        if source_df is None:
            return NumericalValidationResult(
                layer_name="Layer 2: Numerical",
                passed=False,
                status=ValidationStatus.INCONCLUSIVE,
                message="Source DataFrame unavailable in session memory for recalculation.",
                inconclusive_reason="missing_source_dataframe",
            )

        spec = step.structured_intent
        if spec is None:
            return NumericalValidationResult(
                layer_name="Layer 2: Numerical",
                passed=False,
                status=ValidationStatus.INCONCLUSIVE,
                message="Structured QuerySpecification unavailable; recalculation skipped.",
                inconclusive_reason="missing_structured_intent",
            )

        calc_res, inconv_reason = self.calculator.calculate(
            source_df, spec, table_profile=table_profile
        )
        if inconv_reason or calc_res is None:
            return NumericalValidationResult(
                layer_name="Layer 2: Numerical",
                passed=False,
                status=ValidationStatus.INCONCLUSIVE,
                message=f"Independent calculation inconclusive: {inconv_reason}",
                inconclusive_reason=inconv_reason or "calculation_inconclusive",
            )

        # Case 1: Scalar Single-Metric Query
        if spec.is_scalar():
            metric_spec = spec.metrics[0]
            metric_name = metric_spec.name
            expected_val = calc_res.get(metric_name) if isinstance(calc_res, dict) else None

            if execution_result.raw_rows:
                actual_val = execution_result.raw_rows[0][0]
            elif execution_result.scalar_value is not None:
                actual_val = execution_result.scalar_value
            elif execution_result.dataframe is not None and not execution_result.dataframe.empty:
                actual_val = execution_result.dataframe.iloc[0, 0]
            else:
                actual_val = None

            matched, abs_diff, rel_diff = self.compare_numeric_values(actual_val, expected_val)
            if not matched:
                return NumericalValidationResult(
                    layer_name="Layer 2: Numerical",
                    passed=False,
                    status=ValidationStatus.FAILED,
                    duckdb_value=actual_val,
                    independent_value=expected_val,
                    absolute_difference=abs_diff,
                    relative_difference=rel_diff,
                    atol=self.atol,
                    rtol=self.rtol,
                    compared_rows=1,
                    message=(
                        f"Scalar numerical mismatch for '{metric_name}': actual={actual_val}, "
                        f"expected={expected_val}, diff={abs_diff}"
                    ),
                )

            return NumericalValidationResult(
                layer_name="Layer 2: Numerical",
                passed=True,
                status=ValidationStatus.VERIFIED,
                duckdb_value=actual_val,
                independent_value=expected_val,
                absolute_difference=abs_diff,
                relative_difference=rel_diff,
                atol=self.atol,
                rtol=self.rtol,
                compared_rows=1,
                message=f"Scalar metric '{metric_name}' numerically verified within tolerance.",
            )

        # Case 2: Single-Row Multi-Metric Query (Non-grouped)
        if not spec.is_grouped() and isinstance(calc_res, dict):
            duck_df = execution_result.dataframe
            actual_rows = len(duck_df) if duck_df is not None else 0
            if duck_df is None or len(duck_df) != 1:
                return NumericalValidationResult(
                    layer_name="Layer 2: Numerical",
                    passed=False,
                    status=ValidationStatus.FAILED,
                    message=f"Expected exactly 1 row for multi-metric summary, got {actual_rows}",
                )

            cols_lower = {c.lower(): idx for idx, c in enumerate(duck_df.columns)}
            for m in spec.metrics:
                m_name = m.name
                if m_name.lower() not in cols_lower:
                    return NumericalValidationResult(
                        layer_name="Layer 2: Numerical",
                        passed=False,
                        status=ValidationStatus.FAILED,
                        message=f"Metric column alias '{m_name}' not found in query result",
                    )

                col_idx = cols_lower[m_name.lower()]
                actual_val = (
                    execution_result.raw_rows[0][col_idx]
                    if execution_result.raw_rows
                    else duck_df.iloc[0, col_idx]
                )
                expected_val = calc_res.get(m_name)

                matched, abs_diff, rel_diff = self.compare_numeric_values(actual_val, expected_val)
                if not matched:
                    return NumericalValidationResult(
                        layer_name="Layer 2: Numerical",
                        passed=False,
                        status=ValidationStatus.FAILED,
                        duckdb_value=actual_val,
                        independent_value=expected_val,
                        absolute_difference=abs_diff,
                        relative_difference=rel_diff,
                        atol=self.atol,
                        rtol=self.rtol,
                        compared_rows=1,
                        message=(
                            f"Multi-metric mismatch for '{m_name}': "
                            f"actual={actual_val}, expected={expected_val}"
                        ),
                    )

            return NumericalValidationResult(
                layer_name="Layer 2: Numerical",
                passed=True,
                status=ValidationStatus.VERIFIED,
                compared_rows=1,
                atol=self.atol,
                rtol=self.rtol,
                message="All summary metrics independently verified within tolerance.",
            )

        # Case 3: Grouped Summary Query
        if spec.is_grouped() and isinstance(calc_res, list):
            duck_df = execution_result.dataframe
            if duck_df is None:
                return NumericalValidationResult(
                    layer_name="Layer 2: Numerical",
                    passed=False,
                    status=ValidationStatus.FAILED,
                    message="Missing DataFrame for grouped query verification",
                )

            cols_lower = {c.lower(): idx for idx, c in enumerate(duck_df.columns)}
            dim_indices: list[int] = []
            for d in spec.dimensions:
                if d.lower() not in cols_lower:
                    return NumericalValidationResult(
                        layer_name="Layer 2: Numerical",
                        passed=False,
                        status=ValidationStatus.FAILED,
                        message=f"Dimension column '{d}' not found in query result",
                    )
                dim_indices.append(cols_lower[d.lower()])

            metric_indices: dict[str, int] = {}
            for m in spec.metrics:
                if m.name.lower() not in cols_lower:
                    return NumericalValidationResult(
                        layer_name="Layer 2: Numerical",
                        passed=False,
                        status=ValidationStatus.FAILED,
                        message=f"Metric column alias '{m.name}' not found in grouped query result",
                    )
                metric_indices[m.name] = cols_lower[m.name.lower()]

            duck_groups: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
            if execution_result.raw_rows:
                for row in execution_result.raw_rows:
                    key = tuple(row[idx] for idx in dim_indices)
                    metrics_map = {m_name: row[c_idx] for m_name, c_idx in metric_indices.items()}
                    duck_groups.append((key, metrics_map))
            else:
                for _, row in duck_df.iterrows():
                    key = tuple(row.iloc[idx] for idx in dim_indices)
                    metrics_map = {
                        m_name: row.iloc[c_idx] for m_name, c_idx in metric_indices.items()
                    }
                    duck_groups.append((key, metrics_map))

            seen_duck_keys: list[tuple[Any, ...]] = []
            for k, _ in duck_groups:
                for prev_k in seen_duck_keys:
                    if len(k) == len(prev_k) and all(
                        match_group_key_element(a, b, k1_is_source=False, k2_is_source=False)
                        for a, b in zip(k, prev_k)
                    ):
                        return NumericalValidationResult(
                            layer_name="Layer 2: Numerical",
                            passed=False,
                            status=ValidationStatus.FAILED,
                            compared_rows=len(duck_groups),
                            message=f"Duplicate group key detected in DuckDB query output: {k}",
                            details={
                                "failure_type": "group_structure",
                                "failure_reason": "duplicate_duckdb_group_key",
                                "duplicate_key": k,
                                "expected_groups": len(calc_res),
                                "actual_groups": len(duck_groups),
                            },
                        )
                seen_duck_keys.append(k)

            expected_groups: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
            for row_dict in calc_res:
                key = tuple(row_dict.get(d) for d in spec.dimensions)
                metrics_map = {m.name: row_dict.get(m.name) for m in spec.metrics}
                expected_groups.append((key, metrics_map))

            seen_exp_keys: list[tuple[Any, ...]] = []
            for k, _ in expected_groups:
                for prev_k in seen_exp_keys:
                    if len(k) == len(prev_k) and all(
                        match_group_key_element(a, b, k1_is_source=True, k2_is_source=True)
                        for a, b in zip(k, prev_k)
                    ):
                        return NumericalValidationResult(
                            layer_name="Layer 2: Numerical",
                            passed=False,
                            status=ValidationStatus.FAILED,
                            compared_rows=len(duck_groups),
                            message=f"Duplicate group key detected in independent calculation: {k}",
                            details={
                                "failure_type": "group_structure",
                                "failure_reason": "duplicate_expected_group_key",
                                "duplicate_key": k,
                                "expected_groups": len(expected_groups),
                                "actual_groups": len(duck_groups),
                            },
                        )
                seen_exp_keys.append(k)

            # 1. Map each expected group to its matching duck group
            matched_pairs: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]] = []
            missing_group_keys: list[tuple[Any, ...]] = []
            matched_duck_indices: set[int] = set()

            for exp_key, exp_metrics in expected_groups:
                matched_idx = None
                matched_metrics = None
                for idx, (d_key, d_metrics) in enumerate(duck_groups):
                    if (
                        idx not in matched_duck_indices
                        and len(exp_key) == len(d_key)
                        and all(
                            match_group_key_element(d, e, k1_is_source=False, k2_is_source=True)
                            for d, e in zip(d_key, exp_key)
                        )
                    ):
                        matched_idx = idx
                        matched_metrics = d_metrics
                        break
                if matched_metrics is not None and matched_idx is not None:
                    matched_duck_indices.add(matched_idx)
                    matched_pairs.append((exp_key, exp_metrics, matched_metrics))
                else:
                    missing_group_keys.append(exp_key)

            unexpected_group_keys = [
                duck_groups[i][0] for i in range(len(duck_groups)) if i not in matched_duck_indices
            ]

            # 2. Evaluate comparisons for all matched groups
            matched_groups_count = 0
            compared_metrics_count = 0
            passed_comparisons_count = 0
            group_comparisons: list[dict[str, Any]] = []
            flat_comparisons: list[dict[str, Any]] = []

            group_metric_failure = False
            failing_group_key = None
            failing_metric_name = None
            failing_act = None
            failing_exp = None
            failing_abs = None
            failing_rel = None

            for exp_key, exp_metrics, matched_duck_metric in matched_pairs:
                matched_groups_count += 1
                group_metrics: dict[str, Any] = {}
                first_metric_record: dict[str, Any] | None = None
                current_group_failed = False

                for m_name, exp_val in exp_metrics.items():
                    act_val = matched_duck_metric.get(m_name)
                    matched, abs_diff, rel_diff = self.compare_numeric_values(act_val, exp_val)
                    compared_metrics_count += 1

                    metric_record = {
                        "metric": m_name,
                        "duckdb_value": act_val,
                        "independent_value": exp_val,
                        "absolute_difference": abs_diff,
                        "relative_difference": rel_diff,
                        "passed": bool(matched),
                    }
                    group_metrics[m_name] = metric_record
                    if first_metric_record is None:
                        first_metric_record = metric_record

                    flat_comparisons.append(
                        {
                            "group_key": exp_key,
                            **metric_record,
                        }
                    )

                    if matched:
                        passed_comparisons_count += 1
                    else:
                        current_group_failed = True
                        if not group_metric_failure:
                            group_metric_failure = True
                            failing_group_key = exp_key
                            failing_metric_name = m_name
                            failing_act = act_val
                            failing_exp = exp_val
                            failing_abs = abs_diff
                            failing_rel = rel_diff
                        break

                group_entry: dict[str, Any] = {
                    "group_key": exp_key,
                    "metrics": group_metrics,
                    "passed": not current_group_failed,
                }
                if first_metric_record and len(exp_metrics) == 1:
                    group_entry["metric"] = first_metric_record["metric"]
                    group_entry["duckdb_value"] = first_metric_record["duckdb_value"]
                    group_entry["independent_value"] = first_metric_record["independent_value"]
                    group_entry["absolute_difference"] = first_metric_record["absolute_difference"]
                    group_entry["relative_difference"] = first_metric_record["relative_difference"]
                    group_entry["passed"] = first_metric_record["passed"]

                group_comparisons.append(group_entry)

                if current_group_failed:
                    break

            # 3. Handle metric-value failure
            if group_metric_failure:
                details = {
                    "failure_type": "metric_mismatch",
                    "failing_group_key": failing_group_key,
                    "failing_metric": failing_metric_name,
                    "expected_groups": len(expected_groups),
                    "actual_groups": len(duck_groups),
                    "matched_groups": matched_groups_count,
                    "missing_group_keys": tuple(missing_group_keys),
                    "unexpected_group_keys": tuple(unexpected_group_keys),
                    "compared_metrics": compared_metrics_count,
                    "passed_comparisons": passed_comparisons_count,
                    "group_comparisons": group_comparisons,
                    "comparisons": flat_comparisons,
                }
                return NumericalValidationResult(
                    layer_name="Layer 2: Numerical",
                    passed=False,
                    status=ValidationStatus.FAILED,
                    duckdb_value=failing_act,
                    independent_value=failing_exp,
                    absolute_difference=failing_abs,
                    relative_difference=failing_rel,
                    atol=self.atol,
                    rtol=self.rtol,
                    compared_rows=len(duck_groups),
                    message=(
                        f"Group metric mismatch for key {failing_group_key}, metric "
                        f"'{failing_metric_name}': actual={failing_act}, "
                        f"expected={failing_exp}, diff={failing_abs}"
                    ),
                    details=details,
                )

            # 4. Handle group-structure failure (missing groups or extra groups)
            if (
                missing_group_keys
                or unexpected_group_keys
                or len(duck_groups) != len(expected_groups)
            ):
                if len(duck_groups) != len(expected_groups):
                    message = (
                        f"Group count mismatch: DuckDB returned {len(duck_groups)} groups, "
                        f"independent calculation expected {len(expected_groups)} groups"
                    )
                    reason = "group_count_mismatch"
                else:
                    missing_str = ", ".join(str(k) for k in missing_group_keys)
                    unexpected_str = ", ".join(str(k) for k in unexpected_group_keys)
                    message = (
                        f"Group key mismatch: expected group(s) [{missing_str}] missing from "
                        f"DuckDB output, unexpected group(s) [{unexpected_str}] returned"
                    )
                    reason = "group_key_mismatch"

                details = {
                    "failure_type": "group_structure",
                    "failure_reason": reason,
                    "failing_group_key": missing_group_keys[0] if missing_group_keys else None,
                    "expected_groups": len(expected_groups),
                    "actual_groups": len(duck_groups),
                    "matched_groups": matched_groups_count,
                    "missing_group_keys": tuple(missing_group_keys),
                    "unexpected_group_keys": tuple(unexpected_group_keys),
                    "compared_metrics": compared_metrics_count,
                    "passed_comparisons": passed_comparisons_count,
                    "group_comparisons": group_comparisons,
                    "comparisons": flat_comparisons,
                }
                return NumericalValidationResult(
                    layer_name="Layer 2: Numerical",
                    passed=False,
                    status=ValidationStatus.FAILED,
                    compared_rows=len(duck_groups),
                    message=message,
                    details=details,
                )

            # 5. Success: all groups matched and all metrics passed
            details = {
                "failure_type": None,
                "expected_groups": len(expected_groups),
                "actual_groups": len(duck_groups),
                "matched_groups": matched_groups_count,
                "missing_group_keys": (),
                "unexpected_group_keys": (),
                "compared_metrics": compared_metrics_count,
                "passed_comparisons": passed_comparisons_count,
                "group_comparisons": group_comparisons,
                "comparisons": flat_comparisons,
            }

            return NumericalValidationResult(
                layer_name="Layer 2: Numerical",
                passed=True,
                status=ValidationStatus.VERIFIED,
                compared_rows=len(expected_groups),
                atol=self.atol,
                rtol=self.rtol,
                message=(
                    f"All {len(expected_groups)} grouped categories and "
                    "metrics independently verified."
                ),
                details=details,
            )

        return NumericalValidationResult(
            layer_name="Layer 2: Numerical",
            passed=False,
            status=ValidationStatus.INCONCLUSIVE,
            message="Unrecognized query specification structure for numerical validation",
            inconclusive_reason="unrecognized_structure",
        )
