from __future__ import annotations
from typing import Tuple
import re
try:
    import sqlglot  # type: ignore
    _HAVE_SQLGLOT = True
except Exception:
    sqlglot = None  # type: ignore
    _HAVE_SQLGLOT = False


def analyze(sql: str) -> Tuple[bool, bool]:
    """Returns (has_group_or_agg, parsed_ok). Falls back to regex if sqlglot unavailable."""
    if _HAVE_SQLGLOT:
        try:
            expr = sqlglot.parse_one(sql, read="postgres")
            # Detect GROUP BY or aggregate functions
            has_group = bool(list(expr.find_all(sqlglot.expressions.Group)))
            has_agg = False
            for func in expr.find_all(sqlglot.expressions.Func):
                n = (func.name or "").lower()
                if n in {"sum", "avg", "count", "min", "max"}:
                    has_agg = True
                    break
            return has_group or has_agg, True
        except Exception:
            return False, False
    # Fallback: simple regex checks
    s = sql.lower()
    has_group = bool(re.search(r"\bgroup\s+by\b", s))
    has_agg = bool(re.search(r"\b(sum|avg|count|min|max)\s*\(", s))
    return has_group or has_agg, False


def should_inject_limit(sql: str) -> bool:
    has_group_or_agg, parsed = analyze(sql)
    # Do not inject LIMIT for aggregates or when GROUP BY is present
    if has_group_or_agg:
        return False
    return True
