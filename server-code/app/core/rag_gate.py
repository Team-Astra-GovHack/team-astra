from __future__ import annotations
try:
    import sqlglot  # type: ignore
    from sqlglot import expressions as exp  # type: ignore
    _HAVE = True
except Exception:
    sqlglot = None  # type: ignore
    exp = None  # type: ignore
    _HAVE = False


def should_use_rag(planned_sql: str) -> bool:
    """Return False for simple single-table aggregates without GROUP BY."""
    try:
        s = (planned_sql or "").upper()
        is_agg = any(k in s for k in ("COUNT(", "SUM(", "AVG(", "MIN(", "MAX("))
        has_group = " GROUP BY " in s
        if _HAVE:
            tree = sqlglot.parse_one(planned_sql, read="postgres")
            has_join = any(True for _ in tree.find_all(exp.Join))
            if (not has_join) and is_agg and (not has_group):
                return False
        else:
            # Degraded heuristic: no JOIN keyword and agg without GROUP BY
            if (" JOIN " not in s) and is_agg and (not has_group):
                return False
    except Exception:
        pass
    return True

