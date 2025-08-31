from __future__ import annotations
from typing import List
try:
    import sqlglot  # type: ignore
    from sqlglot import expressions as exp  # type: ignore
    _HAVE = True
except Exception:
    sqlglot = None  # type: ignore
    exp = None  # type: ignore
    _HAVE = False


def columns_touched(sql: str) -> List[str]:
    cols: List[str] = []
    if not _HAVE:
        return cols
    try:
        t = sqlglot.parse_one(sql, read="postgres")
        for c in t.find_all(exp.Column):
            cols.append(f"{c.table or '*'}.{c.name}")
    except Exception:
        return []
    return sorted(set(cols))

