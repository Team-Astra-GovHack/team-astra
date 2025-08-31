from __future__ import annotations
from typing import Dict, List, Set, Tuple
import re

try:
    import sqlglot  # type: ignore
    from sqlglot import expressions as exp  # type: ignore
    SQLGLOT_AVAILABLE = True
    SQLGLOT_IMPORT_ERROR = ""
except Exception as e:  # noqa: BLE001
    SQLGLOT_AVAILABLE = False
    SQLGLOT_IMPORT_ERROR = str(e)
    sqlglot = None  # type: ignore
    exp = None  # type: ignore


class SqlValidationError(Exception):
    pass


def _tables_from_tree(tree: "exp.Expression") -> List[str]:
    return [str(t.this) for t in tree.find_all(exp.Table) if getattr(t, "this", None)]


def _is_select_or_with(tree: "exp.Expression") -> bool:
    return isinstance(tree, (exp.Select, exp.With)) or bool(tree.find(exp.Select))


def analyze_sql(sql: str) -> Dict[str, object]:
    out: Dict[str, object] = {
        "tables": set(),
        "columns": set(),
        "has_group": False,
        "has_agg": False,
        "select_only": True,
    }
    if not SQLGLOT_AVAILABLE:
        # degraded analysis
        s = sql
        out["select_only"] = bool(re.match(r'\s*(with|select)\b', s, flags=re.I))
        tables: Set[str] = set()
        for m in re.finditer(r'\b(from|join)\s+([\w\."]+)', s, flags=re.I):
            tables.add(m.group(2).replace('"', ''))
        out["tables"] = tables
        out["columns"] = set()
        out["has_group"] = bool(re.search(r'\bgroup\s+by\b', s, flags=re.I))
        out["has_agg"] = bool(re.search(r'\b(sum|avg|count|min|max)\s*\(', s, flags=re.I))
        return out

    try:
        tree = sqlglot.parse_one(sql, read="postgres")
        out["select_only"] = _is_select_or_with(tree)
        tables: Set[str] = set(_tables_from_tree(tree))
        out["tables"] = tables
        cols: Set[str] = set()
        for c in tree.find_all(exp.Column):
            cols.add(c.sql().replace("`", "").replace('"', ""))
        out["columns"] = cols
        out["has_group"] = bool(list(tree.find_all(exp.Group)))
        out["has_agg"] = any((f.name or "").lower() in {"sum", "avg", "count", "min", "max"}
                              for f in tree.find_all(exp.Func))
        return out
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
        return out


def _validate_with_sqlglot(sql: str, schema_index) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception as e:  # noqa: BLE001
        return False, [f"Parse error: {e}"]
    if tree is None:
        return False, ["Empty or invalid SQL"]
    if not _is_select_or_with(tree):
        return False, [f"Only SELECT/WITH allowed (got {type(tree).__name__})"]

    tables = _tables_from_tree(tree)
    if not tables:
        issues.append("No FROM table found")
    for t in tables:
        if not schema_index or not schema_index.has_table(str(t)):
            issues.append(f"Unknown table: {t}")

    alias_to_table: Dict[str, str] = {}
    for tbl in tree.find_all(exp.Table):
        if tbl.alias:
            alias_to_table[str(tbl.alias.this)] = str(tbl.this)

    for col in tree.find_all(exp.Column):
        col_name = col.name
        tbl = col.table
        if tbl:  # qualified
            base = alias_to_table.get(str(tbl), str(tbl))
            if not schema_index or not schema_index.has_table(base):
                issues.append(f"Unknown table reference: {tbl}")
            elif not schema_index.has_column(base, col_name):
                issues.append(f"Unknown column: {base}.{col_name}")
        else:  # unqualified
            if not any(schema_index and schema_index.has_column(str(t), col_name) for t in tables):
                issues.append(f"Unknown column: {col_name} (unqualified)")
    return (len(issues) == 0), issues


def _validate_degraded(sql: str) -> Tuple[bool, List[str]]:
    s = sql.strip().rstrip(";")
    issues: List[str] = []
    if not re.match(r'\s*(select|with)\b', s, flags=re.I):
        issues.append("Only SELECT/WITH allowed (degraded mode)")
    blacklist = ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "TRUNCATE ", "CREATE ", "\\", "COPY ")
    su = s.upper()
    if any(tok in su for tok in blacklist):
        issues.append("Prohibited keyword detected (degraded mode)")
    return (len(issues) == 0), issues


def validate_sql(sql: str, schema_index) -> Tuple[bool, List[str]]:
    if SQLGLOT_AVAILABLE:
        return _validate_with_sqlglot(sql, schema_index)
    ok, issues = _validate_degraded(sql)
    if not ok:
        issues.append(f"validator_degraded: {SQLGLOT_IMPORT_ERROR}")
    return ok, issues
