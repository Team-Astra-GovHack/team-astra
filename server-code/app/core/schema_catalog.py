# app/core/schema_catalog.py
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple
from sqlalchemy import text
from sqlalchemy.engine import Engine

EXCLUDED_SCHEMAS = {"pg_catalog", "information_schema"}

def build_schema_overview(engine: Engine, max_tables: int = 80, max_cols_per_table: int = 16) -> str:
    """
    Returns a compact, deterministic text summary of the DB schema for LLM grounding.
    Format example:
      - public.my_table (approx_cols=5)
        columns: id(bigint), agency(text), total_requests_received(int), data_source(text), period(text)
    """
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT c.table_schema, c.table_name, c.column_name, c.data_type, c.ordinal_position
            FROM information_schema.columns c
            JOIN information_schema.tables t
              ON t.table_schema = c.table_schema AND t.table_name = c.table_name
            WHERE t.table_type='BASE TABLE'
              AND c.table_schema NOT IN ('pg_catalog','information_schema')
            ORDER BY c.table_schema, c.table_name, c.ordinal_position
        """)).all()

    cols: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    for r in rows:
        cols[(r.table_schema, r.table_name)].append((r.column_name, r.data_type))

    # stable order
    items = sorted(cols.items(), key=lambda k: (k[0][0], k[0][1]))[:max_tables]
    parts: List[str] = []
    for (schema, table), cdefs in items:
        sample = ", ".join([f"{n}({t})" for n, t in cdefs[:max_cols_per_table]])
        parts.append(f"- {schema}.{table}\n  columns: {sample}")
    return "\n".join(parts) if parts else "(no user tables found)"
