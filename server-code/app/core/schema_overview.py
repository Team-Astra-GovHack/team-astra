# app/core/schema_overview.py
from __future__ import annotations
from typing import Dict, List, Tuple
from sqlalchemy import text
from sqlalchemy.engine import Engine
import time

# Very small in-process cache so we don't rebuild the overview every request
_CACHE: Dict[str, Tuple[float, str]] = {}
_TTL_SECONDS = 300  # 5 minutes


def _key(engine: Engine) -> str:
    url = str(engine.url)
    return url


def build_db_overview(engine: Engine, max_tables: int = 60, max_cols_per_table: int = 40) -> str:
    """
    Build a compact overview of schemas/tables/columns from information_schema.
    Includes PK markers when available. Designed for prompts (short, factual).
    """
    k = _key(engine)
    now = time.time()
    hit = _CACHE.get(k)
    if hit and (now - hit[0] < _TTL_SECONDS):
        return hit[1]

    with engine.begin() as conn:
        # list user tables (avoid system schemas)
        tables = conn.execute(text("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type='BASE TABLE'
              AND table_schema NOT IN ('pg_catalog','information_schema')
            ORDER BY table_schema, table_name
            LIMIT :lim
        """), {"lim": max_tables}).all()

        # Preload PK columns (schema, table -> set(pk_columns))
        pk_rows = conn.execute(text("""
            SELECT
              tc.table_schema, tc.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON kcu.constraint_name = tc.constraint_name
             AND kcu.table_schema = tc.table_schema
             AND kcu.table_name = tc.table_name
            WHERE tc.constraint_type='PRIMARY KEY'
        """)).all()
        pk_map: Dict[tuple, set] = {}
        for s, t, c in pk_rows:
            pk_map.setdefault((s, t), set()).add(c)

        # Columns
        lines: List[str] = []
        lines.append("=== Database Overview (read-only) ===")
        for schema, table in tables:
            cols = conn.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema=:s AND table_name=:t
                ORDER BY ordinal_position
                LIMIT :lc
            """), {"s": schema, "t": table, "lc": max_cols_per_table}).all()

            col_descs: List[str] = []
            for name, dtype, nullable in cols:
                is_pk = name in pk_map.get((schema, table), set())
                tag = " PK" if is_pk else ""
                ntag = "" if nullable == "YES" else " NOT NULL"
                col_descs.append(f"{name}:{dtype}{ntag}{tag}")

            lines.append(f"- {schema}.{table} -> [{', '.join(col_descs)}]")

        lines.append("\nGuidance:")
        lines.append("- Use fully-qualified names schema.table and column names exactly as listed.")
        lines.append("- Only SELECT/SHOW/DESCRIBE; queries must be read-only.")
        lines.append("- Prefer aggregations at the DB (SUM/COUNT/etc) and keep results small.")
        overview = "\n".join(lines)

    _CACHE[k] = (now, overview)
    return overview
