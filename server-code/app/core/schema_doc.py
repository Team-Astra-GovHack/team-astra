# app/core/schema_doc.py
from __future__ import annotations
from typing import Iterable
from sqlalchemy import text
from sqlalchemy.engine import Engine

KEYS = (
    "agency", "request", "type", "date", "year", "month",
    "count", "total", "received", "response", "charges", "data_source"
)

def _interesting(cols: list[str]) -> list[str]:
    # Keep key columns first, then a few others (kept short for prompts)
    keys = [c for c in cols if any(k in c.lower() for k in KEYS)]
    rest = [c for c in cols if c not in keys]
    return [*keys[:6], *rest[:4]]  # ~10 columns max

def build_schema_overview(engine: Engine,
                          include_schemas: Iterable[str] = ("public",),
                          max_tables: int = 50) -> str:
    """
    Returns a concise, LLM-friendly overview like:

    - public.table_a (~rows≈12345)
      columns: agency, request_date, total_requests_received, data_source
    - public.table_b (~rows≈52)
      columns: ...

    Uses only catalog metadata; no data scan.
    """
    schemas = tuple(include_schemas)
    with engine.begin() as conn:
        tables = conn.execute(text("""
            SELECT n.nspname AS schema, c.relname AS table,
                   COALESCE(NULLIF(c.reltuples, 0), 0)::bigint AS est_rows
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'r' AND n.nspname = ANY(:schemas)
            ORDER BY est_rows DESC, n.nspname, c.relname
            LIMIT :max_tables
        """), {"schemas": list(schemas), "max_tables": max_tables}).mappings().all()

        out_lines: list[str] = []
        for t in tables:
            schema, table, est = t["schema"], t["table"], int(t["est_rows"])
            cols = conn.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :s AND table_name = :t
                ORDER BY ordinal_position
                LIMIT 100
            """), {"s": schema, "t": table}).scalars().all()
            cols = _interesting(cols)
            out_lines.append(f"- {schema}.{table} (~rows≈{est})")
            if cols:
                out_lines.append(f"  columns: {', '.join(cols)}")
        return "\n".join(out_lines) or "(no tables discovered)"
