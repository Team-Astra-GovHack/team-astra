# app/core/read_only_db_executor.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from decimal import Decimal
from datetime import date, datetime

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.sql_guard import sanitize_sql


def _json_safe(v: Any) -> Any:
    """Coerce DB types into JSON-serializable values."""
    if isinstance(v, Decimal):
        # Use float for analytics; switch to str if you need exact precision
        return float(v)
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return v


class ReadOnlyDbExecutor:
    def __init__(self, engine: Engine, default_limit: int, statement_timeout_ms: int):
        self.engine = engine
        self.default_limit = default_limit
        self.statement_timeout_ms = statement_timeout_ms

    def execute(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        sanitized = sanitize_sql(sql, self.default_limit)
        with self.engine.begin() as conn:
            conn.exec_driver_sql(f"SET LOCAL statement_timeout = {int(self.statement_timeout_ms)}")
            rows = conn.execute(text(sanitized), params or {}).mappings().all()
            out: List[Dict[str, Any]] = []
            for r in rows:
                d = {k: _json_safe(v) for k, v in dict(r).items()}
                out.append(d)
            return out
