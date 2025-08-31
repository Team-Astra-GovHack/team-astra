from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Set, Tuple, Iterable
from sqlalchemy import text
from sqlalchemy.engine import Engine


@dataclass
class SchemaIndex:
    tables: Dict[Tuple[str, str], Set[str]]  # (schema, table) -> columns

    @classmethod
    def from_engine(cls, engine: Engine) -> "SchemaIndex":
        with engine.begin() as conn:
            rows = conn.execute(text(
                """
                SELECT table_schema, table_name, column_name
                FROM information_schema.columns
                WHERE table_schema NOT IN ('pg_catalog','information_schema')
                ORDER BY table_schema, table_name, ordinal_position
                """
            )).all()
        tables: Dict[Tuple[str, str], Set[str]] = {}
        for s, t, c in rows:
            tables.setdefault((s, t), set()).add(c)
        return cls(tables=tables)

    def has_table(self, name: str) -> bool:
        s, t = self._split(name)
        if s:
            return (s, t) in self.tables
        # unqualified: match any schema
        return any(tt == t for (_, tt) in self.tables.keys())

    def has_column(self, table: str, column: str) -> bool:
        s, t = self._split(table)
        if s:
            return column in self.tables.get((s, t), set())
        # unqualified: match any schema
        for (ss, tt), cols in self.tables.items():
            if tt == t and column in cols:
                return True
        return False

    def normalize_table(self, name: str) -> str:
        s, t = self._split(name)
        if s:
            return f"{s}.{t}"
        # choose first matching schema deterministically
        for (ss, tt) in sorted(self.tables.keys()):
            if tt == t:
                return f"{ss}.{tt}"
        return name

    @staticmethod
    def _split(name: str) -> Tuple[str | None, str]:
        name = name.strip().strip('"')
        if "." in name:
            s, t = name.split(".", 1)
            return s.strip('"'), t.strip('"')
        return None, name

