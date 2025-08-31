from __future__ import annotations
from typing import List
from app.core.catalogue import Catalogue


def suggest_followups(cat: Catalogue, limit: int = 5) -> List[str]:
    ideas: List[str] = []
    for full, table in cat.tables.items():
        measures = [c.name for c in table.columns.values() if c.role == "measure"]
        dims = [c.name for c in table.columns.values() if c.role == "dimension"]
        if measures and dims:
            m = measures[0]
            d = dims[0]
            ideas.append(f"Show {m} by {d} for the latest year")
        elif measures:
            m = measures[0]
            ideas.append(f"Give the overall {m} in the dataset")
        if len(ideas) >= limit:
            break
    return ideas[:limit]

