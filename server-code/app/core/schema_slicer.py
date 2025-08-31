from __future__ import annotations
import re
from typing import List
from app.core.catalogue import Catalogue


def shortlist_tables(question: str, cat: Catalogue, k: int = 5) -> List[str]:
    q_tokens = set(re.findall(r"[a-zA-Z0-9_]+", (question or "").lower()))
    scored = []
    for full, table in cat.tables.items():
        name = table.name.lower()
        cols_text = " ".join(c.lower() for c in table.columns.keys())
        text = f"{name} {cols_text}"
        score = sum(1 for tok in q_tokens if tok and tok in text)
        if score > 0:
            scored.append((score, full))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [name for _, name in scored[:k]]

