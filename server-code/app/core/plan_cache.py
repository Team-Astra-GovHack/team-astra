from __future__ import annotations
import hashlib
import time
from typing import Optional, Tuple, Dict, Any

_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_TTL = 60 * 60 * 24  # 24h


def key_for(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()


def get_plan(key: str) -> Optional[Dict[str, Any]]:
    now = time.time()
    hit = _CACHE.get(key)
    if hit and (now - hit[0] < _TTL):
        return hit[1]
    return None


def set_plan(key: str, plan: Dict[str, Any]) -> None:
    _CACHE[key] = (time.time(), plan)

