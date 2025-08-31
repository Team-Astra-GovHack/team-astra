
# =========================
# app/routers/health.py
# =========================
from __future__ import annotations

import logging
from fastapi import APIRouter
from app.deps import use_rag, repo
from app.deps import db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

@router.get("/health", summary="Liveness & readiness checks for ASTRA")
def health_check():
    # You can add deeper checks here (e.g., DB ping, Pinecone ping, etc.)
    return {"status": "ok"}


@router.get("/health/rag", summary="RAG readiness (optional)")
def rag_health():
    if not use_rag():
        return {"enabled": False, "ok": True}
    try:
        r = repo()
        ok = True
        # Best-effort stats query
        try:
            stats = r.index.describe_index_stats()
            total = stats.get("total_vector_count", 0) if isinstance(stats, dict) else None
        except Exception:
            total = None
        return {"enabled": True, "ok": ok, "total_vector_count": total}
    except Exception as ex:
        return {"enabled": True, "ok": False, "error": str(ex)}


@router.get("/health/db", summary="Database connectivity (read-only)")
def db_health():
    try:
        executor = db()
        rows = executor.execute("SELECT 1 AS ok")
        return {"ok": True, "rows": rows}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}
