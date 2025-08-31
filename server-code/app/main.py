
# =========================
# app/main.py
# =========================
from __future__ import annotations

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chat, health
from app.core.sql_validation import SQLGLOT_AVAILABLE, SQLGLOT_IMPORT_ERROR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# Quiet noisy third-party loggers (e.g., Pinecone plugin discovery)
logging.getLogger("pinecone_plugin_interface.logging").setLevel(logging.WARNING)
logging.getLogger("pinecone_plugin_interface").setLevel(logging.WARNING)
logging.getLogger("pinecone").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(
    title="ASTRA API",
    version="0.1.0",
    description=(
        "LLM-planned analytics chat over PostgreSQL (read-only) with Pinecone RAG "
        "and Gemini for planning/answers + embeddings. Streams results via SSE.\n\n"
        "Use /chat/stream (GET) or /chat (POST) to stream answers."
    ),
)

# CORS (dev-open; tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# health router defines its own path ("/health")
app.include_router(health.router)

# chat router already has prefix="/chat"
app.include_router(chat.router)

if not SQLGLOT_AVAILABLE:
    import logging as _logging
    _logging.getLogger(__name__).warning("SQL validator running in DEGRADED MODE: %s", SQLGLOT_IMPORT_ERROR)
