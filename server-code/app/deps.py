# app/deps.py
from __future__ import annotations
from functools import lru_cache
from sqlalchemy import create_engine

from app.settings import Settings
from app.core.gemini_client import GeminiClient
from app.core.embedding_engine import GeminiEmbeddingEngine
from app.core.rag_repository import PineconeRAGRepository
from app.core.read_only_db_executor import ReadOnlyDbExecutor
from app.core.schema_doc import build_schema_overview


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def engine():
    s = settings()
    # Pre-ping keeps connections healthy over time
    return create_engine(s.DB_URL_RO, pool_pre_ping=True)


@lru_cache(maxsize=1)
def gemini() -> GeminiClient:
    s = settings()
    return GeminiClient(
        api_key=s.GEMINI_API_KEY,
        analyst_model=s.GEMINI_MODEL_ANALYST,
        narrator_model=s.GEMINI_MODEL_NARRATOR,
    )


@lru_cache(maxsize=1)
def embedder() -> GeminiEmbeddingEngine:
    s = settings()
    return GeminiEmbeddingEngine(
        api_key=s.GEMINI_API_KEY,
        model_name=s.GEMINI_MODEL_EMBEDDING,
    )


@lru_cache(maxsize=1)
def repo() -> PineconeRAGRepository:
    s = settings()
    top_k = int(getattr(s, "RAG_TOP_K", 8))
    min_score = float(getattr(s, "RAG_MIN_SCORE", 0.0))
    namespace = getattr(s, "PINECONE_NAMESPACE", "default") or "default"
    return PineconeRAGRepository(
        api_key=s.PINECONE_API_KEY,
        index_name=s.PINECONE_INDEX,
        namespace=namespace,
        top_k=top_k,
        min_score=min_score,
    )


@lru_cache(maxsize=1)
def db() -> ReadOnlyDbExecutor:
    s = settings()
    default_limit = int(getattr(s, "SQL_DEFAULT_LIMIT", 5000))
    timeout_ms = int(getattr(s, "SQL_STATEMENT_TIMEOUT_MS", 5000))
    return ReadOnlyDbExecutor(
        engine=engine(),
        default_limit=default_limit,
        statement_timeout_ms=timeout_ms,
    )


@lru_cache(maxsize=1)
def schema_overview() -> str:
    # Build once and cache; refresh requires process restart (simple & fast)
    return build_schema_overview(engine(), include_schemas=("public",), max_tables=80)
