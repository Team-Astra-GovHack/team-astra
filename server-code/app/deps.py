# app/deps.py
from __future__ import annotations
from functools import lru_cache
from sqlalchemy import create_engine
from app.settings import Settings
from app.core.gemini_client import GeminiClient
from app.core.embedding_engine import GeminiEmbeddingEngine
from app.core.rag_repository import PineconeRAGRepository
from app.core.read_only_db_executor import ReadOnlyDbExecutor
from app.core.knowledge import (
    build_additional_schema_text,
    build_value_descriptions_text,
    collect_problem_identifiers,
    build_naming_hints,
)
from app.core.catalogue import Catalogue, load_catalogue
from pathlib import Path


@lru_cache
def settings() -> Settings:
    return Settings()


@lru_cache
def gemini() -> GeminiClient:
    s = settings()
    return GeminiClient(api_key=s.GEMINI_API_KEY, model=s.GEMINI_MODEL)


@lru_cache
def embedder() -> GeminiEmbeddingEngine:
    s = settings()
    return GeminiEmbeddingEngine(
        api_key=s.GEMINI_API_KEY,
        model=s.GEMINI_EMBED_MODEL,  # "text-embedding-004" by default
        dim=768,
        timeout=30.0,
    )


@lru_cache
def repo() -> PineconeRAGRepository:
    s = settings()
    # If you’ve updated your Pinecone client, ENV may be unused; keep index/key.
    return PineconeRAGRepository(
        api_key=s.PINECONE_API_KEY,
        index_name=s.PINECONE_INDEX,
    )


@lru_cache
def db() -> ReadOnlyDbExecutor:
    s = settings()
    engine = create_engine(s.DB_URL_RO, isolation_level="AUTOCOMMIT")
    return ReadOnlyDbExecutor(
        engine=engine,
        default_limit=s.DEFAULT_SQL_LIMIT,
        statement_timeout_ms=s.STATEMENT_TIMEOUT_MS,
    )


@lru_cache
def external_schemas_text() -> str:
    s = settings()
    base = Path(__file__).resolve().parents[2]  # project root (server-code/..)
    return build_additional_schema_text(base, s.DATASET_SCHEMAS_PATH)


@lru_cache
def value_descriptions_text() -> str:
    s = settings()
    base = Path(__file__).resolve().parents[2]
    return build_value_descriptions_text(base, s.VALUE_DESCRIPTIONS_PATH)


@lru_cache
def identifiers_to_quote() -> frozenset[str]:
    s = settings()
    base = Path(__file__).resolve().parents[2]
    ids = collect_problem_identifiers(base, s.DATASET_SCHEMAS_PATH, s.VALUE_DESCRIPTIONS_PATH)
    return frozenset(ids)


@lru_cache
def use_rag() -> bool:
    return bool(settings().USE_RAG)


@lru_cache
def naming_hints_text() -> str:
    s = settings()
    base = Path(__file__).resolve().parents[2]
    return build_naming_hints(base, s.DATASET_SCHEMAS_PATH, s.VALUE_DESCRIPTIONS_PATH)


@lru_cache
def catalogue() -> Catalogue:
    s = settings()
    base = Path(__file__).resolve().parents[2]
    return load_catalogue(base, s.DATASET_SCHEMAS_PATH, s.VALUE_DESCRIPTIONS_PATH)
