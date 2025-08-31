from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Load .env and ignore any extra keys we don't model explicitly
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENV: str = "dev"
    CORS_ALLOW_ORIGINS: str = "*"

    GEMINI_API_KEY: str
    GEMINI_MODEL_ANALYST: str = "gemini-2.5-flash"
    GEMINI_MODEL_NARRATOR: str = "gemini-2.5-flash"
    GEMINI_MODEL_EMBEDDING: str = "models/text-embedding-004"

    DB_URL_RO: str
    SQL_DEFAULT_LIMIT: int = 5000
    SQL_STATEMENT_TIMEOUT_MS: int = 5000
    EXPLAIN_ROW_CAP: int = 2_000_000
    EXPLAIN_COST_CAP: int = 50_000_000

    PINECONE_API_KEY: str
    PINECONE_INDEX: str
    PINECONE_DIM: int = 768
    PINECONE_NAMESPACE: str = "default"
    # Not used with Pinecone v3, but keep to avoid confusion/errors
    PINECONE_ENV: Optional[str] = None

    RAG_TOP_K: int = 10
    RAG_MIN_SCORE: float = 0.0
    PREFER_SCHEMA: bool = True

    SQLITE_PATH: Optional[str] = None
