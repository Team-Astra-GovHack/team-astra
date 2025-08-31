# app/settings.py
from __future__ import annotations
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices


class Settings(BaseSettings):
    # pydantic v2 config
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Gemini ---
    GEMINI_API_KEY: str
    # generation model used by Analyst/Narrator
    GEMINI_MODEL: str = Field(default="gemini-1.5-pro", validation_alias=AliasChoices("GEMINI_MODEL","GEMINI_MODEL_ANALYST"))
    # embedding model used by the embedder
    GEMINI_EMBED_MODEL: str = Field(default="text-embedding-004", validation_alias=AliasChoices("GEMINI_EMBED_MODEL","GEMINI_MODEL_EMBEDDING"))

    # --- Database (read-only) ---
    DB_URL_RO: str
    DEFAULT_SQL_LIMIT: int = Field(default=5000, validation_alias=AliasChoices("DEFAULT_SQL_LIMIT","SQL_DEFAULT_LIMIT"))
    STATEMENT_TIMEOUT_MS: int = Field(default=20000, validation_alias=AliasChoices("STATEMENT_TIMEOUT_MS","SQL_STATEMENT_TIMEOUT_MS"))  # 20s
    USE_RAG: bool = False

    # --- Pinecone (RAG) ---
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX: Optional[str] = None
    # Some installs still keep this around; safe to ignore if unused.
    PINECONE_ENV: Optional[str] = None

    # Misc
    APP_NAME: str = "ASTRA API"
    APP_VERSION: str = "0.1.0"

    # --- Optional external knowledge files ---
    DATASET_SCHEMAS_PATH: Optional[str] = None
    VALUE_DESCRIPTIONS_PATH: Optional[str] = None
