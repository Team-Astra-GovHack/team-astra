# app/docs.py
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from typing import Any, Dict, List, Optional

from app.settings import Settings

TAGS_METADATA: List[Dict[str, Any]] = [
    {
        "name": "health",
        "description": "Liveness & readiness checks for ASTRA.",
    },
    {
        "name": "chat",
        "description": (
            "Ask ASTRA natural language questions. "
            "The `/chat/stream` and `/chat` endpoints stream answers using **Server-Sent Events (SSE)**."
        ),
    },
]

def create_app(settings: Settings) -> FastAPI:
    """
    Central place for Swagger/OpenAPI metadata and docs URLs.
    """
    app = FastAPI(
        title="ASTRA API",
        version="0.1.0",
        description=(
            "LLM-planned analytics chat over **PostgreSQL (read-only)** with **Pinecone** RAG and **Gemini** "
            "for planning/answers + embeddings. Streams results via **SSE**.\n\n"
            "Use `/chat/stream` (GET) or `/chat` (POST) to stream answers."
        ),
        openapi_url="/openapi.json",
        docs_url="/docs",     # Swagger UI
        redoc_url="/redoc",   # ReDoc
        contact={"name": "ASTRA Team"},
        license_info={"name": "MIT"},
    )

    app.openapi_tags = TAGS_METADATA

    # Optional: customize OpenAPI (servers, security schemes, etc.)
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            tags=TAGS_METADATA,
        )

        # Servers block (handy for env-specific docs)
        schema["servers"] = [
            {"url": "http://127.0.0.1:8001", "description": "Local dev"},
            # add staging/prod here if you want them visible in the docs
            # {"url": "https://api.staging.example.com", "description": "Staging"},
            # {"url": "https://api.example.com", "description": "Production"},
        ]

        # Example security scheme (API key in header). Remove if not used.
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["ApiKeyAuth"] = {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        }
        # Apply globally (optional)
        # schema["security"] = [{"ApiKeyAuth": []}]

        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi
    return app
