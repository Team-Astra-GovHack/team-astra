# app/main.py
from fastapi.middleware.cors import CORSMiddleware
from app.settings import Settings
from app.routers import chat, health
from app.docs import create_app

settings = Settings()
app = create_app(settings)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",")] if settings.CORS_ALLOW_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
