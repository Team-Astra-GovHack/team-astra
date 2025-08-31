# app/routers/health.py
from fastapi import APIRouter

router = APIRouter()

@router.get("")
def ok():
    return {"status": "ok"}
