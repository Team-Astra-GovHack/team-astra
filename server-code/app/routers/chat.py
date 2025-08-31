
# app/routers/chat.py
from __future__ import annotations
import json
import threading
import queue
from typing import List, Dict, Any, Iterable

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.deps import gemini, embedder, repo, db, schema_overview
from app.core.ask_astra_chat_executer import AskASTRAChatExecuter, QEmitter

router = APIRouter()


def _sse(events: Iterable[Dict[str, Any]]):
    """
    Server-Sent Events formatter. Each event dict must have:
      {"id": "...", "name": "chat-step" | "chat-result" | "chat-sql", "data": {...}}
    """
    for ev in events:
        eid = ev.get("id", "")
        name = ev.get("name", "message")
        data = ev.get("data", {})
        yield f"id: {eid}\n"
        yield f"event: {name}\n"
        yield "data: " + json.dumps(data, ensure_ascii=False, default=str) + "\n\n"


def _start_stream(queries: List[str]) -> StreamingResponse:
    q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=200)

    def run():
        ex = AskASTRAChatExecuter(
            gemini=gemini(),
            embedder=embedder(),
            repo=repo(),
            db=db(),
            schema_text=schema_overview(),   # << important: inject authoritative schema
            emitter=QEmitter(q=q),
        )
        try:
            ex.execute(queries)
        finally:
            # send sentinel to close stream
            q.put({"id": "done", "name": "end", "data": {}})

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def gen():
        while True:
            ev = q.get()
            if ev.get("name") == "end":
                break
            for chunk in _sse([ev]):
                yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream")

@router.get("/chat/stream")
def stream(q: str, request: Request):
    """
    SSE endpoint. Example:
      /chat/stream?q=How%20many%20distinct%20agencies%20...
    """
    queries = [q]
    return _start_stream(queries)
