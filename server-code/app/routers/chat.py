# app/routers/chat.py
from __future__ import annotations

import json
import logging
from typing import List, Dict, Any, Optional
import uuid
from queue import Queue, Empty
from threading import Thread

from fastapi import APIRouter, Query, Body, HTTPException
from fastapi.responses import StreamingResponse
from anyio import EndOfStream

from app.deps import (
    gemini, embedder, repo, db,
    external_schemas_text, value_descriptions_text,
    identifiers_to_quote, use_rag, naming_hints_text, catalogue,
)
from app.core.ask_astra_chat_executer import AskASTRAChatExecuter, QEmitter
from app.core.schema_overview import build_db_overview
from app.core.schema_index import SchemaIndex

logger = logging.getLogger("app.routers.chat")
router = APIRouter(prefix="/chat", tags=["chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
KEEPALIVE_SECONDS = 15
ALLOWED_FINAL_EVENTS = {"keepalive", "chat-structured", "chat-done"}


class GatingEmitter:
    def __init__(self, q: Queue, final_only: bool = False):
        self.q = q
        self.final_only = final_only

    def send(self, event: Dict[str, Any]) -> None:
        name = event.get("name") or "message"
        if self.final_only and name not in ALLOWED_FINAL_EVENTS:
            return
        # Minify final payload shape for front-end if gating is enabled
        if self.final_only and name == "chat-structured":
            data = event.get("data") or {}
            refs = data.get("references") or {}
            minimal = {
                "answer_markdown": data.get("answer_markdown", ""),
                "references": {
                    "tables": refs.get("tables", []) or [],
                    "source": refs.get("source", "schema_json_prompt"),
                    "rag_doc_ids": refs.get("rag_doc_ids", []) or [],
                },
                "trace_id": data.get("trace_id"),
            }
            event = {"id": event.get("id"), "name": name, "data": minimal}
        self.q.put(event)


def _sse(event: Dict[str, Any]) -> str:
    lines = []
    if "id" in event and event["id"] is not None:
        lines.append(f"id: {event['id']}")
    lines.append(f"event: {event.get('name', 'message')}")
    payload = json.dumps(event.get("data", {}), default=str, ensure_ascii=False)
    lines.append(f"data: {payload}")
    return "\n".join(lines) + "\n\n"


def _start_stream(queries: List[str], *, final_only: bool = False) -> StreamingResponse:
    if not queries or not all(isinstance(q, str) and q.strip() for q in queries):
        raise HTTPException(status_code=400, detail="Empty or invalid query list.")

    q: Queue = Queue(maxsize=1000)
    emitter = GatingEmitter(q, final_only=final_only)

    def worker() -> None:
        """
        Initialize everything here so the endpoint returns SSE immediately.
        If deps/init fail, emit chat-error + chat-done instead of 500.
        """
        try:
            # Build a concise DB schema overview for prompts (fallback if DB unavailable)
            db_ok = True
            try:
                schema_text = build_db_overview(db().engine)
                schema_idx = SchemaIndex.from_engine(db().engine)
            except Exception as e:
                logger.exception("DB overview unavailable; falling back to external schema: %s", e)
                db_ok = False
                schema_idx = None
                # Fallback schema: external schemas + value descriptions
                ext = external_schemas_text() or "(none)"
                vals = value_descriptions_text() or "(none)"
                schema_text = (
                    "=== Fallback Schema (DB unreachable) ===\n"
                    + ext + "\n\n"
                    + vals
                )
            # Keep prompt size reasonable to reduce token/quota pressure
            if len(schema_text) > 16000:
                schema_text = schema_text[:16000]

            # Construct executer with schema overview and emitter
            use_rag_flag = use_rag()
            rag_repo = repo() if use_rag_flag else None  # lazy-load Pinecone only if enabled
            executer = AskASTRAChatExecuter(
                gemini(),
                embedder(),
                rag_repo,
                db(),
                schema_text,
                emitter,
                external_schemas_text=external_schemas_text(),
                value_descriptions_text=value_descriptions_text(),
                identifiers_to_quote=identifiers_to_quote(),
                rag_enabled=use_rag_flag,
                db_available=db_ok,
                schema_index=schema_idx,
                naming_hints=naming_hints_text(),
                catalogue=catalogue(),
            )

        except Exception as exc:
            logger.exception("Failed to initialize executer/deps: %s", exc)
            # Emit a final minimal payload in final-only mode
            if isinstance(emitter, GatingEmitter) and emitter.final_only:
                trace_id = str(uuid.uuid4())
                emitter.send({
                    "name": "chat-structured",
                    "data": {
                        "answer_markdown": "I can’t answer this from the current datasets.",
                        "references": {"tables": [], "source": "error", "rag_doc_ids": []},
                        "trace_id": trace_id,
                    },
                })
            emitter.send({"name": "chat-done", "data": {}})
            return

        try:
            executer.execute(queries)
        except Exception as exc:
            logger.exception("Fatal error in chat worker: %s", exc)
            if isinstance(emitter, GatingEmitter) and emitter.final_only:
                trace_id = str(uuid.uuid4())
                emitter.send({
                    "name": "chat-structured",
                    "data": {
                        "answer_markdown": "I can’t answer this from the current datasets.",
                        "references": {"tables": [], "source": "error", "rag_doc_ids": []},
                        "trace_id": trace_id,
                    },
                })
        finally:
            emitter.send({"name": "chat-done", "data": {}})

    Thread(target=worker, daemon=True, name="astra-chat-worker").start()

    async def gen():
        try:
            while True:
                try:
                    ev = q.get(timeout=KEEPALIVE_SECONDS)
                except Empty:
                    yield "event: keepalive\ndata: {}\n\n"
                    continue
                yield _sse(ev)
                if ev.get("name") == "chat-done":
                    break
        except (EndOfStream, GeneratorExit):
            logger.info("Client disconnected from /chat stream.")
        except Exception as stream_exc:
            logger.exception("Streaming generator crashed: %s", stream_exc)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.get("/stream")
def stream(q: str = Query(..., description="User question"), final_only: bool = Query(False)):
    return _start_stream([q], final_only=final_only)


@router.post("/")
def chat(body: Optional[dict] = Body(...), final_only: bool = Query(False)):
    user_q = (body or {}).get("q") or (body or {}).get("query")
    if not user_q or not isinstance(user_q, str) or not user_q.strip():
        raise HTTPException(status_code=400, detail="Missing 'q' in request body.")
    return _start_stream([user_q], final_only=final_only)
