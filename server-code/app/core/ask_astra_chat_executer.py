# app/core/ask_astra_chat_executer.py
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Dict, Any
from queue import Queue

from app.core.gemini_client import GeminiClient
from app.core.embedding_engine import GeminiEmbeddingEngine
from app.core.rag_repository import PineconeRAGRepository
from app.core.read_only_db_executor import ReadOnlyDbExecutor
from app.core.models import QueryPlan, SqlQuery


@dataclass
class QEmitter:
    q: Queue
    def send(self, event: Dict[str, Any]) -> None:
        self.q.put(event)


class AskASTRAChatExecuter:
    def __init__(
        self,
        gemini: GeminiClient,
        embedder: GeminiEmbeddingEngine,
        repo: PineconeRAGRepository,
        db: ReadOnlyDbExecutor,
        schema_text: str,                 # NEW: authoritative DB overview for prompts
        emitter: QEmitter,
    ):
        self.gemini = gemini
        self.embedder = embedder
        self.repo = repo
        self.db = db
        self.schema_text = schema_text
        self.emitter = emitter
        self._eid = 0

    def _emit(self, name: str, data: Dict[str, Any]) -> None:
        self._eid += 1
        self.emitter.send({"id": str(self._eid), "name": name, "data": data})

    # === Main entry ===
    def execute(self, queries: List[str]) -> None:
        focus = queries[-1] if queries else ""
        self._emit("chat-step", {"step": "Understanding", "message": "Formulating plan..."})

        qp = self._analyze(queries, focus)
        self._emit("chat-step", {"step": "Understanding", "message": "Plan ready", "plan": qp.model_dump()})

        self._emit("chat-step", {"step": "Formulating queries", "message": "Generating SQL candidates..."})
        sqls = self._generate_sql(queries, focus, qp)
        self._emit("chat-sql", {"count": len(sqls), "queries": [s.model_dump() for s in sqls]})

        # Execute SQL(s)
        results: List[Dict[str, Any]] = []
        for i, sq in enumerate(sqls, 1):
            self._emit("chat-step", {"step": "Executing SQL", "message": f"Running {i}/{len(sqls)}"})
            try:
                rows = self.db.execute(sq.query, sq.params)
                results.extend(rows)
            except Exception as ex:
                self._emit("chat-step", {"step": "Executing SQL", "message": f"Query failed: {ex}"})

        # RAG only for reference (doc IDs), not for numeric facts
        doc_ids: List[str] = []
        try:
            merged = "\n".join([*queries, *qp.rewrittenQueries]) or focus
            embs = self.embedder.get_batch_embeddings([merged]).embeddings
            if embs:
                for hit in self.repo.find_nearest_by_cosine_distance(embs[0].values, top_k=10):
                    hid = (hit.get("id") if isinstance(hit, dict) else None) or "?"
                    doc_ids.append(str(hid))
        except Exception as ex:
            self._emit("chat-step", {"step": "Retrieval", "message": f"Vector search failed: {ex}"})

        self._emit("chat-step", {"step": "Retrieval", "message": f"SQL rows: {len(results)}, Docs: {len(doc_ids)}"})

        # Gather references for narrator: data_source values + SQL used
        data_sources = sorted({str(r["data_source"]) for r in results if isinstance(r, dict) and r.get("data_source")})
        sql_used = "\n\n".join(q.query for q in sqls) if sqls else ""

        # Compose narrator prompt
        prompt = self._answer_prompt(
            focus=focus,
            sql_rows=results,
            qp=qp,
            data_sources=", ".join(data_sources) if data_sources else "none",
            doc_ids=", ".join(doc_ids[:10]) if doc_ids else "none",
            sql_used=sql_used,
        )

        # Stream narrated answer; if none, emit a local fallback
        emitted_any = False
        try:
            for chunk in self.gemini.narrator_stream(prompt):
                if chunk:
                    emitted_any = True
                    self._emit("chat-result", {"chunk": chunk})
        except Exception as ex:
            self._emit("chat-step", {"step": "Answering", "message": f"Narrator error: {ex}"})

        if not emitted_any:
            fallback_parts: List[str] = []
            if results:
                fallback_parts.append("### Results (sample)\n```\n")
                fallback_parts.append(json.dumps(results[:10], indent=2, default=str))
                fallback_parts.append("\n```\n")
            if doc_ids:
                fallback_parts.append("\n### Reference doc IDs\n")
                fallback_parts.append(", ".join(doc_ids[:10]))
            if not fallback_parts:
                fallback_parts.append("No data available and the LLM produced no text.")
            self._emit("chat-result", {"chunk": "".join(fallback_parts)})

    # === LLM helpers ===
    def _analyze(self, queries: List[str], focus: str) -> QueryPlan:
        from app.prompts.versioned.v1.analyst import ANALYST_PROMPT
        s = self.gemini.analyst(ANALYST_PROMPT.format(
            SCHEMA_OVERVIEW=self.schema_text,
            QUERIES=json.dumps(queries),
            FOCUS=focus
        ))
        try:
            data = json.loads(s.replace("```json","").replace("```","").strip())
            return QueryPlan(**data)
        except Exception:
            from app.prompts.versioned.v1.repair import REPAIR_PROMPT
            repaired = self.gemini.analyst(REPAIR_PROMPT.format(BAD_JSON=s))
            try:
                data = json.loads(repaired.replace("```json","").replace("```","").strip())
                return QueryPlan(**data)
            except Exception:
                return QueryPlan()

    def _generate_sql(self, queries: List[str], focus: str, qp: QueryPlan) -> List[SqlQuery]:
        from app.prompts.versioned.v1.analyst import SQL_GEN_PROMPT
        s = self.gemini.analyst(SQL_GEN_PROMPT.format(
            SCHEMA_OVERVIEW=self.schema_text,
            FOCUS=focus,
            REWRITTEN=json.dumps(qp.rewrittenQueries),
        ))
        try:
            arr = json.loads(s.replace("```json","").replace("```","").strip())
            return [SqlQuery(**x) for x in arr if isinstance(x, dict) and "query" in x]
        except Exception:
            return []

    def _answer_prompt(self, *, focus, sql_rows, qp, data_sources, doc_ids, sql_used) -> str:
        from app.prompts.versioned.v1.narrator import NARRATOR_PROMPT
        return NARRATOR_PROMPT.format(
            FOCUS=focus,
            STRATEGY=qp.searchStrategy,
            SQL_ROWS=json.dumps(sql_rows[:200], default=str),  # avoid Decimal issues
            DATA_SOURCES=data_sources,
            DOC_IDS=doc_ids,
            SQL_USED=sql_used,
        )
