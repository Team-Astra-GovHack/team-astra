# app/core/ask_astra_chat_executer.py
from __future__ import annotations
import json
from dataclasses import dataclass
import uuid
from typing import List, Dict, Any, Optional, Iterable, Tuple
from queue import Queue

from app.core.gemini_client import GeminiClient
from app.core.embedding_engine import GeminiEmbeddingEngine
from app.core.rag_repository import PineconeRAGRepository
from app.core.followups import suggest_followups
from app.core.needed_data import format_needed_data
from app.core.read_only_db_executor import ReadOnlyDbExecutor
from app.core.models import QueryPlan, SqlQuery
from app.core.schema_index import SchemaIndex
from app.core.redact import redact
import time
from contextlib import contextmanager
from app.core.plan_cache import key_for, get_plan, set_plan
from app.core.knowledge import build_structure_slice
from app.core.schema_slicer import shortlist_tables
from app.core.rag_gate import should_use_rag
from app.core.columns_touched import columns_touched as columns_touched_from_sql


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
        repo: Optional[PineconeRAGRepository],
        db: ReadOnlyDbExecutor,
        schema_text: str,                 # NEW: authoritative DB overview for prompts
        emitter: QEmitter,
        external_schemas_text: str = "",
        value_descriptions_text: str = "",
        identifiers_to_quote: Optional[Iterable[str]] = None,
        rag_enabled: bool = False,
        db_available: bool = True,
        naming_hints: str = "",
        schema_index: Optional[SchemaIndex] = None,
        catalogue: Optional["Catalogue"] = None,
    ):
        self.gemini = gemini
        self.embedder = embedder
        self.repo = repo
        self.db = db
        self.schema_text = schema_text
        self.emitter = emitter
        self.external_schemas_text = external_schemas_text
        self.value_descriptions_text = value_descriptions_text
        self._eid = 0
        self._quote_ids = set(identifiers_to_quote or [])
        self._use_rag = bool(rag_enabled)
        self._db_ok = bool(db_available)
        self._naming_hints = naming_hints
        self._schema_index = schema_index
        self._catalogue = catalogue
        self._trace_id = str(uuid.uuid4())

    @contextmanager
    def _step(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = int((time.perf_counter() - t0) * 1000)
            try:
                self._emit("metric", {"step": name, "latency_ms": dt})
            except Exception:
                pass

    def _emit(self, name: str, data: Dict[str, Any]) -> None:
        self._eid += 1
        payload = {"id": str(self._eid), "name": name, "data": {**data, "trace_id": self._trace_id}}
        self.emitter.send(payload)

    # === Main entry ===
    def execute(self, queries: List[str]) -> None:
        focus = queries[-1] if queries else ""
        self._emit("chat-step", {"step": "Understanding", "message": "Formulating plan..."})

        with self._step("Understanding"):
            # Plan cache (by normalized question)
            # include a simple schema signature so cache invalidates on schema change
            schema_sig = ""
            try:
                if self._catalogue:
                    import hashlib
                    parts: List[str] = []
                    for full, t in sorted(self._catalogue.tables.items()):
                        parts.append(full)
                        parts.extend(sorted(t.columns.keys()))
                    schema_sig = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
            except Exception:
                schema_sig = ""
            k = key_for(f"{focus}|{schema_sig}")
            cached = get_plan(k)
            if cached:
                try:
                    qp = QueryPlan(**cached)
                except Exception:
                    qp = self._analyze(queries, focus)
            else:
                qp = self._analyze(queries, focus)
                try:
                    set_plan(k, qp.model_dump())
                except Exception:
                    pass
        # Emit planning cache metric
        self._emit("metric", {"step": "Planning", "cache_hit": bool(cached)})
        self._emit("chat-step", {"step": "Understanding", "message": "Plan ready", "plan": qp.model_dump()})

        # Build a sliced STRUCTURE CATALOGUE for prompts (reduce tokens)
        structure_text = self.external_schemas_text or self.schema_text
        try:
            if self._catalogue:
                qmerge = " ".join([focus, *qp.rewrittenQueries])
                tops = shortlist_tables(qmerge, self._catalogue, k=5)
                if tops:
                    structure_text = build_structure_slice(self._catalogue, tops)
        except Exception:
            pass

        # Capability: if DB schema is available, validate target tables/columns
        capable, cap_reason, needed_data = self._capability_check(qp, focus)
        near_misses: List[str] = []
        if not capable and self._catalogue:
            keywords = ("employee", "headcount", "department", "status")
            for full, t in self._catalogue.tables.items():
                for cname in t.columns.keys():
                    if any(k in cname.lower() for k in keywords):
                        near_misses.append(f"{full}.{cname}")
                        if len(near_misses) >= 8:
                            break
                if len(near_misses) >= 8:
                    break
        self._emit("capability-check", {"status": "pass" if capable else "fail", "reason": cap_reason, "needed_data": needed_data or "", "near_misses": near_misses})
        if not capable:
            qp.searchStrategy = "no_answer"
            reason = qp.reasoning or ""
            # Avoid duplicating the same reason
            if cap_reason and (cap_reason not in reason):
                qp.reasoning = f"{reason + '; ' if reason else ''}{cap_reason}".strip()

        # Value canonicalization (simple heuristics using catalogue)
        value_map: Dict[str, Dict[str, any]] = {}
        if self._catalogue:
            lower_focus = (focus or "").lower()
            # Example: detect 'it department' phrases
            if "it department" in lower_focus or "information technology" in lower_focus:
                canon, suggestions = self._catalogue.values.canonicalize("department", "IT")
                if (canon != "IT") or suggestions:
                    value_map["department"] = {"input": "IT", "canonical": canon, "suggestions": suggestions}
        if value_map:
            self._emit("value-map", value_map)

        # --- RAG retrieval (optional; schema-first by default) ---
        rag_notes = ""
        doc_ids: List[str] = []
        # Defer RAG until after planning/validation; emit a placeholder status
        self._emit("chat-step", {"step": "Retrieval", "message": {"config_enabled": bool(self._use_rag and self.repo is not None), "used": False, "docs": 0, "reason": "deferred"}})

        sqls: List[SqlQuery] = []
        if qp.searchStrategy != "no_answer":
            # Planning
            with self._step("Planning"):
                self._emit("chat-step", {"step": "Formulating queries", "message": "Generating SQL candidates..."})
                sqls = self._generate_sql(queries, focus, qp, rag_notes, structure_override=structure_text)
            # Validation
            with self._step("Validation"):
                if self._schema_index or self._catalogue:
                    valid_sqls: List[SqlQuery] = []
                    policy_notes: List[str] = []
                    for sq in sqls:
                        bad_tables = []
                        for t in self._extract_tables(sq.query):
                            ok = False
                            if self._schema_index and self._schema_index.has_table(t):
                                ok = True
                            if self._catalogue and self._catalogue.has_table(t):
                                ok = True
                            if not ok:
                                bad_tables.append(t)
                        if bad_tables:
                            self._emit("chat-error", {"message": f"Unknown tables in SQL: {', '.join(bad_tables)}", "sql": sq.query})
                            continue
                        from app.core.sql_validation import validate_sql
                        issues_ok, issues = validate_sql(sq.query, self._schema_index)
                        if not issues_ok:
                            self._emit("schema-validation", {"status": "error", "issues": issues})
                            self._final_no_answer("Validation failed", qp, data_sources_list=[], doc_ids=doc_ids, sqls=[])
                            return
                        agg_errs = self._check_aggregations(sq.query)
                        if agg_errs:
                            self._emit("chat-error", {"message": f"Aggregation policy violation: {', '.join(agg_errs)}", "sql": sq.query})
                            continue
                        # Post-plan policy: require > 0 requests if that measure exists and query has no joins/group
                        try:
                            gate_ok = True
                            try:
                                from app.core.rag_gate import _HAVE as _HAVE_SQLGLOT  # reuse flag
                            except Exception:
                                _HAVE_SQLGLOT = False
                            if _HAVE_SQLGLOT:
                                import sqlglot
                                from sqlglot import expressions as exp
                                t = sqlglot.parse_one(sq.query, read="postgres")
                                has_join = any(True for _ in t.find_all(exp.Join))
                                has_group = any(True for _ in t.find_all(exp.Group))
                                if (not has_join) and (not has_group):
                                    tables = [str(x.this) for x in t.find_all(exp.Table) if x.this]
                                    target_table = tables[0] if tables else None
                                    if target_table and self._catalogue and self._catalogue.has_table(target_table):
                                        tinfo = self._catalogue.tables.get(target_table)
                                        if tinfo and ("total_requests_received" in tinfo.columns) and ("total_requests_received" not in sq.query):
                                            # inject WHERE total_requests_received > 0
                                            new_sql = f"SELECT * FROM ({sq.query}) AS sub WHERE total_requests_received > 0" if " WHERE " in sq.query.upper() else sq.query + " WHERE total_requests_received > 0"
                                            sq = SqlQuery(**{**sq.model_dump(), "query": new_sql})
                                            policy_notes.append("Filtered to rows with total_requests_received > 0")
                        except Exception:
                            pass
                        valid_sqls.append(sq)
                    sqls = valid_sqls
                self._emit("schema-validation", {"status": "ok", "count": len(sqls)})
        else:
            sqls = []
            self._emit("schema-validation", {"status": "skipped", "reason": "no_answer"})

        # Now run Retrieval after planning with a simple gate
        with self._step("Retrieval"):
            if self._use_rag and self.repo is not None and sqls:
                try:
                    gate_ok = should_use_rag(sqls[0].query)
                    if not gate_ok:
                        self._emit("chat-step", {"step": "Retrieval", "message": {"config_enabled": True, "used": False, "docs": 0, "reason": "skipped"}})
                    else:
                        merged = "\n".join([*queries, *qp.rewrittenQueries]) or focus
                        embs = self.embedder.get_batch_embeddings([merged]).embeddings
                        if embs:
                            hits = self.repo.find_nearest_by_cosine_distance(embs[0].values, top_k=8)
                            texts: List[str] = []
                            for h in hits:
                                hid = (h.get("id") if isinstance(h, dict) else None) or "?"
                                doc_ids.append(str(hid))
                                doc_txt = (h.get("document") if isinstance(h, dict) else "") or ""
                                if doc_txt:
                                    texts.append(str(doc_txt))
                            rag_notes = "\n---\n".join(texts)[:3000]
                        used = len(doc_ids) > 0
                        reason = "ok" if used else ("empty_index" if self.repo.is_empty() else "no_match")
                        self._emit("chat-step", {"step": "Retrieval", "message": {"config_enabled": True, "used": used, "docs": len(doc_ids), "reason": reason}})
                except Exception as ex:
                    self._emit("chat-step", {"step": "Retrieval", "message": {"config_enabled": True, "used": False, "docs": 0, "reason": f"error: {redact(str(ex))}"}})
            else:
                # Already emitted placeholder earlier
                pass
        
        self._emit("chat-sql", {"count": len(sqls), "queries": [s.model_dump() for s in sqls]})

        # Execute SQL(s)
        results: List[Dict[str, Any]] = []
        tables_used: set[str] = set()
        per_query_counts: List[int] = []
        with self._step("Execution"):
            if not self._db_ok:
                self._emit("chat-step", {"step": "Executing SQL", "message": "Database unavailable; skipping execution"})
                for i, sq in enumerate(sqls, 1):
                    self._emit("chat-error", {"query_index": i, "message": "Database unavailable; skipped execution", "sql": sq.query})
            else:
                for i, sq in enumerate(sqls, 1):
                    self._emit("chat-step", {"step": "Executing SQL", "message": f"Running {i}/{len(sqls)}"})
                    try:
                        rows = self.db.execute(sq.query, sq.params)
                        results.extend(rows)
                        per_query_counts.append(len(rows))
                        tables_used.update(self._extract_tables(sq.query))
                        # Stream a tiny preview
                        preview = rows[:5]
                        cols = list(preview[0].keys()) if preview else []
                        self._emit("chat-result", {
                            "query_index": i,
                            "returned_count": len(rows),
                            "columns": cols,
                            "preview_rows": preview,
                        })
                    except Exception as ex:
                        # Attempt a single auto-repair by quoting special identifiers
                        repaired = self._maybe_quote_identifiers(sq.query)
                        if repaired and repaired != sq.query:
                            try:
                                rows = self.db.execute(repaired, sq.params)
                                results.extend(rows)
                                per_query_counts.append(len(rows))
                                tables_used.update(self._extract_tables(repaired))
                                preview = rows[:5]
                                cols = list(preview[0].keys()) if preview else []
                                self._emit("chat-step", {"step": "Executing SQL", "message": "Auto-quoted identifiers and succeeded"})
                                self._emit("chat-result", {
                                    "query_index": i,
                                    "returned_count": len(rows),
                                    "columns": cols,
                                    "preview_rows": preview,
                                })
                                continue
                            except Exception as ex2:
                                self._emit("chat-error", {"query_index": i, "message": redact(str(ex2)), "sql": repaired})
                        # Unrepaired or failed after repair
                        self._emit("chat-error", {"query_index": i, "message": redact(str(ex)), "sql": sq.query})

        # Summarize results and docs
        self._emit("chat-step", {"step": "Retrieval", "message": {"sql_rows": len(results), "docs": len(doc_ids)}})

        # Gather references for narrator: data_source values + SQL used
        data_sources_list = sorted({str(r["data_source"]) for r in results if isinstance(r, dict) and r.get("data_source")})
        data_sources = ", ".join(data_sources_list) if data_sources_list else "none"
        sql_used = "\n\n".join(q.query for q in sqls) if sqls else ""

        # Compose narrator prompt
        prompt = self._answer_prompt(
            focus=focus,
            sql_rows=results,
            qp=qp,
            data_sources=data_sources,
            doc_ids=", ".join(doc_ids[:10]) if doc_ids else "none",
            sql_used=sql_used,
            rag_context=rag_notes or "(none)",
        )

        # Numeric truth binding and narration: bypass LLM for small aggregates
        from app.core.sql_validation import analyze_sql
        bypass_llm = False
        # Collect columns touched with best-effort qualifier
        columns_touched: List[str] = []
        for sq in sqls:
            cols_tmp = columns_touched_from_sql(sq.query)
            if cols_tmp:
                columns_touched.extend(cols_tmp)
            else:
                info_tmp = analyze_sql(sq.query)
                columns_touched.extend(sorted(info_tmp.get("columns") or []))

        emitted_any = False
        answer_parts: List[str] = []
        if results:
            is_small = len(results) <= 50
            first_info = analyze_sql(sqls[0].query) if sqls else {"has_agg": False, "has_group": False}
            if is_small and (first_info.get("has_agg") or first_info.get("has_group")):
                # format rows as markdown table directly
                if isinstance(results[0], dict) and len(results[0]) == 1:
                    val = next(iter(results[0].values()))
                    try:
                        n = float(val)
                        n_disp = int(n) if float(n).is_integer() else n
                    except Exception:
                        n_disp = val
                    header = list(results[0].keys())[0]
                    if ("distinct" in (focus or "").lower()) and ("agenc" in (focus or "").lower()):
                        header = "distinct_agencies"
                    answer_md = (
                        f"Answer: **{n_disp}**.\n\n"
                        f"| {header} |\n|---:|\n| {n_disp} |"
                    )
                elif isinstance(results[0], dict):
                    headers = list(results[0].keys())
                    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                    for r in results[:50]:
                        lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
                    answer_md = "\n".join(lines)
                else:
                    answer_md = json.dumps(results[:50], indent=2, default=str)
                emitted_any = True
                answer_parts = [answer_md]
                bypass_llm = True
        if not bypass_llm:
            with self._step("Narration"):
                self._emit("chat-step", {"step": "Narration", "message": "Composing answer..."})
                try:
                    for chunk in self.gemini.narrator_stream(prompt):
                        if chunk:
                            emitted_any = True
                            answer_parts.append(chunk)
                except Exception as ex:
                    self._emit("chat-step", {"step": "Answering", "message": f"Narrator error: {redact(str(ex))}"})

        if not emitted_any:
            if qp.searchStrategy == "no_answer":
                # Constructive guidance copy
                fb = (
                    "I can’t answer this from the current datasets. They cover FOI requests, charges, and FOI/IPS staffing years but not department-level headcount.\n\n"
                    "### What you can do now\n"
                    "- Try one of the suggested questions below (uses available data).\n"
                    "- Or provide the dataset below and I’ll answer automatically.\n"
                )
            else:
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
                fb = "".join(fallback_parts)
            answer_parts.append(fb)
            self._emit("chat-result", {"chunk": fb})

        # Emit a final structured summary for clients that prefer JSON
        try:
            # Build references.tables from final SQL list if not collected during execution
            if not tables_used:
                for sq in sqls:
                    tables_used.update(self._extract_tables(sq.query))
            rag_used = len(doc_ids) > 0
            source = "schema_json_prompt+rag" if rag_used else "schema_json_prompt"
            structured = {
                "answer_markdown": "".join(answer_parts),
                "plan": qp.model_dump(),
                "references": {
                    "tables": sorted(tables_used),
                    "data_sources": data_sources_list,
                    "rag_doc_ids": doc_ids[:20] if rag_used else [],
                    "rag_enabled": self._use_rag,
                    "rag_used": rag_used,
                    "source": source,
                    "needed_data": (needed_data or "") if qp.searchStrategy == "no_answer" else "",
                },
                "sql_used": sql_used,
                "sql_list": [q.query for q in sqls],
                "rows_sample": results[:10],
                "row_counts": per_query_counts,
                "columns_touched": sorted(set(columns_touched)),
                "bypass_llm": bypass_llm,
            }
            # Optional coverage note heuristic (FY vs calendar range)
            try:
                import re as _re
                if any(_re.search(r"2019[\-_]20", t) for t in tables_used):
                    m = _re.search(r"(\d{4}-\d{2}-\d{2}).*(\d{4}-\d{2}-\d{2})", focus)
                    if m:
                        cn = "Answered from FY2019–20 table (2019-07-01..2020-06-30); requested range differs."
                        structured["references"]["coverage_note"] = cn  # type: ignore[index]
            except Exception:
                pass
            # Add detailed needed data scaffold and followups on no_answer
            if qp.searchStrategy == "no_answer" and (needed_data or "").strip():
                structured["references"]["needed_data_details"] = format_needed_data(needed_data)  # type: ignore[index]
            if qp.searchStrategy == "no_answer" and self._catalogue:
                followups = suggest_followups(self._catalogue)
                if followups:
                    self._emit("chat-suggest", {"followups": followups})
                    structured["followups"] = followups
            self._emit("chat-structured", structured)
            total_ms = int((time.perf_counter() - t_total_start) * 1000)
            self._emit("metric", {"step": "Summary", "total_ms": total_ms, "answerable": qp.searchStrategy != "no_answer", "bypass_llm": bypass_llm, "rag_used": rag_used, "db_rows": sum(per_query_counts) if per_query_counts else 0})
        except Exception:
            pass

    # === LLM helpers ===
    def _analyze(self, queries: List[str], focus: str) -> QueryPlan:
        from app.prompts.versioned.v1.analyst import ANALYST_PROMPT
        s = self.gemini.analyst(ANALYST_PROMPT.format(
            SCHEMA_OVERVIEW=self.schema_text,
            STRUCTURE_CATALOGUE=self.external_schemas_text or "(none)",
            VALUE_CATALOGUE=self.value_descriptions_text or "(none)",
            NAMING_HINTS=self._naming_hints or "(none)",
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

    def _generate_sql(self, queries: List[str], focus: str, qp: QueryPlan, rag_notes: str = "", structure_override: Optional[str] = None) -> List[SqlQuery]:
        from app.prompts.versioned.v1.analyst import SQL_GEN_PROMPT
        s = self.gemini.analyst(SQL_GEN_PROMPT.format(
            SCHEMA_OVERVIEW=self.schema_text,
            STRUCTURE_CATALOGUE=structure_override or self.external_schemas_text or "(none)",
            VALUE_CATALOGUE=self.value_descriptions_text or "(none)",
            NAMING_HINTS=self._naming_hints or "(none)",
            FOCUS=focus,
            REWRITTEN=json.dumps(qp.rewrittenQueries),
            RAG_CONTEXT=rag_notes or "(none)",
        ))
        try:
            arr = json.loads(s.replace("```json","").replace("```","").strip())
            return [SqlQuery(**x) for x in arr if isinstance(x, dict) and "query" in x]
        except Exception:
            return []

    def _answer_prompt(self, *, focus, sql_rows, qp, data_sources, doc_ids, sql_used, rag_context: str) -> str:
        from app.prompts.versioned.v1.narrator import NARRATOR_PROMPT
        return NARRATOR_PROMPT.format(
            FOCUS=focus,
            STRATEGY=qp.searchStrategy,
            SQL_ROWS=json.dumps(sql_rows[:200], default=str),  # avoid Decimal issues
            DATA_SOURCES=data_sources,
            DOC_IDS=doc_ids,
            SQL_USED=sql_used,
            EXTERNAL_SCHEMAS=self.external_schemas_text or "(none)",
            VALUE_DESCRIPTIONS=self.value_descriptions_text or "(none)",
            RAG_CONTEXT=rag_context,
            NAMING_HINTS=self._naming_hints or "(none)",
        )

    def _final_no_answer(self, reason: str, qp: QueryPlan, *, data_sources_list: List[str], doc_ids: List[str], sqls: List[SqlQuery]) -> None:
        try:
            structured = {
                "answer_markdown": (
                    "I can’t answer this from the current datasets. "
                    + (reason or "")
                ),
                "plan": qp.model_dump(),
                "references": {
                    "tables": [],
                    "data_sources": data_sources_list,
                    "rag_doc_ids": doc_ids[:20] if self._use_rag else [],
                    "rag_enabled": self._use_rag,
                    "source": "schema_json_prompt" + ("+rag" if self._use_rag else ""),
                },
                "sql_used": "",
                "sql_list": [],
                "rows_sample": [],
                "row_counts": [],
                "columns_touched": [],
                "bypass_llm": True,
            }
            self._emit("chat-structured", structured)
        finally:
            pass

    def _maybe_quote_identifiers(self, sql: str) -> str:
        # Only attempt if we have suspicious identifiers
        if not self._quote_ids:
            return sql
        out = sql
        for ident in self._quote_ids:
            # Skip if already quoted
            q = f'"{ident}"'
            if q in out:
                continue
            # Replace standalone occurrences with quoted variant
            # Use simple boundaries around word characters and underscore
            import re
            pattern = re.compile(rf'(?<!\")\b{re.escape(ident)}\b')
            out = pattern.sub(q, out)
        return out

    def _extract_tables(self, sql: str) -> List[str]:
        import re
        pat = re.compile(r"\b(from|join)\s+((?:\"[^\"]+\"|[a-zA-Z_][\w]*)\.(?:\"[^\"]+\"|[a-zA-Z_][\w]*)|(?:\"[^\"]+\"|[a-zA-Z_][\w]*))", re.I)
        tables: List[str] = []
        for m in pat.finditer(sql):
            t = m.group(2)
            # Normalize quoted identifiers to bare form for references list
            norm = t.replace('"', '')
            tables.append(norm)
        return tables

    def _capability_check(self, qp: QueryPlan, focus: str) -> Tuple[bool, str, Optional[str]]:
        # If planner already decided no_answer, honor it
        # Prepare a needed_data hint if we can infer intent
        needed_hint: Optional[str] = None
        lower_q = (focus or "").lower()
        if any(k in lower_q for k in ("headcount", "people working", "employees", "staff count")):
            needed_hint = "employees(department,status,employment_start_date,employment_end_date)"
        if qp.searchStrategy and qp.searchStrategy.lower() == "no_answer":
            return False, qp.reasoning or "Planner returned no_answer", needed_hint
        # Use catalogue/schema for capability checks (catalogue may be present even if DB is down)
        has_schema = bool(self._schema_index) or bool(self._catalogue)
        if not has_schema:
            return True, "", None
        def has_table(name: str) -> bool:
            if self._schema_index and self._schema_index.has_table(name):
                return True
            if self._catalogue and self._catalogue.has_table(name):
                return True
            return False
        def has_column(table: str, column: str) -> bool:
            if self._schema_index and self._schema_index.has_column(table, column):
                return True
            if self._catalogue and self._catalogue.has_column(table, column):
                return True
            return False
        # Heuristic: detect headcount intent and require employee+department columns
        if any(k in lower_q for k in ("headcount", "people working", "employees", "staff count")):
            # naive search for a table with department-like column
            dept_ok = False
            for tname in (self._catalogue.list_tables() if self._catalogue else []):
                tshort = tname.split(".", 1)[1]
                if has_column(tshort, "department"):
                    dept_ok = True
                    break
            if not dept_ok:
                return False, "headcount requires a person table with a department column; none present", needed_hint
        # If planner specified explicit target tables/columns, validate quickly
        # If planner specified target tables/columns, verify they exist
        bad_tables = [t for t in qp.targetTables if not has_table(t)]
        bad_columns = [c for c in qp.keyColumns if not self._column_exists_any(c)]
        if bad_tables or bad_columns:
            parts: List[str] = []
            if bad_tables:
                parts.append(f"unknown tables: {', '.join(bad_tables)}")
            if bad_columns:
                parts.append(f"unknown columns: {', '.join(bad_columns)}")
            return False, "Schema validation failed: " + "; ".join(parts), None
        return True, "", None

    def _check_aggregations(self, sql: str) -> List[str]:
        if not self._catalogue:
            return []
        import re
        errs: List[str] = []
        # find sum/avg(column)
        for func in ("sum", "avg"):
            for m in re.finditer(rf"{func}\s*\(\s*([\w\.\"]+)\s*\)", sql, flags=re.I):
                ident = m.group(1).replace('"', '')
                # ident may be table.column or just column — try to split
                if "." in ident:
                    t, c = ident.split(".", 1)
                    role = self._catalogue.role(t, c)
                else:
                    # try against any table: accept if any table has it as measure
                    role = None
                    for full in self._catalogue.list_tables():
                        schema, t = full.split(".", 1)
                        rr = self._catalogue.role(t, ident)
                        if rr:
                            role = rr
                            break
                if role and role != "measure":
                    errs.append(f"{func.upper()} on non-measure column {ident}")
        return errs

    def _column_exists_any(self, name: str) -> bool:
        # name may be qualified table.column or just column
        if not self._schema_index:
            return True
        if "." in name:
            t, c = name.rsplit(".", 1)
            return self._schema_index.has_column(t, c)
        # search across tables
        for (s, t), cols in self._schema_index.tables.items():
            if name in cols:
                return True
        return False
