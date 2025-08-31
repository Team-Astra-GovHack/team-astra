# app/jobs/ingest_schema.py
"""
Reads Postgres information_schema, creates short Markdown snippets per table
(+ up to 5 sample rows), embeds with Gemini, and upserts into Pinecone.
"""
from __future__ import annotations
import json, decimal
from sqlalchemy import create_engine, text
from app.settings import Settings
from app.core.embedding_engine import GeminiEmbeddingEngine
from app.core.rag_repository import PineconeRAGRepository


def _json_default(o):
    # Keep numbers numeric: convert Decimal -> float; fallback to str for anything else
    if isinstance(o, decimal.Decimal):
        return float(o)
    return str(o)


def main() -> None:
    s = Settings()

    eng = create_engine(s.DB_URL_RO, isolation_level="AUTOCOMMIT")

    embedder = GeminiEmbeddingEngine(
        api_key=s.GEMINI_API_KEY,
        model=s.GEMINI_MODEL_EMBEDDING,
        dim=s.PINECONE_DIM,
    )
    repo = PineconeRAGRepository(
        api_key=s.PINECONE_API_KEY,
        index_name=s.PINECONE_INDEX,
        namespace=s.PINECONE_NAMESPACE,
    )

    with eng.begin() as conn:
        tables = conn.execute(text("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type='BASE TABLE'
              AND table_schema NOT IN ('pg_catalog','information_schema')
              AND table_name NOT IN ('_prisma_migrations')  -- ignore utility tables
            ORDER BY table_schema, table_name
        """)).all()

        if not tables:
            print("No tables found outside system schemas.")
            return

        for (schema, table) in tables:
            try:
                cols = conn.execute(text("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema=:schema AND table_name=:table
                    ORDER BY ordinal_position
                """), {"schema": schema, "table": table}).mappings().all()

                sample = conn.execute(
                    text(f'SELECT * FROM "{schema}"."{table}" LIMIT 5')
                ).mappings().all()

                md_parts = [
                    f"# {schema}.{table}",
                    "## Columns",
                    *[
                        f"- `{c['column_name']}`: {c['data_type']}, nullable={c['is_nullable']}"
                        for c in cols
                    ],
                    "## Sample rows (up to 5):",
                    "```json",
                    json.dumps([dict(r) for r in sample], ensure_ascii=False, indent=2, default=_json_default),
                    "```",
                ]
                md = "\n".join(md_parts)

                emb = embedder.get_batch_embeddings([md]).embeddings[0].values
                repo.upsert(
                    doc_id=f"{schema}.{table}",
                    content=md,
                    metadata={"type": "table", "schema": schema, "table": table},
                    embedding=emb,
                )
                print("Upserted", f"{schema}.{table}")

            except Exception as ex:
                print(f"Skip {schema}.{table}: {ex}")


if __name__ == "__main__":
    main()
