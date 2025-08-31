# app/core/rag_repository.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec

class PineconeRAGRepository:
    """
    Thin repository over Pinecone v3 client (serverless).
    Assumes the index already exists; use ensure_index() if you want to create it in code.
    """
    def __init__(self, api_key: str, index_name: str, namespace: Optional[str] = None):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.namespace = namespace

    def ensure_index(self, name: str, dimension: int, metric: str = "cosine",
                     cloud: str = "aws", region: str = "us-east-1") -> None:
        existing = {i.name for i in self.pc.list_indexes()}
        if name not in existing:
            self.pc.create_index(
                name=name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

    def upsert(self, doc_id: str, content: str, metadata: Dict[str, Any], embedding: List[float]) -> None:
        meta = {**metadata, "content": content}
        self.index.upsert(
            vectors=[{"id": doc_id, "values": embedding, "metadata": meta}],
            namespace=self.namespace,
        )

    def find_nearest_by_cosine_distance(self, vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        res = self.index.query(
            vector=vector,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            namespace=self.namespace,
        )
        out: List[Dict[str, Any]] = []
        for m in (res.get("matches") or []):
            md = m.get("metadata") or {}
            out.append({
                "id": m.get("id"),
                "score": m.get("score"),
                "document": md.get("content", "") or "",
                "metadata": md,
            })
        return out

    def is_empty(self) -> bool:
        """Best-effort check whether index has any vectors. Returns False on error."""
        try:
            stats = self.index.describe_index_stats()
            # pinecone v3 stats dict may include total_vector_count
            tv = stats.get("total_vector_count") if isinstance(stats, dict) else None
            return bool(tv == 0)
        except Exception:
            return False
