# app/core/embedding_engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests


@dataclass
class Embedding:
    values: List[float]


@dataclass
class BatchEmbedding:
    embeddings: List[Embedding]


class GeminiEmbeddingEngine:
    def __init__(self, api_key: str, model: str = "embedding-001", dim: int = 768, timeout: int = 30):
        """
        Gemini Embedding Engine.

        Args:
            api_key: Google Gemini API key
            model: Embedding model name (default: text-embedding-004)
            dim: Expected embedding dimensions
            timeout: Request timeout (seconds)
        """
        self.api_key = api_key
        # Normalize model name for Generative Language API
        m = (model or "").strip()
        # text-embedding-004 is Vertex model name; for Generative Language API use embedding-001
        if m.endswith("text-embedding-004") or "/text-embedding-004" in m:
            m = "embedding-001"
        # strip any leading "models/"
        if m.startswith("models/"):
            m = m[len("models/"):]
        self.model = m or "embedding-001"
        self.dim = dim
        self.timeout = timeout

        # REST endpoint for batch embeddings
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:batchEmbedContents?key={self.api_key}"

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type((requests.RequestException,)),
    )
    def get_batch_embeddings(self, texts: List[str]) -> BatchEmbedding:
        """Batch-embed a list of texts using Gemini embeddings API."""
        payload = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]}
                }
                for t in texts
            ]
        }

        r = requests.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()

        # Extract embeddings from API response
        responses = r.json().get("responses", []) or []
        embs: List[Embedding] = []
        for resp in responses:
            values = resp.get("embedding", {}).get("values", [])
            emb = Embedding(values=[float(x) for x in values])

            # Safety: pad/trim to expected dimension
            if len(emb.values) > self.dim:
                emb.values = emb.values[: self.dim]
            elif len(emb.values) < self.dim:
                emb.values.extend([0.0] * (self.dim - len(emb.values)))

            embs.append(emb)

        return BatchEmbedding(embeddings=embs)
