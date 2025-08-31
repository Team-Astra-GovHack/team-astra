# app/core/embedding_engine.py
from __future__ import annotations
import requests
from dataclasses import dataclass
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


@dataclass
class Embedding:
    values: List[float]


@dataclass
class BatchEmbedding:
    embeddings: List[Embedding]


class GeminiEmbeddingEngine:
    def __init__(self, api_key: str, model: str = "models/text-embedding-004", dim: int = 768, timeout: int = 60):
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.timeout = timeout

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type((requests.RequestException,)),
    )
    def get_batch_embeddings(self, texts: List[str]) -> BatchEmbedding:
        """Batch-embed a list of texts using Gemini embeddings API."""
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/text-embedding-004:batchEmbedContents?key={self.api_key}"
        )
        payload = {
            "requests": [
                {"model": self.model, "content": {"parts": [{"text": t}]}}
                for t in texts
            ]
        }
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json().get("embeddings", []) or []
        embs = [Embedding(values=[float(x) for x in e.get("values", [])]) for e in data]
        # Optional safety: pad/trim to expected dim
        for e in embs:
            if len(e.values) > self.dim:
                e.values = e.values[: self.dim]
            elif len(e.values) < self.dim:
                e.values.extend([0.0] * (self.dim - len(e.values)))
        return BatchEmbedding(embeddings=embs)
