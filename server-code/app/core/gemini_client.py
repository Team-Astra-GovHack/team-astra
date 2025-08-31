# app/core/gemini_client.py
from __future__ import annotations
import json
import requests
from typing import Iterator, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Optional SDK; we gracefully fall back to REST if it's missing
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None


class GeminiClient:
    def __init__(self, api_key: str, analyst_model: str = "gemini-2.5-flash", narrator_model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.analyst_model = analyst_model
        self.narrator_model = narrator_model
        self._sdk = None
        if genai:
            try:
                genai.configure(api_key=api_key)
                self._sdk = genai
            except Exception:
                # If SDK init fails, continue with REST fallback
                self._sdk = None

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type((requests.RequestException,)),
    )
    def _generate(self, model: str, text: str) -> str:
        """Generate a single non-streamed response from Gemini."""
        if self._sdk:
            resp = self._sdk.GenerativeModel(model).generate_content(text)
            return (getattr(resp, "text", "") or "").strip()

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        r = requests.post(url, json={"contents": [{"parts": [{"text": text}]}]}, timeout=60)
        r.raise_for_status()
        data = r.json()
        cand = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return (cand or "").strip()

    def analyst(self, prompt: str) -> str:
        return self._generate(self.analyst_model, prompt)

    def narrator_stream(self, prompt: str) -> Iterator[str]:
        """Stream narrator output if SDK available; otherwise fall back to one-shot REST."""
        if self._sdk:
            model = self._sdk.GenerativeModel(self.narrator_model)
            for ev in model.generate_content(prompt, stream=True):
                if getattr(ev, "text", None):
                    yield ev.text
            return
        # REST fallback (non-streaming)
        yield self._generate(self.narrator_model, prompt)

    def json_only(self, model: str, prompt: str) -> Optional[dict]:
        raw = self._generate(model, prompt)
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return None
