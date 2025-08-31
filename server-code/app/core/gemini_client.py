# app/core/gemini_client.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Generator, Optional

import google.generativeai as genai
try:
    # Present when google-api-core is installed (pulled by google-generativeai)
    from google.api_core.exceptions import ResourceExhausted
except Exception:  # pragma: no cover
    ResourceExhausted = Exception  # type: ignore


@dataclass
class GeminiClient:
    api_key: str
    # Primary model used for both analyst and narrator unless overridden.
    model: str = "gemini-1.5-pro"
    # Fallback model used when rate limited or quota-exhausted.
    fallback_model: str = "gemini-1.5-flash"
    temperature: float = 0.2

    def __post_init__(self) -> None:
        genai.configure(api_key=self.api_key)
        self._primary = genai.GenerativeModel(self.model)
        self._fallback = (
            genai.GenerativeModel(self.fallback_model)
            if self.fallback_model and self.fallback_model != self.model
            else self._primary
        )

    def _try_generate(self, prompt: str, *, temperature: float, stream: bool = False):
        """Try primary model; on quota (429) fall back once to fallback model."""
        try:
            return self._primary.generate_content(
                prompt,
                generation_config={"temperature": temperature},
                stream=stream,
            )
        except ResourceExhausted:
            # Switch to fallback model once
            if self._fallback is self._primary:
                raise
            return self._fallback.generate_content(
                prompt,
                generation_config={"temperature": temperature},
                stream=stream,
            )
        except Exception as e:
            # Heuristic: if looks like quota/rate limit, fall back once
            msg = str(e).lower()
            if ("429" in msg or "quota" in msg or "rate" in msg) and (self._fallback is not self._primary):
                return self._fallback.generate_content(
                    prompt,
                    generation_config={"temperature": temperature},
                    stream=stream,
                )
            raise

    def analyst(self, prompt: str) -> str:
        """
        Synchronous call used for planning/JSON generation (Analyst, SQL planner).
        Keep temperature low to minimize hallucinations.
        """
        try:
            resp = self._try_generate(
                prompt,
                temperature=0.0,
                stream=False,
            )
            # Prefer .text, fall back to first candidate if needed
            if getattr(resp, "text", None):
                return resp.text
            try:
                return resp.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
            except Exception:
                return ""
        except Exception:
            # Gracefully degrade so upstream can repair/continue
            return ""

    def narrator_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Stream tokens/chunks for the Narrator role.
        """
        try:
            stream = self._try_generate(prompt, temperature=0.3, stream=True)
            for ev in stream:
                chunk = getattr(ev, "text", "") or ""
                if chunk:
                    yield chunk
        except Exception:
            # Let caller decide how to handle narrator failure
            return
