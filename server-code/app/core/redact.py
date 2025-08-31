from __future__ import annotations
import re

API_KEY_RE = re.compile(r"(key=)([^&\s]+)", re.I)

def redact(s: str) -> str:
    try:
        out = API_KEY_RE.sub(r"\1***REDACTED***", s)
        # Heuristic: mask long tokens
        out = re.sub(r"([A-Za-z0-9_\-]{24,})", "***", out)
        return out
    except Exception:
        return "(redacted error)"

