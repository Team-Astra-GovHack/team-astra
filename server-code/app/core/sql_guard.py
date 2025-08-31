# app/core/sql_guard.py
import re

# Allow only safe, single-statement, read-only queries
ALLOW = re.compile(r"^\s*(?:select|show|describe)\b", re.I)
STRIP = re.compile(r"((?s)/\*.*?\*/)|(--.*?$)", re.M)  # strip /* */ and -- comments
DANGERS = re.compile(r"\b(insert|update|delete|merge|alter|drop|truncate|grant|revoke|create)\b", re.I)

def sanitize_sql(sql: str, default_limit: int) -> str:
    s = STRIP.sub(" ", sql).strip()
    if ";" in s:
        raise ValueError("Multiple statements not allowed.")
    if (not ALLOW.search(s)) or DANGERS.search(s):
        raise ValueError("Only read-only queries are allowed.")
    # Inject LIMIT if missing
    if re.search(r"\blimit\s+\d+\b", s, re.I) is None:
        s = f"{s} LIMIT {int(default_limit)}"
    return s
