# app/core/sql_guard.py
import re
from app.core.sql_policy import should_inject_limit

# Allow only safe, single-statement, read-only queries
ALLOW = re.compile(r"^\s*(?:select|with)\b", re.I)
STRIP = re.compile(r"/\*.*?\*/|--.*?$", flags=re.M | re.S) # strip /* */ and -- comments
DANGERS = re.compile(r"\b(insert|update|delete|merge|alter|drop|truncate|grant|revoke|create|copy|vacuum|analyze|explain|commit|rollback|begin|set|reset)\b", re.I)
PSQL_META = re.compile(r"(^|\s)\\\\\w+", re.I)  # \gdesc, \dt, etc.

def sanitize_sql(sql: str, default_limit: int) -> str:
    s = STRIP.sub(" ", sql).strip()
    if ";" in s:
        raise ValueError("Multiple statements not allowed.")
    if (not ALLOW.search(s)) or DANGERS.search(s) or PSQL_META.search(s):
        raise ValueError("Only read-only queries are allowed.")
    # Inject LIMIT if missing and query is not aggregate/grouped
    if re.search(r"\blimit\s+\d+\b", s, re.I) is None and should_inject_limit(s):
        s = f"{s} LIMIT {int(default_limit)}"
    return s
