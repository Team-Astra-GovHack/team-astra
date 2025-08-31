from __future__ import annotations

def format_needed_data(spec: str) -> str:
    """
    Turn a compact spec like "employees(department,status,employment_start_date,employment_end_date)"
    into a schema scaffold + example SELECT.
    """
    spec = (spec or "").strip()
    if not spec or "(" not in spec or ")" not in spec:
        return ""
    tbl = spec.split("(", 1)[0].strip()
    cols_str = spec[spec.find("(") + 1 : spec.rfind(")")]
    cols = [c.strip() for c in cols_str.split(",") if c.strip()]
    if not tbl or not cols:
        return ""
    create_cols = ",\n  ".join(f"{c} TEXT" for c in cols)
    create_sql = f"CREATE TABLE {tbl} (\n  id INTEGER PRIMARY KEY,\n  {create_cols}\n);"
    example = f"SELECT COUNT(*) AS headcount FROM {tbl} WHERE department = 'IT' AND status = 'active';"
    return (
        "**Required dataset**\n\n" +
        "```sql\n" + create_sql + "\n```\n\n" +
        "**Example query once available**\n\n" +
        "```sql\n" + example + "\n```\n"
    )

