from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _try_read_json(paths: List[Path]) -> object | None:
    for p in paths:
        try:
            if p.is_file():
                return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def build_additional_schema_text(base_dir: Path, filename: str | None = None,
                                 char_limit: int = 6000) -> str:
    """
    Load dataset_schemas.json and render a compact, prompt-friendly overview.
    Limits size to keep token usage reasonable.
    """
    candidates: List[Path] = []
    if filename:
        candidates.append(Path(filename))
    # Try alongside server-code, then repo root
    candidates.extend([
        base_dir / "dataset_schemas.json",
        base_dir.parent / "dataset_schemas.json",
    ])
    data = _try_read_json(candidates)
    if not isinstance(data, dict):
        return "(no external dataset schemas)"

    lines: List[str] = ["=== External Dataset Schemas ==="]
    # Render at most 8 datasets, 12 columns each
    for ds_name, cols in list(data.items())[:8]:
        if not isinstance(cols, list):
            continue
        sample = []
        for c in cols[:12]:
            if isinstance(c, dict):
                n = str(c.get("column_name", "?"))
                t = str(c.get("data_type", ""))
                sample.append(f"{n}({t})" if t else n)
        lines.append(f"- {ds_name}: {', '.join(sample)}")
        if sum(len(x) for x in lines) > char_limit:
            break

    out = "\n".join(lines)
    return out[:char_limit]


def build_value_descriptions_text(base_dir: Path, filename: str | None = None,
                                  char_limit: int = 6000) -> str:
    """
    Load value_descriptions_min.json and render concise column descriptions.
    """
    candidates: List[Path] = []
    if filename:
        candidates.append(Path(filename))
    candidates.extend([
        base_dir / "value_descriptions_min.json",
        base_dir.parent / "value_descriptions_min.json",
    ])
    data = _try_read_json(candidates)
    if not isinstance(data, dict):
        return "(no value descriptions)"

    lines: List[str] = ["=== Column Value Descriptions (subset) ==="]
    shown = 0
    # Render up to 20 descriptions across sections
    for section, items in data.items():
        if shown >= 20:
            break
        if not isinstance(items, list):
            continue
        for it in items:
            if shown >= 20:
                break
            if not isinstance(it, dict):
                continue
            name = str(it.get("column_name", "?"))
            desc = str(it.get("value_description", "")).strip()
            if not desc:
                continue
            # Keep each line short
            desc = desc.replace("\n", " ")
            if len(desc) > 280:
                desc = desc[:277] + "..."
            lines.append(f"- {name}: {desc}")
            shown += 1

    out = "\n".join(lines)
    return out[:char_limit]


def collect_problem_identifiers(base_dir: Path,
                                dataset_path: str | None = None,
                                value_path: str | None = None) -> Set[str]:
    """
    Collect identifier names from JSON that likely require quoting in SQL
    (e.g., start with a digit or contain spaces/hyphens). Dots are ignored
    because they act as qualifiers in SQL and rarely represent actual column names.
    """
    out: Set[str] = set()
    ds = _try_read_json([Path(dataset_path)] if dataset_path else [])
    if not ds:
        ds = _try_read_json([base_dir / "dataset_schemas.json", base_dir.parent / "dataset_schemas.json"]) or {}
    if isinstance(ds, dict):
        for _, cols in ds.items():
            if isinstance(cols, list):
                for c in cols:
                    if isinstance(c, dict):
                        n = str(c.get("column_name", ""))
                        if not n:
                            continue
                        if n[0].isdigit() or (" " in n) or ("-" in n):
                            if "." not in n:
                                out.add(n)
    vd = _try_read_json([Path(value_path)] if value_path else [])
    if not vd:
        vd = _try_read_json([base_dir / "value_descriptions_min.json", base_dir.parent / "value_descriptions_min.json"]) or {}
    if isinstance(vd, dict):
        for _, items in vd.items():
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        n = str(it.get("column_name", ""))
                        if not n:
                            continue
                        if n[0].isdigit() or (" " in n) or ("-" in n):
                            if "." not in n:
                                out.add(n)
    return out


def build_naming_hints(base_dir: Path,
                       dataset_path: str | None = None,
                       value_path: str | None = None,
                       char_limit: int = 4000) -> str:
    """
    Produce concise naming hints to help the LLM map dataset column names
    to DB-safe identifiers and know which require quoting.

    - Suggest aliases for dotted-suffix columns like foo.1 -> foo1
    - List identifiers likely needing quotes (start with digit / contain spaces-hyphens)
    - Suggest likely table names by stripping .csv from dataset names
    """
    ds = _try_read_json([Path(dataset_path)] if dataset_path else [])
    if not ds:
        ds = _try_read_json([base_dir / "dataset_schemas.json", base_dir.parent / "dataset_schemas.json"]) or {}
    vd = _try_read_json([Path(value_path)] if value_path else [])
    if not vd:
        vd = _try_read_json([base_dir / "value_descriptions_min.json", base_dir.parent / "value_descriptions_min.json"]) or {}

    dotted_aliases: Set[Tuple[str, str]] = set()
    quote_needed: Set[str] = set()
    table_candidates: Set[str] = set()

    def _scan_cols(items):
        nonlocal dotted_aliases, quote_needed
        if not isinstance(items, list):
            return
        for c in items:
            if not isinstance(c, dict):
                continue
            name = str(c.get("column_name", "")).strip()
            if not name:
                continue
            # Dotted numeric suffix -> suggest compact alias without dot
            if "." in name:
                base, _, suf = name.rpartition(".")
                if suf.isdigit() and base:
                    dotted_aliases.add((name, f"{base}{suf}"))
            # Quote-needed identifiers
            if name[0:1].isdigit() or (" " in name) or ("-" in name):
                quote_needed.add(name)

    # From dataset schemas
    if isinstance(ds, dict):
        for ds_name, cols in ds.items():
            if isinstance(ds_name, str) and ds_name.endswith(".csv"):
                table_candidates.add(ds_name[:-4])
            _scan_cols(cols)

    # From value descriptions
    if isinstance(vd, dict):
        for _, items in vd.items():
            _scan_cols(items)

    lines: List[str] = ["=== Naming Hints ==="]
    if table_candidates:
        tc = ", ".join(sorted(table_candidates)[:30])
        lines.append(f"Dataset -> table candidates (strip .csv): {tc}")
    if quote_needed:
        qn = ", ".join(sorted(quote_needed)[:30])
        lines.append(f"Quote-needed identifiers: {qn}")
    if dotted_aliases:
        lines.append("Dotted suffix aliases (suggested):")
        for src, dst in list(sorted(dotted_aliases))[:40]:
            lines.append(f"- {src} -> {dst}")

    out = "\n".join(lines)
    return out[:char_limit]


def build_structure_slice(cat: "Catalogue", table_names: List[str], max_cols: int = 12, char_limit: int = 6000) -> str:
    try:
        from app.core.catalogue import Catalogue  # type: ignore
    except Exception:
        pass
    lines: List[str] = ["=== Structure Catalogue (sliced) ==="]
    for full in table_names:
        t = cat.tables.get(full)
        if not t:
            continue
        cols = ", ".join(list(t.columns.keys())[:max_cols])
        lines.append(f"- {full}\n  columns: {cols}")
        if sum(len(x) for x in lines) > char_limit:
            break
    return "\n".join(lines)[:char_limit]
