from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import json


@dataclass
class ColumnInfo:
    name: str
    type: str = ""
    nullable: bool = True
    description: str = ""
    role: str = "dimension"  # "dimension" | "measure"


@dataclass
class TableInfo:
    name: str  # unqualified table name (e.g., agency_foi_data...)
    schema: str = "public"
    description: str = ""
    columns: Dict[str, ColumnInfo] = field(default_factory=dict)
    pk: Set[str] = field(default_factory=set)
    # fk: column -> referenced table (heuristic)
    fk: Dict[str, str] = field(default_factory=dict)


@dataclass
class ValueDict:
    # column_name (lower) -> alias_map (alias_lower -> canonical)
    aliases: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def canonicalize(self, column: str, value: str) -> Tuple[str, List[str]]:
        amap = self.aliases.get(column.lower()) or {}
        v = value.strip()
        k = v.lower().strip()
        if k in amap:
            return amap[k], []
        # basic normalization
        norm = k.replace(".", "").replace(" ", "")
        for alias, canon in amap.items():
            if alias.replace(".", "").replace(" ", "") == norm:
                return canon, []
        # suggestions (top 5 by simple substring)
        suggestions = []
        for alias in amap.keys():
            if k in alias or alias in k:
                suggestions.append(alias)
            if len(suggestions) >= 5:
                break
        return v, suggestions


@dataclass
class Catalogue:
    tables: Dict[str, TableInfo] = field(default_factory=dict)  # key: schema.table
    values: ValueDict = field(default_factory=ValueDict)

    def has_table(self, name: str) -> bool:
        if "." in name:
            key = name
        else:
            key = f"public.{name}"
        return key in self.tables

    def has_column(self, table: str, column: str) -> bool:
        tkey = table if "." in table else f"public.{table}"
        t = self.tables.get(tkey)
        return bool(t and (column in t.columns))

    def list_tables(self) -> List[str]:
        return list(self.tables.keys())

    def role(self, table: str, column: str) -> Optional[str]:
        tkey = table if "." in table else f"public.{table}"
        t = self.tables.get(tkey)
        if not t:
            return None
        c = t.columns.get(column)
        return c.role if c else None


def _guess_role(name: str, dtype: str) -> str:
    nm = name.lower()
    dt = (dtype or "").lower()
    if nm.endswith("_id") or nm == "id" or "id" == nm:
        return "dimension"
    if any(k in nm for k in ("date", "year", "month", "agency", "department", "source", "status")):
        return "dimension"
    if dt in ("int", "integer", "bigint", "numeric", "float", "double", "real", "int64"):
        return "measure"
    return "dimension"


def _guess_fk(table: str, col: str) -> Optional[str]:
    if not col.lower().endswith("_id"):
        return None
    base = col[:-3]
    # naive pluralization: try exact, then +s
    return f"public.{base}"  # caller may adjust


def load_catalogue(base_dir: Path,
                   dataset_path: Optional[str] = None,
                   value_path: Optional[str] = None) -> Catalogue:
    # Load datasets
    ds_paths: List[Path] = []
    if dataset_path:
        ds_paths.append(Path(dataset_path))
    ds_paths += [base_dir / "dataset_schemas.json", base_dir.parent / "dataset_schemas.json"]
    ds_data = None
    for p in ds_paths:
        try:
            if p.is_file():
                ds_data = json.loads(p.read_text(encoding="utf-8"))
                break
        except Exception:
            continue
    cat = Catalogue()
    if isinstance(ds_data, dict):
        for ds_name, cols in ds_data.items():
            if not isinstance(cols, list):
                continue
            tname = ds_name[:-4] if isinstance(ds_name, str) and ds_name.endswith(".csv") else str(ds_name)
            key = f"public.{tname}"
            tinfo = TableInfo(name=tname)
            for c in cols:
                if not isinstance(c, dict):
                    continue
                cname = str(c.get("column_name", "")).strip()
                ctype = str(c.get("data_type", "")).strip()
                if not cname:
                    continue
                role = _guess_role(cname, ctype)
                tinfo.columns[cname] = ColumnInfo(name=cname, type=ctype, role=role)
            # Heuristic PK
            if "id" in tinfo.columns:
                tinfo.pk.add("id")
            # Heuristic FK
            for cname in list(tinfo.columns.keys()):
                fk = _guess_fk(tname, cname)
                if fk:
                    tinfo.fk[cname] = fk
            cat.tables[key] = tinfo

    # Load values / aliases
    val_paths: List[Path] = []
    if value_path:
        val_paths.append(Path(value_path))
    val_paths += [base_dir / "value_descriptions_min.json", base_dir.parent / "value_descriptions_min.json"]
    val_data = None
    for p in val_paths:
        try:
            if p.is_file():
                val_data = json.loads(p.read_text(encoding="utf-8"))
                break
        except Exception:
            continue
    aliases: Dict[str, Dict[str, str]] = {}
    # Heuristic: seed common department synonyms if present
    dept_aliases = {
        "information technology": "IT",
        "info tech": "IT",
        "i.t.": "IT",
        "it": "IT",
    }
    # Only add if any table likely contains a department-like column
    has_dept = any(any("department" in c.lower() for c in t.columns) for t in cat.tables.values())
    if has_dept:
        aliases["department"] = dept_aliases
    # Future: parse val_data when it contains explicit enums/aliases
    cat.values.aliases = aliases
    return cat

