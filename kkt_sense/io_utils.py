"""I/O helpers for JSONL label files."""

from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import List
from typing import Union


def ensure_parent_dir(path: Union[str, Path]) -> Path:
    """Ensure parent directory exists and return a Path instance."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


def _normalize_record(record: Any) -> dict:
    if hasattr(record, "to_jsonable"):
        return record.to_jsonable()
    if is_dataclass(record):
        return asdict(record)
    if isinstance(record, dict):
        return record
    return {"value": record}


def save_jsonl(records: Iterable[Any], output_path: Union[str, Path]) -> Path:
    """Save records to JSONL, one JSON object per line."""
    path_obj = ensure_parent_dir(output_path)
    with path_obj.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = _normalize_record(record)
            handle.write(json.dumps(payload, ensure_ascii=True))
            handle.write("\n")
    return path_obj


def load_jsonl(path: Union[str, Path]) -> List[dict]:
    """Load a JSONL file into a list of dicts."""
    path_obj = Path(path)
    records: List[dict] = []
    with path_obj.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
