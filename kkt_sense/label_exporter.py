"""Episode-level export helpers."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Iterable
from typing import List

from .io_utils import save_jsonl


def _get_field(record: Any, name: str) -> Any:
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def export_episode(records: Iterable[Any], output_path: str) -> None:
    """Export an episode of records to JSONL."""
    save_jsonl(records, output_path)


def summarize_episode(records: Iterable[Any]) -> Dict[str, Any]:
    """Summarize a list of step records."""
    record_list: List[Any] = list(records)
    has_nominal_actions = any(_get_field(r, "action_nominal") is not None for r in record_list)
    has_safe_actions = any(_get_field(r, "action_safe") is not None for r in record_list)
    has_dual_variables = any(_get_field(r, "dual_variables") is not None for r in record_list)
    return {
        "num_steps": len(record_list),
        "has_nominal_actions": has_nominal_actions,
        "has_safe_actions": has_safe_actions,
        "has_dual_variables": has_dual_variables,
    }
