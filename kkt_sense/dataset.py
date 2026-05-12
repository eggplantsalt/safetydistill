"""Lightweight dataset loader for KKT JSONL datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np


class KKTJsonlDataset:
    """Dataset loader for KKT JSONL datasets backed by a manifest.json file."""

    def __init__(
        self,
        manifest_path: str,
        require_kkt: bool = True,
        load_into_memory: bool = False,
        include_no_safety_control: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.require_kkt = require_kkt
        self.load_into_memory = load_into_memory
        self.include_no_safety_control = include_no_safety_control
        self.skipped_files: List[str] = []

        self._records: List[Dict[str, Any]] = []
        self._index: List[Tuple[Path, int]] = []

        manifest = self._load_manifest()
        files = manifest.get("files", [])

        for file_entry in files:
            file_has_kkt = bool(file_entry.get("has_kkt"))
            if self.require_kkt and not file_has_kkt:
                self.skipped_files.append(str(file_entry.get("path")))
                continue

            file_path = self._resolve_path(file_entry.get("path"))
            if self.load_into_memory:
                self._records.extend(self._load_records(file_path))
            else:
                self._index.extend(self._index_records(file_path))

    def __len__(self) -> int:
        if self.load_into_memory:
            return len(self._records)
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.load_into_memory:
            record = self._records[idx]
            return self._normalize_record(record, None, None)

        file_path, line_number = self._index[idx]
        record = self._read_record(file_path, line_number)
        return self._normalize_record(record, file_path, line_number)

    def _load_manifest(self) -> Dict[str, Any]:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _resolve_path(self, path_value: Optional[str]) -> Path:
        if path_value is None:
            raise ValueError("Manifest entry missing path")
        path = Path(path_value)
        if path.is_absolute():
            return path
        return (self.manifest_path.parent / path).resolve()

    def _load_records(self, file_path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not self.include_no_safety_control and record.get("qp_status") == "no_safety_control":
                    continue
                records.append(record)
        return records

    def _index_records(self, file_path: Path) -> List[Tuple[Path, int]]:
        index: List[Tuple[Path, int]] = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not self.include_no_safety_control and record.get("qp_status") == "no_safety_control":
                    continue
                index.append((file_path, line_number))
        return index

    def _read_record(self, file_path: Path, line_number: int) -> Dict[str, Any]:
        with file_path.open("r", encoding="utf-8") as handle:
            for current_line, line in enumerate(handle):
                if current_line != line_number:
                    continue
                return json.loads(line)
        raise IndexError(f"Line {line_number} not found in {file_path}")

    def _normalize_record(
        self, record: Dict[str, Any], file_path: Optional[Path], line_number: Optional[int]
    ) -> Dict[str, Any]:
        if self.require_kkt:
            missing = []
            for field in ["dual_variables", "active_set", "constraint_values", "constraint_gradients"]:
                if record.get(field) is None:
                    missing.append(field)
            if missing:
                location = "unknown"
                if file_path is not None and line_number is not None:
                    location = f"{file_path}:{line_number}"
                raise ValueError(f"Missing KKT fields {missing} in {location}")

        action_nominal = self._to_numpy(record.get("action_nominal"))
        action_safe = self._to_numpy(record.get("action_safe"))
        action_delta = self._to_numpy(record.get("action_delta"))

        dual_cbf_main = self._extract_scalar(record.get("dual_variables"), "cbf_main")
        active_cbf_main = self._extract_bool(record.get("active_set"), "cbf_main")
        h_value = self._extract_scalar(record.get("constraint_values"), "h")
        linear_cbf_lhs = self._extract_scalar(record.get("constraint_values"), "linear_cbf_lhs")
        a_u_v = self._to_numpy(self._extract_value(record.get("constraint_gradients"), "a_u_v"))
        a_uz = self._to_numpy(self._extract_value(record.get("constraint_gradients"), "a_uz"))

        return {
            "instruction": record.get("instruction"),
            "task_suite_name": record.get("task_suite_name"),
            "safety_level": record.get("safety_level"),
            "task_index": record.get("task_index"),
            "episode_index": record.get("episode_index"),
            "step_index": record.get("step_index"),
            "action_nominal": action_nominal,
            "action_safe": action_safe,
            "action_delta": action_delta,
            "dual_cbf_main": dual_cbf_main,
            "active_cbf_main": active_cbf_main,
            "h": h_value,
            "linear_cbf_lhs": linear_cbf_lhs,
            "a_u_v": a_u_v,
            "a_uz": a_uz,
            "qp_status": record.get("qp_status"),
            "raw": record,
        }

    def _to_numpy(self, value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        return arr

    def _extract_value(self, container: Any, key: str) -> Any:
        if isinstance(container, dict):
            return container.get(key)
        return None

    def _extract_scalar(self, container: Any, key: str) -> Optional[float]:
        value = self._extract_value(container, key)
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, (list, tuple, np.ndarray)):
            flat = np.asarray(value).ravel()
            if flat.size == 0:
                return None
            return float(flat[0])
        return None

    def _extract_bool(self, container: Any, key: str) -> Optional[bool]:
        value = self._extract_value(container, key)
        if isinstance(value, bool):
            return value
        return None
