"""Build a manifest.json for KKT label datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np


def iter_jsonl_files(input_path: Path, max_files: Optional[int]) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    files = sorted(input_path.rglob("*.jsonl"))
    if max_files is not None:
        return files[:max_files]
    return files


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _safe_minmax(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    return float(arr.min()), float(arr.max())


def _extract_numbers(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in np.asarray(value).ravel()]
    return []


def summarize_jsonl_file(path: Path, require_kkt_fields: bool, base_dir: Path, absolute_paths: bool) -> Dict[str, Any]:
    num_records = 0
    qp_status_counts: Dict[str, int] = {}
    has_action_nominal = False
    has_action_safe = False
    has_action_delta = False
    has_dual_variables = True
    has_active_set = True
    has_constraint_values = True
    has_constraint_gradients = True
    active_set_cbf_main_true = 0

    dual_values: List[float] = []
    h_values: List[float] = []

    task_suite_name = None
    safety_level = None
    task_index = None
    episode_index = None
    instruction = None

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            num_records += 1

            if num_records == 1:
                task_suite_name = record.get("task_suite_name")
                safety_level = record.get("safety_level")
                task_index = record.get("task_index")
                episode_index = record.get("episode_index")
                instruction = record.get("instruction")

            qp_status = record.get("qp_status")
            qp_status_counts[qp_status] = qp_status_counts.get(qp_status, 0) + 1

            if record.get("action_nominal") is not None:
                has_action_nominal = True
            if record.get("action_safe") is not None:
                has_action_safe = True
            if record.get("action_delta") is not None:
                has_action_delta = True

            dual_variables = record.get("dual_variables")
            if dual_variables is None:
                has_dual_variables = False
            active_set = record.get("active_set")
            if active_set is None:
                has_active_set = False
            constraint_values = record.get("constraint_values")
            if constraint_values is None:
                has_constraint_values = False
            constraint_gradients = record.get("constraint_gradients")
            if constraint_gradients is None:
                has_constraint_gradients = False

            if isinstance(active_set, dict) and active_set.get("cbf_main") is True:
                active_set_cbf_main_true += 1

            if isinstance(dual_variables, dict):
                dual_values.extend(_extract_numbers(dual_variables.get("cbf_main")))

            if isinstance(constraint_values, dict):
                h_values.extend(_extract_numbers(constraint_values.get("h")))

    has_kkt = (
        num_records > 0
        and has_dual_variables
        and has_active_set
        and has_constraint_values
        and has_constraint_gradients
    )

    dual_min, dual_max = _safe_minmax(dual_values)
    h_min, h_max = _safe_minmax(h_values)

    if absolute_paths:
        path_value = str(path.resolve())
    else:
        try:
            path_value = str(path.resolve().relative_to(base_dir.resolve()))
        except ValueError:
            path_value = str(path)

    return {
        "path": path_value,
        "num_records": num_records,
        "task_suite_name": task_suite_name,
        "safety_level": safety_level,
        "task_index": task_index,
        "episode_index": episode_index,
        "instruction": instruction,
        "qp_status_counts": qp_status_counts,
        "has_action_nominal": has_action_nominal,
        "has_action_safe": has_action_safe,
        "has_action_delta": has_action_delta,
        "has_dual_variables": has_dual_variables,
        "has_active_set": has_active_set,
        "has_constraint_values": has_constraint_values,
        "has_constraint_gradients": has_constraint_gradients,
        "has_kkt": has_kkt,
        "active_set_cbf_main_true": active_set_cbf_main_true,
        "dual_cbf_main_min": dual_min,
        "dual_cbf_main_max": dual_max,
        "dual_cbf_main_mean": _safe_mean(dual_values),
        "h_min": h_min,
        "h_max": h_max,
        "h_mean": _safe_mean(h_values),
    }


def merge_file_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_records = 0
    num_files_with_kkt = 0
    num_records_with_kkt = 0
    qp_status_counts: Dict[str, int] = {}

    has_action_nominal = False
    has_action_safe = False
    has_action_delta = False
    has_dual_variables = False
    has_active_set = False
    has_constraint_values = False
    has_constraint_gradients = False

    for summary in summaries:
        total_records += summary["num_records"]
        for status, count in summary["qp_status_counts"].items():
            qp_status_counts[status] = qp_status_counts.get(status, 0) + count

        if summary["has_kkt"]:
            num_files_with_kkt += 1
            num_records_with_kkt += summary["num_records"]

        has_action_nominal = has_action_nominal or summary["has_action_nominal"]
        has_action_safe = has_action_safe or summary["has_action_safe"]
        has_action_delta = has_action_delta or summary["has_action_delta"]
        has_dual_variables = has_dual_variables or summary["has_dual_variables"]
        has_active_set = has_active_set or summary["has_active_set"]
        has_constraint_values = has_constraint_values or summary["has_constraint_values"]
        has_constraint_gradients = has_constraint_gradients or summary["has_constraint_gradients"]

    return {
        "num_records": total_records,
        "num_files_with_kkt": num_files_with_kkt,
        "num_records_with_kkt": num_records_with_kkt,
        "qp_status_counts": qp_status_counts,
        "fields": {
            "has_action_nominal": has_action_nominal,
            "has_action_safe": has_action_safe,
            "has_action_delta": has_action_delta,
            "has_dual_variables": has_dual_variables,
            "has_active_set": has_active_set,
            "has_constraint_values": has_constraint_values,
            "has_constraint_gradients": has_constraint_gradients,
        },
    }


def build_manifest(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    require_kkt_fields: bool,
    absolute_paths: bool,
    max_files: Optional[int],
) -> Tuple[Dict[str, Any], int]:
    files = iter_jsonl_files(input_path, max_files)
    summaries: List[Dict[str, Any]] = []

    for path in files:
        summaries.append(
            summarize_jsonl_file(path, require_kkt_fields, output_path.parent, absolute_paths)
        )

    merged = merge_file_summaries(summaries)

    manifest = {
        "dataset_name": dataset_name,
        "input_path": str(input_path),
        "num_files": len(files),
        "num_records": merged["num_records"],
        "num_files_with_kkt": merged["num_files_with_kkt"],
        "num_records_with_kkt": merged["num_records_with_kkt"],
        "qp_status_counts": merged["qp_status_counts"],
        "fields": merged["fields"],
        "files": summaries,
    }

    missing_kkt_files = [summary for summary in summaries if not summary["has_kkt"]]
    return manifest, len(missing_kkt_files)


def write_manifest(manifest: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build KKT dataset manifest.json")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--require-kkt-fields", action="store_true")
    parser.add_argument("--absolute-paths", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if args.output_path is None:
        if input_path.is_file():
            output_path = input_path.parent / "manifest.json"
        else:
            output_path = input_path / "manifest.json"
    else:
        output_path = Path(args.output_path)

    dataset_name = args.dataset_name
    if dataset_name is None:
        dataset_name = input_path.resolve().name

    manifest, missing_kkt_files = build_manifest(
        input_path=input_path,
        output_path=output_path,
        dataset_name=dataset_name,
        require_kkt_fields=args.require_kkt_fields,
        absolute_paths=args.absolute_paths,
        max_files=args.max_files,
    )

    write_manifest(manifest, output_path)

    if args.require_kkt_fields:
        if manifest["num_files"] == 0:
            print(f"error: no JSONL files found under input path: {input_path}")
            return 1
        if manifest["num_records"] == 0:
            print(f"error: zero JSONL records found under input path: {input_path}")
            return 1
        if missing_kkt_files > 0:
            print(f"missing_kkt_files: {missing_kkt_files}")
            return 1

    print(f"manifest_path: {output_path}")
    print(f"num_files: {manifest['num_files']}")
    print(f"num_records: {manifest['num_records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
