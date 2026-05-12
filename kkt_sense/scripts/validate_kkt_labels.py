"""Validate and summarize KKT label JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List

import numpy as np


def _iter_jsonl_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.jsonl"))


def _get_dict(record: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = record.get(key)
    if isinstance(value, dict):
        return value
    return {}


def _extract_numbers(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in np.asarray(value).ravel()]
    return []


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _print_records(records: Iterable[Dict[str, Any]], max_print: int) -> None:
    for idx, record in enumerate(records):
        if idx >= max_print:
            break
        print(json.dumps(record, ensure_ascii=True, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate KKT label JSONL files")
    parser.add_argument("--path", required=True)
    parser.add_argument("--require-kkt-fields", action="store_true")
    parser.add_argument("--max-print-records", type=int, default=1)
    args = parser.parse_args()

    input_path = Path(args.path)
    files = _iter_jsonl_files(input_path)

    stats = {
        "num_files": len(files),
        "num_records": 0,
        "qp_status_counts": {},
        "action_nominal_non_null": 0,
        "action_safe_non_null": 0,
        "action_delta_non_null": 0,
        "dual_variables_non_null": 0,
        "active_set_non_null": 0,
        "constraint_values_non_null": 0,
        "constraint_gradients_non_null": 0,
        "active_set_cbf_main_true": 0,
    }

    action_delta_norms = []
    dual_values = []
    h_values = []
    linear_lhs_values = []

    missing_kkt_fields = 0
    sample_records = []

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                stats["num_records"] += 1

                if len(sample_records) < args.max_print_records:
                    sample_records.append(record)

                qp_status = record.get("qp_status")
                stats["qp_status_counts"][qp_status] = stats["qp_status_counts"].get(qp_status, 0) + 1

                if record.get("action_nominal") is not None:
                    stats["action_nominal_non_null"] += 1
                if record.get("action_safe") is not None:
                    stats["action_safe_non_null"] += 1
                if record.get("action_delta") is not None:
                    stats["action_delta_non_null"] += 1

                dual = record.get("dual_variables")
                if dual is not None:
                    stats["dual_variables_non_null"] += 1
                active = record.get("active_set")
                if active is not None:
                    stats["active_set_non_null"] += 1
                cvals = record.get("constraint_values")
                if cvals is not None:
                    stats["constraint_values_non_null"] += 1
                cgrads = record.get("constraint_gradients")
                if cgrads is not None:
                    stats["constraint_gradients_non_null"] += 1

                if args.require_kkt_fields:
                    if dual is None or active is None or cvals is None or cgrads is None:
                        missing_kkt_fields += 1

                active_dict = _get_dict(record, "active_set")
                if active_dict.get("cbf_main") is True:
                    stats["active_set_cbf_main_true"] += 1

                action_delta = record.get("action_delta")
                if action_delta is not None:
                    delta_arr = np.asarray(action_delta, dtype=float)
                    action_delta_norms.append(float(np.linalg.norm(delta_arr)))

                dual_dict = _get_dict(record, "dual_variables")
                dual_values.extend(_extract_numbers(dual_dict.get("cbf_main")))

                cvals_dict = _get_dict(record, "constraint_values")
                h_values.extend(_extract_numbers(cvals_dict.get("h")))
                linear_lhs_values.extend(_extract_numbers(cvals_dict.get("linear_cbf_lhs")))

    print("num_files:", stats["num_files"])
    print("num_records:", stats["num_records"])
    print("qp_status_counts:", stats["qp_status_counts"])
    print("action_nominal_non_null:", stats["action_nominal_non_null"])
    print("action_safe_non_null:", stats["action_safe_non_null"])
    print("action_delta_non_null:", stats["action_delta_non_null"])
    print("dual_variables_non_null:", stats["dual_variables_non_null"])
    print("active_set_non_null:", stats["active_set_non_null"])
    print("constraint_values_non_null:", stats["constraint_values_non_null"])
    print("constraint_gradients_non_null:", stats["constraint_gradients_non_null"])
    print("active_set.cbf_main true count:", stats["active_set_cbf_main_true"])
    print("action_delta norm stats:", _summary_stats(action_delta_norms))
    print("dual_variables.cbf_main stats:", _summary_stats(dual_values))
    print("constraint_values.h stats:", _summary_stats(h_values))
    print("constraint_values.linear_cbf_lhs stats:", _summary_stats(linear_lhs_values))

    _print_records(sample_records, args.max_print_records)

    if args.require_kkt_fields and missing_kkt_fields:
        print("missing_kkt_fields:", missing_kkt_fields)
        sys.exit(1)


if __name__ == "__main__":
    main()
