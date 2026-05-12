"""Inspect OpenVLA-style KKT sample exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple


def _load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _record_has_kkt(record: Dict[str, Any]) -> bool:
    return (
        record.get("dual_variables") is not None
        and record.get("active_set") is not None
        and record.get("constraint_values") is not None
        and record.get("constraint_gradients") is not None
    )


def _iter_records(episode_dir: Path, steps_path: Path, num_samples: int) -> List[Tuple[Path, Dict[str, Any]]]:
    records = []
    with steps_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append((episode_dir, json.loads(line)))
            if len(records) >= num_samples:
                break
    return records


def _resolve_record_path(episode_dir: Path, record: Dict[str, Any], key: str) -> Path:
    rel = record.get(key, "")
    return (episode_dir / rel).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect OpenVLA-style samples")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--require-kkt", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    manifest = _load_manifest(manifest_path)
    base_dir = manifest_path.parent

    episodes = manifest.get("episodes", [])
    print("dataset_name:", manifest.get("dataset_name"))
    print("format:", manifest.get("format"))
    print("num_episodes:", manifest.get("num_episodes"))
    print("num_records:", manifest.get("num_records"))

    samples = []
    for episode in episodes:
        if len(samples) >= args.num_samples:
            break

        episode_dir_rel = episode.get("episode_dir")
        steps_path_rel = episode.get("steps_path")
        if episode_dir_rel is None or steps_path_rel is None:
            continue

        episode_dir = (base_dir / episode_dir_rel).resolve()
        steps_file = (base_dir / steps_path_rel).resolve()
        if not steps_file.exists():
            print("missing_steps_file:", steps_file)
            continue

        samples.extend(_iter_records(episode_dir, steps_file, args.num_samples - len(samples)))

    if not samples:
        print("error: no samples found.")
        return 1

    for idx, (episode_dir, record) in enumerate(samples):
        agentview_path = _resolve_record_path(episode_dir, record, "agentview_image_path")
        wrist_path = _resolve_record_path(episode_dir, record, "wrist_image_path")

        state = record.get("state") or []
        state_fields = record.get("state_fields") or []
        action_nominal = record.get("action_nominal") or []
        action_safe = record.get("action_safe") or []
        action_delta = record.get("action_delta") or []
        kkt_present = _record_has_kkt(record)

        print("sample_index:", idx)
        print("  task_index:", record.get("task_index"))
        print("  episode_index:", record.get("episode_index"))
        print("  step_index:", record.get("step_index"))
        print("  instruction:", record.get("instruction"))
        print("  agentview_image_path:", agentview_path)
        print("  wrist_image_path:", wrist_path)
        print("  agentview_exists:", agentview_path.exists())
        print("  wrist_exists:", wrist_path.exists())
        print("  state_length:", len(state))
        print("  state_fields_length:", len(state_fields))
        print("  action_nominal_length:", len(action_nominal))
        print("  action_safe_length:", len(action_safe))
        print("  action_delta_length:", len(action_delta))
        print("  qp_status:", record.get("qp_status"))
        print("  kkt_fields_present:", kkt_present)

        if not agentview_path.exists() or not wrist_path.exists():
            print("error: missing image file for sample", idx)
            return 1

        if len(state) != len(state_fields):
            print("error: state/state_fields length mismatch for sample", idx)
            return 1

        if args.require_kkt and not kkt_present:
            print("error: missing KKT fields in sample", idx)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
