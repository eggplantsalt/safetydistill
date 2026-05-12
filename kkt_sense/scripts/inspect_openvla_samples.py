"""Inspect OpenVLA-style KKT sample exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List


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


def _iter_records(steps_path: Path, num_samples: int) -> List[Dict[str, Any]]:
    records = []
    with steps_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= num_samples:
                break
    return records


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
    print("num_episodes:", manifest.get("num_episodes"))
    print("num_records:", manifest.get("num_records"))

    samples = []
    for episode in episodes:
        if len(samples) >= args.num_samples:
            break
        steps_path = episode.get("steps_path")
        if steps_path is None:
            continue
        steps_file = (base_dir / steps_path).resolve()
        samples.extend(_iter_records(steps_file, args.num_samples - len(samples)))

    if not samples:
        print("No samples found.")
        return 1

    first = samples[0]
    agentview_path = base_dir / first.get("agentview_image_path", "")
    wrist_path = base_dir / first.get("wrist_image_path", "")

    print("first_sample_instruction:", first.get("instruction"))
    print("agentview_image_path:", agentview_path)
    print("wrist_image_path:", wrist_path)
    print("agentview_exists:", agentview_path.exists())
    print("wrist_exists:", wrist_path.exists())

    state = first.get("state") or []
    action_nominal = first.get("action_nominal") or []
    action_safe = first.get("action_safe") or []
    action_delta = first.get("action_delta") or []

    print("state_length:", len(state))
    print("action_nominal_length:", len(action_nominal))
    print("action_safe_length:", len(action_safe))
    print("action_delta_length:", len(action_delta))

    kkt_present = _record_has_kkt(first)
    print("kkt_fields_present:", kkt_present)

    if args.require_kkt and not kkt_present:
        print("Missing KKT fields in sample.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
