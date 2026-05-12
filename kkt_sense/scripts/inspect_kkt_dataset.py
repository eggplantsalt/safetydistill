"""Inspect KKT JSONL dataset loader outputs."""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np

from kkt_sense.dataset import KKTJsonlDataset


def _shape(value: Optional[np.ndarray]) -> str:
    if value is None:
        return "None"
    return str(tuple(value.shape))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect KKT dataset loader")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--require-kkt", action="store_true")
    parser.add_argument("--load-into-memory", action="store_true")
    parser.add_argument("--include-no-safety-control", action="store_true")
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()

    dataset = KKTJsonlDataset(
        manifest_path=args.manifest_path,
        require_kkt=args.require_kkt,
        load_into_memory=args.load_into_memory,
        include_no_safety_control=args.include_no_safety_control,
    )

    print("dataset_length:", len(dataset))
    print("skipped_files:", dataset.skipped_files)

    for idx in range(min(args.num_samples, len(dataset))):
        sample = dataset[idx]
        action_delta = sample.get("action_delta")
        delta_norm = float(np.linalg.norm(action_delta)) if action_delta is not None else None

        print("sample_index:", idx)
        print("task_index:", sample.get("task_index"))
        print("episode_index:", sample.get("episode_index"))
        print("step_index:", sample.get("step_index"))
        print("qp_status:", sample.get("qp_status"))
        print("action_nominal shape:", _shape(sample.get("action_nominal")))
        print("action_safe shape:", _shape(sample.get("action_safe")))
        print("action_delta norm:", delta_norm)
        print("dual_cbf_main:", sample.get("dual_cbf_main"))
        print("active_cbf_main:", sample.get("active_cbf_main"))
        print("h:", sample.get("h"))
        print("linear_cbf_lhs:", sample.get("linear_cbf_lhs"))
        print("a_u_v shape:", _shape(sample.get("a_u_v")))
        print("a_uz shape:", _shape(sample.get("a_uz")))


if __name__ == "__main__":
    main()
