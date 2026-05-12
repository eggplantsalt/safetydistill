"""Inspect KKT training target batches."""

from __future__ import annotations

import argparse
from typing import Any
from typing import Dict

import numpy as np

from kkt_sense.dataset import KKTJsonlDataset
from kkt_sense.training_targets import build_training_target
from kkt_sense.training_targets import collate_training_targets


def _stats(array: np.ndarray) -> Dict[str, Any]:
    return {
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _mask_stats(array: np.ndarray) -> Dict[str, Any]:
    return {
        "shape": tuple(array.shape),
        "sum": float(array.sum()),
        "mean": float(array.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect KKT training batch")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--require-kkt", action="store_true")
    parser.add_argument("--include-no-safety-control", action="store_true")
    parser.add_argument("--action-target", default="safe", choices=["safe", "delta", "nominal"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    dataset = KKTJsonlDataset(
        manifest_path=args.manifest_path,
        require_kkt=args.require_kkt,
        load_into_memory=False,
        include_no_safety_control=args.include_no_safety_control,
    )

    end_index = min(args.start_index + args.batch_size, len(dataset))
    targets = []
    for idx in range(args.start_index, end_index):
        sample = dataset[idx]
        targets.append(build_training_target(sample, action_target=args.action_target))

    batch = collate_training_targets(targets)

    print("dataset_length:", len(dataset))
    print("batch_size:", len(targets))
    print("action_target:", args.action_target)
    print("inputs.instruction[0]:", batch["inputs"]["instruction"][0] if targets else None)
    print("metadata[0]:", batch["metadata"][0] if targets else None)

    for name, array in batch["targets"].items():
        print("targets.", name, _stats(array))
    for name, array in batch["masks"].items():
        print("masks.", name, _mask_stats(array))


if __name__ == "__main__":
    main()
