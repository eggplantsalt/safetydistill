"""Batch-generate synthetic KKT labels using the translational AEGIS entrypoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _parse_indices(raw: str) -> List[int]:
    raw = raw.strip()
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _build_command(args: argparse.Namespace, task_index: int, episode_index: int) -> List[str]:
    return [
        args.python_bin,
        "main/main_aegis_translational.py",
        "--task-suite-name",
        args.task_suite_name,
        "--safety-level",
        args.safety_level,
        "--task-index",
        str(task_index),
        "--episode-index",
        str(episode_index),
        "--num-trials-per-task",
        str(args.num_trials_per_task),
        "--num-steps-wait",
        str(args.num_steps_wait),
        "--replan-steps",
        str(args.replan_steps),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--debug-synthetic-safety-obstacle",
        "--enable-kkt-label-export",
        "--kkt-label-output-dir",
        args.output_dir,
        "--video-out-path",
        args.video_out_path,
    ]


def _run_command(command: List[str], dry_run: bool) -> int:
    print("Command:", " ".join(command))
    if dry_run:
        return 0
    result = subprocess.run(command)
    if result.returncode != 0:
        print("Command failed:", " ".join(command))
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic KKT mini-dataset.")
    parser.add_argument("--task-suite-name", default="safelibero_spatial")
    parser.add_argument("--safety-level", default="I")
    parser.add_argument("--task-indices", default="0")
    parser.add_argument("--episode-indices", default="0")
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument("--num-steps-wait", type=int, default=20)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", default="data/kkt_safelibero_synthetic_debug")
    parser.add_argument("--video-out-path", default="results_kkt_synthetic_debug")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    task_indices = _parse_indices(args.task_indices)
    episode_indices = _parse_indices(args.episode_indices)
    if not task_indices or not episode_indices:
        print("No task or episode indices provided.")
        return 1

    for task_index in task_indices:
        for episode_index in episode_indices:
            command = _build_command(args, task_index, episode_index)
            returncode = _run_command(command, args.dry_run)
            if returncode != 0:
                return returncode

    if args.validate and not args.dry_run:
        validate_cmd = [
            args.python_bin,
            "-m",
            "kkt_sense.scripts.validate_kkt_labels",
            "--path",
            args.output_dir,
            "--require-kkt-fields",
            "--max-print-records",
            "0",
        ]
        returncode = _run_command(validate_cmd, args.dry_run)
        if returncode != 0:
            return returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
