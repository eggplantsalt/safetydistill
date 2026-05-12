"""Generate dummy KKT label files for Phase 1 validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from kkt_sense.label_exporter import export_episode
from kkt_sense.label_exporter import summarize_episode
from kkt_sense.qp_interface import extract_qp_certificate
from kkt_sense.rollout_capture import build_step_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dummy KKT-SafeLIBERO labels.")
    parser.add_argument("--task-suite-name", default="safelibero_spatial")
    parser.add_argument("--safety-level", default="I")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output-dir", default="data/kkt_safelibero_debug")
    parser.add_argument("--num-dummy-steps", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_dir) / (
        f"{args.task_suite_name}_level_{args.safety_level}_task_{args.task_index}_episode_{args.episode_index}.jsonl"
    )

    records = []
    qp_certificate = extract_qp_certificate()
    for step_index in range(args.num_dummy_steps):
        record = build_step_record(
            task_suite_name=args.task_suite_name,
            safety_level=args.safety_level,
            task_index=args.task_index,
            episode_index=args.episode_index,
            step_index=step_index,
            instruction=None,
            observation_metadata={},
            action_nominal=[0.1, 0.0, 0.0],
            action_safe=[0.08, 0.02, 0.0],
            constraint_values=None,
            constraint_gradients=None,
            dual_variables=qp_certificate["dual_variables"],
            active_set=qp_certificate["active_set"],
            qp_status=qp_certificate["qp_status"],
            collision_info=None,
            extra_debug={"dummy": True},
        )
        records.append(record)

    export_episode(records, str(output_path))
    summary = summarize_episode(records)

    print(f"output_path: {output_path}")
    print(f"episode_summary: {summary}")
    print(f"records_written: {len(records)}")


if __name__ == "__main__":
    main()
