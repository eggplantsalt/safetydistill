"""OpenVLA-style sample exporter for KKT teacher rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import imageio


def build_state_fields(gripper_qpos: Any) -> List[str]:
    base_fields = [
        "eef_pos_x",
        "eef_pos_y",
        "eef_pos_z",
        "eef_axis_angle_x",
        "eef_axis_angle_y",
        "eef_axis_angle_z",
    ]
    try:
        length = len(gripper_qpos)
    except TypeError:
        length = 1

    if length <= 1:
        return base_fields + ["gripper_qpos"]

    return base_fields + ["gripper_qpos_{}".format(i) for i in range(length)]


def make_openvla_episode_dir(
    output_dir: str,
    task_suite_name: str,
    safety_level: str,
    task_index: int,
    episode_index: int,
) -> Path:
    base_dir = Path(output_dir)
    episode_dir = base_dir / "{}_level_{}_task_{}_episode_{}".format(
        task_suite_name,
        safety_level,
        task_index,
        episode_index,
    )
    (episode_dir / "images").mkdir(parents=True, exist_ok=True)
    return episode_dir


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True))
        handle.write("\n")


def save_openvla_step_sample(
    *,
    episode_dir: Path,
    step_index: int,
    agentview_img: Any,
    wrist_img: Any,
    task_suite_name: str,
    safety_level: str,
    task_index: int,
    episode_index: int,
    instruction: Optional[str],
    state: List[float],
    state_fields: List[str],
    action_nominal: Optional[List[float]],
    action_safe: Optional[List[float]],
    action_delta: Optional[List[float]],
    dual_variables: Optional[Dict[str, Any]],
    active_set: Optional[Dict[str, Any]],
    constraint_values: Optional[Dict[str, Any]],
    constraint_gradients: Optional[Dict[str, Any]],
    qp_status: Optional[str],
    extra_debug: Optional[Dict[str, Any]],
) -> None:
    images_dir = episode_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    agentview_name = "step_{:06d}_agentview.png".format(step_index)
    wrist_name = "step_{:06d}_wrist.png".format(step_index)
    agentview_path = images_dir / agentview_name
    wrist_path = images_dir / wrist_name

    imageio.imwrite(agentview_path, agentview_img)
    imageio.imwrite(wrist_path, wrist_img)

    record = {
        "task_suite_name": task_suite_name,
        "safety_level": safety_level,
        "task_index": task_index,
        "episode_index": episode_index,
        "step_index": step_index,
        "instruction": instruction,
        "agentview_image_path": "images/{}".format(agentview_name),
        "wrist_image_path": "images/{}".format(wrist_name),
        "state": state,
        "state_fields": state_fields,
        "action_nominal": action_nominal,
        "action_safe": action_safe,
        "action_delta": action_delta,
        "dual_variables": dual_variables,
        "active_set": active_set,
        "constraint_values": constraint_values,
        "constraint_gradients": constraint_gradients,
        "qp_status": qp_status,
        "extra_debug": extra_debug or {},
    }

    steps_path = episode_dir / "steps.jsonl"
    _append_jsonl(steps_path, record)


def _record_has_kkt(record: Dict[str, Any]) -> bool:
    return (
        record.get("dual_variables") is not None
        and record.get("active_set") is not None
        and record.get("constraint_values") is not None
        and record.get("constraint_gradients") is not None
    )


def build_openvla_sample_manifest(output_dir: str) -> Dict[str, Any]:
    base_dir = Path(output_dir)
    steps_files = sorted(base_dir.rglob("steps.jsonl"))

    episodes = []
    total_records = 0

    for steps_path in steps_files:
        num_records = 0
        first_record = None
        has_kkt = True
        with steps_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                num_records += 1
                if first_record is None:
                    first_record = record
                if not _record_has_kkt(record):
                    has_kkt = False

        total_records += num_records
        episode_dir = steps_path.parent
        try:
            episode_rel = str(episode_dir.relative_to(base_dir))
            steps_rel = str(steps_path.relative_to(base_dir))
        except ValueError:
            episode_rel = str(episode_dir)
            steps_rel = str(steps_path)

        episodes.append(
            {
                "episode_dir": episode_rel,
                "steps_path": steps_rel,
                "task_suite_name": None if first_record is None else first_record.get("task_suite_name"),
                "safety_level": None if first_record is None else first_record.get("safety_level"),
                "task_index": None if first_record is None else first_record.get("task_index"),
                "episode_index": None if first_record is None else first_record.get("episode_index"),
                "num_records": num_records,
                "has_kkt": bool(has_kkt and num_records > 0),
            }
        )

    manifest = {
        "dataset_name": base_dir.name,
        "format": "kkt_openvla_samples_v1",
        "num_episodes": len(episodes),
        "num_records": total_records,
        "episodes": episodes,
    }

    manifest_path = base_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)

    return manifest
