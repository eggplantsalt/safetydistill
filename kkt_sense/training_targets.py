"""Training target builders for KKT datasets."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List

import numpy as np


def _as_float32_array(value: Any, shape: List[int], fill: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(shape, fill, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    try:
        arr = arr.reshape(shape)
    except ValueError as exc:
        raise ValueError(f"Expected shape {shape}, got {arr.shape}") from exc
    return arr


def build_training_target(sample: Dict[str, Any], *, action_target: str = "safe") -> Dict[str, Any]:
    """Build a standardized training target from a dataset sample."""
    if action_target not in {"safe", "delta", "nominal"}:
        raise ValueError(f"Unsupported action_target: {action_target}")

    action_nominal = sample.get("action_nominal")
    action_safe = sample.get("action_safe")
    action_delta = sample.get("action_delta")

    has_action = action_nominal is not None and action_safe is not None and action_delta is not None

    dual_cbf_main = sample.get("dual_cbf_main")
    active_cbf_main = sample.get("active_cbf_main")
    h_value = sample.get("h")
    linear_cbf_lhs = sample.get("linear_cbf_lhs")
    a_u_v = sample.get("a_u_v")
    a_uz = sample.get("a_uz")

    has_kkt = (
        dual_cbf_main is not None
        and active_cbf_main is not None
        and h_value is not None
        and a_u_v is not None
        and a_uz is not None
    )

    qp_status = sample.get("qp_status")
    qp_valid = qp_status is not None and qp_status != "no_safety_control"

    if action_target == "safe":
        action_value = action_safe
    elif action_target == "delta":
        action_value = action_delta
    else:
        action_value = action_nominal

    constraint_direction = None
    if a_u_v is not None and a_uz is not None:
        constraint_direction = np.concatenate(
            [np.asarray(a_u_v, dtype=np.float32), np.asarray(a_uz, dtype=np.float32)], axis=0
        )

    targets = {
        "action": _as_float32_array(action_value, [7]),
        "action_nominal": _as_float32_array(action_nominal, [7]),
        "action_safe": _as_float32_array(action_safe, [7]),
        "action_delta": _as_float32_array(action_delta, [7]),
        "dual_cbf_main": _as_float32_array(dual_cbf_main, [1]),
        "active_cbf_main": _as_float32_array(1.0 if active_cbf_main else 0.0, [1]),
        "h": _as_float32_array(h_value, [1]),
        "linear_cbf_lhs": _as_float32_array(linear_cbf_lhs, [1]),
        "constraint_direction": _as_float32_array(constraint_direction, [6]),
    }

    masks = {
        "has_action": _as_float32_array(1.0 if has_action else 0.0, [1]),
        "has_kkt": _as_float32_array(1.0 if has_kkt else 0.0, [1]),
        "active_cbf_main": _as_float32_array(1.0 if active_cbf_main else 0.0, [1]),
        "qp_valid": _as_float32_array(1.0 if qp_valid else 0.0, [1]),
    }

    return {
        "metadata": {
            "task_suite_name": sample.get("task_suite_name"),
            "safety_level": sample.get("safety_level"),
            "task_index": sample.get("task_index"),
            "episode_index": sample.get("episode_index"),
            "step_index": sample.get("step_index"),
            "qp_status": qp_status,
            "instruction": sample.get("instruction"),
        },
        "inputs": {"instruction": sample.get("instruction")},
        "targets": targets,
        "masks": masks,
        "raw": sample.get("raw"),
    }


def collate_training_targets(targets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of training targets into batch tensors."""
    if not targets:
        raise ValueError("No targets provided for collation")

    def stack(name: str, group: str, shape: List[int]) -> np.ndarray:
        arrays = []
        for item in targets:
            value = item[group][name]
            arr = np.asarray(value, dtype=np.float32)
            if list(arr.shape) != shape:
                raise ValueError(f"Expected {name} shape {shape}, got {arr.shape}")
            arrays.append(arr)
        return np.stack(arrays, axis=0)

    batch = {
        "metadata": [item["metadata"] for item in targets],
        "inputs": {"instruction": [item["inputs"]["instruction"] for item in targets]},
        "targets": {
            "action": stack("action", "targets", [7]),
            "action_nominal": stack("action_nominal", "targets", [7]),
            "action_safe": stack("action_safe", "targets", [7]),
            "action_delta": stack("action_delta", "targets", [7]),
            "dual_cbf_main": stack("dual_cbf_main", "targets", [1]),
            "active_cbf_main": stack("active_cbf_main", "targets", [1]),
            "h": stack("h", "targets", [1]),
            "linear_cbf_lhs": stack("linear_cbf_lhs", "targets", [1]),
            "constraint_direction": stack("constraint_direction", "targets", [6]),
        },
        "masks": {
            "has_action": stack("has_action", "masks", [1]),
            "has_kkt": stack("has_kkt", "masks", [1]),
            "active_cbf_main": stack("active_cbf_main", "masks", [1]),
            "qp_valid": stack("qp_valid", "masks", [1]),
        },
        "raw": [item.get("raw") for item in targets],
    }

    return batch
