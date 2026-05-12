"""Adapter helpers for integrating AEGIS rollouts with KKT label export."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from .rollout_capture import build_step_record
from .schema import StepRecord


def _to_jsonable(value: Any) -> Any:
    """Convert values to JSON-friendly types for label export."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    return str(value)


def build_aegis_step_record(
    *,
    task_suite_name: str,
    safety_level: str,
    task_index: int,
    episode_index: int,
    step_index: int,
    instruction: Optional[str],
    observation_metadata: Optional[Dict[str, Any]],
    action_nominal: Optional[List[float]],
    action_safe: Optional[List[float]],
    qp_status: Optional[str],
    constraint_values: Optional[Dict[str, Any]] = None,
    constraint_gradients: Optional[Dict[str, Any]] = None,
    dual_variables: Optional[Dict[str, Any]] = None,
    active_set: Optional[Dict[str, Any]] = None,
    collision_info: Optional[Dict[str, Any]] = None,
    extra_debug: Optional[Dict[str, Any]] = None,
) -> StepRecord:
    """Build a StepRecord for an AEGIS rollout step.

    Constraint and dual fields are left empty for Phase 2 scaffolding.
    """
    return build_step_record(
        task_suite_name=task_suite_name,
        safety_level=safety_level,
        task_index=task_index,
        episode_index=episode_index,
        step_index=step_index,
        instruction=instruction,
        observation_metadata=observation_metadata or {},
        action_nominal=action_nominal,
        action_safe=action_safe,
        constraint_values=constraint_values,
        constraint_gradients=constraint_gradients,
        dual_variables=dual_variables,
        active_set=active_set,
        qp_status=qp_status,
        collision_info=collision_info,
        extra_debug=extra_debug or {},
    )


def extract_cvxpy_qp_certificate(
    *,
    prob: Any,
    constraints: Optional[List[Any]],
    h: Any = None,
    a_u_v: Any = None,
    a_uz: Any = None,
    u_ref_vec: Any = None,
    u_value: Any = None,
) -> Dict[str, Any]:
    """Extract a minimal QP certificate from CVXPY objects.

    This is a Phase 3A placeholder and does not represent a full KKT certificate.
    """
    try:
        qp_status = prob.status
    except Exception:
        qp_status = "unknown"

    dual_value = None
    if constraints:
        try:
            dual_value = constraints[0].dual_value
        except Exception:
            dual_value = None

    dual_json = _to_jsonable(dual_value)
    if dual_value is not None:
        dual_variables = {"cbf_main": dual_json}
        try:
            dual_abs = abs(float(dual_json))
            active = bool(dual_abs > 1e-6)
        except Exception:
            active = None
        active_set = {"cbf_main": active}
    else:
        dual_variables = None
        active_set = None

    constraint_values = {"h": _to_jsonable(h)}
    if u_value is not None and a_u_v is not None and a_uz is not None and h is not None:
        try:
            u_value_np = u_value
            linear_lhs = a_u_v @ u_value_np[:3] + a_uz @ u_value_np[3:6] + 10 * h
            constraint_values["linear_cbf_lhs"] = _to_jsonable(linear_lhs)
        except Exception:
            pass

    constraint_gradients = {
        "a_u_v": _to_jsonable(a_u_v),
        "a_uz": _to_jsonable(a_uz),
    }

    extra_debug = {
        "u_ref_vec": _to_jsonable(u_ref_vec),
        "u_value": _to_jsonable(u_value),
        "prob_value": _to_jsonable(getattr(prob, "value", None)),
    }

    return {
        "dual_variables": dual_variables,
        "active_set": active_set,
        "constraint_values": constraint_values,
        "constraint_gradients": constraint_gradients,
        "qp_status": qp_status,
        "extra_debug": extra_debug,
    }


def make_episode_output_path(
    output_dir: Union[str, Path],
    task_suite_name: str,
    safety_level: str,
    task_index: int,
    episode_index: int,
) -> Path:
    """Create an output path for an episode JSONL file."""
    base_dir = Path(output_dir)
    filename = (
        f"{task_suite_name}_level_{safety_level}_task_{task_index}_episode_{episode_index}.jsonl"
    )
    return base_dir / filename
