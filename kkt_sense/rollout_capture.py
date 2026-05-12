"""Helpers to build per-step records from rollouts."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .schema import StepRecord


def compute_action_delta(
    action_safe: Optional[List[float]],
    action_nominal: Optional[List[float]],
) -> Optional[List[float]]:
    """Compute action delta (safe - nominal) with validation."""
    if action_safe is None or action_nominal is None:
        return None
    if len(action_safe) != len(action_nominal):
        raise ValueError("action_safe and action_nominal must have the same length")
    return [float(safe) - float(nominal) for safe, nominal in zip(action_safe, action_nominal)]


def build_step_record(
    *,
    task_suite_name: str,
    safety_level: str,
    task_index: int,
    episode_index: int,
    step_index: int,
    instruction: Optional[str] = None,
    observation_metadata: Optional[Dict[str, Any]] = None,
    action_nominal: Optional[List[float]] = None,
    action_safe: Optional[List[float]] = None,
    constraint_values: Optional[Dict[str, Any]] = None,
    constraint_gradients: Optional[Dict[str, Any]] = None,
    dual_variables: Optional[Dict[str, Any]] = None,
    active_set: Optional[Dict[str, Any]] = None,
    qp_status: Optional[str] = None,
    collision_info: Optional[Dict[str, Any]] = None,
    extra_debug: Optional[Dict[str, Any]] = None,
) -> StepRecord:
    """Build a StepRecord from rollout inputs."""
    action_delta = compute_action_delta(action_safe, action_nominal)
    return StepRecord(
        task_suite_name=task_suite_name,
        safety_level=safety_level,
        task_index=task_index,
        episode_index=episode_index,
        step_index=step_index,
        instruction=instruction,
        observation_metadata=observation_metadata or {},
        action_nominal=action_nominal,
        action_safe=action_safe,
        action_delta=action_delta,
        constraint_values=constraint_values,
        constraint_gradients=constraint_gradients,
        dual_variables=dual_variables,
        active_set=active_set,
        qp_status=qp_status,
        collision_info=collision_info,
        extra_debug=extra_debug or {},
    )
