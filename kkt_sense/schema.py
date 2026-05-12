"""Schema definitions for KKT-SafeLIBERO labels."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import is_dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


try:
    import numpy as np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_NUMPY = False


def _to_jsonable(value: Any) -> Any:
    """Convert a value to a JSON-serializable form."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if _HAS_NUMPY:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    return str(value)


@dataclass
class StepRecord:
    """Single-step record for KKT-SafeLIBERO label export."""

    task_suite_name: str
    safety_level: str
    task_index: int
    episode_index: int
    step_index: int
    instruction: Optional[str] = None
    observation_metadata: Dict[str, Any] = field(default_factory=dict)

    action_nominal: Optional[List[float]] = None
    action_safe: Optional[List[float]] = None
    action_delta: Optional[List[float]] = None

    constraint_values: Optional[Dict[str, Any]] = None
    constraint_gradients: Optional[Dict[str, Any]] = None
    dual_variables: Optional[Dict[str, Any]] = None
    active_set: Optional[Dict[str, Any]] = None
    qp_status: Optional[str] = None
    collision_info: Optional[Dict[str, Any]] = None
    extra_debug: Dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict for storage."""
        return _to_jsonable(asdict(self))
