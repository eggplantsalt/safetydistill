"""QP solver extraction placeholder APIs."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional


def extract_qp_certificate(
    action_nominal: Optional[List[float]] = None,
    action_safe: Optional[List[float]] = None,
    solver_result: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Extract QP-related certificates from a solver result.

    Returns a dict with dual variables, active set, qp_status, and extra_debug.
    """
    if solver_result is None:
        return {
            "dual_variables": None,
            "active_set": None,
            "qp_status": "placeholder_no_solver_result",
            "extra_debug": {},
        }

    return {
        "dual_variables": None,
        "active_set": None,
        "qp_status": "placeholder_unparsed_solver_result",
        "extra_debug": {"solver_result_type": str(type(solver_result))},
    }
