"""Constraint placeholder APIs.

These functions will be wired to CBF/QP constraint values and gradients in Phase 2.
"""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Optional


def compute_constraint_values(*args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Return constraint values for the current step.

    TODO: Populate values from main/utils.py CBF/QP helpers or lightweight constraints.
    """
    return None


def compute_constraint_gradients(*args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Return constraint gradients for the current step.

    TODO: Populate gradients from main/utils.py CBF/QP helpers or lightweight constraints.
    """
    return None
