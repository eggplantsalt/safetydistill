# KKT Action Flow Notes

This document summarizes where actions are produced, modified, and executed in AEGIS.

## main/main_aegis_translational.py

**Policy action (action_nominal)**
- The policy action is taken from the websocket client output and queued in `action_plan`.
- The per-step nominal action is pulled with `action = action_plan.popleft()`.

**Safety layer action (action_safe)**
- If `flag_safety_control` is true, a QP is solved and the executed action is assembled into `action_input`.
- If `flag_safety_control` is false, the executed action is the original `action`.

**env.step action**
- With safety control: `env.step(action_input.tolist())`.
- Without safety control: `env.step(action.tolist())`.

**Episode loop**
- Episode loop is `while t < max_steps:` with `t` incremented each step.
- Task/episode indices come from the outer loops `for task_id in task_index` and `for episode_idx in episode_index`.
- `task_suite_name` and `safety_level` are read from `args` at the top of `eval_libero`.

## main/main_aegis.py

**Policy action (action_nominal)**
- Same pattern: `action = action_plan.popleft()` uses policy outputs from websocket.

**Safety layer action (action_safe)**
- When `flag_safety_control` is true, `action_input` is built from QP outputs.
- When false, the executed action is `action`.

**env.step action**
- With safety control: `env.step(action_input.tolist())`.
- Without safety control: `env.step(action.tolist())`.

**Episode loop**
- Same structure as translational: `while t < max_steps:` inside task/episode loops.

## utils.py

No direct action selection logic. This file provides CBF/QP helpers and perception utilities.

## Recommended Phase 2 hook point

`main/main_aegis_translational.py` is the better first hook point because it already uses a
reduced action space and simpler safety layer, which aligns with the Phase 2 minimal export.

## Variable summary

- action_nominal: `action` from `action_plan.popleft()`.
- action_safe: `action_input` when safety control is active; otherwise `action`.
- env.step action: `action_input.tolist()` or `action.tolist()` depending on `flag_safety_control`.

## Open questions

- [需确认] Whether `action` is always a numpy array or can be a plain list. The export code
  should safely handle both by converting with `tolist()` when available.

## Phase 3A QP certificate export

- Current translational QP has a single main CBF constraint exported as `cbf_main`.
- `dual_variables` are read from `constraints[0].dual_value` when available.
- `active_set` is computed with `abs(dual_value) > 1e-6`.
- `constraint_values` include `h` and an optional `linear_cbf_lhs` at `u_value`.
- `constraint_gradients` include `a_u_v` and `a_uz`.
- This is a first-pass engineering label, not the final paper-grade KKT certificate.

## Phase 3B robustness fixes

- Repo root path is injected into `sys.path` before importing `kkt_sense` to support
  running `python main/main_aegis_translational.py`.
- Added `_action_to_list` and `_action_to_numpy` helpers to handle numpy/list/tuple actions.
- QP no-solution path now falls back to the nominal reference and logs a warning.
- Fallback is marked in `extra_debug.used_nominal_fallback` and `qp_status` is suffixed
  with `_fallback_nominal`.
- These fixes do not change the default behavior because KKT export remains opt-in.

## Phase 4A synthetic safety obstacle smoke test

- This mode is a smoke test to verify QP/KKT certificate export, not a formal experiment.
- It skips GroundingDINO and real point-cloud perception.
- It injects a synthetic obstacle ellipsoid close to the end-effector.
- It forces `flag_safety_control=True` so the existing CBF-QP branch runs.
- JSONL output should include non-empty `dual_variables`, `active_set`,
  `constraint_values`, and `constraint_gradients`.
- For real experiments, restore the full perception pipeline.
