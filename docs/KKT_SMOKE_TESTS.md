# KKT Smoke Tests

These are smoke tests for label export stability. They are not formal experiments.

- The dummy policy server does not represent a real VLA.
- The synthetic obstacle does not represent real GroundingDINO perception.
- The goal is to verify label export and QP certificate export.

## Environment variable example

```
export PYTHONPATH=$PWD
```

## Start dummy policy server

```
python -m kkt_sense.scripts.dummy_policy_server \
  --host 0.0.0.0 \
  --port 8000 \
  --chunk-size 10 \
  --action-dim 7 \
  --action-mode zero
```

## Smoke test A: action-only export

```
python main/main_aegis_translational.py \
  --task-suite-name safelibero_spatial \
  --safety-level I \
  --task-index 0 \
  --episode-index 0 \
  --num-trials-per-task 1 \
  --num-steps-wait 20 \
  --replan-steps 5 \
  --disable-groundingdino \
  --enable-kkt-label-export \
  --kkt-label-output-dir data/kkt_safelibero_labels_debug \
  --video-out-path results_kkt_debug
```

Expected:
- 300 JSONL records
- `qp_status = no_safety_control`
- `dual_variables = null`

## Smoke test B: synthetic QP/KKT export

```
python main/main_aegis_translational.py \
  --task-suite-name safelibero_spatial \
  --safety-level I \
  --task-index 0 \
  --episode-index 0 \
  --num-trials-per-task 1 \
  --num-steps-wait 20 \
  --replan-steps 5 \
  --debug-synthetic-safety-obstacle \
  --enable-kkt-label-export \
  --kkt-label-output-dir data/kkt_safelibero_labels_synth_qp_debug \
  --video-out-path results_kkt_synth_qp_debug
```

Expected:
- 300 JSONL records
- `qp_status` mostly or fully `optimal`
- `dual_variables` non-null
- `active_set` non-null
- `constraint_values` non-null
- `constraint_gradients` non-null

## Validate labels

```
python -m kkt_sense.scripts.validate_kkt_labels \
  --path data/kkt_safelibero_labels_synth_qp_debug \
  --require-kkt-fields
```
