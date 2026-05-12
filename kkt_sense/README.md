# kkt_sense

KKT-SenseVLA label generation scaffolding for exporting KKT-SafeLIBERO distillation data
from VLSA/AEGIS teacher rollouts.

## Phase 1 scope
- Defines a lightweight schema for step records.
- Supports JSONL export and dummy label generation.
- No real SafeLIBERO or MuJoCo runs.
- No QP dual variable extraction yet.

## Phase 2 (planned)
- Hook into AEGIS evaluation to capture `action_nominal` and `action_safe`.
- Populate `action_delta` from the safety layer outputs.

## Phase 3 (planned)
- Extract QP solver certificates (`dual_variables`, `active_set`).
- Add `constraint_values` and `constraint_gradients` from CBF/QP or lightweight constraints.

## Dummy label generation
```
python -m kkt_sense.scripts.generate_kkt_labels \
  --task-suite-name safelibero_spatial \
  --safety-level I \
  --task-index 0 \
  --episode-index 0 \
  --output-dir data/kkt_safelibero_debug
```

Expected output file name:
```
{task_suite_name}_level_{safety_level}_task_{task_index}_episode_{episode_index}.jsonl
```
