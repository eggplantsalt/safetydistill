# Phase 11E/11F Healthybase Recovery and Tier-1 Student Audit

Date: 2026-05-14

## Context

This note records the recovery of the KKT-SenseVLA / OpenVLA-OFT Tier-1 student pipeline after discovering corrupted 1000-step checkpoint component files and broken base model shard state.

The original 1000-step deployable checkpoint had corrupted zero-byte component files:
- action_head--1000_checkpoint.pt
- proprio_projector--1000_checkpoint.pt
- kkt_head--1000_checkpoint.pt

A healthybase repair path was used to restore a valid OpenVLA-OFT base checkpoint and rebuild a usable KKT-trained student checkpoint.

## Recovered student checkpoint

The healthybase student checkpoint was validated by:
- OpenVLA student server startup
- /act smoke test
- single-episode safetydistill rollout probe
- 4-task Tier-1 student eval
- teacher/student comparison audit

## Healthybase student eval result

Dataset:
- data/kkt_openvla_goal_student_eval_healthybase_tier1/manifest.json

Coverage:
- 4 tasks × 1 episode
- 1200 records
- task indices: 0, 1, 2, 3
- 300 records per task

Health:
- qp_status: optimal 1200/1200
- state_lens: 8 for 1200/1200
- has_kkt: 1200/1200
- nominal_zero: 0/1200
- safe_zero: 0/1200
- delta_zero: 0/1200
- nominal_gripper: {-1.0: 623, 1.0: 577}

Action delta norm:
- min: 7.850768917460118e-09
- p50: 9.207769990474856e-09
- p90: 0.04097781119593314
- max: 0.9280785078180626

## Teacher pilot reference

Dataset:
- data/kkt_openvla_goal_pilot_tier1/manifest.json

Coverage:
- 4 tasks × 2 episodes
- 2400 records

Health:
- qp_status: optimal 2400/2400
- has_kkt: 2400/2400
- nominal_zero: 0/2400
- safe_zero: 0/2400
- delta_zero: 0/2400
- nominal_gripper: {1.0: 1279, -1.0: 1121}

Teacher action delta norm:
- p50: 0.05118033960329934
- p90: 0.4591465568930989
- max: 1.6053557041546207

## Teacher vs healthybase student comparison

Student / teacher action_delta ratio:
- p50 ratio: 1.799083410122835e-07
- p90 ratio: 0.08924778065029425
- max ratio: 0.5781139378744651

Interpretation:
The healthybase student requires much smaller AEGIS/QP correction than the original OpenVLA-OFT teacher under the same Tier-1 synthetic-obstacle setup. This is a positive sanity signal that the student has learned the safe-action distribution from the Tier-1 KKT/OpenVLA samples.

## Caveats

This remains a Tier-1 synthetic-obstacle pilot/sanity result.

It does not yet prove final real perception safety because:
- GroundingDINO is still skipped.
- fit_ellipse is still skipped in synthetic mode.
- The obstacle is synthetic/debug, not perception-derived.
- SafeLIBERO task success is not the primary metric for this stage.
- Teacher and student rollouts are not strict same-trajectory paired evaluations.

## Current status

Phase 11E-repair: completed.
Phase 11F teacher/student Tier-1 audit: completed.

The recovered healthybase student pipeline is ready for the next phase.

## Recommended next phase

Before entering real perception, run one controlled Tier-1 ablation/eval expansion if needed:
- teacher baseline
- healthybase student
- optional no-AEGIS rollout
- compare collision, success, QP correction magnitude, action_delta distribution, and gripper behavior

Then enter Tier 2:
- enable GroundingDINO
- inspect detection quality
- inspect fit_ellipse quality
- run tiny 1-task perception probe
- only then scale to multi-task real-perception evaluation
