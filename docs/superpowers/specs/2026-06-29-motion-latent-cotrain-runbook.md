# Motion-Latent Cotrain / Morph Descriptor Runbook

**Date:** 2026-06-29

This runbook covers the committed motion-latent cotrain and morph-descriptor code path.
The GR3 zero-shot evaluation experiment is intentionally parked for now.

## Scope

- Core model lives in `GBC/utils/data_preparation/morphology_conditioned_retargeter.py`.
- Fit-derived descriptors live in `GBC/utils/data_preparation/morphology_descriptor.py`.
- Single-robot and cotrain trainers live in
  `GBC_MUJICA/GBC_MUJICA/data_preparation/gbc_retargeting/`.
- The current cotrain target is G1 + H1-2. GR3 zero-shot render/eval scripts are not part of
  the required run path.

## Environment

Run commands from `/cpfs/user/yaoyifei/codespace/gbc`.

```bash
source /cpfs/user/yaoyifei/miniconda3/etc/profile.d/conda.sh
conda activate GBC
export HUMANOID_NAME=unitree_g1
export PYTHONPATH=/cpfs/user/yaoyifei/codespace/gbc/GBC:/cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

For DLC CLI calls, unset lab proxies first:

```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
```

## Verification

Targeted tests for the committed surface:

```bash
cd /cpfs/user/yaoyifei/codespace/gbc/GBC
PYTHONPATH=/cpfs/user/yaoyifei/codespace/gbc/GBC:/cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA \
  /cpfs/user/yaoyifei/miniconda3/envs/GBC/bin/python -m pytest -q \
  tests/test_morphology_descriptor.py \
  tests/test_morph_descriptor_injection.py \
  tests/test_root_kinematics.py \
  tests/test_unified_human_decoder.py \
  tests/test_robot_motion_encoder.py \
  tests/test_retarget_metrics.py \
  tests/test_loss_guards.py \
  tests/test_h1_2_urdf_parses.py
```

Expected current result: `40 passed`.

Syntax check for trainer entrypoints:

```bash
PYTHONPATH=/cpfs/user/yaoyifei/codespace/gbc/GBC:/cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA \
  /cpfs/user/yaoyifei/miniconda3/envs/GBC/bin/python -m py_compile \
  /cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA/GBC_MUJICA/data_preparation/gbc_retargeting/train_graph_latent.py \
  /cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA/GBC_MUJICA/data_preparation/gbc_retargeting/train_cotrain_motion_latent.py \
  /cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA/GBC_MUJICA/data_preparation/gbc_retargeting/train_motion_latent.py \
  /cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA/GBC_MUJICA/data_preparation/gbc_retargeting/train_multi_morph.py
```

Note: full `pytest -q tests` is not a valid gate yet. Some untracked legacy tests still use the old
3-positional-argument `MotionLatentRetargeter` API or script-style pytest fixtures.

## Local Cotrain Smoke

Use a tiny window-limited run to verify the b0 path without launching a full experiment:

```bash
cd /cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA
CUDA_VISIBLE_DEVICES=0 \
HUMANOID_NAME=unitree_g1 \
PYTHONPATH=/cpfs/user/yaoyifei/codespace/gbc/GBC:/cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/cpfs/user/yaoyifei/miniconda3/envs/GBC/bin/python \
  -m GBC_MUJICA.data_preparation.gbc_retargeting.train_cotrain_motion_latent \
  --robot g1:/cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA/GBC_MUJICA/data_preparation/humanoid_cfg/unitree_g1.yaml:/cpfs/user/yaoyifei/datasets/amass_gmr_convert/reference_actions \
  --robot h1_2:/cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA/GBC_MUJICA/data_preparation/humanoid_cfg/unitree_h1_2_handless.yaml:/cpfs/user/yaoyifei/datasets/amass_gmr_convert_h1_2_xl/reference_actions \
  --init-ckpt /cpfs/user/yaoyifei/codespace/gbc/gbc_runs/unified_human_g1_0605/models/09_14_29_latent_best_epoch_4_step_24088.pt \
  --cotrain-balance min \
  --device cuda:0 \
  --model-arch motion_latent \
  --unified-human-decoder \
  --epochs 1 \
  --batch-size 2 \
  --files-per-batch 1 \
  --window 60 \
  --stride 30 \
  --num-workers 0 \
  --max-train-windows 1 \
  --max-val-windows 1 \
  --val-batches 1 \
  --lr 2e-5 \
  --latent-architecture structured \
  --output-mode latent \
  --slot-z 12 \
  --graph-layers 2 \
  --decoder-context masked_graph_hidden \
  --decoder-root-slot-into-arms \
  --no-robot-dof-root-edges \
  --temporal-module causal_transformer \
  --sample-latent \
  --require-reference-actions \
  --no-tqdm \
  --save-dir /cpfs/user/yaoyifei/gbc_runs/debug_morph_b0_smoke_0629 \
  --reference-action-weight 0 \
  --latent-reference-action-weight 0 \
  --ref-action-ee-weight 1 \
  --latent-root-z-weight 50 \
  --latent-foot-slide-weight 100 \
  --latent-ground-penetration-weight 100 \
  --latent-contact-height-weight 100 \
  --latent-feet-distance-weight 50 \
  --latent-action-smooth-weight 50 \
  --latent-action-jerk-weight 50 \
  --latent-human-recon-weight 100 \
  --use-transform-loss \
  --transform-loss-rot-weight 1.0 \
  --fk-target-from-gmr \
  --body-plane-weight 1000 \
  --morph-descriptor-kind none
```

Expected smoke signal: `metrics.jsonl` ends with `{"stage": "done", ...}`.

## DLC Cotrain Submission

The existing workspace wrapper can submit the four descriptor cells:

```bash
cd /cpfs/user/yaoyifei/codespace/gbc
CELL=b0 DATE=MMDD scripts_phaseA/submit_cotrain_morph_dlc.sh
CELL=add DATE=MMDD scripts_phaseA/submit_cotrain_morph_dlc.sh
CELL=addroot DATE=MMDD scripts_phaseA/submit_cotrain_morph_dlc.sh
CELL=nodeaug DATE=MMDD scripts_phaseA/submit_cotrain_morph_dlc.sh
```

Cell meanings:

- `b0`: no descriptor, matched cotrain control.
- `add`: bone-length descriptor added to `m_cls` and `m_per_dof`.
- `addroot`: bone-length descriptor added to `m_cls` only.
- `nodeaug`: bone-length descriptor injected before the morphology graph.

## Parked

Do not spend more time on the GR3 zero-shot render/eval path for this branch unless it is explicitly
reactivated. The parked surface includes `eval_gr3_zeroshot_morph.py`, `render_gr3_zeroshot.py`,
GR3 render outputs, and zero-shot status reports.
