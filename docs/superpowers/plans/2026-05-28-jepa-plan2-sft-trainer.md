# Input Contracts + Real-Data SFT Trainer (scratch baseline) — Implementation Plan (Plan 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the fixed input contracts (P4 root yaw-canonicalization, P5 canonical-FPS) and a real-data CMAD SFT trainer that trains `MotionLatentRetargeter` on AMASS + GMR-G1 reference actions, producing the `scratch` (from-scratch) baseline — the design's dominant gate.

**Architecture:** Reuse the existing GMR data pipeline (`AMASSSequence10Dataset` + loader, root-feature builder, SMPL/robot FK loss leaf helpers) but drive the NEW `MotionLatentRetargeter` (which the old `train_graph_latent.py` cannot, since its `compute_losses` is bound to the old model). Add a self-contained yaw-canonicalization preprocessing step and a G1 `robot_batch` builder. The trainer supports `--mfme-ckpt`/`--freeze-mfme` so Plan 3 can feed it pretrained encoders unchanged.

**Tech Stack:** PyTorch, `unittest` (run via `pytest`), reused modules in `GBC_MUJICA/.../gbc_retargeting/`, the committed foundation on branch `jepa-gonogo-ablation`, `human_body_prior` BodyModel.

**Scope note — Plan 2 of 3.** Plan 1 (done) built the Motion-JEPA module + synthetic smoke. This plan implements design §3 P1/P4/P5 and produces the `scratch` row. Plan 3 adds the JEPA/MAE pretraining arms and the matrix. **Plan 2 exercises only the from-scratch (no `--mfme-ckpt`) path**; the freeze/ckpt flags are wired but used in Plan 3.

---

## ADDENDUM (2026-05-28): locality config

Plan 1 Task 0 makes `MotionLatentRetargeter` default to `encoder_graph_layers=1`, `encoder_per_slot_temporal=True`, `encoder_root_global_hub=False` (the landed locality fix; memory `project-latent-locality-fix`). The SFT trainer here constructs `MotionLatentRetargeter(d_model=..., z_dim=...)` with those defaults — **correct as written; do not add `encoder_graph_layers=2`.** Checkpoints saved/loaded by this trainer therefore carry the locality-respecting encoder. Plan 1 must land before this plan runs. The FK-frame verification (Task 7 Step 2) is independent of this config.

---

## Verified interface facts (from a code audit — do not re-derive)

**Reused data pipeline (all in `GBC_MUJICA/GBC_MUJICA/data_preparation/gbc_retargeting/`):**
- `amass_sequence10_dataset.py:602` — `build_sequence10_datasets(dataset_path, window, stride, val_ratio, seed, target_fps=20.0, max_files=0, ..., reference_actions_dir=None, require_reference_actions=False) -> (train_ds, val_ds, split)`.
- `amass_sequence10_dataset.py:240` — `collate_sequence_windows`; `:79` — `resample_motion_arrays(poses, trans, source_fps, target_fps)` (pure, NumPy).
- `train_sequence_retargeter.py:182` — `make_loader(dataset, args, shuffle)`; `:146` — `load_fit(smpl_fits_dir, device)` → dict with `beta (1,16)`; `:93` — `make_root_features(poses, trans)` → `(B,T,9)` = `cat([root_orient(3), trans-trans[:,:1] (3), trans_vel(3)])`; `:154` — `compute_gt_joints(body_model, poses, fit, reshape_idx, device) -> (B,T,L,3)`; `:222` — `fk_positions(fk, actions) -> (B,T,L,3)` (root-relative); `:49` — `SMPLH_BODY_NAMES` (22-tuple).

**Batch dict** (after collate): `poses (B,T,66)` float32 (root orient `[...,:3]`, body `[...,3:66]`=21×3), `trans (B,T,3)`, `reference_actions (B,T,29)` + `reference_action_mask (B,)` bool (optional), `reference_root_pos*` (optional), `file_index`/`start`/`target_fps`/`source_fps`. **No `fit` in batch** — load once via `load_fit`.

**G1 cfg** `GBC_MUJICA/GBC_MUJICA/data_preparation/humanoid_cfg/unitree_g1.yaml`: `base.urdf_path=/cpfs/user/yaoyifei/datasets/unitree_ros/robots/g1_description/g1_29dof.urdf` (K=29), `base.smplh_model_path`, `base.dmpls_model_path`, `base.smpl_fits_dir`, `base.fps=20`, `smplh.mapping_table` (13 entries: keys=SMPL-H names, values=robot link names), `poseformer.joint_weights` (13 floats).

**FK loss recipe (reused leaf helpers):** `mapping_table = cfg["smplh"]["mapping_table"]`; `target_links=list(values)`, `mapping_keys=list(keys)`, `reshape_idx=[SMPLH_BODY_NAMES.index(k) for k in mapping_keys]`; `fk = CachedTargetLinkKinematics(urdf_path, target_links, device)` (`GBC/utils/base/cached_fk.py:59`); `body_model = BodyModel(bm_fname=smplh_model_path, dmpl_fname=dmpls_model_path, num_betas=16, num_dmpls=8)`; `fk_loss = ((fk_positions(fk, actions) - compute_gt_joints(body_model, poses, fit, reshape_idx, device)) * jw.view(1,1,-1,1)).square().mean()`.

**New model** `MotionLatentRetargeter.forward(pose_aa (B,T,21,3), root_pose (B,T,9), robot_batch: dict, beta (B,16), part_mask=None, sample=None) -> {actions (B,T,29), z, mu, logvar, graph_hidden, slot_latents, m_per_dof, m_cls, part_mask, future_slots?}` (`morphology_conditioned_retargeter.py:805`). `robot_batch` keys: `node_features, slot_assignment, hop_matrix, control_mode, dof_padding_mask?` — built from the G1 URDF, NOT produced by the data pipeline.

**Rotation helpers (wxyz convention, `GBC/utils/base/math_utils.py`):** `angle_axis_to_quaternion (774)`, `quaternion_to_angle_axis (833)`, `matrix_from_quat (1619)`, `quat_from_matrix (1770)`, `yaw_quat (1850)`, `quat_inv (1714)`, `quat_from_euler_xyz (1727)`.

---

## File Structure

| File | Responsibility |
|---|---|
| `GBC/utils/data_preparation/motion_canonicalize.py` (create) | `canonicalize_motion(poses, trans)` — remove frame-0 world yaw. Self-contained, no model dep. |
| `GBC/utils/data_preparation/g1_sft_helpers.py` (create) | `build_g1_robot_batch(cfg_path, device, batch_size)`; `make_g1_fk_bundle(cfg_path, device)`; `g1_fk_link_loss(bundle, actions, poses, fit, device)`. |
| `scripts/data_preparation/train_morph_latent_sft.py` (create) | The SFT trainer: dataset → canonicalize → model → loss → optim → numeric logging → checkpoint. |
| `tests/test_motion_canonicalize.py` (create) | Yaw-invariance of canonicalization + of MFME `z`. |
| `tests/test_g1_sft_helpers.py` (create) | robot_batch shape/K (asset-gated); FK-loss finite (asset-gated). |
| `tests/test_train_morph_latent_sft.py` (create) | Loss/step smoke on a synthetic batch (no real assets); resample frame-alignment. |

---

## Task 1: P4 — `canonicalize_motion` (root yaw-canonicalization)

**Files:**
- Create: `GBC/utils/data_preparation/motion_canonicalize.py`
- Test: `tests/test_motion_canonicalize.py`

- [ ] **Step 1: Write the failing test** (this is the exact contract)

```python
from __future__ import annotations

import unittest

import torch

from GBC.utils.base.math_utils import (
    angle_axis_to_quaternion, quaternion_to_angle_axis,
    matrix_from_quat, quat_from_matrix, quat_from_euler_xyz,
)
from GBC.utils.data_preparation.motion_canonicalize import canonicalize_motion


def _random_motion(b=2, t=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    poses = torch.zeros(b, t, 66)
    poses[..., :3] = 0.3 * torch.randn(b, t, 3, generator=g)   # root orient (aa)
    poses[..., 3:66] = 0.1 * torch.randn(b, t, 63, generator=g)  # body
    trans = torch.randn(b, t, 3, generator=g)
    return poses, trans


def _apply_world_yaw(poses, trans, phi):
    b = poses.shape[0]
    yaw = torch.full((b,), float(phi))
    zeros = torch.zeros(b)
    q_phi = quat_from_euler_xyz(zeros, zeros, yaw)              # (B,4) wxyz
    M_phi = matrix_from_quat(q_phi)                            # (B,3,3)
    q = angle_axis_to_quaternion(poses[..., :3])               # (B,T,4)
    M = matrix_from_quat(q)                                    # (B,T,3,3)
    M_rot = M_phi[:, None] @ M
    poses2 = poses.clone()
    poses2[..., :3] = quaternion_to_angle_axis(quat_from_matrix(M_rot))
    trans2 = torch.einsum("bij,btj->bti", M_phi, trans)
    return poses2, trans2


class TestCanonicalize(unittest.TestCase):
    def test_yaw_invariance(self):
        poses, trans = _random_motion()
        p_c, t_c = canonicalize_motion(poses, trans)
        for phi in (0.7, -2.1, 3.0):
            pr, tr = _apply_world_yaw(poses, trans, phi)
            pr_c, tr_c = canonicalize_motion(pr, tr)
            self.assertTrue(torch.allclose(p_c[..., :3], pr_c[..., :3], atol=1e-4),
                            f"orient not yaw-invariant at phi={phi}")
            self.assertTrue(torch.allclose(t_c, tr_c, atol=1e-4),
                            f"trans not yaw-invariant at phi={phi}")

    def test_body_pose_untouched(self):
        poses, trans = _random_motion()
        p_c, _ = canonicalize_motion(poses, trans)
        self.assertTrue(torch.allclose(p_c[..., 3:66], poses[..., 3:66]))

    def test_frame0_heading_zeroed(self):
        # After canonicalization, frame-0 root orientation has ~zero yaw.
        poses, trans = _random_motion()
        p_c, _ = canonicalize_motion(poses, trans)
        from GBC.utils.base.math_utils import yaw_quat
        q0 = angle_axis_to_quaternion(p_c[:, 0, :3])
        yq0 = yaw_quat(q0)
        # yaw-only quat ~ identity (w≈±1, z≈0)
        self.assertTrue(torch.allclose(yq0[:, 3].abs(), torch.zeros_like(yq0[:, 3]), atol=1e-3))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /cpfs/user/yaoyifei/codespace/gbc/GBC && PYTHONPATH=.:../GBC_MUJICA python -m pytest tests/test_motion_canonicalize.py -v`
Expected: FAIL — `ModuleNotFoundError: ... motion_canonicalize`.

- [ ] **Step 3: Implement `canonicalize_motion`**

```python
"""Remove frame-0 world yaw from a motion so the encoder latent is heading-invariant.

Body-local joint axis-angles (poses[..., 3:66]) are already invariant to a global
rotation; only the root orientation (poses[..., :3]) and translation carry world yaw.
"""

from __future__ import annotations

from typing import Tuple

import torch

from GBC.utils.base.math_utils import (
    angle_axis_to_quaternion,
    matrix_from_quat,
    quat_from_matrix,
    quat_inv,
    quaternion_to_angle_axis,
    yaw_quat,
)


def canonicalize_motion(poses: torch.Tensor, trans: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``poses (B,T,66)``, ``trans (B,T,3)`` -> yaw-canonicalized copies.

    Rotates the world frame by the inverse of frame-0 heading yaw. Body pose
    (``poses[..., 3:66]``) is returned unchanged.
    """
    orient_aa = poses[..., :3]
    q = angle_axis_to_quaternion(orient_aa)            # (B,T,4) wxyz
    M = matrix_from_quat(q)                             # (B,T,3,3)
    yq0 = yaw_quat(q[:, 0])                              # (B,4) frame-0 yaw-only
    M_inv = matrix_from_quat(quat_inv(yq0))             # (B,3,3) = R_z(-yaw0)
    M_canon = M_inv[:, None] @ M                         # (B,T,3,3)
    orient_canon = quaternion_to_angle_axis(quat_from_matrix(M_canon))
    trans_canon = torch.einsum("bij,btj->bti", M_inv, trans)
    poses_canon = poses.clone()
    poses_canon[..., :3] = orient_canon
    return poses_canon, trans_canon
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=.:../GBC_MUJICA python -m pytest tests/test_motion_canonicalize.py -v`
Expected: PASS (3 tests). If yaw-invariance fails, the quaternion convention is the suspect — but all helpers used are wxyz (verified), so a failure means a transpose in `M_inv @ M`; per systematic-debugging, print intermediate `yq0`/`M_inv` before changing the math.

- [ ] **Step 5: Commit**

```bash
git add GBC/utils/data_preparation/motion_canonicalize.py tests/test_motion_canonicalize.py
git commit -m "feat(jepa): add motion_canonicalize (root yaw-canonicalization, P4)"
```

---

## Task 2: P5 — canonical-FPS contract

P5 needs no new resampling code: `build_sequence10_datasets(target_fps=...)` already resamples every window to the canonical grid via `resample_motion_arrays` (`amass_sequence10_dataset.py:79`), and reference actions are resampled to the same frame count. P5 reduces to **(a) a pinned constant and (b) a test that the resample helper frame-aligns**.

**Files:**
- Modify: `GBC/utils/data_preparation/motion_canonicalize.py` (add the constant)
- Test: `tests/test_motion_canonicalize.py` (append)

- [ ] **Step 1: Write the failing test** (append)

```python
import numpy as np

from GBC.utils.data_preparation.motion_canonicalize import CANONICAL_FPS


class TestCanonicalFps(unittest.TestCase):
    def test_constant_value(self):
        self.assertEqual(CANONICAL_FPS, 20.0)  # matches unitree_g1.yaml base.fps

    def test_resample_aligns_length(self):
        from GBC_MUJICA.data_preparation.gbc_retargeting.amass_sequence10_dataset import (
            resample_motion_arrays,
        )
        poses = np.random.randn(120, 66).astype(np.float32)
        trans = np.random.randn(120, 3).astype(np.float32)
        p2, t2 = resample_motion_arrays(poses, trans, source_fps=120.0, target_fps=CANONICAL_FPS)
        self.assertEqual(p2.shape[0], t2.shape[0])
        self.assertAlmostEqual(p2.shape[0], int(round(120 * CANONICAL_FPS / 120.0)), delta=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=.:../GBC_MUJICA python -m pytest tests/test_motion_canonicalize.py::TestCanonicalFps -v`
Expected: FAIL — `ImportError: cannot import name 'CANONICAL_FPS'`.

- [ ] **Step 3: Add the constant** (top of `motion_canonicalize.py`, after imports)

```python
# Canonical frame rate for all JEPA go/no-go rows. Matches unitree_g1.yaml base.fps.
# Held constant across scratch/mae/jepa so temporal masks cover identical physical horizons.
CANONICAL_FPS: float = 20.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=.:../GBC_MUJICA python -m pytest tests/test_motion_canonicalize.py::TestCanonicalFps -v`
Expected: PASS (2 tests). If `resample_motion_arrays` signature differs, read `amass_sequence10_dataset.py:79` and adapt the call — do not change the constant.

- [ ] **Step 5: Commit**

```bash
git add GBC/utils/data_preparation/motion_canonicalize.py tests/test_motion_canonicalize.py
git commit -m "feat(jepa): pin CANONICAL_FPS=20 contract (P5)"
```

---

## Task 3: G1 `robot_batch` builder

**Files:**
- Create: `GBC/utils/data_preparation/g1_sft_helpers.py`
- Test: `tests/test_g1_sft_helpers.py`

- [ ] **Step 1: Write the failing test** (asset-gated)

```python
from __future__ import annotations

import os
import unittest

import torch
import yaml

CFG = "GBC_MUJICA/GBC_MUJICA/data_preparation/humanoid_cfg/unitree_g1.yaml"


def _urdf_present():
    if not os.path.exists(CFG):
        return False
    base = yaml.safe_load(open(CFG))["base"]
    return os.path.exists(base["urdf_path"])


@unittest.skipUnless(_urdf_present(), "G1 URDF asset not present")
class TestG1RobotBatch(unittest.TestCase):
    def test_keys_and_k(self):
        from GBC.utils.data_preparation.g1_sft_helpers import build_g1_robot_batch
        rb, spec = build_g1_robot_batch(CFG, device="cpu", batch_size=3)
        for key in ("node_features", "slot_assignment", "hop_matrix", "control_mode"):
            self.assertIn(key, rb)
        self.assertEqual(rb["node_features"].shape[0], 3)        # batched
        self.assertEqual(spec.num_dofs, 29)                       # G1 = 29 DoF
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=.:../GBC_MUJICA python -m pytest tests/test_g1_sft_helpers.py -v`
Expected: FAIL (import error) or SKIP if the URDF is absent on this machine. If skipped, also run Step 4 on a machine with assets before relying on it.

- [ ] **Step 3: Implement the builder** (create `g1_sft_helpers.py`)

```python
"""G1-specific glue for the morphology-conditioned SFT trainer: robot_batch + FK loss."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import yaml

from GBC.utils.base.base_fk import RobotKinematics
from GBC.utils.base.cached_fk import CachedTargetLinkKinematics
from GBC.utils.data_preparation.morphology_conditioned_retargeter import (
    RobotSpec,
    batch_robot_specs,
    build_robot_spec_from_kinematics,
)


def _load_cfg(cfg_path: str) -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def build_g1_robot_batch(cfg_path: str, device, batch_size: int) -> Tuple[Dict[str, torch.Tensor], RobotSpec]:
    cfg = _load_cfg(cfg_path)
    urdf = cfg["base"]["urdf_path"]
    rk = RobotKinematics(urdf_path=urdf, device=str(device))
    spec = build_robot_spec_from_kinematics(rk, name="unitree_g1").to(device)
    rb = batch_robot_specs([spec] * batch_size)
    rb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in rb.items()}
    return rb, spec
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=.:../GBC_MUJICA python -m pytest tests/test_g1_sft_helpers.py -v`
Expected: PASS on a machine with the G1 URDF (else SKIP).

- [ ] **Step 5: Commit**

```bash
git add GBC/utils/data_preparation/g1_sft_helpers.py tests/test_g1_sft_helpers.py
git commit -m "feat(jepa): G1 robot_batch builder for SFT"
```

---

## Task 4: G1 FK link loss (reused leaf helpers)

**Files:**
- Modify: `GBC/utils/data_preparation/g1_sft_helpers.py`
- Test: `tests/test_g1_sft_helpers.py` (append)

- [ ] **Step 1: Write the failing test** (asset-gated)

```python
def _assets_present():
    if not os.path.exists(CFG):
        return False
    base = yaml.safe_load(open(CFG))["base"]
    return all(os.path.exists(base[k]) for k in
               ("urdf_path", "smplh_model_path", "dmpls_model_path", "smpl_fits_dir"))


@unittest.skipUnless(_assets_present(), "G1/SMPLH assets not present")
class TestG1FkLoss(unittest.TestCase):
    def test_fk_loss_finite(self):
        from GBC.utils.data_preparation.g1_sft_helpers import make_g1_fk_bundle, g1_fk_link_loss
        from GBC_MUJICA.data_preparation.gbc_retargeting.train_sequence_retargeter import load_fit
        bundle = make_g1_fk_bundle(CFG, device="cpu")
        fit = load_fit(yaml.safe_load(open(CFG))["base"]["smpl_fits_dir"], "cpu")
        b, t = 2, 8
        poses = torch.zeros(b, t, 66)
        actions = torch.zeros(b, t, 29)
        loss = g1_fk_link_loss(bundle, actions, poses, fit, device="cpu")
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=.:../GBC_MUJICA python -m pytest tests/test_g1_sft_helpers.py::TestG1FkLoss -v`
Expected: FAIL (import error) or SKIP if assets absent.

- [ ] **Step 3: Implement the FK bundle + loss** (append to `g1_sft_helpers.py`)

```python
from dataclasses import dataclass


@dataclass
class G1FkBundle:
    fk: CachedTargetLinkKinematics
    body_model: object
    reshape_idx: List[int]
    joint_weights: torch.Tensor


def make_g1_fk_bundle(cfg_path: str, device) -> G1FkBundle:
    from human_body_prior.body_model.body_model import BodyModel
    from GBC_MUJICA.data_preparation.gbc_retargeting.train_sequence_retargeter import (
        SMPLH_BODY_NAMES,
    )

    cfg = _load_cfg(cfg_path)
    base = cfg["base"]
    mapping = cfg["smplh"]["mapping_table"]
    mapping_keys = list(mapping.keys())
    target_links = list(mapping.values())
    reshape_idx = [SMPLH_BODY_NAMES.index(k) for k in mapping_keys]

    fk = CachedTargetLinkKinematics(base["urdf_path"], target_links, device=str(device))
    body_model = BodyModel(
        bm_fname=base["smplh_model_path"], dmpl_fname=base["dmpls_model_path"],
        num_betas=16, num_dmpls=8,
    ).to(device)
    jw = torch.tensor(cfg["poseformer"]["joint_weights"], dtype=torch.float32, device=device)
    return G1FkBundle(fk=fk, body_model=body_model, reshape_idx=reshape_idx, joint_weights=jw)


def g1_fk_link_loss(bundle: G1FkBundle, actions: torch.Tensor, poses: torch.Tensor,
                    fit: Dict[str, torch.Tensor], device) -> torch.Tensor:
    from GBC_MUJICA.data_preparation.gbc_retargeting.train_sequence_retargeter import (
        compute_gt_joints, fk_positions,
    )
    gt = compute_gt_joints(bundle.body_model, poses, fit, bundle.reshape_idx, device)  # (B,T,L,3)
    pred = fk_positions(bundle.fk, actions)                                            # (B,T,L,3)
    jw = bundle.joint_weights.view(1, 1, -1, 1)
    return ((pred - gt) * jw).square().mean()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=.:../GBC_MUJICA python -m pytest tests/test_g1_sft_helpers.py::TestG1FkLoss -v`
Expected: PASS on a machine with assets (else SKIP).

- [ ] **Step 5: Commit**

```bash
git add GBC/utils/data_preparation/g1_sft_helpers.py tests/test_g1_sft_helpers.py
git commit -m "feat(jepa): G1 FK link loss reusing compute_gt_joints/fk_positions"
```

---

## Task 5: The SFT trainer

**Files:**
- Create: `scripts/data_preparation/train_morph_latent_sft.py`

Loss (design §3 P1): `1000 * masked_action_l2(actions, ref) + fk_weight * fk_link_loss + 1e-3 * latent_smoothness_loss(z)`. The model is fed **canonicalized** `root_pose` (P4) and `pose_aa`; the FK loss is computed on the **same canonicalized poses** for frame consistency. `--require-reference-actions` guarantees every batch has GMR targets (so `masked_action_l2` needs no per-row mask).

- [ ] **Step 1: Write the trainer**

```python
"""SFT trainer for MotionLatentRetargeter on AMASS + GMR-G1 reference actions.

Drives the NEW morphology-conditioned model (the old train_graph_latent.py cannot).
Reuses the AMASSSequence10Dataset pipeline + SMPL/robot FK leaf helpers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from GBC.utils.data_preparation.morphology_conditioned_retargeter import (
    MotionLatentRetargeter, masked_action_l2, latent_smoothness_loss,
)
from GBC.utils.data_preparation.motion_canonicalize import canonicalize_motion, CANONICAL_FPS
from GBC.utils.data_preparation.g1_sft_helpers import (
    build_g1_robot_batch, make_g1_fk_bundle, g1_fk_link_loss,
)
from GBC_MUJICA.data_preparation.gbc_retargeting.amass_sequence10_dataset import (
    build_sequence10_datasets,
)
from GBC_MUJICA.data_preparation.gbc_retargeting.train_sequence_retargeter import (
    make_loader, load_fit, make_root_features,
)


def batch_to_model_inputs(batch, fit, device):
    poses = batch["poses"].to(device)
    trans = batch["trans"].to(device)
    poses, trans = canonicalize_motion(poses, trans)             # P4
    b = poses.shape[0]
    pose_aa = poses[..., 3:66].reshape(b, poses.shape[1], 21, 3)
    root_pose = make_root_features(poses, trans)                 # (B,T,9)
    beta = fit["beta"].to(device).expand(b, -1)
    ref = batch["reference_actions"].to(device)
    return pose_aa, root_pose, beta, ref, poses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="GBC_MUJICA/GBC_MUJICA/data_preparation/humanoid_cfg/unitree_g1.yaml")
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--reference-actions-dir", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--z-dim", type=int, default=256)
    ap.add_argument("--action-weight", type=float, default=1000.0)
    ap.add_argument("--fk-weight", type=float, default=1.0)
    ap.add_argument("--smooth-weight", type=float, default=1e-3)
    ap.add_argument("--mfme-ckpt", default=None, help="pretrained MFME state_dict (Plan 3)")
    ap.add_argument("--freeze-mfme", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-steps", type=int, default=0, help="0=full epochs")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = yaml.safe_load(open(args.cfg))

    train_ds, val_ds, _ = build_sequence10_datasets(
        dataset_path=args.dataset_path, window=args.window, stride=args.stride,
        val_ratio=0.1, seed=0, target_fps=CANONICAL_FPS,
        reference_actions_dir=args.reference_actions_dir, require_reference_actions=True,
    )
    loader = make_loader(train_ds, args, shuffle=True)
    fit = load_fit(cfg["base"]["smpl_fits_dir"], device)
    rb, _ = build_g1_robot_batch(args.cfg, device, args.batch_size)
    fk_bundle = make_g1_fk_bundle(args.cfg, device) if args.fk_weight > 0 else None

    model = MotionLatentRetargeter(d_model=args.d_model, z_dim=args.z_dim).to(device)
    if args.mfme_ckpt:
        state = torch.load(args.mfme_ckpt, map_location=device)
        missing, unexpected = model.mfme.load_state_dict(state, strict=False)
        print(f"loaded MFME ckpt: missing={list(missing)} unexpected={list(unexpected)}")
    if args.freeze_mfme:
        for p in model.mfme.parameters():
            p.requires_grad_(False)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    step, t0 = 0, time.time()
    log = []
    for epoch in range(args.epochs):
        for batch in loader:
            bs = batch["poses"].shape[0]
            rb_b = rb if bs == args.batch_size else build_g1_robot_batch(args.cfg, device, bs)[0]
            pose_aa, root_pose, beta, ref, poses_canon = batch_to_model_inputs(batch, fit, device)
            out = model(pose_aa=pose_aa, root_pose=root_pose, robot_batch=rb_b, beta=beta, sample=False)
            recon = masked_action_l2(out["actions"], ref, rb_b.get("dof_padding_mask"))
            fk = (g1_fk_link_loss(fk_bundle, out["actions"], poses_canon, fit, device)
                  if fk_bundle is not None else out["actions"].new_zeros(()))
            smooth = latent_smoothness_loss(out["z"])
            loss = args.action_weight * recon + args.fk_weight * fk + args.smooth_weight * smooth
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 50 == 0:
                rec = {"step": step, "epoch": epoch, "loss": float(loss.item()),
                       "recon": float(recon.item()), "fk": float(fk.item()),
                       "smooth": float(smooth.item()), "elapsed_sec": time.time() - t0}
                log.append(rec); print(json.dumps(rec))
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    torch.save({"model": model.state_dict(), "args": vars(args)}, save_dir / "last.pt")
    (save_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"saved {save_dir/'last.pt'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit (script only; tested in Task 6)**

```bash
git add scripts/data_preparation/train_morph_latent_sft.py
git commit -m "feat(jepa): MotionLatentRetargeter SFT trainer (scratch baseline path)"
```

---

## Task 6: Trainer loss/step smoke test (no real assets)

**Files:**
- Create: `tests/test_train_morph_latent_sft.py`

Verifies the loss assembly + one optimizer step on a **synthetic** batch, without the real dataset/FK assets (FK disabled). Catches shape/wiring bugs in CI.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import unittest

import torch

from GBC.utils.data_preparation.morphology_conditioned_retargeter import (
    MotionLatentRetargeter, masked_action_l2, latent_smoothness_loss,
    build_synthetic_robot_spec, batch_robot_specs,
)
from GBC.utils.data_preparation.motion_canonicalize import canonicalize_motion


def _toy_rb(batch):
    names = [f"j{i}" for i in range(8)]
    parents = [-1, 0, 1, 2, 0, 4, 0, 6]
    spec = build_synthetic_robot_spec("toy", names, parents, seed=0)
    return batch_robot_specs([spec] * batch)


class TestSftStep(unittest.TestCase):
    def test_one_step_decreases_or_finite(self):
        torch.manual_seed(0)
        model = MotionLatentRetargeter(d_model=64, z_dim=64, slot_latent_dim=16,
                                       encoder_graph_layers=1, encoder_temporal_layers=1,
                                       encoder_num_heads=4, morphology_num_heads=4,
                                       decoder_num_heads=4)
        b, t = 2, 8
        poses = torch.randn(b, t, 66); trans = torch.randn(b, t, 3)
        poses, trans = canonicalize_motion(poses, trans)
        pose_aa = poses[..., 3:66].reshape(b, t, 21, 3)
        root_pose = torch.cat([poses[..., :3], trans, torch.zeros(b, t, 3)], dim=-1)  # (B,T,9)
        beta = torch.randn(b, 16)
        rb = _toy_rb(b)
        k = rb["node_features"].shape[1]
        ref = torch.randn(b, t, k)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        out = model(pose_aa=pose_aa, root_pose=root_pose, robot_batch=rb, beta=beta, sample=False)
        loss = 1000.0 * masked_action_l2(out["actions"], ref, rb.get("dof_padding_mask")) \
               + 1e-3 * latent_smoothness_loss(out["z"])
        self.assertTrue(torch.isfinite(loss))
        opt.zero_grad(); loss.backward(); opt.step()
        self.assertTrue(any(p.grad is not None for p in model.cmad.parameters()))
```

- [ ] **Step 2: Run test to verify it passes**

Run: `PYTHONPATH=.:../GBC_MUJICA python -m pytest tests/test_train_morph_latent_sft.py -v`
Expected: PASS (exercises already-implemented model + canonicalization; no real assets).

- [ ] **Step 3: Commit**

```bash
git add tests/test_train_morph_latent_sft.py
git commit -m "test(jepa): SFT trainer loss/step smoke (synthetic, asset-free)"
```

---

## Task 7: Real `scratch` baseline run

- [ ] **Step 1: Train the scratch baseline on GMR G1**

Run:
```bash
PYTHONPATH=.:../GBC_MUJICA python scripts/data_preparation/train_morph_latent_sft.py \
  --dataset-path /path/to/amass \
  --reference-actions-dir /cpfs/user/yaoyifei/datasets/amass_gmr_convert/reference_actions \
  --window 120 --stride 20 --batch-size 32 --epochs 20 \
  --save-dir outputs/morph_latent_sft_scratch_g1_v1
```
Expected: `loss`/`recon` fall steadily; `outputs/morph_latent_sft_scratch_g1_v1/last.pt` + `train_log.json` written. This is the `scratch` row of the matrix and the equal-budget reference for Plan 3.

- [ ] **Step 2: Verification checkpoint — FK loss sanity**

Confirm `fk` in `train_log.json` decreases (not flat-high). If it plateaus high while `recon` falls, the human/robot FK frame convention is misaligned (design §4 risk): compare `compute_gt_joints` output frame to `fk_positions`; align by feeding non-canonicalized poses to `compute_gt_joints` OR rotating robot pred into the human root frame. Do not silence the term — fix the frame.

- [ ] **Step 3: Record the baseline number**

Note the final `recon` (action-L2) and wall-clock. Plan 3 reuses this exact config + budget for `mae`/`jepa`.

---

## Final verification

- [ ] Asset-free tests pass: `PYTHONPATH=.:../GBC_MUJICA python -m pytest tests/test_motion_canonicalize.py tests/test_train_morph_latent_sft.py -v` → PASS.
- [ ] Asset-gated tests pass on a machine with G1/SMPLH assets: `tests/test_g1_sft_helpers.py` → PASS (not SKIP).
- [ ] Foundation + Plan 1 tests still pass: `PYTHONPATH=.:../GBC_MUJICA python -m pytest tests/test_morphology_conditioned_retargeter.py tests/test_motion_jepa.py -q` → PASS.
- [ ] `scratch` baseline checkpoint + log exist with falling `recon`.

**Exit state:** a working real-data SFT trainer and a recorded `scratch` G1 baseline. The `--mfme-ckpt`/`--freeze-mfme` flags are wired for Plan 3 but unused here.

---

## Self-Review

**Spec coverage (vs `2026-05-28-jepa-gonogo-ablation-design.md`):**
- P4 root yaw-canonicalization + yaw-invariance test → Task 1. ✓
- P5 canonical-FPS contract → Task 2 (reuses existing `target_fps` resampling). ✓
- P1 real-data CMAD SFT trainer (reuse dataset + FK helpers, build G1 robot_batch, loss = 1000·action-L2 + fk + smooth, `--mfme-ckpt`/`--freeze-mfme`) → Tasks 3–5. ✓
- `scratch` row produced on real GMR G1 → Task 7. ✓
- Deferred to Plan 3: real-AMASS JEPA pretraining, MAE head + pretraining, the `mae`/`jepa` rows, Level-1 probe, report/decision. ✓

**Placeholder scan:** no TBD/TODO; the only non-literal value is `--dataset-path /path/to/amass` (a runtime path the user supplies), and the FK-frame verification (Task 7 Step 2) is an explicit checkpoint, not a code gap. ✓

**Type/name consistency:** `canonicalize_motion(poses, trans) -> (poses, trans)`, `CANONICAL_FPS=20.0`, `build_g1_robot_batch(cfg_path, device, batch_size) -> (rb_dict, spec)`, `make_g1_fk_bundle(cfg_path, device) -> G1FkBundle`, `g1_fk_link_loss(bundle, actions, poses, fit, device)`, `batch_to_model_inputs(...) -> (pose_aa, root_pose, beta, ref, poses_canon)`. The trainer feeds the model `forward(pose_aa, root_pose, robot_batch, beta, sample)` and reads `out["actions"]`/`out["z"]` — matches `morphology_conditioned_retargeter.py:805`. `masked_action_l2(actions, ref, dof_padding_mask)` matches `:842`. Reused helper signatures match the audit (`build_sequence10_datasets`, `make_loader`, `load_fit`, `make_root_features`, `compute_gt_joints`, `fk_positions`). ✓

**Known risk flagged in-plan:** human/robot FK-frame alignment under canonicalization (Task 7 Step 2) — gated by a verification checkpoint, not assumed correct.
