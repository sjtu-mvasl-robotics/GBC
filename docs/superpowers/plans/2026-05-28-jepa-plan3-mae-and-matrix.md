# MAE Baseline + scratch/mae/jepa Matrix — Implementation Plan (Plan 3 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the MAE input-recon pretraining baseline, the real-AMASS pretraining driver for both objectives, a val evaluator, the Level-1 probe gate, and the matrix runner that produces the `scratch`/`mae`/`jepa` go/no-go decision.

**Architecture:** Reuse Plan 1's `MotionJEPAModule` (jepa arm) and Plan 2's dataset + canonicalization + SFT trainer. Add a parallel `MaeReconModule` (mae arm) that shares the exact mask machinery but predicts raw input pose instead of target features. One pretraining driver emits an MFME `state_dict` per objective; the SFT trainer (Plan 2, `--mfme-ckpt`/`--freeze-mfme`) consumes it. A matrix runner orchestrates the three rows at equal budget and applies the decision rule.

**Tech Stack:** PyTorch, `unittest` (run via `pytest`), Plan 1 + Plan 2 modules on branch `jepa-gonogo-ablation`.

**Scope note — Plan 3 of 3.** Depends on Plan 1 (`motion_jepa.py`, `slot_time_mask`) and Plan 2 (`motion_canonicalize.py`, `g1_sft_helpers.py`, `train_morph_latent_sft.py`, dataset reuse). Implements design §3 P3, §6 matrix, §7 Level-1+2, §9 decision rule.

---

## Reused interfaces (verified in Plans 1–2)

- `MorphologyFreeMotionEncoder(d_model, z_dim, slot_latent_dim, ...)` with `forward(pose_aa, root_pose, beta, part_mask, sample, slot_time_mask)`; `.z_dim`, `.num_human_joints` (`=21`), `.root_dim` (`=9`) set in `__init__` (`morphology_conditioned_retargeter.py:351-353`).
- `motion_jepa.py`: `MotionJEPAModule(encoder, mask_cfg, num_slots, ...)` `.online_encoder`/`.update_target(m)`; `MotionMaskCfg`, `MotionMaskGenerator(cfg, num_slots)`.
- `motion_canonicalize.py`: `canonicalize_motion(poses, trans)`, `CANONICAL_FPS=20.0`.
- Dataset: `build_sequence10_datasets(dataset_path, window, stride, val_ratio, seed, target_fps, reference_actions_dir, require_reference_actions)`, `make_loader(ds, args, shuffle)`, `load_fit(dir, device)`, `make_root_features(poses, trans)`.
- SFT trainer CLI (Plan 2): `scripts/data_preparation/train_morph_latent_sft.py` with `--mfme-ckpt`, `--freeze-mfme`, `--max-steps`, `--save-dir`, writing `last.pt` + `train_log.json`.
- `MotionLatentRetargeter` forward returns `out["actions"] (B,T,29)`, `out["z"]`; spec from `build_g1_robot_batch` (Plan 2) gives `dof_names` for lower-body selection.

---

## File Structure

| File | Responsibility |
|---|---|
| `GBC/utils/data_preparation/motion_jepa.py` (modify) | Add `MaeReconModule` (shares `MotionMaskGenerator`; reconstructs input pose). |
| `scripts/data_preparation/pretrain_mfme.py` (create) | Real-AMASS pretraining driver, `--objective {jepa,mae}`, canonicalized inputs, saves MFME `state_dict`. |
| `scripts/data_preparation/evaluate_morph_latent.py` (create) | Val action-L2 + lower-body amplitude ratio for a checkpoint. |
| `scripts/data_preparation/probe_mfme_level1.py` (create) | Frozen-MFME future-slot probe; compares jepa/mae/untrained (§7 gate 2). |
| `scripts/data_preparation/run_jepa_matrix.py` (create) | Orchestrate scratch/mae/jepa at equal budget; collect metrics + deltas; apply decision rule. |
| `tests/test_motion_jepa.py` (modify) | Add `MaeReconModule` tests. |
| `tests/test_jepa_matrix.py` (create) | Decision-rule logic unit test (pure, no training). |

---

## Task 1: `MaeReconModule` (MAE input-recon baseline)

**Files:**
- Modify: `GBC/utils/data_preparation/motion_jepa.py`
- Test: `tests/test_motion_jepa.py` (append)

Same `slot_time_mask` masking as JEPA; the online encoder sees the masked clip; a head on `z` reconstructs the full input pose (`21*3 + root_dim`); loss = L2 to the true input pose. No EMA target — this is the input-space contrast to JEPA's feature prediction.

- [ ] **Step 1: Write the failing test**

```python
from GBC.utils.data_preparation.motion_jepa import MaeReconModule, MotionMaskCfg


class TestMaeModule(unittest.TestCase):
    def test_forward_finite_and_grad(self):
        torch.manual_seed(0)
        mfme = _mfme_small()  # defined earlier in this test file (Plan 1)
        mod = MaeReconModule(mfme, MotionMaskCfg(), num_slots=6)
        pose, root, beta = _synth(b=2, t=16)
        out = mod(pose, root, beta, generator=torch.Generator().manual_seed(0))
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        self.assertTrue(any(p.grad is not None for p in mod.encoder.parameters()))

    def test_recon_head_shape(self):
        mfme = _mfme_small()
        mod = MaeReconModule(mfme, MotionMaskCfg(), num_slots=6)
        pose, root, beta = _synth(b=2, t=16)
        recon = mod.reconstruct(pose, root, beta)
        self.assertEqual(recon.shape, (2, 16, 21 * 3 + 9))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /cpfs/user/yaoyifei/codespace/gbc/GBC && PYTHONPATH=.:../GBC_MUJICA python -m pytest tests/test_motion_jepa.py::TestMaeModule -v`
Expected: FAIL — `ImportError: cannot import name 'MaeReconModule'`.

- [ ] **Step 3: Implement `MaeReconModule`** (append to `motion_jepa.py`)

```python
class MaeReconModule(nn.Module):
    """MAE-style input-space reconstruction baseline (the recon contrast to JEPA).

    Masks slot-time tokens (same generator as JEPA), encodes the masked clip,
    and reconstructs the full input pose ``[pose_aa(21*3) | root_pose(root_dim)]``.
    """

    def __init__(self, encoder: nn.Module, mask_cfg: MotionMaskCfg, num_slots: int,
                 hidden: int = 256):
        super().__init__()
        self.encoder = encoder
        self.mask_gen = MotionMaskGenerator(mask_cfg, num_slots=num_slots)
        out_dim = encoder.num_human_joints * 3 + encoder.root_dim
        self.recon_head = nn.Sequential(
            nn.LayerNorm(encoder.z_dim), nn.Linear(encoder.z_dim, hidden),
            nn.GELU(), nn.Linear(hidden, out_dim),
        )

    def reconstruct(self, pose_aa, root_pose, beta=None, slot_time_mask=None):
        enc = self.encoder(pose_aa, root_pose, beta, sample=False, slot_time_mask=slot_time_mask)
        return self.recon_head(enc["z"])  # (B, T, 21*3 + root_dim)

    def forward(self, pose_aa, root_pose, beta=None, generator=None):
        b, t = pose_aa.shape[0], pose_aa.shape[1]
        mask = self.mask_gen(b, t, generator=generator).to(pose_aa.device)
        recon = self.reconstruct(pose_aa, root_pose, beta, slot_time_mask=mask)
        target = torch.cat([pose_aa.reshape(b, t, -1), root_pose], dim=-1)  # (B,T,21*3+root)
        loss = F.smooth_l1_loss(recon, target)
        with torch.no_grad():
            enc_z = self.encoder(pose_aa, root_pose, beta, sample=False, slot_time_mask=mask)["z"]
            z_std = enc_z.std().detach()
        return {"loss": loss, "z_std": z_std, "mask_ratio": mask.float().mean().detach()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=.:../GBC_MUJICA python -m pytest tests/test_motion_jepa.py::TestMaeModule -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add GBC/utils/data_preparation/motion_jepa.py tests/test_motion_jepa.py
git commit -m "feat(jepa): add MaeReconModule (input-space recon baseline)"
```

---

## Task 2: Real-AMASS pretraining driver (`pretrain_mfme.py`)

**Files:**
- Create: `scripts/data_preparation/pretrain_mfme.py`

Reuses the dataset (no reference actions needed) + canonicalization. `--objective jepa` trains `MotionJEPAModule` (with EMA), `--objective mae` trains `MaeReconModule`. Saves `mfme_state.pt` (the encoder `state_dict`) consumable by the Plan 2 SFT trainer's `--mfme-ckpt`.

- [ ] **Step 1: Write the script**

```python
"""Pretrain MFME on AMASS with JEPA (feature prediction) or MAE (input recon).

Saves the encoder state_dict for the Plan 2 SFT trainer (--mfme-ckpt).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from GBC.utils.data_preparation.morphology_conditioned_retargeter import MorphologyFreeMotionEncoder
from GBC.utils.data_preparation.motion_jepa import (
    MotionJEPAModule, MaeReconModule, MotionMaskCfg,
)
from GBC.utils.data_preparation.motion_canonicalize import canonicalize_motion, CANONICAL_FPS
from GBC_MUJICA.data_preparation.gbc_retargeting.amass_sequence10_dataset import (
    build_sequence10_datasets,
)
from GBC_MUJICA.data_preparation.gbc_retargeting.train_sequence_retargeter import make_loader


def _inputs(batch, device):
    poses = batch["poses"].to(device)
    trans = batch["trans"].to(device)
    poses, trans = canonicalize_motion(poses, trans)
    b, t = poses.shape[0], poses.shape[1]
    pose_aa = poses[..., 3:66].reshape(b, t, 21, 3)
    from GBC_MUJICA.data_preparation.gbc_retargeting.train_sequence_retargeter import make_root_features
    root_pose = make_root_features(poses, trans)
    beta = None  # beta optional for pretraining; encoder handles None
    return pose_aa, root_pose, beta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["jepa", "mae"], required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--z-dim", type=int, default=256)
    ap.add_argument("--ema-start", type=float, default=0.996)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    train_ds, _, _ = build_sequence10_datasets(
        dataset_path=args.dataset_path, window=args.window, stride=args.stride,
        val_ratio=0.1, seed=0, target_fps=CANONICAL_FPS,
        reference_actions_dir=None, require_reference_actions=False,
    )
    loader = make_loader(train_ds, args, shuffle=True)

    mfme = MorphologyFreeMotionEncoder(d_model=args.d_model, z_dim=args.z_dim).to(device)
    if args.objective == "jepa":
        mod = MotionJEPAModule(mfme, MotionMaskCfg(), num_slots=6,
                               pred_d_model=128, pred_layers=2, pred_heads=4,
                               max_window=args.window).to(device)
        encoder_for_ckpt = lambda: mod.online_encoder
    else:
        mod = MaeReconModule(mfme, MotionMaskCfg(), num_slots=6).to(device)
        encoder_for_ckpt = lambda: mod.encoder

    opt = torch.optim.Adam([p for p in mod.parameters() if p.requires_grad], lr=args.lr)
    gen = torch.Generator(device=device).manual_seed(0)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    log, step, t0 = [], 0, time.time()

    for epoch in range(args.epochs):
        for batch in loader:
            pose_aa, root_pose, beta = _inputs(batch, device)
            out = mod(pose_aa, root_pose, beta, generator=gen)
            opt.zero_grad(); out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
            opt.step()
            if args.objective == "jepa":
                mod.update_target(momentum=args.ema_start)
            if step % 50 == 0:
                rec = {"step": step, "loss": float(out["loss"].item()),
                       "z_std": float(out["z_std"].item()),
                       "elapsed_sec": time.time() - t0}
                if "target_std" in out:
                    rec["target_std"] = float(out["target_std"].item())
                log.append(rec); print(json.dumps(rec))
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    torch.save(encoder_for_ckpt().state_dict(), save_dir / "mfme_state.pt")
    (save_dir / "pretrain_log.json").write_text(json.dumps(log, indent=2))
    # Non-collapse guard (design §7).
    assert log[-1]["z_std"] > 1e-3, "COLLAPSE: z std ~ 0"
    print(f"saved {save_dir/'mfme_state.pt'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/data_preparation/pretrain_mfme.py
git commit -m "feat(jepa): real-AMASS MFME pretraining driver (jepa|mae)"
```

---

## Task 3: Level-1 frozen-MFME probe (`probe_mfme_level1.py`)

**Files:**
- Create: `scripts/data_preparation/probe_mfme_level1.py`

§7 gate 2: a frozen-MFME future-slot probe (predict `z[t+k]` from `z[t]`). Run for jepa/mae/untrained; gate = jepa & mae beat untrained. Emits JSON.

- [ ] **Step 1: Write the script**

```python
"""Level-1 frozen-MFME probe: future-z prediction error for jepa/mae/untrained."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from GBC.utils.data_preparation.morphology_conditioned_retargeter import MorphologyFreeMotionEncoder
from GBC.utils.data_preparation.motion_canonicalize import canonicalize_motion, CANONICAL_FPS
from GBC_MUJICA.data_preparation.gbc_retargeting.amass_sequence10_dataset import build_sequence10_datasets
from GBC_MUJICA.data_preparation.gbc_retargeting.train_sequence_retargeter import (
    make_loader, make_root_features,
)


def _batch_inputs(batch, device):
    poses, trans = canonicalize_motion(batch["poses"].to(device), batch["trans"].to(device))
    b, t = poses.shape[0], poses.shape[1]
    return poses[..., 3:66].reshape(b, t, 21, 3), make_root_features(poses, trans)


def probe(encoder, loader, device, k=5, steps=200):
    encoder.eval()
    head = nn.Linear(encoder.z_dim, encoder.z_dim).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    it, last = iter(loader), 0.0
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        pose_aa, root_pose = _batch_inputs(batch, device)
        with torch.no_grad():
            z = encoder(pose_aa, root_pose, None, sample=False)["z"]
        pred = head(z[:, :-k])
        loss = ((pred - z[:, k:].detach()) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        last = loss.item()
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--jepa-ckpt", required=True)
    ap.add_argument("--mae-ckpt", required=True)
    ap.add_argument("--out", default="outputs/jepa_level1_probe.json")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--z-dim", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    _, val_ds, _ = build_sequence10_datasets(
        dataset_path=args.dataset_path, window=64, stride=16, val_ratio=0.1, seed=0,
        target_fps=CANONICAL_FPS, reference_actions_dir=None, require_reference_actions=False)
    loader = make_loader(val_ds, args, shuffle=True)

    def load(ckpt):
        enc = MorphologyFreeMotionEncoder(d_model=args.d_model, z_dim=args.z_dim).to(device)
        if ckpt != "untrained":
            enc.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        return enc

    res = {
        "probe_untrained": probe(load("untrained"), loader, device),
        "probe_mae": probe(load(args.mae_ckpt), loader, device),
        "probe_jepa": probe(load(args.jepa_ckpt), loader, device),
    }
    res["jepa_beats_untrained"] = bool(res["probe_jepa"] < res["probe_untrained"])
    res["mae_beats_untrained"] = bool(res["probe_mae"] < res["probe_untrained"])
    res["level1_pass"] = bool(res["jepa_beats_untrained"] and res["mae_beats_untrained"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/data_preparation/probe_mfme_level1.py
git commit -m "feat(jepa): Level-1 frozen-MFME future-z probe gate"
```

---

## Task 4: Val evaluator (`evaluate_morph_latent.py`)

**Files:**
- Create: `scripts/data_preparation/evaluate_morph_latent.py`

Primary go/no-go metric: val `action_l2` to GMR, plus lower-body dynamic amplitude ratio (predicted vs GMR std over time on hip/knee/ankle DoFs).

- [ ] **Step 1: Write the script**

```python
"""Evaluate a MotionLatentRetargeter SFT checkpoint on the G1 val split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from GBC.utils.data_preparation.morphology_conditioned_retargeter import (
    MotionLatentRetargeter, masked_action_l2,
)
from GBC.utils.data_preparation.motion_canonicalize import canonicalize_motion, CANONICAL_FPS
from GBC.utils.data_preparation.g1_sft_helpers import build_g1_robot_batch
from GBC_MUJICA.data_preparation.gbc_retargeting.amass_sequence10_dataset import build_sequence10_datasets
from GBC_MUJICA.data_preparation.gbc_retargeting.train_sequence_retargeter import (
    make_loader, load_fit, make_root_features,
)


def _lower_body_idx(dof_names):
    return [i for i, n in enumerate(dof_names)
            if any(s in n.lower() for s in ("hip", "knee", "ankle"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="GBC_MUJICA/GBC_MUJICA/data_preparation/humanoid_cfg/unitree_g1.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--reference-actions-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    cfg = yaml.safe_load(open(args.cfg))

    _, val_ds, _ = build_sequence10_datasets(
        dataset_path=args.dataset_path, window=120, stride=20, val_ratio=0.1, seed=0,
        target_fps=CANONICAL_FPS, reference_actions_dir=args.reference_actions_dir,
        require_reference_actions=True)
    loader = make_loader(val_ds, args, shuffle=False)
    fit = load_fit(cfg["base"]["smpl_fits_dir"], device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = MotionLatentRetargeter(d_model=ckpt["args"]["d_model"], z_dim=ckpt["args"]["z_dim"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    rb, spec = build_g1_robot_batch(args.cfg, device, args.batch_size)
    lb = _lower_body_idx(spec.dof_names)

    al2, amp_pred, amp_ref, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            bs = batch["poses"].shape[0]
            rb_b = rb if bs == args.batch_size else build_g1_robot_batch(args.cfg, device, bs)[0]
            poses, trans = canonicalize_motion(batch["poses"].to(device), batch["trans"].to(device))
            pose_aa = poses[..., 3:66].reshape(bs, poses.shape[1], 21, 3)
            root_pose = make_root_features(poses, trans)
            beta = fit["beta"].to(device).expand(bs, -1)
            ref = batch["reference_actions"].to(device)
            out = model(pose_aa=pose_aa, root_pose=root_pose, robot_batch=rb_b, beta=beta, sample=False)
            al2 += masked_action_l2(out["actions"], ref, rb_b.get("dof_padding_mask")).item() * bs
            amp_pred += out["actions"][..., lb].std(dim=1).mean().item() * bs
            amp_ref += ref[..., lb].std(dim=1).mean().item() * bs
            n += bs
    report = {
        "action_l2": al2 / n,
        "lower_body_amp_ratio": (amp_pred / n) / max(amp_ref / n, 1e-9),
        "n_windows": n, "ckpt": args.ckpt,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/data_preparation/evaluate_morph_latent.py
git commit -m "feat(jepa): val evaluator (action-L2 + lower-body amplitude ratio)"
```

---

## Task 5: Matrix runner + decision rule

**Files:**
- Create: `scripts/data_preparation/run_jepa_matrix.py`
- Test: `tests/test_jepa_matrix.py`

Orchestrates the three rows at equal `--max-steps`, then applies the design §9 decision rule. The decision logic is unit-tested in isolation (pure function) so it is verified without training.

- [ ] **Step 1: Write the failing decision-rule test**

```python
from __future__ import annotations

import unittest

from scripts.data_preparation.run_jepa_matrix import decide


class TestDecision(unittest.TestCase):
    def test_keep_jepa(self):
        # jepa within 10% of scratch AND beats mae on amplitude
        d = decide(scratch_l2=0.40, mae_l2=0.42, jepa_l2=0.41,
                   mae_amp=0.70, jepa_amp=0.95)
        self.assertEqual(d["verdict"], "keep_jepa")

    def test_park_jepa_when_tied_with_mae(self):
        d = decide(scratch_l2=0.40, mae_l2=0.41, jepa_l2=0.41,
                   mae_amp=0.90, jepa_amp=0.90)
        self.assertEqual(d["verdict"], "park_jepa_use_mae")

    def test_decoder_issue_when_jepa_regresses(self):
        d = decide(scratch_l2=0.40, mae_l2=0.45, jepa_l2=0.60,
                   mae_amp=0.8, jepa_amp=0.8)
        self.assertEqual(d["verdict"], "investigate_decoder")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /cpfs/user/yaoyifei/codespace/gbc/GBC && PYTHONPATH=.:../GBC_MUJICA python -m pytest tests/test_jepa_matrix.py -v`
Expected: FAIL — `ModuleNotFoundError: run_jepa_matrix`.

- [ ] **Step 3: Implement the runner + `decide`**

```python
"""Orchestrate the scratch/mae/jepa go/no-go matrix at equal budget + decide."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict

PY = "python"


def decide(scratch_l2: float, mae_l2: float, jepa_l2: float,
           mae_amp: float, jepa_amp: float, tol: float = 0.10) -> Dict[str, object]:
    within = jepa_l2 <= scratch_l2 * (1 + tol)
    beats_mae = (jepa_l2 < mae_l2) or (abs(jepa_l2 - mae_l2) <= 0.02 and jepa_amp > mae_amp)
    if not within and jepa_l2 > scratch_l2 * (1 + tol):
        verdict = "investigate_decoder"
    elif within and beats_mae:
        verdict = "keep_jepa"
    elif within and not beats_mae:
        verdict = "park_jepa_use_mae"
    else:
        verdict = "investigate_decoder"
    return {"verdict": verdict, "within_tol": bool(within), "beats_mae": bool(beats_mae),
            "delta_jepa_minus_scratch": jepa_l2 - scratch_l2,
            "delta_jepa_minus_mae": jepa_l2 - mae_l2}


def _run(cmd):
    print("+", " ".join(cmd)); subprocess.run(cmd, check=True)


def _final_recon(log_path: str) -> float:
    log = json.loads(Path(log_path).read_text())
    return float(log[-1]["recon"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--reference-actions-dir", required=True)
    ap.add_argument("--out-dir", default="outputs/jepa_matrix")
    ap.add_argument("--pretrain-steps", type=int, default=20000)
    ap.add_argument("--sft-steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)
    pre = ["scripts/data_preparation/pretrain_mfme.py"]
    sft = ["scripts/data_preparation/train_morph_latent_sft.py"]
    ev = ["scripts/data_preparation/evaluate_morph_latent.py"]
    common_data = ["--dataset-path", args.dataset_path]

    # Pretrain mae + jepa.
    for obj in ("mae", "jepa"):
        _run([PY, *pre, "--objective", obj, *common_data,
              "--max-steps", str(args.pretrain_steps),
              "--save-dir", str(od / f"pretrain_{obj}")])

    # SFT the three rows at equal budget.
    rows = {
        "scratch": [],
        "mae": ["--mfme-ckpt", str(od / "pretrain_mae" / "mfme_state.pt"), "--freeze-mfme"],
        "jepa": ["--mfme-ckpt", str(od / "pretrain_jepa" / "mfme_state.pt"), "--freeze-mfme"],
    }
    for row, extra in rows.items():
        _run([PY, *sft, *common_data, "--reference-actions-dir", args.reference_actions_dir,
              "--max-steps", str(args.sft_steps), "--save-dir", str(od / f"sft_{row}"), *extra])
        _run([PY, *ev, "--ckpt", str(od / f"sft_{row}" / "last.pt"), *common_data,
              "--reference-actions-dir", args.reference_actions_dir,
              "--out", str(od / f"eval_{row}.json")])

    evals = {r: json.loads((od / f"eval_{r}.json").read_text()) for r in rows}
    verdict = decide(
        scratch_l2=evals["scratch"]["action_l2"], mae_l2=evals["mae"]["action_l2"],
        jepa_l2=evals["jepa"]["action_l2"], mae_amp=evals["mae"]["lower_body_amp_ratio"],
        jepa_amp=evals["jepa"]["lower_body_amp_ratio"])
    report = {"evals": evals, "decision": verdict}
    (od / "matrix_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=.:../GBC_MUJICA python -m pytest tests/test_jepa_matrix.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/data_preparation/run_jepa_matrix.py tests/test_jepa_matrix.py
git commit -m "feat(jepa): matrix runner + decision rule (scratch/mae/jepa)"
```

---

## Task 6: Run the full matrix + record the decision

- [ ] **Step 1: Level-1 gate first (cheap)**

Run pretraining for both objectives (Task 2 command), then:
```bash
PYTHONPATH=.:../GBC_MUJICA python scripts/data_preparation/probe_mfme_level1.py \
  --dataset-path /path/to/amass \
  --jepa-ckpt outputs/jepa_matrix/pretrain_jepa/mfme_state.pt \
  --mae-ckpt  outputs/jepa_matrix/pretrain_mae/mfme_state.pt
```
Gate: `level1_pass == true` (jepa AND mae beat untrained). If jepa fails, fix mask/target (design §11) before the matrix.

- [ ] **Step 2: Run the matrix at equal budget**

```bash
PYTHONPATH=.:../GBC_MUJICA python scripts/data_preparation/run_jepa_matrix.py \
  --dataset-path /path/to/amass \
  --reference-actions-dir /cpfs/user/yaoyifei/datasets/amass_gmr_convert/reference_actions \
  --pretrain-steps 20000 --sft-steps 20000
```
Expected: `outputs/jepa_matrix/matrix_report.json` with per-row `action_l2`/`lower_body_amp_ratio` and a `decision.verdict` in {`keep_jepa`, `park_jepa_use_mae`, `investigate_decoder`}.

- [ ] **Step 3: Cross-check against the Plan 2 baseline**

Confirm the matrix `scratch` `action_l2` matches the Plan 2 standalone `scratch` baseline (same config/budget) within noise. A mismatch means the runner's budget/seed differs — fix before trusting the verdict.

---

## Final verification

- [ ] Asset-free unit tests pass: `PYTHONPATH=.:../GBC_MUJICA python -m pytest tests/test_motion_jepa.py tests/test_jepa_matrix.py -v` → PASS.
- [ ] Level-1 gate JSON shows `level1_pass: true`.
- [ ] `outputs/jepa_matrix/matrix_report.json` exists with a verdict.
- [ ] Full suite (Plans 1–3) green: `PYTHONPATH=.:../GBC_MUJICA python -m pytest tests/test_motion_jepa.py tests/test_jepa_retarget_integration.py tests/test_motion_canonicalize.py tests/test_train_morph_latent_sft.py tests/test_jepa_matrix.py -v`.

**Exit state:** the go/no-go decision is recorded in `matrix_report.json`. Per design §9: `keep_jepa` → proceed to joint-finetune (`jepa-ft`) + design Phases 4–5; `park_jepa_use_mae` → keep the recon objective; `investigate_decoder` → freeze longer / lower LR / warm up CMAD.

---

## Self-Review

**Spec coverage (vs `2026-05-28-jepa-gonogo-ablation-design.md`):**
- P3 MAE-recon head + masked-pose objective (shares slot_time_mask) → Task 1. ✓
- Real-AMASS pretraining for jepa + mae (canonicalized, CANONICAL_FPS, non-collapse guard) → Task 2. ✓
- §7 gate 2 Level-1 frozen-MFME probe (jepa/mae vs untrained) → Task 3, Task 6 Step 1. ✓
- Primary metric (val action-L2) + lower-body amplitude ratio → Task 4. ✓
- §6 matrix at equal budget (scratch/mae/jepa, frozen mae/jepa) → Task 5 runner. ✓
- §9 decision rule (within 10% of scratch AND beats mae; else park/investigate) → `decide()` Task 5, tested. ✓

**Placeholder scan:** no TBD/TODO; the only non-literals are runtime `--dataset-path /path/to/amass` (user-supplied) and the reference-actions dir (a real path). ✓

**Type/name consistency:** `MaeReconModule(encoder, mask_cfg, num_slots, hidden)` with `.encoder`/`.reconstruct(...)`/`.forward(...)` returning `loss`/`z_std`; `pretrain_mfme.py` saves the encoder `state_dict` consumed by Plan 2's `--mfme-ckpt` (`model.mfme.load_state_dict(..., strict=False)`); `decide(scratch_l2, mae_l2, jepa_l2, mae_amp, jepa_amp)` matches the runner's call and the test; eval keys `action_l2`/`lower_body_amp_ratio` match the runner's reads. MFME attrs used (`num_human_joints`, `root_dim`, `z_dim`) are set in `morphology_conditioned_retargeter.py:351-355`. ✓

**Cross-plan dependency check:** Plan 3 imports only symbols created in Plan 1 (`MotionJEPAModule`, `MotionMaskCfg`, `MaeReconModule`), Plan 2 (`canonicalize_motion`, `CANONICAL_FPS`, `build_g1_robot_batch`, the SFT trainer CLI), and the verified reused pipeline. No forward references to unbuilt symbols. ✓
