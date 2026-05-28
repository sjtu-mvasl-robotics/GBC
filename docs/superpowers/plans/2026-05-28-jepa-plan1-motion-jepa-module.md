# Motion-JEPA Module + Synthetic Smoke Gate — Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the self-supervised Motion-JEPA module (`slot_time_mask` on MFME, mask generator, EMA target encoder, predictor, feature-prediction loss) and a synthetic non-collapse smoke gate, all without any real-data / GMR dependency.

**Architecture:** Wrap the already-committed `MorphologyFreeMotionEncoder` (MFME) in an online/EMA-target JEPA module. The online encoder sees a slot-time-masked clip; an EMA target encoder sees the full clip (detached). A narrow transformer predictor refines the online `graph_hidden` and is trained to match the target `graph_hidden` (+ `slot_latents` aux) at masked positions, in normalized feature space. A synthetic smoke script proves non-collapse and frozen-probe transfer.

**Tech Stack:** PyTorch, `unittest` (run via `pytest`), the committed `morphology_conditioned_retargeter.py` foundation on branch `jepa-gonogo-ablation`.

**Scope note — this is Plan 1 of 3.** The full go/no-go design (`2026-05-28-jepa-gonogo-ablation-design.md`) spans independent subsystems; per the writing-plans scope check it is split:
- **Plan 1 (this doc):** Motion-JEPA module + synthetic Phase-0 smoke gate. No real data. Implements design §3 P2 and §7 gate 1.
- **Plan 2 (later):** P4 root canonicalization + P5 FPS resample + P1 real-data CMAD SFT trainer producing the `scratch` baseline on GMR G1.
- **Plan 3 (later):** P3 MAE-recon head + the `scratch`/`mae`/`jepa` matrix + Level-1 probe + report/decision.

---

## ADDENDUM (2026-05-28): align MFME with the landed locality fix

The structured-path locality-leak fix has landed in `graph_retarget_latent.py` (memory: `project-latent-locality-fix`, `feedback-graph-layers-one-for-locality`). `GraphTemporalEncoder` now takes `per_slot_temporal: bool=False` and `root_global_hub: bool=True` (both default to **legacy/leaky**); MFME's call sites are unchanged, so MFME is not broken — but by default it produces a cross-region-leaky `graph_hidden`, which is exactly the JEPA target. To keep the JEPA latent locality-respecting, MFME must opt in:

- `graph_layers = 1` (≥2 reintroduces a 2-hop cross-region leak via the joint-tree tokenizer + slot trunk)
- `per_slot_temporal = True` (temporal attention within each slot, not across the full T×S grid)
- `root_global_hub = False` (tree slot edges; `global_root` is not a star hub)

**Task 0 below makes these the MFME / `MotionLatentRetargeter` defaults.** After Task 0, all constructions in Plans 1–3 inherit them; wherever a plan body shows `graph_layers=2`, it is superseded by `=1`. The unit-test MFMEs in this plan already pass `graph_layers=1`. **In Task 7 `run_smoke`, delete `graph_layers=2` from both `MorphologyFreeMotionEncoder(...)` calls** so they inherit the locality default (keep `temporal_layers=2`, `num_heads=8`).

**Out of scope (open; user still deciding — do NOT implement):** per-slot VAE bottleneck for MFME (`z_slots` / derived `z_dim`) and an independent pure-pelvis root slot. Locality is enforced at the encoder/`graph_hidden` level (the JEPA target); MFME keeps its pooled `z` for CMAD/MAE/probes.

---

## File Structure

| File | Responsibility |
|---|---|
| `GBC/utils/data_preparation/morphology_conditioned_retargeter.py` (modify) | Add optional `slot_time_mask` + a learned mask token to `MorphologyFreeMotionEncoder`. Default behavior unchanged when `slot_time_mask is None`. |
| `GBC/utils/data_preparation/motion_jepa.py` (create) | `MotionMaskCfg`, `MotionMaskGenerator`, `MotionJEPAPredictor`, `MotionJEPAModule`, `motion_jepa_loss`, `ema_update`. |
| `tests/test_motion_jepa.py` (create) | Unit tests for mask, predictor, module, loss, EMA. |
| `tests/test_jepa_retarget_integration.py` (create) | Load JEPA-trained MFME into `MotionLatentRetargeter.mfme`; default schema unchanged; frozen-MFME still passes CMAD gradients. |
| `scripts/data_preparation/pretrain_motion_jepa_smoke.py` (create) | Synthetic Phase-0 smoke: train JEPA a few hundred steps, log collapse monitors + frozen future-slot probe vs untrained, emit JSON gate. |

**Verified interface facts (do not re-derive):**
- `MorphologyFreeMotionEncoder.forward(pose_aa, root_pose, beta=None, part_mask=None, sample=None)` returns dict with `mu/logvar/z` `(B,T,256)`, `slot_latents (B,T,6,32)`, `graph_hidden (B,T,6,256)`, `part_mask (B,6)` (`morphology_conditioned_retargeter.py:408-431`).
- Internally: `slot_tokens, part_mask = self.tokenizer(...)` → `(B,T,6,d_model)`; then `graph_hidden = self.encoder(slot_tokens, part_mask)` (`:416-417`). The mask injection goes **between** these two calls.
- `DEFAULT_BODY_SLOTS` = `(global_root, trunk_head, left_arm, right_arm, left_leg, right_leg)`, so `S=6`, slot 0 is `global_root` (`graph_retarget_latent.py:14`).
- Synthetic-motion helper pattern (mirror in tests/smoke): `pose (B,T,21,3)`, `root (B,T,9)`, `beta (B,16)` (`scripts/data_preparation/run_morph_latent_experiments.py:184-189`).
- `MotionLatentRetargeter().mfme` is a `MorphologyFreeMotionEncoder` (`morphology_conditioned_retargeter.py:744`).

---

## Task 0: Make MFME locality-aware (foundation edit)

**Files:**
- Modify: `GBC/utils/data_preparation/morphology_conditioned_retargeter.py` (`MorphologyFreeMotionEncoder.__init__`, `MotionLatentRetargeter.__init__`)
- Test: `tests/test_motion_jepa.py`

- [ ] **Step 1: Write the failing test** (prepend to `tests/test_motion_jepa.py`, after the imports/`_synth` helper)

```python
class TestLocalityDefaults(unittest.TestCase):
    def test_mfme_locality_config(self):
        from GBC.utils.data_preparation.morphology_conditioned_retargeter import (
            MotionLatentRetargeter,
        )
        mfme = MorphologyFreeMotionEncoder(d_model=64, z_dim=64, slot_latent_dim=16, num_heads=4)
        self.assertTrue(mfme.encoder.per_slot_temporal)
        self.assertEqual(len(mfme.encoder.graph_blocks), 1)            # graph_layers default = 1
        self.assertFalse(mfme.encoder.graph_blocks[0].root_global_hub)  # tree slot edges
        rt = MotionLatentRetargeter(d_model=64, z_dim=64, slot_latent_dim=16,
                                    encoder_num_heads=4, morphology_num_heads=4, decoder_num_heads=4)
        self.assertTrue(rt.mfme.encoder.per_slot_temporal)
        self.assertEqual(len(rt.mfme.encoder.graph_blocks), 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /cpfs/user/yaoyifei/codespace/gbc/GBC && PYTHONPATH=. python -m pytest tests/test_motion_jepa.py::TestLocalityDefaults -v`
Expected: FAIL — `per_slot_temporal` is `False` and there are 2 graph blocks (legacy defaults).

- [ ] **Step 3: Edit `MorphologyFreeMotionEncoder.__init__`**

Change the `graph_layers` default and add two flags to the signature:

```python
        graph_layers: int = 1,            # was 2 — graph_layers>=2 reintroduces 2-hop leak
        temporal_layers: int = 4,
        num_heads: int = 8,
        slots: Sequence[str] = DEFAULT_BODY_SLOTS,
        edges: Iterable[Tuple[str, str]] = DEFAULT_BODY_EDGES,
        slot_joints: Optional[Dict[str, Sequence[int]]] = None,
        tokenizer: str = "structured",
        dropout: float = 0.1,
        per_slot_temporal: bool = True,   # locality: temporal attn within each slot
        root_global_hub: bool = False,    # locality: tree slot edges (root not a hub)
```

And pass both flags to the encoder construction:

```python
        self.encoder = GraphTemporalEncoder(
            d_model=d_model,
            num_heads=num_heads,
            slots=self.slots,
            edges=edges,
            graph_layers=graph_layers,
            temporal_layers=temporal_layers,
            dropout=dropout,
            per_slot_temporal=per_slot_temporal,
            root_global_hub=root_global_hub,
        )
```

- [ ] **Step 4: Edit `MotionLatentRetargeter.__init__`**

Change `encoder_graph_layers` default and add two flags:

```python
        encoder_graph_layers: int = 1,    # was 2
        encoder_temporal_layers: int = 4,
        encoder_num_heads: int = 8,
        ...
        encoder_per_slot_temporal: bool = True,
        encoder_root_global_hub: bool = False,
```

And thread them into the `self.mfme = MorphologyFreeMotionEncoder(...)` construction:

```python
            graph_layers=encoder_graph_layers,
            ...
            per_slot_temporal=encoder_per_slot_temporal,
            root_global_hub=encoder_root_global_hub,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_motion_jepa.py::TestLocalityDefaults -v`
Expected: PASS.

- [ ] **Step 6: Foundation regression**

Run: `PYTHONPATH=. python -m pytest tests/test_morphology_conditioned_retargeter.py -q`
Expected: PASS. The structural tests forward/check gradients and tolerate `graph_layers=1`. If one asserts a layer-count or exact-value, investigate per systematic-debugging — do NOT revert the locality default to satisfy a stale assertion.

- [ ] **Step 7: Commit**

```bash
git add GBC/utils/data_preparation/morphology_conditioned_retargeter.py tests/test_motion_jepa.py
git commit -m "feat(jepa): MFME locality config (graph_layers=1, per_slot_temporal, tree edges)"
```

---

## Task 1: Add `slot_time_mask` to MFME (default-preserving)

**Files:**
- Modify: `GBC/utils/data_preparation/morphology_conditioned_retargeter.py` (`MorphologyFreeMotionEncoder.__init__` `:333-399`, `.forward` `:408-431`)
- Test: `tests/test_motion_jepa.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_motion_jepa.py`:

```python
from __future__ import annotations

import unittest

import torch

from GBC.utils.data_preparation.morphology_conditioned_retargeter import (
    MorphologyFreeMotionEncoder,
)


def _synth(b=2, t=8, device="cpu", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    pose = torch.randn(b, t, 21, 3, generator=g, device=device)
    root = torch.randn(b, t, 9, generator=g, device=device)
    beta = torch.randn(b, 16, generator=g, device=device)
    return pose, root, beta


class TestSlotTimeMask(unittest.TestCase):
    def test_none_mask_is_identity(self):
        torch.manual_seed(0)
        mfme = MorphologyFreeMotionEncoder(d_model=64, z_dim=64, slot_latent_dim=16,
                                           graph_layers=1, temporal_layers=1, num_heads=4)
        mfme.eval()
        pose, root, beta = _synth()
        out_a = mfme(pose, root, beta, sample=False)
        out_b = mfme(pose, root, beta, slot_time_mask=None, sample=False)
        self.assertTrue(torch.allclose(out_a["graph_hidden"], out_b["graph_hidden"], atol=1e-6))

    def test_mask_changes_output(self):
        torch.manual_seed(0)
        mfme = MorphologyFreeMotionEncoder(d_model=64, z_dim=64, slot_latent_dim=16,
                                           graph_layers=1, temporal_layers=1, num_heads=4)
        mfme.eval()
        pose, root, beta = _synth()
        b, t, s = 2, 8, 6
        mask = torch.zeros(b, t, s, dtype=torch.bool)
        mask[:, 4:, 2] = True  # hide left_arm in second half
        out_full = mfme(pose, root, beta, sample=False)
        out_masked = mfme(pose, root, beta, slot_time_mask=mask, sample=False)
        self.assertFalse(torch.allclose(out_full["graph_hidden"], out_masked["graph_hidden"], atol=1e-5))
        self.assertEqual(out_masked["graph_hidden"].shape, (b, t, s, 64))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /cpfs/user/yaoyifei/codespace/gbc/GBC && PYTHONPATH=. python -m pytest tests/test_motion_jepa.py::TestSlotTimeMask -v`
Expected: FAIL — `TypeError: forward() got an unexpected keyword argument 'slot_time_mask'`.

- [ ] **Step 3: Add the mask token in `__init__`**

In `MorphologyFreeMotionEncoder.__init__`, after `self.logvar_head = ...` (`:399`), add:

```python
        self.masked_slot_time_token = nn.Parameter(torch.zeros(d_model))
```

- [ ] **Step 4: Add `slot_time_mask` to `forward`**

Replace the `forward` signature and the first two body lines (`:408-417`) with:

```python
    def forward(
        self,
        pose_aa: torch.Tensor,
        root_pose: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
        sample: Optional[bool] = None,
        slot_time_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        slot_tokens, part_mask = self.tokenizer(pose_aa, root_pose, beta, part_mask)
        if slot_time_mask is not None:
            tok = self.masked_slot_time_token.to(slot_tokens.dtype)
            slot_tokens = torch.where(slot_time_mask.unsqueeze(-1), tok, slot_tokens)
        graph_hidden = self.encoder(slot_tokens, part_mask)
```

(Keep the rest of `forward` from `pooled = _masked_mean(...)` onward unchanged.)

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_motion_jepa.py::TestSlotTimeMask -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Confirm no regression in existing tests**

Run: `PYTHONPATH=. python -m pytest tests/test_morphology_conditioned_retargeter.py -q`
Expected: PASS (mask defaults to `None`, so behavior is unchanged).

- [ ] **Step 7: Commit**

```bash
git add GBC/utils/data_preparation/morphology_conditioned_retargeter.py tests/test_motion_jepa.py
git commit -m "feat(jepa): add slot_time_mask to MorphologyFreeMotionEncoder"
```

---

## Task 2: `MotionMaskCfg` + `MotionMaskGenerator`

**Files:**
- Create: `GBC/utils/data_preparation/motion_jepa.py`
- Test: `tests/test_motion_jepa.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_motion_jepa.py`)

```python
from GBC.utils.data_preparation.motion_jepa import MotionMaskCfg, MotionMaskGenerator


class TestMaskGenerator(unittest.TestCase):
    def test_shape_and_dtype(self):
        gen = MotionMaskGenerator(MotionMaskCfg(), num_slots=6)
        mask = gen(batch_size=4, window=16, generator=torch.Generator().manual_seed(0))
        self.assertEqual(mask.shape, (4, 16, 6))
        self.assertEqual(mask.dtype, torch.bool)

    def test_some_but_not_all_masked(self):
        gen = MotionMaskGenerator(MotionMaskCfg(), num_slots=6)
        mask = gen(batch_size=8, window=32, generator=torch.Generator().manual_seed(1))
        ratio = mask.float().mean().item()
        self.assertGreater(ratio, 0.0)
        self.assertLess(ratio, 1.0)

    def test_root_mostly_visible(self):
        # global_root (slot 0) is kept visible most of the time.
        cfg = MotionMaskCfg(keep_global_root_prob=1.0)
        gen = MotionMaskGenerator(cfg, num_slots=6)
        mask = gen(batch_size=64, window=16, generator=torch.Generator().manual_seed(2))
        self.assertEqual(mask[:, :, 0].any().item(), False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/test_motion_jepa.py::TestMaskGenerator -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'GBC.utils.data_preparation.motion_jepa'`.

- [ ] **Step 3: Create `motion_jepa.py` with the mask generator**

```python
"""Motion-JEPA: self-supervised feature-prediction pretraining for the MFME.

See docs/superpowers/plans/2026-05-28-jepa-gonogo-ablation-design.md (P2) and
2026-05-27-jepa-universal-motion-latent-plan.md (section 4).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MotionMaskCfg:
    temporal_block_lengths: Tuple[int, ...] = (8, 16, 32)
    num_temporal_blocks: int = 2
    slot_block_sizes: Tuple[int, ...] = (1, 2, 3)
    keep_global_root_prob: float = 0.75
    future_only_prob: float = 0.50
    mask_ratio_range: Tuple[float, float] = (0.25, 0.55)


class MotionMaskGenerator:
    """Produces ``Bool[B, T, S]`` slot-time masks (True = hidden from context)."""

    def __init__(self, cfg: MotionMaskCfg, num_slots: int, root_slot: int = 0):
        self.cfg = cfg
        self.num_slots = int(num_slots)
        self.root_slot = int(root_slot)

    def __call__(
        self, batch_size: int, window: int, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        cfg = self.cfg
        mask = torch.zeros(batch_size, window, self.num_slots, dtype=torch.bool)

        def randint(hi: int) -> int:
            return int(torch.randint(hi, (1,), generator=generator).item())

        def rand() -> float:
            return float(torch.rand((1,), generator=generator).item())

        for b in range(batch_size):
            future_only = rand() < cfg.future_only_prob
            for _ in range(cfg.num_temporal_blocks):
                blk = cfg.temporal_block_lengths[randint(len(cfg.temporal_block_lengths))]
                blk = min(blk, window)
                if future_only:
                    start = max(0, window - blk)
                else:
                    start = randint(max(1, window - blk + 1))
                sblk = cfg.slot_block_sizes[randint(len(cfg.slot_block_sizes))]
                s0 = randint(self.num_slots)
                slots = [(s0 + i) % self.num_slots for i in range(sblk)]
                for s in slots:
                    if s == self.root_slot and rand() < cfg.keep_global_root_prob:
                        continue
                    mask[b, start:start + blk, s] = True

            # Enforce mask-ratio band by random add/drop on this sample.
            lo, hi = cfg.mask_ratio_range
            total = window * self.num_slots
            target = lo + rand() * (hi - lo)
            cur = mask[b].float().mean().item()
            if cur > hi:
                idx = mask[b].nonzero(as_tuple=False)
                drop = int((cur - target) * total)
                perm = torch.randperm(idx.shape[0], generator=generator)[:max(0, drop)]
                for j in perm.tolist():
                    t_, s_ = idx[j].tolist()
                    mask[b, t_, s_] = False
        return mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_motion_jepa.py::TestMaskGenerator -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add GBC/utils/data_preparation/motion_jepa.py tests/test_motion_jepa.py
git commit -m "feat(jepa): add MotionMaskCfg + MotionMaskGenerator"
```

---

## Task 3: `MotionJEPAPredictor`

**Files:**
- Modify: `GBC/utils/data_preparation/motion_jepa.py`
- Test: `tests/test_motion_jepa.py` (append)

Design: predict refined features at **every** `(b,t,s)` position; the module gathers masked positions for the loss (avoids ragged batching). Inputs are the online encoder's `graph_hidden`; outputs are per-position graph + slot predictions.

- [ ] **Step 1: Write the failing test**

```python
from GBC.utils.data_preparation.motion_jepa import MotionJEPAPredictor


class TestPredictor(unittest.TestCase):
    def test_output_shapes(self):
        torch.manual_seed(0)
        pred = MotionJEPAPredictor(d_model=64, slot_latent_dim=16, num_slots=6,
                                   pred_d_model=32, num_layers=2, num_heads=4, max_window=64)
        ctx = torch.randn(2, 8, 6, 64)
        out_graph, out_slot = pred(ctx)
        self.assertEqual(out_graph.shape, (2, 8, 6, 64))
        self.assertEqual(out_slot.shape, (2, 8, 6, 16))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/test_motion_jepa.py::TestPredictor -v`
Expected: FAIL — `ImportError: cannot import name 'MotionJEPAPredictor'`.

- [ ] **Step 3: Implement the predictor** (append to `motion_jepa.py`)

```python
class MotionJEPAPredictor(nn.Module):
    """Refine context ``graph_hidden`` into target-space predictions per position."""

    def __init__(
        self,
        d_model: int,
        slot_latent_dim: int,
        num_slots: int,
        pred_d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_window: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_slots = int(num_slots)
        self.in_proj = nn.Linear(d_model, pred_d_model)
        self.time_embed = nn.Embedding(max_window, pred_d_model)
        self.slot_embed = nn.Embedding(num_slots, pred_d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=pred_d_model, nhead=num_heads, dim_feedforward=4 * pred_d_model,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.graph_head = nn.Sequential(nn.LayerNorm(pred_d_model), nn.Linear(pred_d_model, d_model))
        self.slot_head = nn.Sequential(nn.LayerNorm(pred_d_model), nn.Linear(pred_d_model, slot_latent_dim))

    def forward(self, context_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, s, _ = context_hidden.shape
        x = self.in_proj(context_hidden)  # (B, T, S, P)
        tid = torch.arange(t, device=x.device)
        sid = torch.arange(s, device=x.device)
        x = x + self.time_embed(tid).view(1, t, 1, -1) + self.slot_embed(sid).view(1, 1, s, -1)
        x = x.reshape(b, t * s, -1)
        x = self.encoder(x)
        x = x.reshape(b, t, s, -1)
        return self.graph_head(x), self.slot_head(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_motion_jepa.py::TestPredictor -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add GBC/utils/data_preparation/motion_jepa.py tests/test_motion_jepa.py
git commit -m "feat(jepa): add MotionJEPAPredictor"
```

---

## Task 4: `motion_jepa_loss` + `ema_update`

**Files:**
- Modify: `GBC/utils/data_preparation/motion_jepa.py`
- Test: `tests/test_motion_jepa.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from GBC.utils.data_preparation.motion_jepa import motion_jepa_loss, ema_update


class TestLossAndEMA(unittest.TestCase):
    def test_loss_zero_when_equal(self):
        x = torch.randn(10, 64)
        loss = motion_jepa_loss(x, x.clone())
        self.assertLess(loss.item(), 1e-6)

    def test_loss_positive_and_target_detached(self):
        pred = torch.randn(10, 64, requires_grad=True)
        target = torch.randn(10, 64, requires_grad=True)
        loss = motion_jepa_loss(pred, target)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNone(target.grad)  # target is detached inside the loss
        self.assertIsNotNone(pred.grad)

    def test_ema_moves_target_toward_online(self):
        online = torch.nn.Linear(4, 4)
        target = torch.nn.Linear(4, 4)
        with torch.no_grad():
            for p in target.parameters():
                p.zero_()
        before = target.weight.clone()
        ema_update(online, target, momentum=0.9)
        after = target.weight.clone()
        self.assertFalse(torch.allclose(before, after))
        # after = 0.9*0 + 0.1*online
        self.assertTrue(torch.allclose(after, 0.1 * online.weight, atol=1e-6))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/test_motion_jepa.py::TestLossAndEMA -v`
Expected: FAIL — `ImportError: cannot import name 'motion_jepa_loss'`.

- [ ] **Step 3: Implement loss + EMA** (append to `motion_jepa.py`)

```python
def motion_jepa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 between layer-normalized prediction and detached target."""
    d = pred.shape[-1]
    p = F.layer_norm(pred, (d,))
    y = F.layer_norm(target.detach(), (d,))
    return F.smooth_l1_loss(p, y)


@torch.no_grad()
def ema_update(online: nn.Module, target: nn.Module, momentum: float) -> None:
    for po, pt in zip(online.parameters(), target.parameters()):
        pt.data.mul_(momentum).add_(po.data, alpha=1.0 - momentum)
    for bo, bt in zip(online.buffers(), target.buffers()):
        bt.data.copy_(bo.data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_motion_jepa.py::TestLossAndEMA -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add GBC/utils/data_preparation/motion_jepa.py tests/test_motion_jepa.py
git commit -m "feat(jepa): add motion_jepa_loss + ema_update"
```

---

## Task 5: `MotionJEPAModule` (online + EMA target + predictor)

**Files:**
- Modify: `GBC/utils/data_preparation/motion_jepa.py`
- Test: `tests/test_motion_jepa.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from GBC.utils.data_preparation.morphology_conditioned_retargeter import MorphologyFreeMotionEncoder
from GBC.utils.data_preparation.motion_jepa import MotionJEPAModule, MotionMaskCfg


def _mfme_small():
    return MorphologyFreeMotionEncoder(d_model=64, z_dim=64, slot_latent_dim=16,
                                       graph_layers=1, temporal_layers=1, num_heads=4)


class TestJEPAModule(unittest.TestCase):
    def test_forward_returns_finite_loss(self):
        torch.manual_seed(0)
        mod = MotionJEPAModule(_mfme_small(), MotionMaskCfg(), num_slots=6,
                               pred_d_model=32, pred_layers=2, pred_heads=4, max_window=64)
        pose, root, beta = _synth(b=2, t=16)
        out = mod(pose, root, beta, generator=torch.Generator().manual_seed(0))
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertGreater(out["loss"].item(), 0.0)

    def test_target_encoder_has_no_grad(self):
        mod = MotionJEPAModule(_mfme_small(), MotionMaskCfg(), num_slots=6,
                               pred_d_model=32, pred_layers=2, pred_heads=4, max_window=64)
        self.assertFalse(any(p.requires_grad for p in mod.target_encoder.parameters()))
        pose, root, beta = _synth(b=2, t=16)
        out = mod(pose, root, beta, generator=torch.Generator().manual_seed(0))
        out["loss"].backward()
        self.assertTrue(all(p.grad is None for p in mod.target_encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in mod.online_encoder.parameters()))

    def test_ema_step_changes_target(self):
        mod = MotionJEPAModule(_mfme_small(), MotionMaskCfg(), num_slots=6,
                               pred_d_model=32, pred_layers=2, pred_heads=4, max_window=64)
        with torch.no_grad():
            for p in mod.online_encoder.parameters():
                p.add_(0.5)
        before = [p.clone() for p in mod.target_encoder.parameters()]
        mod.update_target(momentum=0.99)
        after = list(mod.target_encoder.parameters())
        self.assertTrue(any(not torch.allclose(a, b) for a, b in zip(after, before)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/test_motion_jepa.py::TestJEPAModule -v`
Expected: FAIL — `ImportError: cannot import name 'MotionJEPAModule'`.

- [ ] **Step 3: Implement the module** (append to `motion_jepa.py`)

```python
class MotionJEPAModule(nn.Module):
    """Online MFME + EMA target MFME + predictor, trained with feature prediction."""

    def __init__(
        self,
        encoder: nn.Module,           # a MorphologyFreeMotionEncoder (online)
        mask_cfg: MotionMaskCfg,
        num_slots: int,
        slot_loss_weight: float = 0.25,
        pred_d_model: int = 256,
        pred_layers: int = 2,
        pred_heads: int = 4,
        max_window: int = 256,
    ):
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = MotionJEPAPredictor(
            d_model=encoder.d_model, slot_latent_dim=encoder.slot_projector[-1].out_features,
            num_slots=num_slots, pred_d_model=pred_d_model, num_layers=pred_layers,
            num_heads=pred_heads, max_window=max_window,
        )
        self.mask_gen = MotionMaskGenerator(mask_cfg, num_slots=num_slots)
        self.slot_loss_weight = float(slot_loss_weight)

    def forward(
        self,
        pose_aa: torch.Tensor,
        root_pose: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        b, t = pose_aa.shape[0], pose_aa.shape[1]
        mask = self.mask_gen(b, t, generator=generator).to(pose_aa.device)

        with torch.no_grad():
            tgt = self.target_encoder(pose_aa, root_pose, beta, sample=False)
        ctx = self.online_encoder(pose_aa, root_pose, beta, sample=False, slot_time_mask=mask)

        pred_graph, pred_slot = self.predictor(ctx["graph_hidden"])
        m = mask  # (B, T, S)
        if m.any():
            lg = motion_jepa_loss(pred_graph[m], tgt["graph_hidden"][m])
            ls = motion_jepa_loss(pred_slot[m], tgt["slot_latents"][m])
        else:
            lg = pred_graph.sum() * 0.0
            ls = pred_slot.sum() * 0.0
        loss = lg + self.slot_loss_weight * ls

        with torch.no_grad():
            tgt_std = tgt["graph_hidden"].std().detach()
            pred_std = pred_graph.std().detach()
            z_std = ctx["z"].std().detach()
        return {
            "loss": loss, "loss_graph": lg.detach(), "loss_slot": ls.detach(),
            "target_std": tgt_std, "pred_std": pred_std, "z_std": z_std,
            "mask_ratio": m.float().mean().detach(),
        }

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        ema_update(self.online_encoder, self.target_encoder, momentum)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_motion_jepa.py::TestJEPAModule -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full module test file**

Run: `PYTHONPATH=. python -m pytest tests/test_motion_jepa.py -v`
Expected: PASS (all classes).

- [ ] **Step 6: Commit**

```bash
git add GBC/utils/data_preparation/motion_jepa.py tests/test_motion_jepa.py
git commit -m "feat(jepa): add MotionJEPAModule with EMA target + feature-prediction loss"
```

---

## Task 6: Retargeter integration test (checkpoint reuse + frozen gradients)

**Files:**
- Test: `tests/test_jepa_retarget_integration.py` (create)

This proves the JEPA-trained MFME drops into `MotionLatentRetargeter.mfme`, the default forward schema is unchanged, and freezing the MFME still lets CMAD train.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import unittest

import torch

from GBC.utils.data_preparation.morphology_conditioned_retargeter import (
    MorphologyFreeMotionEncoder,
    MotionLatentRetargeter,
    build_synthetic_robot_spec,
    batch_robot_specs,
    masked_action_l2,
)
from GBC.utils.data_preparation.motion_jepa import MotionJEPAModule, MotionMaskCfg


def _synth(b=2, t=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, t, 21, 3, generator=g),
            torch.randn(b, t, 9, generator=g),
            torch.randn(b, 16, generator=g))


def _spec(batch):
    names = [f"j{i}" for i in range(6)]
    parents = [-1, 0, 1, 1, 0, 0]
    spec = build_synthetic_robot_spec("toy", names, parents, seed=0)
    sb = batch_robot_specs([spec] * batch)
    return sb


class TestJEPAIntegration(unittest.TestCase):
    def test_load_jepa_mfme_into_retargeter(self):
        torch.manual_seed(0)
        mfme = MorphologyFreeMotionEncoder(d_model=64, z_dim=64, slot_latent_dim=16,
                                           graph_layers=1, temporal_layers=1, num_heads=4)
        jepa = MotionJEPAModule(mfme, MotionMaskCfg(), num_slots=6,
                                pred_d_model=32, pred_layers=1, pred_heads=4, max_window=32)
        pose, root, beta = _synth()
        jepa(pose, root, beta, generator=torch.Generator().manual_seed(0))["loss"].backward()

        rt = MotionLatentRetargeter(d_model=64, z_dim=64, slot_latent_dim=16,
                                    encoder_graph_layers=1, encoder_temporal_layers=1,
                                    encoder_num_heads=4, morphology_num_heads=4,
                                    decoder_num_heads=4)
        missing, unexpected = rt.mfme.load_state_dict(jepa.online_encoder.state_dict(), strict=False)
        self.assertEqual(list(unexpected), [])  # mask token is extra-tolerant via strict=False

    def test_default_schema_unchanged(self):
        rt = MotionLatentRetargeter(d_model=64, z_dim=64, slot_latent_dim=16,
                                    encoder_graph_layers=1, encoder_temporal_layers=1,
                                    encoder_num_heads=4, morphology_num_heads=4,
                                    decoder_num_heads=4)
        pose, root, beta = _synth()
        out = rt(pose_aa=pose, root_pose=root, robot_batch=_spec(2), beta=beta, sample=False)
        for k in ("mu", "logvar", "z", "slot_latents", "graph_hidden", "part_mask",
                  "m_per_dof", "m_cls", "actions"):
            self.assertIn(k, out)

    def test_frozen_mfme_still_trains_cmad(self):
        rt = MotionLatentRetargeter(d_model=64, z_dim=64, slot_latent_dim=16,
                                    encoder_graph_layers=1, encoder_temporal_layers=1,
                                    encoder_num_heads=4, morphology_num_heads=4,
                                    decoder_num_heads=4)
        for p in rt.mfme.parameters():
            p.requires_grad_(False)
        pose, root, beta = _synth()
        sb = _spec(2)
        out = rt(pose_aa=pose, root_pose=root, robot_batch=sb, beta=beta, sample=False)
        target = torch.zeros_like(out["actions"])
        loss = masked_action_l2(out["actions"], target, sb.get("dof_padding_mask"))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in rt.mfme.parameters()))
        self.assertTrue(any(p.grad is not None for p in rt.cmad.parameters()))
```

- [ ] **Step 2: Run test to verify it fails (then passes)**

Run: `PYTHONPATH=. python -m pytest tests/test_jepa_retarget_integration.py -v`
Expected: This test exercises only already-implemented code (Tasks 1–5 + committed foundation), so it should PASS once those land. If `unexpected` keys appear from the mask token, that is the signal the load path needs `strict=False` (already used). Investigate any failure per systematic-debugging before changing assertions.

- [ ] **Step 3: Commit**

```bash
git add tests/test_jepa_retarget_integration.py
git commit -m "test(jepa): integration test for MFME checkpoint reuse + frozen-MFME CMAD gradients"
```

---

## Task 7: Synthetic Phase-0 smoke gate

**Files:**
- Create: `scripts/data_preparation/pretrain_motion_jepa_smoke.py`
- Test: `tests/test_motion_jepa.py` (append a fast smoke-run test)

Implements design §7 gate 1: train JEPA on synthetic motion, assert non-collapse (`target_std`, `z_std` stay non-trivial), and show a frozen future-slot probe beats an untrained encoder. No GMR / real data. Emits `outputs/jepa_motion_smoke.json`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_motion_jepa.py`)

```python
class TestSmokeGate(unittest.TestCase):
    def test_short_run_non_collapse(self):
        from scripts.data_preparation.pretrain_motion_jepa_smoke import run_smoke
        report = run_smoke(steps=30, batch_size=4, window=16, d_model=64, z_dim=64,
                           device="cpu", seed=0)
        self.assertGreater(report["target_std_final"], 1e-3)
        self.assertGreater(report["z_std_final"], 1e-3)
        self.assertLess(report["loss_final"], report["loss_init"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/test_motion_jepa.py::TestSmokeGate -v`
Expected: FAIL — `ModuleNotFoundError: scripts.data_preparation.pretrain_motion_jepa_smoke`.

- [ ] **Step 3: Implement the smoke script**

Create `scripts/data_preparation/pretrain_motion_jepa_smoke.py`:

```python
"""Synthetic Phase-0 smoke gate for Motion-JEPA (no real data).

Trains the JEPA module on random synthetic motion and checks non-collapse +
a frozen future-slot probe vs an untrained encoder. Structural sanity only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from GBC.utils.data_preparation.morphology_conditioned_retargeter import (
    MorphologyFreeMotionEncoder,
)
from GBC.utils.data_preparation.motion_jepa import MotionJEPAModule, MotionMaskCfg


def _synth(b, t, device, generator):
    pose = torch.randn(b, t, 21, 3, device=device, generator=generator)
    root = torch.randn(b, t, 9, device=device, generator=generator)
    beta = torch.randn(b, 16, device=device, generator=generator)
    return pose, root, beta


def _probe_error(encoder: nn.Module, device: str, generator: torch.Generator) -> float:
    """Frozen-encoder linear probe: predict next-frame z-pool from current frame."""
    encoder.eval()
    probe = nn.Linear(encoder.z_dim, encoder.z_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    last = 0.0
    for _ in range(50):
        pose, root, beta = _synth(8, 16, device, generator)
        with torch.no_grad():
            z = encoder(pose, root, beta, sample=False)["z"]  # (B, T, z)
        pred = probe(z[:, :-1])
        loss = ((pred - z[:, 1:].detach()) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        last = loss.item()
    return last


def run_smoke(steps=300, batch_size=16, window=32, d_model=256, z_dim=256,
              device="cpu", seed=0) -> Dict[str, float]:
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    mfme = MorphologyFreeMotionEncoder(d_model=d_model, z_dim=z_dim, slot_latent_dim=32,
                                       graph_layers=2, temporal_layers=2, num_heads=8).to(device)
    mod = MotionJEPAModule(mfme, MotionMaskCfg(), num_slots=6,
                           pred_d_model=128, pred_layers=2, pred_heads=4,
                           max_window=window).to(device)
    opt = torch.optim.Adam([p for p in mod.parameters() if p.requires_grad], lr=2e-4)

    # Untrained-encoder probe baseline (before training).
    untrained_probe = _probe_error(MorphologyFreeMotionEncoder(
        d_model=d_model, z_dim=z_dim, slot_latent_dim=32,
        graph_layers=2, temporal_layers=2, num_heads=8).to(device),
        device, torch.Generator(device=device).manual_seed(seed + 1))

    losses, tgt_std, z_std = [], 0.0, 0.0
    for step in range(steps):
        pose, root, beta = _synth(batch_size, window, device, gen)
        out = mod(pose, root, beta, generator=gen)
        opt.zero_grad(); out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        opt.step()
        mod.update_target(momentum=0.996)
        losses.append(out["loss"].item())
        tgt_std, z_std = out["target_std"].item(), out["z_std"].item()

    trained_probe = _probe_error(mod.online_encoder, device,
                                 torch.Generator(device=device).manual_seed(seed + 1))
    report = {
        "loss_init": float(sum(losses[:10]) / max(1, len(losses[:10]))),
        "loss_final": float(sum(losses[-10:]) / max(1, len(losses[-10:]))),
        "target_std_final": float(tgt_std),
        "z_std_final": float(z_std),
        "probe_untrained": float(untrained_probe),
        "probe_jepa": float(trained_probe),
        "probe_improves": bool(trained_probe < untrained_probe),
        "steps": steps,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="outputs/jepa_motion_smoke.json")
    args = ap.parse_args()
    report = run_smoke(steps=args.steps, device=args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert report["target_std_final"] > 1e-3, "COLLAPSE: target std ~ 0"
    assert report["z_std_final"] > 1e-3, "COLLAPSE: z std ~ 0"
    print("SMOKE GATE PASSED" if report["loss_final"] < report["loss_init"] else "SMOKE GATE: loss did not fall")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_motion_jepa.py::TestSmokeGate -v`
Expected: PASS (a 30-step CPU run is fast; asserts non-collapse + falling loss).

- [ ] **Step 5: Run the real smoke gate on GPU**

Run: `PYTHONPATH=. python scripts/data_preparation/pretrain_motion_jepa_smoke.py --steps 300`
Expected: prints JSON + `SMOKE GATE PASSED`; writes `outputs/jepa_motion_smoke.json`. If collapse asserts fire, follow design §11 (increase batch diversity, add variance hinge, lower EMA start) before proceeding — do **not** weaken the asserts.

- [ ] **Step 6: Commit**

```bash
git add scripts/data_preparation/pretrain_motion_jepa_smoke.py tests/test_motion_jepa.py
git commit -m "feat(jepa): synthetic Phase-0 smoke gate (non-collapse + frozen-probe)"
```

---

## Final verification

- [ ] Run the full new test suite:

Run: `PYTHONPATH=. python -m pytest tests/test_motion_jepa.py tests/test_jepa_retarget_integration.py -v`
Expected: all PASS.

- [ ] Confirm no regression in the foundation tests:

Run: `PYTHONPATH=. python -m pytest tests/test_morphology_conditioned_retargeter.py -q`
Expected: PASS.

- [ ] The Phase-0 gate JSON exists and passed: `outputs/jepa_motion_smoke.json` with `target_std_final > 1e-3` and `loss_final < loss_init`.

**Exit state:** a tested, importable Motion-JEPA module and a passing synthetic non-collapse gate. Plan 2 (real-data SFT trainer + P4/P5 input contracts) consumes `MotionJEPAModule.online_encoder` checkpoints via `MotionLatentRetargeter.mfme.load_state_dict(...)`.

---

## Self-Review

**Spec coverage (against `2026-05-28-jepa-gonogo-ablation-design.md`):**
- P2 Motion-JEPA module (`slot_time_mask`, mask gen, predictor, EMA, loss) → Tasks 1–5. ✓
- §7 gate 1 (synthetic Phase-0 smoke, non-collapse + frozen probe) → Task 7. ✓
- §11.1-style tests (mask shapes; `slot_time_mask=None` identity; finite loss; target no-grad; EMA changes target; default schema unchanged; frozen-MFME CMAD gradients) → Tasks 1,5,6. ✓
- Deferred by design to Plans 2–3: P1 SFT trainer, P3 MAE head, P4 canonicalization, P5 FPS, the `scratch`/`mae`/`jepa` matrix. (Out of scope here — noted in header.) ✓

**Placeholder scan:** no TBD/TODO; every code step has complete, runnable code and exact commands. ✓

**Type/name consistency:** `MotionMaskCfg`, `MotionMaskGenerator(cfg, num_slots)`, `MotionJEPAPredictor(d_model, slot_latent_dim, num_slots, pred_d_model, num_layers, num_heads, max_window)`, `MotionJEPAModule(encoder, mask_cfg, num_slots, ...)` with `.online_encoder`/`.target_encoder`/`.update_target(momentum)`, `motion_jepa_loss(pred, target)`, `ema_update(online, target, momentum)`, `run_smoke(...)` — names match across all tasks. The module reads `slot_latent_dim` via `encoder.slot_projector[-1].out_features` (matches MFME `:394-397`) and `encoder.d_model`/`encoder.z_dim` (set in MFME `:354-355`). ✓
