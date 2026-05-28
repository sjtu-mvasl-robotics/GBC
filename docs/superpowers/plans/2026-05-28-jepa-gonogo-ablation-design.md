# JEPA Go/No-Go Ablation — Design

> **Companion to** `2026-05-27-jepa-universal-motion-latent-plan.md`. That plan defines the full JEPA path (Phases 0–5, models B0/B1/J1–J4). **This document completes and operationalizes its "Recommended First Milestone" (§13) as a decision-grade ablation.** It does not supersede the plan; it narrows it to a single yes/no question and pins everything else.

**Date:** 2026-05-28
**Target model:** `MotionLatentRetargeter` (`GBC/GBC/utils/data_preparation/morphology_conditioned_retargeter.py:693`) — *not* the production `UnifiedLatentRetargeter`.
**Robot:** Unitree G1 (29 DoF), single-robot only.

---

## 1. The decision

Keep the JEPA pretraining path **iff** a JEPA-pretrained `MorphologyFreeMotionEncoder` (MFME), frozen during supervised CMAD training, satisfies **both**:

1. **Go/no-go vs current practice** — within **10%** (ideally 5%) of a from-scratch `MotionLatentRetargeter` on the G1 robot's action-L2-to-GMR at **equal wall-clock budget**, with no regression in walking-render coherence or jumping-jacks lower-body amplitude.
2. **Objective-family win** — beats an MAE-style **input-space** reconstruction pretraining baseline on dynamic / held-out motion at the same budget.

If (1) passes but (2) fails, JEPA is no better than the simpler reconstruction objective and we keep recon (consistent with the standing VAE+L2-reconstruction design choice). If (1) fails while Level-1 probes pass (§7), the decoder/curriculum is wrong, not the representation.

**One axis is varied — *pretraining objective*. Everything else is pinned (§5).** Mixing in other knobs (target space, mask policy, EMA schedule, freeze schedule) is explicitly second-pass and would confound the go/no-go signal.

---

## 2. Repository status (code-anchored)

What already exists and what must be built before any row runs. Read the code before implementing from this table.

| Component | Status | Reference |
|---|---|---|
| `MotionLatentRetargeter` (MFME + MGE + CMAD + FutureSlotHead) | ✅ landed | `morphology_conditioned_retargeter.py:693` (`.mfme:744`, `.mge:759`, `.cmad:768`, `.future_head:778`) |
| MFME forward → `mu/logvar/z/slot_latents/graph_hidden/part_mask` | ✅ landed | `:408`; shapes in docstring `:325-330` (`graph_hidden (B,T,6,d_model)`, `slot_latents (B,T,6,slot_latent_dim)`) |
| 6 canonical slots incl. `global_root` | ✅ landed | `graph_retarget_latent.py:14` `DEFAULT_BODY_SLOTS = (global_root, trunk_head, left_arm, right_arm, left_leg, right_leg)` |
| `FutureSlotHead` (future-slot probe target) | ✅ landed | imported `morphology_conditioned_retargeter.py:39`, used `:825`; defined in `graph_retarget_latent.py` |
| Loss helpers `masked_action_l2`, `kl_divergence_loss`, `latent_smoothness_loss` | ✅ landed | `morphology_conditioned_retargeter.py:842 / :861 / :875` |
| Real-G1 robot spec builder `build_robot_spec_from_kinematics` | ✅ landed | `:149` (uses `RobotKinematics.get_dof_features` `base/base_fk.py:409`, `URDF_FEATURE_DIM=21` `:49`) |
| Spec batching `batch_robot_specs` | ✅ landed | `:237` |
| GMR G1 reference-action dataset | ✅ exists | `/cpfs/user/yaoyifei/datasets/amass_gmr_convert/reference_actions` |
| Reference-action loss plumbing (reuse source) | ✅ landed (old model) | MUJICA `train_graph_latent.py`; `_load_reference_actions`, `optional_reference_action_loss` (`retarget_loss_utils.py:144-155`) |
| Yaw primitives for canonicalization | ✅ landed | `get_yaw_only_rot_mat` `amass_action_converter.py:168`; `quat_apply_yaw` `math_utils.py:1874` |
| FPS resample primitive | ✅ landed | `amass_loader.py` `interpolate_fps` (`:68/72`), `interpolate_trans` (`:136`) |
| **Real-data CMAD SFT trainer for `MotionLatentRetargeter`** | ❌ **NOT landed** | only synthetic probes exist: `scripts/data_preparation/run_morph_latent_experiments.py`, `run_morph_ablation_multiseed.py` (both `TargetOracle`-synthetic) |
| **Motion-JEPA module** (`slot_time_mask`, EMA target, predictor, loss) | ❌ NOT landed | plan §4, Phase 1 |
| **MAE-recon head** on MFME (input-space pose reconstruction) | ❌ NOT landed | new model has no `human_decoder`; old `UnifiedLatentRetargeter` does |
| **Root yaw-canonicalization** wired upstream of `MFME.forward` | ❌ NOT landed | recommended in `MORPHOLOGY_CONDITIONED_LATENT_PROPOSAL.md:46` |

**The new `MotionLatentRetargeter` has no real-data training path and no real G1 checkpoints.** All existing real-G1 evidence (`outputs/render/CHECKPOINT_VALIDATION_REPORT.md`) is the *old* `UnifiedLatentRetargeter`. Building P1 below is the dominant prerequisite.

---

## 3. Prerequisites — halt if not landed

These must land and be verified before the matrix in §6 begins. If P1 cannot be built on real GMR G1 data, **halt and re-scope to the synthetic smoke gate (§7 Phase-0) only**; do not report a go/no-go from synthetic data.

### P1 — Real-data CMAD SFT trainer (dominant gate)
`scripts/data_preparation/train_morph_latent_sft.py`:
- Load AMASS `pose_body`/`root_pose`/`beta` + GMR G1 reference actions (reuse MUJICA `_load_reference_actions`).
- Build the G1 `RobotSpec` once via `build_robot_spec_from_kinematics(g1_kinematics)`, replicate to batch with `batch_robot_specs` (`:237`).
- Forward `MotionLatentRetargeter(... robot_batch=spec_batch ...)` (`:805`); loss = `1000 * masked_action_l2(actions, ref, dof_padding_mask)` `+ fk_link_loss` `+ 1e-3 * latent_smoothness_loss(z)`. FK-only for the first pass; physlite reserved.
- Flags: `--mfme-ckpt` (load pretrained MFME state into `.mfme`), `--freeze-mfme` / `--freeze-mfme-steps`, `--pretrain-objective {none,mae,jepa}` (bookkeeping only).
- Numeric-only W&B scalars (§8).

### P2 — Motion-JEPA module
`GBC/utils/data_preparation/motion_jepa.py` per plan §4: `MotionMaskCfg`, `MotionMaskGenerator` (slot-time `Bool[B,T,S]`), `MotionJEPAPredictor`, `MotionJEPAModule` (online + EMA target + predictor), `motion_jepa_loss`. Add `slot_time_mask: Optional[Tensor]=None` to `MorphologyFreeMotionEncoder` with **no change to default behavior when `None`**. Target = detached `graph_hidden` (+ `slot_latents` aux), normalized smooth-L1.

### P3 — MAE-recon head (the `mae` baseline arm)
A small head on MFME `graph_hidden`/`z` → `(num_human_joints*3 + root_dim)`. Objective: mask 15% of slot-time tokens (**reuse P2's `slot_time_mask`**), L2-reconstruct the masked joints in **input space**. This is the only intentional difference from the `jepa` row — same masking, recon target instead of feature target — so the contrast is clean.

### P4 — Root yaw-canonicalization (fixed default, all rows)
`_canonicalize_motion(pose_aa, root_pose)` upstream of `MFME.forward`: remove world yaw + xy-translation from `root_pose` (via `get_yaw_only_rot_mat`/`quat_apply_yaw`), keep body-local pose + a canonicalized root trajectory stream. **Unit test:** random yaw rotation of the input leaves `z` unchanged within tolerance (yaw-invariance).

### P5 — FPS resample to canonical 30 Hz (fixed default, all rows)
Set the AMASS loader to resample every window to 30 Hz (`interpolate_fps`/`interpolate_trans`). Applied identically to pretraining and SFT data.

---

## 4. Why P4/P5 are prerequisites, not axes

Per the temporal-ablations precedent ("inputs must be held constant across rows or the gain is confounded"): if root yaw leaks into `z`, the JEPA future-prediction task is partially solvable by tracking global drift, inflating the `jepa` row; if clips arrive at mixed frame rates, the temporal masks `(8,16,32)` cover different physical horizons per clip. Both would confound the objective contrast. They are pinned ON for **every** row, not tested here.

---

## 5. Pinned configuration (confound control)

Held identical across `scratch`/`mae`/`jepa`:

- **Data:** pretrain = AMASS only (no robot); SFT = GMR G1 (`/cpfs/.../reference_actions`).
- **Inputs:** root yaw-canonicalized (P4) ON; resampled to 30 Hz (P5).
- **SFT config template** (from MUJICA `train_graph_latent.py` mainline, the old model's real-data config): `window=120`, `stride=20`, `batch-size` to fit, `lr=2e-4`, AMP, FK-only loss, `latent-reference-action-weight=1000`, **no teacher / no teacher-init / no teacher-eval**.
- **Model size:** identical `d_model`/`z_dim` (256), identical layer counts, identical `future_offsets`.
- **Budget:** equal wall-clock for the SFT stage of every row.
- **Seeds:** ≥3 seeds per row, report mean ± stdev (per `run_morph_ablation_multiseed.py` convention).

---

## 6. First-pass matrix

`MotionLatentRetargeter`, real GMR G1 robot data, FK-only loss, config §5. Rows are labeled by pretraining objective to avoid collision with the G1 robot name.

| Row | MFME pretrain (AMASS only) | MFME during SFT | Isolates |
|---|---|---|---|
| **`scratch`** (control) | none — random init | trainable (joint) | from-scratch baseline = plan's B1 |
| **`mae`** | MAE masked-pose **input-space** recon (P3) | frozen | reconstruction-family pretraining |
| **`jepa`** | Motion-JEPA **feature-space** prediction (P2) | frozen | JEPA-family pretraining |

**Contrasts:**
- **`jepa` vs `mae`** — objective family, with freeze held constant.
- **`jepa` vs `scratch`** — go/no-go vs current practice.

**Reserved second-pass rows (not first pass):** `jepa-ft` = JEPA pretrain → joint-finetune with low MFME LR + JEPA warm loss (plan Phase 4); `scratch-frozen` = frozen random-init MFME (degenerate control proving "frozen" alone is not the source of any gain). Run only for the top row of the first pass.

---

## 7. Nested gates (cheap → expensive, mirrors plan §10 Levels 1–3)

Run in order; a failed gate stops the spend before the next.

1. **Phase-0 smoke (synthetic).** Reuse the `TargetOracle` harness (`run_morph_latent_experiments.py` style). Pretrain none/MAE/JEPA on synthetic motion; gate = JEPA non-collapsed (target/pred std non-trivial) **and** frozen-MFME probe > random. No GMR dependency. Catches broken masking/EMA early.
2. **Level-1 probe (real AMASS, no SFT).** Freeze the pretrained MFME; train only future-slot (`FutureSlotHead`) and contact probes. Gate = JEPA **and** MAE beat untrained MFME. If JEPA fails here, fix mask/target before touching CMAD.
3. **Level-2 matrix.** The `scratch`/`mae`/`jepa` runs of §6. This is the decision.

---

## 8. Metrics (numeric-only)

W&B `log_wandb_scalars` silently drops non-numeric values; all run labels (objective, robot, freeze) go in `run_config.json` / W&B `config`, never as logged scalars.

**Retargeting (per row):** `retarget/action_l2` (to GMR, G1 robot val split), `retarget/fk_mean`, `retarget/fk_max`, `retarget/foot_slide`, `retarget/ground_pen`, `retarget/lower_body_amp_ratio`.
**JEPA-side (`jepa` row; MAE recon loss for `mae` row):** `loss/motion_jepa`, `stats/target_std`, `stats/pred_std`, `stats/z_std`, `stats/cosine_pred_target`, `probe/future_slot`, `probe/contact_f1`.
**Cross-row deltas:** `delta/jepa_minus_scratch`, `delta/jepa_minus_mae` (per primary metric), with seed stdev.

---

## 9. Decision rule & outcome interpretation

Primary metric: `retarget/action_l2` on the G1 robot **val** split (held-out motions).

| Outcome | Reading | Action |
|---|---|---|
| `jepa` ≤ `scratch`+10% **and** `jepa` < `mae` on dynamic/held-out | JEPA transfers and beats recon | **Keep JEPA.** Proceed to `jepa-ft` (joint-finetune), then plan Phases 4–5. |
| `jepa` ≤ `scratch`+10% but `jepa` ≈ `mae` | JEPA no better than recon | Keep the simpler recon objective; **park JEPA.** |
| Level-1 probes pass but `jepa` ≫ `scratch` (regresses) | representation fine, decoder/curriculum wrong | freeze MFME longer / lower joint LR / warm up CMAD; re-run. |
| Level-1 probes fail | mask policy or target choice wrong | fix P2 (larger temporal masks, keep root visible, shorter horizon) before any SFT. |

---

## 10. Implementation checklist

**Prerequisites**
- [ ] P4 `_canonicalize_motion` + yaw-invariance unit test
- [ ] P5 30 Hz resample wired into the loader for both stages
- [ ] P1 `train_morph_latent_sft.py` (real GMR G1, from-scratch path first) — verify it trains the `scratch` row to sane action-L2 before adding pretraining
- [ ] P2 `motion_jepa.py` + `slot_time_mask` on MFME (default-behavior-preserving)
- [ ] P3 MAE-recon head + masked-pose objective (shares P2 masking)

**Tests** (per plan §11.1)
- [ ] `tests/test_motion_jepa.py` — mask shapes `(B,T,S)`; `slot_time_mask=None` ⇒ identical MFME output; finite loss; target encoder gets no gradient; EMA changes target weights
- [ ] `tests/test_jepa_retarget_integration.py` — load MFME-JEPA ckpt into `MotionLatentRetargeter.mfme`; frozen MFME still passes CMAD gradients; default `forward` output schema unchanged
- [ ] `tests/test_root_canonicalization.py` — yaw-invariance of `z`

**Runs**
- [ ] Phase-0 smoke (synthetic, `scratch`/`mae`/`jepa`)
- [ ] Level-1 probes (AMASS frozen MFME)
- [ ] Level-2 matrix `scratch`/`mae`/`jepa` ×3 seeds, FK-only, equal budget
- [ ] Report table + deltas → `outputs/jepa_gonogo_ablation.json`

---

## 11. Risks & mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| P1 build slips; tempted to decide on synthetic | go/no-go claimed without real G1 robot data | Hard rule §3: no go/no-go from synthetic; halt and re-scope to smoke gate. |
| Row budgets drift | unequal wall-clock confounds result | fix a single wall-clock cap; log `elapsed_sec`; reject rows that overrun. |
| Frozen vs joint confound (`scratch` trainable, `mae`/`jepa` frozen) | "JEPA wins" could be a freeze artifact | the clean objective contrast is **`jepa` vs `mae`** (both frozen); `scratch-frozen` reserved row isolates freeze. |
| Root leakage despite P4 | yaw-rotated clip gives different `z` | yaw-invariance test gates P4; block matrix until it passes. |
| EMA / mask collapse | feature std → 0, flat JEPA loss | Phase-0 smoke catches it; variance hinge only if needed (plan §5.2). |

---

## 12. Scope

**In scope:** single-robot G1; per-frame latent; the pretraining-objective axis only.
**Out of scope (deferred to later docs):** MGE / Graph-JEPA and multi-robot (plan §6, §8); FSQ (plan Phase 5); the chunk-based / FPS-invariant latent track — P5's 30 Hz resampling sidesteps FPS-invariance here rather than solving it, and reconciling per-frame JEPA masking with chunk latents is a separate question.
