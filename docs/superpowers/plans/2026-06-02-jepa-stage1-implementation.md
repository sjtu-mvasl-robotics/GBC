# Stage 1 — Motion-JEPA Latent Learner (Module A), grounded in the D4 baseline

**Date:** 2026-06-02
**Implements:** Module A of `2026-06-01-motion-jepa-latent-design.md` (the JEPA latent learner only — no decoder, no generation).
**Supersedes:** `2026-05-28-jepa-plan1-motion-jepa-module.md` (the synthetic-only Plan 1). That plan's slot-level `slot_time_mask` + full-grid refiner are replaced here by **per-joint** masking + a spatial completion predictor, and the encoder is now config-matched to the real D4 production baseline rather than a toy.

**Baseline of record:** `gbc_runs/_d4_config/` (`D4_rot_plane`, `phaseA_20260601_b_corrected_dlc`). D4 is the current best **`UnifiedLatentRetargeter`** (structured) trained by MUJICA `train_graph_latent.py`; `model_config_version = unified_latent_robot_adapter_v2_temporal_transformer`; selection `latent_mean_l2_m=0.0823`.

---

## 1. What "base on D4" means here

Stage-1 JEPA reuses D4's **encoder building blocks, config, and data pipeline**, and adds a self-supervised objective on top. It does **not** reuse D4's decoder/teacher/retargeting losses (those are Module B2). Concretely inherited from `gbc_runs/_d4_config/run_config.json`:

| Axis | D4 value (inherited) | Source |
|---|---|---|
| tokenizer | `structured` (`SkeletalJointTokenizer`) | `args.latent_architecture` |
| `d_model` | 256 | `args.d_model` |
| `graph_layers` | **2** | `args.graph_layers` |
| `temporal_layers` | 4 | `args.temporal_layers` |
| `num_heads` | 8 | `args.num_heads` |
| `temporal_module` | `transformer` (`is_causal=False`) | `start.arch.temporal_module` |
| `slot_latent_dim` | 32 | `args.slot_latent_dim` |
| `slot_z` → derived z | 48 → **336** per frame | `start.slot_z`, `start.z_dim_derived` |
| `root_feature_mode` | `scaled_abs_z` | `args.root_feature_mode` |
| window / stride / fps | 60 / 30 / 20 | `args.window/stride/fps` |
| data | AMASS via `AMASSSequence10Dataset`, 13011 train / 1446 val files | `start.train_files/val_files` |
| robot cfg | `unitree_g1.yaml` | `args.cfg` |

The 7 structured slots (from `dof_slot_correspondence` + root): `global_root, trunk_head, left_arm, right_arm, left_leg, right_leg` plus the independent `root_state` slot — `z_dim_derived = slot_z(48) × 7 = 336`.

---

## 2. Decisions resolved by D4 (these close design-doc §9)

- **§9.2 `graph_layers`: RESOLVED → 2.** D4 uses `graph_layers=2`, consistent with `locality-grid-results` ("keep glayers2; glayers=1 regressed +2.2%"). The Plan-1 addendum's `graph_layers=1` is **rejected**; do not set it.
- **Temporal module → `transformer`** (match D4). `causal_conv` (the locality-grid FPS winner) is a noted later option for FPS-invariance/causality but is **not** Stage-1 scope.
- **Latent of record = the structured 336-dim per-frame z** (`slot_z=48 × 7`), not MFME's pooled 256-dim z. Stage-1 wraps **D4's structured encoder**, so the JEPA-shaped latent IS the production latent.

---

## 3. Three forks D4 surfaces — RECOMMENDATIONS, confirm at review

These are genuine choices where D4 and the design constraints pull differently. I've taken a position on each; flag any to change.

### Fork A — Encoder inputs: add velocity + contact, or match D4's `none`?
- D4: `encoder_geometry_input='none'`, `encoder_foot_contact_input=False`.
- `project-unified-latent-constraints` (point 5): the latent must be **physics-extendable**, so velocity + contact channels "must be included at training time, because re-fitting the encoder later inside RL is far more expensive." JEPA pretraining is exactly that "now."
- The wiring already landed (`compute_encoder_body_joints` `encoder_geometry.py:34`, `finite_diff_velocity` `perframe_features.py:58`, `build_encoder_inputs` in `train_graph_latent.py`; tokenizer ctor accepts `joint_geometry_dim/joint_velocity_dim/joint_contact_dim` `graph_retarget_latent.py:290-292`).
- **Recommendation: ADD `joint_pos` + `joint_vel` + `joint_contact`** to the JEPA encoder (`--encoder-geometry-input fk_pos_vel --encoder-foot-contact-input`). The JEPA latent should be physics-ready from the start; this is the cheapest moment to fix inputs. Divergence from D4 is intentional and constraint-driven.

### Fork B — Per-joint-temporal encoder (needed for V-JEPA-faithful joint masking)
- D4's encoder **pools joints → 6 slots, then runs the temporal module over slots** (`SkeletalJointTokenizer` pools at `graph_retarget_latent.py:441-457`; `GraphTemporalEncoder` runs over slots).
- V-JEPA's "mask joints, **full temporal info visible**" requires that, when completing a masked joint, the predictor can use the visible joints' representations **across all frames** — i.e., per-joint representations must carry temporal context. With pool-then-temporal, per-joint hidden is **pre-temporal** (single frame), which violates "full temporal info visible."
- **Recommendation: run the temporal module per-joint, before pooling** — a `JointTemporalEncoder` producing per-joint temporally-contextualized hidden `h[B,T,J,d]`; pool → slot z (336) happens **after**, downstream. JEPA then trains the full spatial+temporal encoder and honors V-JEPA's masking. This is the §9.1 decision, taken as option (i).
- *Cost:* temporal runs over 22 joint-tokens×T instead of 6 slots×T (heavier; window=60 keeps it tractable). *Divergence from D4* is required by per-joint granularity + the random-vs-slot ablation the user asked for. The **fallback** (slot-level JEPA on D4's encoder as-is) is cheaper but drops the random-joint arm — rejected unless Fork B is too invasive.

### Fork C — JEPA learning rate
- D4 `lr=2e-5` is a *fine-tune* rate. JEPA is **from-scratch representation pretraining** → use a higher LR.
- **Recommendation: `lr=2e-4`**, `weight_decay=0.05`, cosine with warmup (V-JEPA-like), keep D4's `grad_clip` (10.0 → tighten to 1.0 for JEPA stability).

---

## 4. Architecture (Module A)

### 4.1 Encoder (online + EMA target)
Config-matched to D4 (§1) with Fork-B restructure and Fork-A inputs.

- **Spatial:** `SkeletalJointTokenizer` (structured) builds 22 per-joint tokens (idx 0=root, 1–21 SMPL-H joints), joint-tree graph attention ×`graph_layers=2` (`graph_retarget_latent.py:432-439`). With Fork A, joint input = `[pose_aa(3) | joint_pos(3) | joint_vel(3) | joint_contact(1)]` concatenated (`:317-324`).
- **Temporal (Fork B):** new `JointTemporalEncoder` — the D4 temporal transformer (4 layers, 8 heads) applied **per joint over T** (+ optional cross-joint graph mixing), producing `h[B,T,22,d_model]`.
- **Pool → latent (downstream of JEPA target):** mean per slot → per-slot z (`slot_z=48`) → concat 7 slots → `z[B,T,336]`. Used by B1/B2, not by the JEPA loss. VAE sampling **off** during JEPA target computation (`sample=False`); KL stays a downstream concern.
- **Two copies:** `online_encoder` (gradients) and `target_encoder` (EMA, `requires_grad=False`).

### 4.2 Masking (per-joint, V-JEPA tubelet)
`joint_mask: Bool[B, J]` (J=21 body joints), **constant across all T** (tubelet; temporal mask ratio ≈ 1.0 — visible joints keep full temporal info). Masked joints' input tokens → a learned `jepa_mask_token` before the graph attention (online encoder only; target encoder sees full input).

**Ablation axis (the user's):**
- `random`: random subset of the 21 joints.
- `slot`: whole anatomical slots (contiguous) = V-JEPA blocks.
- mask-ratio band **0.4–0.7** (lower than V-JEPA's 0.9 — only 21 joints; swept).

### 4.3 Predictor (spatial completion)
`MotionJEPAPredictor`: narrow transformer over the 22 joint tokens (depth 2–4, width 128–256, 4 heads). Input = online per-joint hidden `h_ctx[B,T,22,d]` + learned mask tokens at masked positions (with joint-id embedding). Output = predicted per-joint targets at masked joints. **No temporal attention** — completion is spatial; temporal context already lives in `h_ctx` from §4.1. Shared across T.

### 4.4 Loss + EMA
- `L = smooth_l1( LayerNorm(pred[mask]), LayerNorm(target_h[mask]).detach() )`, target from the EMA encoder.
- EMA momentum `0.996 → 1.0` (cosine), `target ← m·target + (1−m)·online`.
- Anti-collapse monitors: per-joint feature std, pred/target std, cosine(pred,target), z std. No contrastive, no decoder.

---

## 5. Data + training (reuse D4 pipeline)

- **Dataset:** `build_sequence10_datasets(dataset_path, window=60, stride=30, val_ratio=0.1, seed=20260521, target_fps=20.0, reference_actions_dir=None)` — **no reference actions** (JEPA is self-supervised). `make_loader` + `collate_sequence_windows`. (`amass_sequence10_dataset.py`, `train_sequence_retargeter.py`.)
- **Inputs per batch:** canonicalize (`canonicalize_motion`, Plan 2 / P4) → `pose_aa=poses[...,3:66].reshape(B,T,21,3)`, `root_pose=make_graph_root_features(..., mode='scaled_abs_z')` (match D4), `beta=fit['beta']`; build `joint_pos/joint_vel/joint_contact` via `build_encoder_inputs` (Fork A).
- **Config:** window 60 / stride 30 / fps 20 (D4); `batch_size` to fit; `lr=2e-4`, `wd=0.05`, cosine+warmup, `grad_clip=1.0`, epochs ~20 (JEPA, longer than D4's 8 retargeting epochs).
- **Script:** new `scripts/data_preparation/pretrain_motion_jepa.py` — saves the encoder `state_dict` (consumable by B1/B2) + EMA target + config.

---

## 6. Validation gates (Stage-1 exit)

1. **Non-collapse:** per-joint target std + z std stay non-trivial through training; `L` drops ≥35% from first-10% to last-10%.
2. **Frozen probe:** a frozen-encoder linear probe (future-z / contact) beats an untrained encoder at equal probe budget.
3. **Ablation reported:** `random` vs `slot` masking — JEPA loss, probe error, and per-joint std-by-region. Hypothesis (V-JEPA): `slot` ≥ `random`.
4. **Latent sanity:** round-trip / linear-probe predictability of the pooled 336-d z is ≥ the D4 encoder's at init (no worse than the production encoder pre-JEPA).

A synthetic smoke (`TargetOracle`-style, no real data) runs CI-fast first; the real gates use AMASS.

---

## 7. Files

| File | Action |
|---|---|
| `GBC/utils/data_preparation/graph_retarget_latent.py` | add per-joint mask + expose per-joint hidden in `SkeletalJointTokenizer`; add `JointTemporalEncoder` (Fork B) |
| `GBC/utils/data_preparation/motion_jepa.py` | revise: `JointMaskCfg`, `JointMaskGenerator` (random/slot), `MotionJEPAPredictor` (spatial), `MotionJEPAModule` (online+EMA), `motion_jepa_loss`, `ema_update` |
| `GBC/utils/data_preparation/motion_canonicalize.py` | reuse (Plan 2, P4) |
| `scripts/data_preparation/pretrain_motion_jepa.py` | new real-AMASS pretraining driver |
| `tests/test_motion_jepa.py` | revise for per-joint masking + JointTemporalEncoder |
| `tests/test_jepa_probe.py` | new: frozen-probe gate |

---

## 8. Open confirmations before TDD breakdown

Confirm Forks A/B/C (§3). On confirmation I'll invoke writing-plans to convert this into the bite-sized TDD task list. Untouched by these: the canonicalization contract (P4), the dataset reuse, and the EMA/loss core.

---

## 9. Self-review

- Grounded in `gbc_runs/_d4_config/run_config.json` (values cited inline); resolves design-doc §9.1 (Fork B = option i) and §9.2 (`graph_layers=2`).
- Scope = Module A only; B1/B2 unchanged. No real-data *decoder* dependency in Stage 1.
- Three forks are explicit decisions with rationale, not buried defaults. The one genuine architectural cost (Fork B per-joint-temporal) is called out with its fallback.
