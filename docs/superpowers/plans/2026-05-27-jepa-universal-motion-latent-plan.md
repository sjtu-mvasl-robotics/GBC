# JEPA Universal Motion Latent Retargeting Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for single-worker implementation or superpowers:subagent-driven-development when splitting the work across MFME, MGE, CMAD, and training scripts. Track progress by updating the checkbox items in this document.

**Date:** 2026-05-27
**Goal:** Add a JEPA-style self-supervised pretraining path that makes the GBC motion latent more universal, while keeping the existing GMR-supervised retargeting losses as the quality anchor.

**One-sentence decision:** Use JEPA for the **morphology-free motion encoder** and optionally for the **morphology graph encoder**; do not use JEPA as a replacement for robot-action supervision.

---

## 1. Scope

This plan covers a complete JEPA path for the current retargeting stack:

1. Literature-grounded design choices from I-JEPA, V-JEPA, V-JEPA 2, and MC-JEPA.
2. A Motion-JEPA pretraining task for `MorphologyFreeMotionEncoder`.
3. A Graph-JEPA pretraining task for `MorphologyGraphEncoder`.
4. A supervised retargeting curriculum that reuses JEPA-pretrained modules.
5. Tests, metrics, ablations, and acceptance gates.

This plan does **not** replace the existing GMR reference-action training. JEPA learns a predictive motion representation; GMR/FK/physics losses still decide whether robot motion is usable.

---

## 2. Literature Findings

Primary sources:

- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA: https://arxiv.org/abs/2404.08471
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- MC-JEPA: https://arxiv.org/abs/2307.12698

### 2.1 I-JEPA takeaways

I-JEPA predicts target block representations from a context block in learned representation space, not raw input space. The target encoder is updated by EMA from the context encoder; the predictor receives context features plus positional target tokens and predicts target representations.

Actionable for GBC:

- Predict **slot-time latent targets**, not angle-axis vectors or robot actions.
- Use an EMA target MFME. Stop gradient through target features.
- The mask policy is central. Mask large semantic units: body slots and time segments, not isolated scalar coordinates.
- Use enough visible context. If the context is too small, the task becomes ambiguous; if too large, the task becomes trivial.

### 2.2 V-JEPA takeaways

V-JEPA validates feature prediction as a standalone video pretraining objective without negative pairs, pixel reconstruction, text supervision, or pretrained image encoders. Its useful signal for us is temporal: feature prediction trained on video transfers to both motion-sensitive and appearance-sensitive tasks.

Actionable for GBC:

- Motion latent pretraining should emphasize temporal blocks and future prediction.
- Use multi-mask prediction to amortize the target encoder forward pass.
- Evaluate with frozen encoder probes before committing to expensive retarget training.
- Do not add a decoder just because MAE-style reconstruction is familiar; the JEPA objective should stay in representation space.

### 2.3 V-JEPA 2 takeaways

V-JEPA 2 first trains an action-free JEPA representation, then post-trains an action-conditioned world model for robot planning. The important design separation is: observation-only representation first, action-conditioned planning/control later.

Actionable for GBC:

- Pretrain MFME **without robot data**.
- Train CMAD / robot adapters after the representation has motion semantics.
- If we later add a robot-action-conditioned world model, it belongs after the motion latent is stable, not in the first JEPA milestone.

### 2.4 MC-JEPA takeaways

MC-JEPA shows that content and motion objectives can share an encoder. Its direct lesson is not optical flow, but the multi-task structure: a shared encoder can carry both appearance/content and motion if objectives are balanced.

Actionable for GBC:

- Add contact, velocity, and future-slot auxiliary probes around JEPA rather than only action MSE.
- Keep these as heads/probes on MFME representations, not as extra inputs that leak robot identity into MFME.

---

## 3. Current Repository Diagnosis

Relevant files:

- `GBC/utils/data_preparation/pose_transformer.py`
- `GBC/utils/data_preparation/pose_transformer_trainer.py`
- `GBC/utils/data_preparation/graph_retarget_latent.py`
- `GBC/utils/data_preparation/morphology_conditioned_retargeter.py`
- `GBC/utils/base/base_fk.py`
- `scripts/data_preparation/run_morph_latent_experiments.py`
- `outputs/render/CHECKPOINT_VALIDATION_REPORT.md`
- `outputs/MORPH_LATENT_VS_MUJICA_COMPARISON.md`

### 3.1 What is already aligned with JEPA

- `SkeletalJointTokenizer` and `GraphTemporalEncoder` already produce structured slot-time representations.
- `FutureSlotHead` already asks the latent to predict future motion-side information.
- `MotionLatentRetargeter` cleanly separates:
  - MFME: motion only.
  - MGE: morphology only.
  - CMAD: motion latent + morphology -> actions.
- `RobotKinematics.get_dof_features()` exposes URDF-derived static features suitable for morphology graph pretraining.

### 3.2 What blocks universality today

- `UnifiedLatentRetargeter` still includes a G1 teacher/base path through `PoseTransformer`, so the action path can keep latent geometry tied to one robot.
- The VAE KL is per-frame and can encourage averaging on dynamic lower-body motion.
- The current production-quality evidence is single-robot G1; the multi-robot path has synthetic validation but not real paired retarget data yet.
- The current encoder has no explicit root canonicalization contract, so world yaw/translation can leak into `z`.

### 3.3 Design stance

The JEPA work should target `MotionLatentRetargeter`, not the old `PoseTransformer` trainer. The old trainer remains a baseline and teacher generator. The universal path should be:

```text
AMASS/SMPL-H motion
    -> MFME-JEPA pretraining
    -> optional MGE-JEPA pretraining over URDFs
    -> CMAD supervised by GMR/reference actions
    -> joint fine-tune with JEPA warm loss + physics/contact losses
```

---

## 4. Target Architecture

### 4.1 Motion-JEPA

Add a JEPA wrapper around `MorphologyFreeMotionEncoder`.

```text
full motion clip
    -> target MFME EMA
    -> target graph_hidden / slot_latents for masked slot-time targets

masked context clip
    -> online MFME
    -> context graph_hidden
    -> JEPA predictor(context, target query tokens)
    -> predicted target graph_hidden / slot_latents
```

Core rule: the target features are computed from the unmasked full motion and detached. The online encoder and predictor receive gradients.

### 4.2 Motion target representation

Use two target spaces:

1. `graph_hidden`: `(B, T, 6, d_model)`.
   - Main JEPA target.
   - Higher capacity and directly consumed by CMAD.

2. `slot_latents`: `(B, T, 6, slot_latent_dim)`.
   - Lightweight auxiliary target.
   - Easier to probe and visualize.

Do **not** use `z` as the only JEPA target in the first version. `z` is pooled across slots and loses the body-part locality that makes masking meaningful.

### 4.3 Motion mask policy

Represent masks as `slot_time_mask: Bool[B, T, S]`, where `True` means the slot-time token is hidden from the context encoder and predicted by the predictor.

Initial mask families:

```python
MotionMaskCfg(
    temporal_block_lengths=(8, 16, 32),
    num_temporal_blocks=2,
    slot_block_sizes=(1, 2, 3),
    keep_global_root_prob=0.75,
    bilateral_pair_prob=0.35,
    future_only_prob=0.50,
    mask_ratio_range=(0.25, 0.55),
)
```

Mask semantics:

- **Future mask:** hide future frames for selected slots; forces motion prediction.
- **Limb mask:** hide one or more body slots over a temporal block; forces cross-body prediction.
- **Bilateral mask:** hide paired left/right limbs sometimes; prevents trivial mirroring.
- **Root-aware mask:** usually keep `global_root`, but sometimes mask it to force local-body prediction.

This is the motion equivalent of I-JEPA's large semantic target blocks and V-JEPA's temporal multi-block masking.

### 4.4 Predictor

Add `MotionJEPAPredictor`.

Inputs:

```python
context_hidden:      Tensor[B, T, S, d_model]
target_indices:      Tensor[N, 3]  # b, t, slot
target_time_embed:   learned or sinusoidal
target_slot_embed:   learned
```

Output:

```python
pred_graph_hidden: Tensor[N, d_model]
pred_slot_latent:  Tensor[N, slot_latent_dim]
```

Recommended first implementation:

- Project context hidden to a flat sequence `(B, T*S, d_model)`.
- Build target query tokens from time embedding + slot embedding + learned mask token.
- Use a 2-4 layer transformer decoder / cross-attention predictor.
- Keep predictor narrow, e.g. `pred_d_model=128 or 256`, `num_layers=2`, `num_heads=4 or 8`.

### 4.5 Encoder interface changes

Keep normal `MotionLatentRetargeter.forward(...)` stable. Add MFME-only methods:

```python
class MorphologyFreeMotionEncoder(nn.Module):
    def tokenize(..., slot_time_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        ...

    def encode_tokens(slot_tokens, part_mask, slot_time_mask=None) -> dict[str, Tensor]:
        ...

    def forward(..., slot_time_mask: Optional[Tensor] = None, sample: Optional[bool] = None):
        ...
```

Implementation note:

- If `slot_time_mask[b, t, s]` is true, replace that slot token with a learned `masked_slot_time_token` before `GraphTemporalEncoder`.
- Existing `part_mask` stays semantic presence/missing body-part mask. Do not overload it for JEPA masks.

### 4.6 EMA target encoder

Add a wrapper:

```python
class MotionJEPAModule(nn.Module):
    online_encoder: MorphologyFreeMotionEncoder
    target_encoder: MorphologyFreeMotionEncoder
    predictor: MotionJEPAPredictor

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        target = momentum * target + (1 - momentum) * online
```

Momentum schedule:

```python
ema_start = 0.996  # small models / early probes
ema_final = 1.0
```

For production-length windows, use V-JEPA-like stronger smoothing:

```python
ema_start = 0.998
ema_final = 1.0
```

---

## 5. Losses

### 5.1 Core JEPA loss

Normalize target and prediction before loss:

```python
p = layer_norm(pred_graph_hidden)
y = layer_norm(target_graph_hidden.detach())
L_jepa_graph = smooth_l1(p, y) or mse(p, y)
```

Use a second auxiliary loss:

```python
L_jepa_slot = smooth_l1(pred_slot_latent, target_slot_latent.detach())
```

Default:

```python
L_motion_jepa = L_jepa_graph + 0.25 * L_jepa_slot
```

### 5.2 Anti-collapse monitoring and optional regularization

I-JEPA/V-JEPA rely mainly on EMA target asymmetry and mask design. Still add monitors:

```python
target_std_mean
pred_std_mean
z_std_mean
cosine_pred_target
slot_variance_by_part
```

Only if collapse appears, add a small variance hinge:

```python
L_var = mean(relu(1.0 - std(batch_features)))
```

Do not add contrastive negatives in v1. It changes the method family and complicates interpretation.

### 5.3 Auxiliary motion heads

Keep these optional in pretraining:

```python
L_future_slot = existing FutureSlotHead target loss
L_contact = BCE(pred_contact, foot_contact)
L_velocity = smooth_l1(pred_delta_slot, target_delta_slot)
```

Initial total:

```python
L_pretrain_mfme =
    1.0 * L_motion_jepa
  + 0.2 * L_future_slot
  + 0.1 * L_contact
  + 0.1 * L_velocity
```

If foot contacts are noisy or unavailable, set `contact_weight=0` and keep the interface.

---

## 6. Graph-JEPA for Morphology

This is optional but low-cost.

### 6.1 Objective

Pretrain `MorphologyGraphEncoder` using only URDF-derived robot specs.

```text
full robot graph
    -> target MGE EMA
    -> target per-DoF embeddings

masked robot graph
    -> online MGE
    -> graph predictor
    -> predicted masked DoF embeddings
```

### 6.2 Masks

Use `dof_mask: Bool[B, K]`.

Mask families:

- Random DoF nodes.
- Whole limb/slot nodes.
- Distal subtree nodes.
- Limit/axis feature dropout while keeping topology.

### 6.3 Losses

```python
L_graph_jepa = smooth_l1(norm(pred_m_per_dof), norm(target_m_per_dof.detach()))
L_slot_cls = cross_entropy(slot_head(m_per_dof), slot_assignment)
L_parent = cross_entropy(parent_head(m_per_dof), parent_index_bucket)
L_feature = mse(feature_head(m_per_dof), node_features)  # optional, low weight
```

Default:

```python
L_pretrain_mge =
    1.0 * L_graph_jepa
  + 0.2 * L_slot_cls
  + 0.1 * L_parent
  + 0.05 * L_feature
```

### 6.4 Data requirement

Start with available URDFs in the repo and local robot assets. Target minimum:

- G1
- H1/H1-2 if available
- GR1/Fourier
- Turin/Turing variants in the local lab assets

If fewer than 4 real robots are available, keep Graph-JEPA as a smoke-test only and do not claim cross-morphology quality from it.

---

## 7. Training Curriculum

### Phase 0: Baseline freeze and metrics snapshot

- [ ] Re-run or load the current best `UnifiedLatentRetargeter` G1 metrics.
- [ ] Re-run `MotionLatentRetargeter` synthetic ablation if code has changed.
- [ ] Record current G1 reference metrics:
  - `latent_mean_l2_m`
  - action L2 to GMR
  - FK mean/max error
  - foot slide
  - ground penetration
  - dynamic lower-body amplitude on jumping jacks
- [ ] Save one CPU inference/render sample for walking and one for jumping jacks.

Exit gate:

- A reproducible baseline table exists before JEPA changes are evaluated.

### Phase 1: Motion-JEPA standalone pretraining

- [ ] Add `GBC/utils/data_preparation/motion_jepa.py`.
- [ ] Add `MotionMaskCfg`, `MotionMaskGenerator`, `MotionJEPAPredictor`, `MotionJEPAModule`.
- [ ] Add `slot_time_mask` support to `MorphologyFreeMotionEncoder` without changing default behavior.
- [ ] Add `scripts/data_preparation/pretrain_motion_jepa.py`.
- [ ] Train on AMASS windows only, no robot actions, no CMAD.
- [ ] Save encoder checkpoint with config and EMA target state.

Default hyperparameters:

```python
window = 64
stride = 16
d_model = 256
z_dim = 256
batch_size = 64  # adjust by GPU memory
lr = 2e-4
weight_decay = 0.05
epochs = 20
ema_start = 0.996
ema_final = 1.0
mask_ratio_range = (0.25, 0.55)
```

Exit gates:

- `L_motion_jepa` decreases by at least 35% from first 10% to last 10% of training.
- `target_std_mean` and `z_std_mean` stay non-trivial; no constant embedding collapse.
- Frozen MFME probe predicts future slot/contact better than an untrained encoder.

### Phase 2: Graph-JEPA standalone pretraining

- [ ] Add `GBC/utils/data_preparation/morphology_jepa.py`.
- [ ] Add `scripts/data_preparation/pretrain_morphology_jepa.py`.
- [ ] Build robot specs through `build_robot_spec_from_kinematics`.
- [ ] Pretrain MGE on real URDFs and synthetic perturbations.
- [ ] Save MGE checkpoint.

Exit gates:

- Masked DoF embedding prediction loss decreases.
- Slot classifier accuracy exceeds a name-only heuristic baseline when using real URDF features.
- Zeroing URDF node features degrades held-out synthetic morphology recon, matching the existing ablation direction.

### Phase 3: Supervised CMAD training with frozen JEPA encoders

- [ ] Add a train script variant, e.g. `scripts/data_preparation/train_morph_latent_jepa_sft.py`.
- [ ] Load pretrained MFME into `MotionLatentRetargeter.mfme`.
- [ ] Optionally load pretrained MGE.
- [ ] Freeze MFME for the first training segment.
- [ ] Train CMAD against GMR/reference actions.
- [ ] Keep physics-lite losses available.

Default loss:

```python
L_sft =
    1000.0 * masked_action_l2(actions, ref_actions)
  + 1.0    * fk_link_loss
  + 0.5    * foot_l2
  + 0.2    * foot_slide
  + 0.1    * ground_penalty
  + 1e-3   * latent_smoothness_loss(z)
```

Exit gates:

- On G1, frozen-MFME CMAD is within 10% of the best non-JEPA `MotionLatentRetargeter` action/FK metric after the same wall-clock budget.
- It must not regress walking visual quality.
- Dynamic lower-body amplitude should not be worse than the current `physlite` checkpoint.

### Phase 4: Joint fine-tuning with JEPA warm loss

- [ ] Unfreeze MFME with lower LR.
- [ ] Keep a small JEPA loss on the same batch or an interleaved AMASS-only batch.
- [ ] Keep target encoder EMA during fine-tuning.
- [ ] Add optional cross-morphology consistency if multi-robot paired GMR data exists.

Default optimizer groups:

```python
mfme_lr = 2e-5
mge_lr = 5e-5
cmad_lr = 2e-4
predictor_lr = 1e-4
```

Default loss:

```python
L_total =
    L_sft
  + 0.05 * L_motion_jepa
  + 0.01 * L_graph_jepa
```

Exit gates:

- G1 is not worse than the frozen-MFME run.
- Held-out motion categories improve or stay stable.
- Latent probing metrics stay above Phase 1 frozen probe.

### Phase 5: FSQ/token ablation

Only start this after continuous JEPA proves useful.

- [ ] Add optional FSQ bottleneck after MFME pooled `z`.
- [ ] Keep JEPA target on pre-quantization `graph_hidden`.
- [ ] Compare continuous JEPA vs FSQ-JEPA.

Exit gate:

- FSQ gives a more compact downstream interface without degrading action/FK quality beyond 5%.

---

## 8. Cross-Morphology Consistency

When real multi-robot paired retarget data exists:

```text
same human motion -> G1 action target
same human motion -> H1/GR1 action target
```

Use one MFME encoding and different robot batches through CMAD.

Add two consistency checks:

1. **Shared latent stability:** `z` should be identical because it is computed before robot conditioning.
2. **Slot attention consistency:** CMAD cross-attention should route leg DoFs to leg slots and arm DoFs to arm slots across robots.

Optional later addition:

```text
robot actions + robot spec -> inverse robot motion encoder -> z_robot
```

Then add:

```python
L_cycle_z = mse(z_robot_A, z_human.detach()) + mse(z_robot_B, z_human.detach())
```

Do not add the inverse encoder before CMAD is stable; it creates a second moving target.

---

## 9. Evaluation Matrix

### 9.1 Models

| ID | Model | Purpose |
|---|---|---|
| B0 | `UnifiedLatentRetargeter structured` | production G1 reference |
| B1 | `MotionLatentRetargeter` from scratch | universal architecture baseline |
| J1 | MFME-JEPA + frozen MFME + CMAD | tests representation transfer |
| J2 | MFME-JEPA + MGE-JEPA + frozen encoders + CMAD | tests both pretrainers |
| J3 | J2 + joint fine-tune with JEPA warm loss | final continuous JEPA candidate |
| J4 | J3 + FSQ bottleneck | compact-token ablation |

### 9.2 Datasets

| Split | Data | Use |
|---|---|---|
| AMASS train | SMPL-H windows | MFME-JEPA |
| AMASS val | held-out clips | JEPA probe and retarget validation |
| URDF train | available robot URDFs | MGE-JEPA |
| GMR G1 train | AMASS + G1 ref actions | CMAD SFT |
| GMR G1 val | held-out motions | apples-to-apples with current G1 |
| Multi-robot train | G1 + H1/GR1 if available | cross-morphology |
| Held-out robot | H1-2 or another humanoid | zero-shot/adapter evaluation |

### 9.3 Metrics

JEPA representation:

- `motion_jepa_loss`
- target/pred feature std
- cosine pred-target
- frozen future-slot probe error
- frozen contact probe F1
- latent yaw-invariance check after root canonicalization

Retargeting:

- action L2 to GMR
- FK mean and max link error
- foot slide
- ground penetration
- contact-height error
- lower-body dynamic amplitude ratio
- render sanity on walking and jumping jacks

Cross-morphology:

- masked action L2 normalized per valid DoF
- held-out robot loss
- zero-node / zero-slot / zero-all morphology ablations
- attention routing entropy by body slot

---

## 10. Acceptance Criteria

The JEPA path is worth keeping only if all three levels pass.

### Level 1: Representation

- [ ] No embedding collapse.
- [ ] JEPA loss decreases meaningfully.
- [ ] Frozen probes beat untrained MFME and from-scratch MFME at equal probe budget.

### Level 2: Single-robot G1

- [ ] G1 action/FK quality is within 5-10% of the best current structured baseline after equal training budget.
- [ ] Walking render remains visually coherent.
- [ ] Jumping-jacks lower-body amplitude does not regress and ideally improves.

### Level 3: Cross-morphology

- [ ] Multi-robot training with shared MFME is stable.
- [ ] Held-out robot performance beats no-morphology controls.
- [ ] URDF feature ablation remains monotonic: FULL better than ZERO_NODE better than ZERO_ALL, or any deviation is explained by data quality.

If Level 1 passes but Level 2 fails, JEPA is useful as a representation pretrainer but the decoder/training curriculum is wrong. If Level 1 fails, stop and fix masks/EMA before touching CMAD.

---

## 11. Implementation Tasks

### 11.1 Tests first

- [ ] `tests/test_motion_jepa.py`
  - [ ] mask generator produces valid `(B, T, S)` masks.
  - [ ] masked tokens do not alter default MFME output when mask is `None`.
  - [ ] `MotionJEPAModule.forward` returns finite loss and predictions.
  - [ ] target encoder receives no gradient.
  - [ ] EMA update changes target weights in the expected direction.

- [ ] `tests/test_morphology_jepa.py`
  - [ ] graph masks preserve padding.
  - [ ] masked MGE forward supports variable K.
  - [ ] graph JEPA loss ignores padded DoFs.
  - [ ] parent/slot auxiliary heads produce expected shapes.

- [ ] `tests/test_jepa_retarget_integration.py`
  - [ ] load MFME-JEPA checkpoint into `MotionLatentRetargeter`.
  - [ ] frozen MFME still allows CMAD gradients.
  - [ ] default `MotionLatentRetargeter.forward` output schema is unchanged.

### 11.2 Code modules

- [ ] Modify `morphology_conditioned_retargeter.py`
  - [ ] add optional `slot_time_mask` to `MorphologyFreeMotionEncoder`.
  - [ ] expose tokenization/encoding helper methods.
  - [ ] preserve all current tests.

- [ ] Add `GBC/utils/data_preparation/motion_jepa.py`
  - [ ] `MotionMaskCfg`
  - [ ] `MotionMaskGenerator`
  - [ ] `MotionJEPAPredictor`
  - [ ] `MotionJEPAModule`
  - [ ] `motion_jepa_loss`
  - [ ] EMA helper

- [ ] Add `GBC/utils/data_preparation/morphology_jepa.py`
  - [ ] `GraphMaskCfg`
  - [ ] `GraphMaskGenerator`
  - [ ] `MorphologyJEPAPredictor`
  - [ ] `MorphologyJEPAModule`

- [ ] Add scripts
  - [ ] `scripts/data_preparation/pretrain_motion_jepa.py`
  - [ ] `scripts/data_preparation/pretrain_morphology_jepa.py`
  - [ ] `scripts/data_preparation/train_morph_latent_jepa_sft.py`
  - [ ] `scripts/data_preparation/evaluate_jepa_latent.py`

### 11.3 Logging

Log at minimum:

```python
{
    "loss/motion_jepa": ...,
    "loss/jepa_graph": ...,
    "loss/jepa_slot": ...,
    "stats/target_std": ...,
    "stats/pred_std": ...,
    "stats/z_std": ...,
    "stats/mask_ratio": ...,
    "stats/ema_momentum": ...,
    "probe/future_slot": ...,
    "probe/contact_f1": ...,
}
```

For retargeting:

```python
{
    "retarget/action_l2": ...,
    "retarget/fk_mean": ...,
    "retarget/foot_slide": ...,
    "retarget/lower_body_amp_ratio": ...,
}
```

---

## 12. Risks and Mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| Mask task too easy | JEPA loss low, probes weak | larger temporal masks, more limb masks, less root visibility |
| Mask task too hard | loss flat, noisy representations | lower mask ratio, keep root visible, shorter target horizon |
| EMA collapse | feature std approaches zero | increase batch diversity, add variance hinge, adjust EMA start lower |
| JEPA hurts retargeting | probes improve but CMAD worsens | freeze MFME longer, lower joint-finetune LR, keep supervised decoder warmup |
| Root leakage | yaw-transformed same motion gives different z | add root canonicalization and yaw-invariance eval |
| MGE overfits few URDFs | graph loss good, held-out bad | use synthetic URDF perturbations and feature dropout; do not overclaim |
| Contact labels noisy | contact probe unstable | keep contact auxiliary off until labels are validated |

---

## 13. Recommended First Milestone

The smallest high-signal milestone is **Motion-JEPA only**:

1. Implement `slot_time_mask` and `MotionJEPAModule`.
2. Pretrain MFME on AMASS for a short run.
3. Freeze MFME and train CMAD on the existing G1 GMR reference-action data.
4. Compare against `MotionLatentRetargeter` from scratch with the same CMAD setup.

Success means:

- JEPA pretraining is non-collapsed.
- Frozen-MFME CMAD reaches comparable G1 quality faster or improves dynamic-motion generalization.

Failure is still informative:

- If JEPA probes are good but CMAD is bad, the decoder/curriculum is the issue.
- If JEPA probes are bad, the mask policy or target choice is wrong.

Do not start FSQ or inverse robot encoders until this milestone is complete.

---

## 14. Concrete Command Sketches

Pretrain MFME:

```bash
PYTHONPATH=. python scripts/data_preparation/pretrain_motion_jepa.py \
  --dataset-path /path/to/amass \
  --window 64 \
  --stride 16 \
  --batch-size 64 \
  --epochs 20 \
  --d-model 256 \
  --z-dim 256 \
  --save-dir outputs/motion_jepa_mfme_v1
```

Pretrain MGE:

```bash
PYTHONPATH=. python scripts/data_preparation/pretrain_morphology_jepa.py \
  --urdf-list configs/robot_urdfs.txt \
  --epochs 2000 \
  --batch-size 16 \
  --save-dir outputs/morphology_jepa_mge_v1
```

Train CMAD with pretrained MFME:

```bash
PYTHONPATH=. python scripts/data_preparation/train_morph_latent_jepa_sft.py \
  --gmr-dataset /path/to/gmr_sft_dataset.pt \
  --robot unitree_g1 \
  --mfme-ckpt outputs/motion_jepa_mfme_v1/best.pt \
  --freeze-mfme-steps 20000 \
  --window 120 \
  --batch-size 32 \
  --save-dir outputs/morph_latent_jepa_sft_g1_v1
```

Evaluate:

```bash
PYTHONPATH=. python scripts/data_preparation/evaluate_jepa_latent.py \
  --ckpt outputs/morph_latent_jepa_sft_g1_v1/best.pt \
  --gmr-val /path/to/gmr_val_dataset.pt \
  --render-clips walking_01,jumping_jacks
```

---

## 15. Bottom Line

JEPA is a good fit for this repository only if it is treated as a **robot-free predictive motion pretraining objective**. The universal latent should be defined by what it predicts about human/canonical motion, not by how well it reconstructs one robot's action vector.

The right architecture is:

```text
MFME-JEPA learns motion semantics.
MGE-JEPA optionally learns morphology semantics.
CMAD learns the supervised translation from motion semantics + morphology semantics to robot actions.
```

That separation is the practical route to a universal latent in GBC.
