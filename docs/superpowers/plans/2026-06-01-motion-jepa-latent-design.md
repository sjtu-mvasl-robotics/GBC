# Motion-JEPA Latent + Flow-Matching Rollout — Design

**Date:** 2026-06-01
**Status:** supersedes the *framing* of `2026-05-28-jepa-gonogo-ablation-design.md`. That doc asked a narrow question ("does JEPA pretraining help CMAD retargeting?"). This doc states **what the latent is, what JEPA predicts, and how generation/rollout is added** — and demotes the old `scratch`/`mae`/`jepa` matrix to an *extrinsic transfer check* of the learned latent.

---

## 1. What we are building (and the clean separation)

A **per-frame motion latent z_t** learned by JEPA, plus the machinery to roll it out and decode it. Three modules with hard boundaries:

| Module | Role | Trains what |
|---|---|---|
| **A — Motion-JEPA (the latent learner)** | Self-supervised. Learns z_t by **spatial joint completion** in representation space. No decoder, no generation. | The encoder. |
| **B1 — FM latent predictor** | Autoregressive **flow-matching** model over frozen z_t; rolls out future latents; text-conditionable. | A separate sequence model on z_t. |
| **B2 — Decoder** | Reconstructs robot motion from z_t. The near-term consumer that proves the latent. | A separate decoder. |

**Decision (committed): Approach 2 — staged.** JEPA learns the latent; then, on the *frozen* latent, train the FM predictor (B1) and the decoder (B2). The JEPA predictor is **spatial only**; temporal/autoregressive prediction lives entirely in B1. Approach 1 (fuse temporal prediction into the JEPA predictor) is the **considered alternative**, rejected — see §8.

**Why this separation (user's framing):** the encoder is "only a temporal compressor/extractor, not an AR predictor." JEPA shapes the latent; the FM model adds dynamics; the decoder adds reconstruction. This mirrors V-JEPA 2 (action-free representation first, world-model second) and reuses the repo's existing flow-matching stack rather than reinventing temporal prediction inside JEPA.

---

## 2. Constraints the latent must satisfy (imported as objectives, not left implicit)

From `project-latent-constraints` — these are now design requirements, not background:

1. **Per-frame** z_t (one latent per frame; no temporal compression of the *addressing* — the temporal module gives context, but each frame keeps its own z_t).
2. **MDM/flow-prior-compatible** — z_t is the operand of a flow-matching predictor (B1) that reuses the repo's FM recipe, so it inherits those priors.
3. **Cross-platform** — z_t is robot-agnostic (encoder is morphology-free); only B2 is per-robot.
4. **Physics-extendable** — encoder ingests velocity + contact channels at training time (`SkeletalJointTokenizer` already supports `joint_velocity_dim`, `joint_contact_dim`, `joint_geometry_dim`, `graph_retarget_latent.py:290-292`).

---

## 3. Module A — Motion-JEPA (the latent learner)

### 3.1 V-JEPA technique, ported to motion (confirmed against the paper)

V-JEPA: multi-block masking where **spatial blocks are repeated across the full temporal extent** (temporal mask ratio ≈ 1.0); short-range (8 blocks ~15%) + long-range (2 blocks ~70%) → ~90% effective; narrow predictor transformer over visible tokens + mask tokens; **EMA target**; feature-space **L1**, targets LayerNorm'd + stop-grad; no contrastive; the paper finds **structured/contiguous masking beats random**.

Motion port: **spatial = joints, temporal = full window.** Mask a subset of **joints across all T frames** (tubelet-style: a masked joint is hidden at every frame; visible joints keep full temporal info). The predictor completes the masked joints' latent targets from the visible joints. The predictor is **spatial only** — it never predicts a fully-hidden future frame, so nothing autoregressive enters Module A.

### 3.2 Granularity (decided): per-joint targets, pooling downstream

`SkeletalJointTokenizer` builds **22 per-joint tokens** (index 0 = root, 1–21 = SMPL-H body joints), runs joint-tree graph attention (`graph_retarget_latent.py:432-439`), then mean-pools into the **6 slots** (`:441-457`). JEPA masks and predicts at the **per-joint** level (the post-graph-attention hidden at `:439`); slot pooling → z_t stays **downstream** of the JEPA target. This is the faithful "mask joints" reading and the only granularity that supports both arms of the ablation below.

### 3.3 Masking ablation (the axis the user called for)

- **random-joint** masking: mask a random subset of the 21 body joints.
- **slot-structured** masking: mask whole anatomical slots (left_arm, right_leg, …) = V-JEPA's contiguous blocks.
- **Mask-ratio band:** swept, **lower than V-JEPA's 90%** — with only 21 joints, 90% is degenerate; start in the 40–70% range and tune. The mask is constant across T (tubelet).
- Hypothesis (V-JEPA's finding): slot-structured > random. This ablation tests it on motion.

### 3.4 Architecture

- **Online encoder + EMA target encoder** (two copies; target is EMA of online, params `requires_grad=False`). Momentum schedule 0.996/0.998 → 1.0.
- **Masking injection:** masked joints' input tokens replaced by a learned `jepa_mask_token` before the joint-tree graph attention (analogue of the existing `missing_token` mechanism, but at joint granularity and JEPA-specific). The online encoder sees masked input; the target encoder sees full input (detached).
- **Predictor** (`MotionJEPAPredictor`): a **narrow** transformer over the 22 joint tokens (V-JEPA: narrow, ~depth 2–4, width 128–256, 4 heads), input = online per-joint hidden + learned mask tokens at masked positions (carrying joint-identity embedding). Output = predicted per-joint targets at masked joints. **No temporal attention** (spatial completion; shared across T).
- **Loss:** `smooth_l1(LayerNorm(pred[masked]), LayerNorm(target[masked]).detach())`. Anti-collapse via EMA + structured masking; monitor per-joint feature std, pred/target std, cosine.

### 3.5 KEY OPEN DECISION — temporal-module placement (flagged for review)

The current encoder pools to slots **before** the temporal module (`MFME`: tokenizer → pool → `GraphTemporalEncoder` over slots×time). So per-joint hidden is **pre-temporal** (per-frame spatial). Two ways to make JEPA train a temporally-contextualized latent:

- **(i) Per-joint temporal (recommended):** restructure so the temporal module operates **per-joint before pooling** → per-joint hidden carries temporal context; JEPA targets these; pooling → z_t happens after. JEPA then trains the **full encoder** (spatial + temporal). Cost: temporal module runs over 22 joints×T instead of 6 slots×T (more compute); diverges from the slot-level locality work.
- **(ii) Pre-temporal target (simpler):** JEPA targets the pre-temporal per-joint hidden (trains only the spatial tokenizer); the temporal module + pooling → z_t is trained in Stage 2 with B1/B2. Cheaper, less invasive, but JEPA does **not** shape the temporal latent.

**Recommendation: (i)** — it makes z_t (the thing B1/B2 consume) actually a JEPA-shaped latent, consistent with "the encoder is the temporal compressor JEPA learns." Confirm at review; it sets the encoder refactor scope.

### 3.6 Deliverable

A frozen encoder producing z_t (dim D, per frame), validated by: non-collapse (feature std), and a frozen-encoder probe (future-z / contact). **No decoder in Module A.**

---

## 4. Module B1 — Flow-matching latent predictor (autoregressive rollout)

Built almost entirely by **reuse** of the repo's flow-matching stack, repointed from raw motion to z_t.

**Repo facts (audited):** the existing "AR predictor" (`HumanoidMDM_BPE`) is a **bidirectional** transformer; autoregression is realized **externally** by **prefix completion** — predict `pred_len` frames from a clean `context_len` prefix, slide, feed back (`utils/sampler_util.py:43-92`, `AutoRegressiveSampler.sample`). FM recipe: **velocity** prediction, Huber loss, **CondOT** linear interpolant, **beta** time sampling, **Blended Positional Encoding** (FlowMDM; APE↔RPE over the diffusion timestep). The backbone is **effectively a generic `[B,T,feat]` sequence model** — I/O are plain `Linear` projections (`HNInputProcess`/`HNOutputProcess`); `latent_dim=256`, `layers=8`, `heads=4`.

**Port to z_t `[B,T,D]`:** set `njoints=D, njoints_feats=1, nlinks=D, nlinks_feats=0`, trivial `sep_func`, `data_rep='dof'`; swap the I/O Linears to `Linear(D↔latent_dim)`; **disable the FK-dependent aux losses** (`lambda_target_loc`, `lambda_data_consistency`, `hn_get_target_location`) — they assume raw HML/URDF layout; keep velocity + smoothness/accel losses (layout-agnostic).

**Reuse surface:**
- `GBC_MUJICA.diffusion_planner.flow_matching.model_wrapper_bpe.HumanoidFlowModelBPE` (backbone + BPE), instantiated with `njoints=D, njoints_feats=1`.
- `…flow_matching.flow_wrapper.FlowMatchingWrapper` (FM loss/interpolant) + `HumanoidVelocityModel` + `p_sample_loop`.
- `…utils.sampler_util.AutoRegressiveSampler` (prefix rollout — operates on `[...,T]`, works unchanged on a latent sequence).
- `…model.bpe_components.BlendedPositionalEncoding` (pulled in automatically).

**Trained on the frozen z_t** produced by Module A. **Text conditioning** uses the existing prepended-token text hook (BERT/CLIP) + classifier-free guidance (`y['text']`, `cond_mask_prob`) — HumanML3D captions; this is how instruction-following AR rollout is enabled, and it is **B1's** concern, never Module A's.

---

## 5. Module B2 — Decoder (z_t → robot motion)

A decoder that reconstructs robot motion from z_t — "the only thing JEPA needs to prove itself." Two instantiations:
- **Reconstruction decoder:** z_t → SMPL-H/robot motion (generic), for latent-fidelity eval (round-trip).
- **Retargeting decoder (CMAD):** z_t + robot morphology → per-DoF actions — the existing `CrossMorphologyActionDecoder` path; this is also the **extrinsic transfer eval** (§6). Trained on GMR-G1 reference actions via the SFT trainer from the prior Plan 2.

B2 is per-robot; z_t is shared/frozen.

---

## 6. Evaluation (the old go/no-go, demoted)

- **Intrinsic (Module A):** non-collapse; frozen-probe future-z/contact beats untrained; **random vs slot masking** ablation (§3.3); latent round-trip fidelity via B2 reconstruction.
- **Generative (B1):** AR rollout quality (prefix-completion), distribution coverage on twist/null-space DOFs (the FM model should *not* collapse them to conditional mean — the failure of deterministic prediction in `latent-collapse-diagnosis`); text-conditioned rollout adherence.
- **Extrinsic transfer (B2 = CMAD):** the prior `scratch`/`mae`/`jepa` matrix on GMR-G1 — now one transfer check of the frozen JEPA latent, not the headline. MAE input-recon remains a useful baseline encoder.

---

## 7. Impact on the existing JEPA plans

- `2026-05-28-jepa-gonogo-ablation-design.md`: framing **superseded** by this doc (kept for audit). Its prerequisites P4 (canonicalization) and P5 (canonical-FPS) survive unchanged as input contracts.
- **Plan 1** (`…plan1-motion-jepa-module.md`): **substantially revised** — masking becomes per-joint tubelet (random vs slot ablation) and the predictor becomes a spatial masked-token completion transformer over the 22 joint tokens, replacing the full-grid refiner and the `future_only` temporal masking. Plan 1 Task 0 (locality defaults) interacts with §3.5 and the conflict in §9.
- **Plan 2** (`…plan2-sft-trainer.md`): becomes **Module B2** (the decoder / extrinsic-eval harness) — largely intact (canonicalization, G1 robot_batch, FK loss, SFT trainer).
- **Plan 3** (`…plan3-mae-and-matrix.md`): the MAE arm + matrix become the **extrinsic transfer eval** (§6), not the primary objective.
- **New plan needed:** Module B1 (FM latent predictor) — a thin wrapper over the reused flow-matching stack on frozen z_t.

---

## 8. Approach 1 (rejected alternative): fused temporal JEPA predictor

Make the single JEPA predictor also predict **future-frame** latents (add causal/future masking); the predictor becomes the latent dynamics model; rollout = recurse it, then decode. **Rejected because:**
- Deterministic feature-matching → **conditional-mean collapse on exactly the twist/null-space DOFs** already diagnosed (`latent-collapse-diagnosis`); a deterministic dynamics model bakes that collapse into every rollout. FM (B1) models the distribution and avoids it.
- Autoregressing against an **EMA-moving target** is unstable; long-horizon drift.
- **Reinvents** temporal prediction instead of reusing the repo's working FM/AR stack (§4).
- Kept on record because it is simpler (one model, one objective) and may be revisited if B1 rollout proves unnecessary.

---

## 9. Open items / conflicts to resolve before plans

> **Items 1–3 RESOLVED 2026-06-02** by grounding Stage 1 in the D4 baseline (`gbc_runs/_d4_config`). See `2026-06-02-jepa-stage1-implementation.md`.

1. **§3.5 temporal-module placement** — RESOLVED: **option (i) per-joint-temporal** (a `JointTemporalEncoder` before pooling), required for V-JEPA-faithful "mask joints, full temporal info visible." Stage-1 Fork B.
2. **`graph_layers` conflict** — RESOLVED: **`graph_layers=2`** (D4 uses 2; matches `locality-grid-results`). The Plan-1 addendum's `=1` is rejected.
3. **Latent dim D** — RESOLVED: **D = 336** (D4's structured `slot_z=48 × 7 slots`). B1's FM backbone `latent_dim` (its transformer width) stays independent; B1 I/O Linears map `336 ↔ latent_dim`.
4. **Mask-ratio band** for 21 joints (§3.3) — sweep target 0.4–0.7 (open hyperparameter).
5. **NEW — Fork A (encoder inputs):** add velocity + contact channels (physics-extendability) vs match D4's `none`. Recommended: add. See Stage-1 §3.

---

## 10. Self-review notes

- Scope: this is a **design/architecture** doc spanning three modules; each module (A, B1, B2) becomes its own implementation plan (writing-plans), per the scope-check. Module A is the first and is independent of real data.
- All repo claims are `file:line`-anchored from a code audit (encoder internals in `graph_retarget_latent.py`; FM stack in `GBC_MUJICA/diffusion_planner/`).
- Two decisions are deliberately left open (§9 items 1–2) because they need the user's call and/or an empirical check; everything else is committed.
