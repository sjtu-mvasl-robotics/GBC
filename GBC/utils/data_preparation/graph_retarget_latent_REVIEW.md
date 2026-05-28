# Review: `graph_retarget_latent.py` (universal-latent framing)

**Target file:** `GBC/utils/data_preparation/graph_retarget_latent.py`
**Stated project goal:** learn a **universal motion representation latent for any [robot] architecture**, inspired by NVIDIA SONIC (arXiv 2511.07820)
**Current implementation reality:** single-robot (Unitree G1, 29 DoF) retargeter with multi-robot scaffolding that is plumbed but unused
**Date:** 2026-05-26

---

## TL;DR

**Goal–implementation gap is the headline issue, not the engineering details.**

The class is named `UnifiedLatentRetargeter` and exposes `robot_embedding_dim` / `num_robot_types`, but in practice it is a **G1-specific retargeter built around a frozen G1 teacher with G1 joint names hardcoded into the decoder graph**. None of the universal-latent scaffolding is actually used (`num_robot_types=1` everywhere, `robot_embedding_dim=0` by default, teacher trained on G1 only, no downstream policy consumes the latent).

The architectural choices that look reasonable for a single-robot retargeter are exactly the choices that **block** a universal latent:

1. **Teacher–residual coupling** — the latent is parameterized as a correction to a robot-specific teacher, so the latent space is *defined relative to G1 actions*, not relative to motion semantics.
2. **Robot-specific output head** — `RetargetDecoder` and `RobotGraphActionDecoder` produce `num_actions`-dimensional vectors with G1-hardcoded kinematic edges. Universality requires *decoders* to be morphology-conditioned, or the *latent* to be detached from any action space entirely.
3. **VAE bottleneck with action-reconstruction supervision** — SONIC explicitly avoids this; the universality story in SONIC comes from **FSQ-quantized tokens + multi-source input encoders + RL tracking**, not from a VAE trained on action MSE.

Also important: **SONIC's "universal" is NOT cross-robot.** SONIC is universal across *input modalities* (video, MoCap, VR, text, music, VLA) feeding *one* robot (G1). If "any architecture" means "any robot morphology", SONIC is the wrong lodestar — **AnyTop**, **URDF-conditioned decoders**, or **MaskedMimic**-style universal-conditioning are closer.

Recommendation: **fork the design**. Keep this file as the G1 retargeter (rename it `g1_latent_retargeter.py`), and stand up a separate universal-latent module whose latent is *defined morphology-free* and whose decoders are *plug-in*.

---

## SONIC reality check (matters for design choices)

What SONIC (NVIDIA GEAR, Nov 2025) actually does:

| Claim | Reality |
|---|---|
| "Universal latent" | Universal across **input sources** (video, MoCap, VR, text, music, VLA, robot tracking targets) — all map into one shared FSQ token space, targeting one robot. |
| Robot scope | **Unitree G1 only**. No cross-morphology evidence. |
| Bottleneck | **FSQ** (Finite Scalar Quantization, Mentzer 2023) — discrete tokens, no auxiliary losses, full codebook utilization. Not a VAE. |
| Training | **Single-stage PPO** (RL motion tracking in sim) + reconstruction + token-alignment loss. **No teacher–student distillation.** |
| Scale | 700 hrs MoCap, >100M frames @ 50 Hz, 1.2M → 42M params sweep, 9k GPU-hours on 128 GPUs. |
| The citable takeaway | **Scale > inductive bias.** A unified token interface + RL + lots of data beats clever architectures. |

**Implication for GBC.** If GBC genuinely wants to follow SONIC, the diff against current code is large: drop the VAE for FSQ, drop the distillation for end-to-end RL or richer self-supervision, add multi-source input encoders (not just SMPL-H), and aim for orders-of-magnitude more data. If GBC instead wants cross-robot universality, SONIC is not the right reference — **AnyTop** (per-skeleton-graph), **MaskedMimic**, and **PULSE** are.

---

## Critical gap analysis — what's universal in the code vs what isn't

### Universal already (good)
- **`HumanSlotTokenizer`** is morphology-agnostic on the **input side**: any SMPL-H pose maps to 6 fixed slot tokens. This is the right shape for a motion encoder.
- **`SkeletalJointTokenizer`** parameterizes the human skeleton via `joint_parents` and `slot_joints` dicts. With minor work it could ingest different human formats (markered MoCap, OpenPose 3D).
- **`FutureSlotHead`** predicts future slot states — robot-free supervision. Keep.
- **`slot_latents`** is exposed as a per-slot intermediate latent (`:1104`), which is the right kind of artifact to feed downstream policies.

### Coupled to G1 (blocking)
| Issue | Location | Why it blocks universality |
|---|---|---|
| Hardcoded joint names | `_build_robot_dof_edges`, `:529-568` | The G1 chain (`waist_yaw_joint`, `left_hip_pitch_joint`, …) is baked in. Any other robot silently falls back to a linear chain (`:594-595`). |
| Fixed action dimension in decoder | `RetargetDecoder.output_dim`, `:484-491`; `RobotGraphActionDecoder.output_dim`, `:639,644` | Both decoders emit a single fixed-dim vector; different robots need different output heads. |
| Robot conditioning is an embedding lookup, not a morphology spec | `robot_embed = nn.Embedding(num_robot_types, …)`, `:480, 646` | Treating robots as enum IDs cannot generalize to unseen morphologies. URDF-derived features (parent indices, axis types, limits) are what AnyTop / Implicit Kinodynamic Retargeting use. |
| Teacher trained on G1 only | `frame_model = PoseTransformer(num_actions=29)`, `:954-961`; `load_teacher(...)`, `:1176-1178` | The "base_actions" anchor is robot-specific. The latent residual is supervised relative to G1, so the latent's geometry is G1-shaped. |
| Action-reconstruction supervision implied by losses | `recon_pose`, `kl`, `phys_total` per the trainer | A latent trained mostly to reconstruct G1 actions will not be robot-agnostic, regardless of `robot_embedding_dim`. |
| `num_robot_types=1` everywhere | `UnifiedLatentRetargeter.__init__`, `:920-921`; never overridden in repo | Multi-robot path is dead code. |

### Coupled to retargeting (versus a true motion prior)
- The encoder consumes `pose_aa` *and* `root_pose` together (`:218-220`, `:1097-1099`). For a motion prior you typically want **canonicalized, root-relative** body kinematics in the encoder, with root trajectory handled by a separate stream (PHC, MaskedMimic, PULSE all do this).
- The encoder has no explicit canonicalization. Two clips of the same motion at different world rotations / translations will produce different latents. SONIC and PULSE canonicalize.

---

## Design directions

There are two reasonable interpretations of "universal motion representation latent for any architecture." They require **different** changes. Pick one.

### Interpretation A — "SONIC-style: many input sources → one robot"
This is what SONIC actually delivered. Best fit if the long-term goal is whole-body control of G1 from video/text/VR/MoCap.

**Required changes (high-effort):**

1. **Replace the VAE bottleneck with FSQ** (Mentzer 2023). Discrete tokens scale better, give a natural interface for cross-source alignment, and remove the KL/posterior-collapse failure mode entirely. Token-alignment loss across sources is what makes the latent "universal" in SONIC's sense.
2. **Add per-source input encoders** sharing the FSQ codebook: SMPL-H pose (this file), 3D keypoints (MoCap markers), 2D keypoints (lifted video via GENMO/HMR), text (CLIP/Stick), music (Encodec features), etc. Each gets a lightweight encoder that maps into the same token space.
3. **Drop the teacher–residual scheme** for the long-term path. SONIC trains end-to-end with PPO + reconstruction. The teacher is appropriate as a *bootstrap* (current Stage 1–2) but should not be the architecture's identity.
4. **Canonicalize root** before encoding. Root trajectory becomes a separate stream the decoder consumes.
5. **Scale data**. 100M frames is the actual lever. Architectural cleverness will not close the gap.

### Interpretation B — "cross-morphology: one input → any robot"
This is what the user's phrasing literally says, and what GBC seems to imply with `num_robot_types`. SONIC does not address this.

**Required changes (medium-effort):**

1. **Make decoders robot-conditional on a URDF, not an embedding ID.** Parse URDF → per-DoF feature vector (parent index, axis, type, limits, link length). Feed these as tokens to a *single* decoder. AnyTop and Implicit Kinodynamic Retargeting (2509.15443) do exactly this. Replace `_build_robot_dof_edges` with a URDF-driven builder.
2. **Make the encoder strictly morphology-free.** Train it only on human-side reconstruction targets: future poses, contacts, root trajectory, masked-pose recovery. No robot action target during pretraining.
3. **Robot-specific heads are lightweight adapters** trained on top of a frozen latent. This is the "motion prior → robot adapter" recipe from PULSE / MaskedMimic.
4. **Drop the G1 teacher from the latent pretraining**. It can come back as a baseline action path *per robot* in the adapter stage.
5. **Define the universal latent in canonical motion space** (global keypoint trajectories or local-frame joint rotations + root trajectory). It must be definable without referencing any robot.

### Pragmatic hybrid (recommended for now)
Given the current state (G1 retargeter working), spend the next milestone on **two reversible, decoupling refactors** that benefit either interpretation:

1. **Strip robot-specific knowledge out of the encoder.** The encoder, `mu_head`, `logvar_head`, `slot_projector`, and `FutureSlotHead` should compile and run with `num_actions=0`. Verify by constructing the encoder without any decoder.
2. **Move the G1 decoder behind a `RobotAdapter` interface.** Signature: `actions = adapter(z, beta, robot_spec)`, where `robot_spec` is a URDF-derived feature pack. The G1 adapter is then *one* implementation; a second robot needs only a new adapter + new URDF, not encoder changes.

These two changes preserve the current G1 results and unblock both Interpretation A and Interpretation B.

---

## Concerns from the previous review that still apply (engineering layer)

(Filed against the current G1 retargeter; still worth fixing regardless of universality.)

1. **Hard kinematic attention masks should be a curriculum**, not a permanent constraint. `_attention_mask_from_edges` (`:109-115`), `_edge_attention_mask` (`:76-97`). SONIC's empirical lesson: at scale, learned attention beats hard masks.
2. **Slot pooling collapses limb-internal detail.** `SkeletalJointTokenizer` uses unweighted mean over joints per slot (`:377`). Use attention pooling or keep a per-joint residual stream.
3. **Per-frame VAE on top of a temporal encoder** invites posterior collapse on high-frequency DOFs (wrists). Add free-bits or KL annealing if you see `latent_actions` regressing to the teacher.
4. **Residual is unregularized.** `residual_head` (`:1029-1038`) is zero-init but unbounded. Add an L2 penalty on `‖residual‖` and a small LR multiplier on `residual_head` (ASAP / FAST best practice).
5. **No foot-contact head.** "Retargeting Matters" (2510.02252) flags this as the cheapest documented foot-skating mitigation.
6. **`teacher_base_blend=1.0` default** means at init the latent path receives no gradient through `actions`. Confirm the trainer anneals it.
7. **Root canonicalization is absent.** For a motion prior, encode body-local kinematics; handle root trajectory separately.

---

## Reframed experiment order

**Phase 0 — Decouple (cheap, do first regardless of interpretation):**
1. Move the G1 decoder behind a `RobotAdapter` interface. Make the encoder compile without it.
2. Replace `_build_robot_dof_edges`'s hardcoded chains with URDF-parsed parent indices.
3. Add a pretraining objective on the encoder that does **not** see any robot: predict future slot states (already implemented), masked-pose recovery (new), foot contacts (new).

**Phase 1 — Pick interpretation:**

*If Interpretation A (SONIC-style multi-source → G1):*
4A. Swap the VAE for FSQ. Verify codebook utilization, drop the KL term.
5A. Add a second input encoder (e.g., 3D MoCap markers, or a video-lifted keypoint stream) and a token-alignment loss across sources.
6A. End-to-end fine-tune with PPO tracking on G1 once the FSQ token interface is stable.

*If Interpretation B (cross-morphology):*
4B. Train a second `RobotAdapter` (H1 or GR1) on top of the frozen encoder. The encoder must not change.
5B. Replace the per-robot `nn.Embedding` with URDF-feature tokens fed to a single shared decoder transformer.
6B. Measure cross-robot zero-shot transfer of the latent (encode SMPL-H, decode H1 / GR1 with their respective adapters).

**Phase 2 — Engineering hardening** (still useful):
7. Curriculum on attention masks; attention pooling on slots; residual L2; foot-contact head.

---

## Decision to make before any code change

| Question | Why it matters |
|---|---|
| Is the long-term target "any robot" (B) or "any input source for G1" (A)? | Determines whether the decoder must become URDF-conditioned (B) or the bottleneck must become FSQ + multi-encoder (A). |
| Do we have a multi-robot dataset, or will we generate one? | If no, Interpretation B is research-grade for now and Interpretation A is more tractable. |
| Will the latent feed an RL policy, or only offline-generated reference trajectories? | The current code only does the latter (no downstream consumer found). For a *universal latent* to be meaningful, at least one policy must consume it; otherwise you cannot validate universality. |

I'd recommend writing the answers explicitly into the existing plan doc (`docs/superpowers/plans/2026-05-15-retarget-gmr-root-latent-plan.md`) before further architectural work. The plan doc already cites SONIC for a future Stage 4 — the discussion above should determine what Stage 4 actually means.

---

## Key references (updated)

- [SONIC paper, arXiv 2511.07820](https://arxiv.org/abs/2511.07820) — multi-input universality, FSQ, single-stage PPO, G1-only
- [SONIC project page](https://nvlabs.github.io/SONIC/) and [GR00T-WholeBodyControl repo](https://github.com/NVlabs/GR00T-WholeBodyControl)
- [FSQ — Finite Scalar Quantization (Mentzer et al., 2309.15505)](https://arxiv.org/abs/2309.15505) — the bottleneck SONIC uses
- [PULSE — universal motion latent for humanoids (Luo et al., 2024)](https://arxiv.org/abs/2310.04582) — closest open-source structural ancestor
- [MaskedMimic (Tessler et al., 2024)](https://research.nvidia.com/labs/par/maskedmimic/) — universal conditioning into a motion prior
- [AnyTop — per-skeleton-graph motion model](https://anytop2025.github.io/Anytop-page/) — closer to true cross-morphology
- [Implicit Kinodynamic Retargeting (arXiv 2509.15443)](https://arxiv.org/abs/2509.15443) — skeleton-aware encoder + kinematic-aware pretraining
- [PHC / PHC+ (Luo et al.)](https://zhengyiluo.github.io/PHC/) — tracking-as-pretraining
- [ASAP (arXiv 2502.01143)](https://arxiv.org/abs/2502.01143) — the teacher-residual pattern the current code uses
- [Retargeting Matters / GMR (arXiv 2510.02252)](https://arxiv.org/abs/2510.02252) — foot-skating, root-height

---

## Bottom line

**Approvable as a single-robot retargeter for the G1 path. Not yet a universal motion latent in any meaningful sense.**

The most important next step is *not* an engineering fix to this file — it's a **product decision** about whether "universal" means SONIC-style multi-input on one robot, or AnyTop-style cross-morphology. Then refactor the encoder/decoder boundary so the universal piece can be trained and validated independently of any robot.

If forced to pick: the GBC repo already has multiple robot configs (G1, H1, H1-2, GR1) and no multi-input modality work. **Interpretation B (cross-morphology) is the more natural fit for the codebase**, even though SONIC is not the right inspiration for it. Lean on AnyTop / MaskedMimic / PULSE for that direction.
