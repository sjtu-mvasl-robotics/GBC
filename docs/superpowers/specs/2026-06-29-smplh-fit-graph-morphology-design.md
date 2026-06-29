# SMPL-H Fit Graph Morphology Conditioning Design

**Date:** 2026-06-29
**Status:** approved design, ready for implementation plan
**Scope:** GBC motion-latent retargeter decoder-side morphology conditioning

---

## 1. Motivation

The current motion-latent retargeter has the correct high-level separation:

- the source-motion encoder is morphology-free;
- the decoder is morphology-conditioned.

The current morphology path is still too weak for the intended height-transfer
behavior. `MorphologyGraphEncoder` sees only URDF DoF-level static features, and
the optional fit-derived descriptor is a compact global vector (`bonelen` or
`betascale`) broadcast into those tokens. That path does not expose the target
robot's SMPL-H fit as a structured skeleton. Separately, the factorized root-z
path can consume `robot_batch["rest_root_height"]`, which makes standing height
an external scalar rather than something inferred from the target morphology.

The goal is to let the decoder infer scale and root height from the target
robot's SMPL-H fit graph while keeping the source-motion encoder free of
robot-fit leakage.

## 2. Non-Goals

- Do not feed target robot fit data into `MorphologyFreeMotionEncoder`.
- Do not continue the GR3 zero-shot descriptor ablation before this morphology
  contract is corrected.
- Do not require an external root-height scalar at inference.
- Do not remove the existing descriptor flags in the first implementation; keep
  them for compatibility and controlled ablation.
- Do not implement BVH support in the first patch, but design the graph provider
  interface so BVH skeleton offsets can use the same path later.

## 3. Current Contract

Current data flow:

```text
SMPL-H source motion
  -> MFME
  -> z_slots / graph_hidden

URDF DoF features
  -> MGE
  -> m_per_dof, m_cls

optional bonelen descriptor
  -> MorphDescriptorAdapter
  -> bias m_per_dof and/or m_cls

z_slots + m_per_dof -> CMAD -> robot actions
root_slot_z + m_cls + optional rest_root_height -> RootDecoder -> root_xyz
```

Problems:

- the fit-derived morphology signal is not graph-structured;
- `bonelen` loses joint-level layout and parent-child geometry;
- `m_cls` does not necessarily encode enough leg/torso structure to infer root
  height robustly;
- `root_height_factorized=True` currently requires a separate
  `rest_root_height` entry, which violates the desired inference contract.

## 4. Proposed Contract

Add a decoder-side target morphology graph from the robot's own SMPL-H fit:

```text
target robot SMPL-H fit
  -> SMPLHFitGraphSpec
  -> SMPLHFitGraphEncoder
  -> fit_per_joint, fit_per_slot, fit_cls

URDF DoF graph
  -> MorphologyGraphEncoder
  -> urdf_per_dof, urdf_cls

fusion:
  m_per_dof = urdf_per_dof + zero_init_fuse_dof(urdf_per_dof, fit_per_slot[slot_assignment])
  m_cls     = urdf_cls     + zero_init_fuse_cls(urdf_cls, fit_cls)

z_slots + m_per_dof -> CMAD -> robot actions
root_slot_z + m_cls -> RootDecoder -> root_xyz
```

The source-motion encoder still receives only source SMPL-H motion features.
The target robot fit enters only the decoder-side morphology path.

## 5. SMPLHFitGraphSpec

Introduce a small data structure that can be built once per robot context and
stored in `robot_batch`:

```python
SMPLHFitGraphSpec:
    node_features: FloatTensor[22, F_fit]
    parent_index: LongTensor[22]
    slot_assignment: LongTensor[22]
    hop_matrix: LongTensor[22, 22]
    joint_names: tuple[str, ...]
```

The initial SMPL-H provider builds the graph from:

- `fit["beta"]`;
- `fit["dmpls"]`;
- `fit["scale"]`;
- the existing `BodyModel`;
- the fixed SMPL-H body kinematic tree.

Node features:

- root-relative, scale-applied joint xyz;
- parent-relative bone vector xyz;
- bone length;
- bone length divided by neutral-human bone length;
- one-hot or embedding id for semantic body slot;
- root flag and leaf flag.

All geometry is target morphology only. It is constant across a training run for
one robot context.

## 6. SMPLHFitGraphEncoder

Add `SMPLHFitGraphEncoder` in `morphology_conditioned_retargeter.py` or a small
adjacent module if the file becomes unwieldy.

Encoder shape:

```python
fit_tokens = feature_proj(node_features) + slot_embed(slot_assignment)
fit_tokens = GraphormerLayer(..., hop_bias)(fit_tokens)
fit_cls = masked_mean_or_cls(fit_tokens)
fit_per_slot = scatter_mean(fit_tokens, slot_assignment)
```

Output:

- `fit_per_joint`: `(B, 22, d_model)`;
- `fit_per_slot`: `(B, S, d_model)`;
- `fit_cls`: `(B, d_model)`.

`fit_per_slot` is the primary bridge to robot DoFs. Each robot DoF already has a
`slot_assignment`; the decoder gathers the corresponding target-fit slot token
and fuses it into that DoF's URDF token.

## 7. Fusion

Fusion must preserve old checkpoint behavior at step 0.

Use residual zero-initialized MLPs:

```python
dof_delta = zero_init_mlp(torch.cat([urdf_per_dof, fit_slot_for_dof], dim=-1))
cls_delta = zero_init_mlp(torch.cat([urdf_cls, fit_cls], dim=-1))
m_per_dof = urdf_per_dof + dof_delta
m_cls = urdf_cls + cls_delta
```

With zero last-layer weights, loading a morph-free checkpoint and enabling the
fit graph path starts from the old behavior. The model can learn to use the fit
graph during training.

## 8. Root Height

Remove `rest_root_height` from the inference contract.

Recommended first implementation:

```text
fit_cls -> stand_height_head -> z_stand_hat
root_slot_z + m_cls -> root_motion_head -> h_t and xy_t
root_z = z_stand_hat * (1 + h_t)
root_xyz = [xy_t, root_z]
```

The analytic standing height computed from robot FK may still be used as a
training label:

```text
L_stand = mse(z_stand_hat, analytic_rest_height)
```

That label is not a model input and is not needed at render or inference time.
The model must be callable with a robot batch that contains fit graph fields but
does not contain `rest_root_height`.

Compatibility mode:

- keep `legacy_rest_height` as an explicit fallback for old experiments;
- make `learned_fit_graph` the path used by the new experiment.

## 9. Training Data Flow

For single-robot training:

1. Load source AMASS/SMPL-H motion normally.
2. Build root features with a fit-free mode such as `abs_z` or
   `abs_z_clearance`.
3. Build `robot_batch` from the target robot URDF as before.
4. Build `fit_graph_batch` from the target robot's SMPL-H fit and add it to
   `robot_batch`.
5. Set `model.set_robot_batch(robot_batch)`.

For G1/H1 cotrain:

1. Each `RobotContext` carries its own URDF batch and SMPL-H fit graph batch.
2. Before each step, `model.set_robot_batch(ctx.robot_batch)` selects the
   target morphology.
3. The source motion encoder path is unchanged and does not receive the target
   fit graph.

For validation/render:

1. The checkpoint reconstructs the model with `morph_fit_graph=smplh`.
2. The robot config points to the correct mesh-anchor SMPL-H fit.
3. The renderer builds the same fit graph and passes it through `set_robot_batch`.
4. No root-height scalar is required.

## 10. BVH Compatibility

The first implementation supports only SMPL-H fit graphs. The interface should
not be named so narrowly that BVH support becomes a rewrite.

Future BVH provider:

```text
BVH rest offsets + parent tree + joint names
  -> SkeletonMorphSpec
  -> same graph encoder
```

BVH lacks SMPL-H beta/dmpls but has rest offsets and parent-child skeleton
geometry. That is sufficient for limb proportions and standing-height cues.

## 11. Flags

Add flags:

- `--morph-fit-graph {none,smplh}` with default `none`;
- `--morph-fit-graph-layers`, default `1`;
- `--root-height-source {legacy_rest_height,learned_fit_graph}`, default
  `legacy_rest_height` for compatibility;
- `--fit-stand-height-weight`, default `0.0`.

Existing descriptor flags remain:

- `--morph-descriptor-kind`;
- `--morph-descriptor-inject`;
- `--morph-descriptor-target`.

The first target experiment should use:

```text
--morph-fit-graph smplh
--root-height-source learned_fit_graph
--fit-stand-height-weight > 0
--morph-descriptor-kind none
```

## 12. Error Handling

- If `--morph-fit-graph smplh` is enabled and the robot config lacks a usable
  `smpl_fits_dir`, fail during context construction.
- If the fit file is the deprecated joint-only H1 fit, fail with a clear error
  that points to the mesh-anchor fit requirement.
- If `root_height_source=learned_fit_graph` but the robot batch lacks fit graph
  fields, fail before the first forward pass.
- If `root_height_source=legacy_rest_height` but `rest_root_height` is missing,
  keep the current loud error.
- If a checkpoint was trained with fit graph enabled, render/eval must require
  the same graph provider unless an explicit compatibility override is passed.

## 13. Tests

Unit tests:

- build `SMPLHFitGraphSpec` from synthetic body-model output and check shapes;
- graph features are finite and root-relative;
- changing fit scale or beta changes fit graph tokens;
- zero-init fusion makes outputs match the old no-fit-graph path at step 0;
- encoder morph-freeness: same source motion produces identical `z` under two
  different target fit graphs;
- decoder sensitivity: after making fusion nonzero, two target fit graphs change
  actions and root xyz;
- `root_height_source=learned_fit_graph` forward runs without
  `rest_root_height`;
- `root_height_source=legacy_rest_height` keeps the old missing-key error;
- deprecated H1 joint-only fit is rejected by the SMPL-H graph provider.

Smoke tests:

- small CPU or single-GPU forward smoke for G1 and H1;
- one tiny train epoch with fit graph enabled and finite losses;
- render normal motion for G1 and H1 and inspect root height and contacts.

## 14. Experiment Plan

First implementation validation:

1. Unit tests and forward smoke.
2. G1/H1 local tiny cotrain smoke.
3. Render normal motion for G1 and H1 from the best available fit-graph ckpt.
4. Compare root-z mean/min/max, penetration, floating, and visual contact.

Only after G1/H1 are stable:

1. Run full G1/H1 cotrain with fit graph enabled.
2. Compare against the matched morph-free cotrain control.
3. Use GR3 as a held-out morphology evaluation after the in-distribution height
   contract is correct.

## 15. Acceptance Criteria

Implementation is acceptable when:

- MFME outputs are independent of target fit graph;
- new fit graph path starts as a zero-init no-op under old checkpoints;
- learned-fit-graph root mode no longer requires `rest_root_height` at inference;
- G1/H1 render does not show the severe floating seen before the H1 fit fix;
- fit graph enabled training logs finite losses and finite root-height metrics;
- old descriptor and legacy root-height paths remain runnable for comparison.

