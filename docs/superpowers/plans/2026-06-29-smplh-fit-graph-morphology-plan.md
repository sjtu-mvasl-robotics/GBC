# SMPL-H Fit Graph Morphology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement decoder-side SMPL-H fit graph morphology conditioning, locally verify it, and submit a DLC smoke run when tests pass.

**Architecture:** Add a fixed SMPL-H fit graph spec built from each target robot fit, encode it with a small graph encoder, and zero-init fuse its tokens into the existing URDF MGE tokens. Add a learned-fit-graph root-height source that predicts standing height from the fit graph rather than consuming `rest_root_height`, while preserving the existing raw and legacy-rest-height paths.

**Tech Stack:** PyTorch modules in `GBC.utils.data_preparation.morphology_conditioned_retargeter`, GBC_MUJICA training contexts, pytest, DLC submit scripts.

---

### Task 1: Fit Graph Spec and Encoder

**Files:**
- Modify: `GBC/GBC/utils/data_preparation/morphology_conditioned_retargeter.py`
- Test: `GBC/tests/test_smplh_fit_graph_morphology.py`

- [x] **Step 1: Write failing tests**

```python
def test_build_smplh_fit_graph_spec_shapes_and_scale_sensitivity():
    body_model = FakeBodyModel()
    fit = {"beta": torch.zeros(1, 16), "dmpls": torch.zeros(1, 8), "scale": torch.tensor([1.0])}
    spec = build_smplh_fit_graph_spec(fit, body_model)
    assert spec.node_features.shape == (22, SMPLH_FIT_GRAPH_FEATURE_DIM)
    assert spec.hop_matrix.shape == (22, 22)
    fit2 = {**fit, "scale": torch.tensor([1.2])}
    spec2 = build_smplh_fit_graph_spec(fit2, body_model)
    assert not torch.allclose(spec.node_features, spec2.node_features)
```

Run:

```bash
PYTHONPATH=/cpfs/user/yaoyifei/codespace/gbc/GBC /cpfs/user/yaoyifei/miniconda3/envs/GBC/bin/python -m pytest GBC/tests/test_smplh_fit_graph_morphology.py -q
```

Expected: fail because the fit graph API does not exist.

- [x] **Step 2: Implement fit graph data path**

Add `SMPLHFitGraphSpec`, `build_smplh_fit_graph_spec`, `batch_smplh_fit_graph_specs`, `validate_smplh_fit_graph_path`, and `SMPLHFitGraphEncoder`. Use 22 SMPL-H body joints, fixed parent indices, fixed slot assignment, and feature dimension 10:

```python
node_feature = [
    root_relative_xyz(3),
    parent_bone_vector(3),
    bone_length(1),
    bone_ratio_to_neutral(1),
    is_root(1),
    is_leaf(1),
]
```

- [x] **Step 3: Verify green**

Run the same pytest command. Expected: tests pass.

### Task 2: Fuse Fit Graph Into Decoder Morphology

**Files:**
- Modify: `GBC/GBC/utils/data_preparation/morphology_conditioned_retargeter.py`
- Test: `GBC/tests/test_smplh_fit_graph_morphology.py`

- [x] **Step 1: Write failing tests**

```python
def test_fit_graph_zero_init_does_not_change_outputs():
    model = tiny_model(morph_fit_graph="smplh")
    robot_batch = tiny_robot_batch_with_fit_graph(scale=1.0)
    no_graph = tiny_robot_batch_without_fit_graph()
    pose, root, beta = tiny_motion()
    model.set_robot_batch(robot_batch)
    out_graph = model(pose, root, beta=beta, sample=False)
    model.morph_fit_graph = "none"
    model.set_robot_batch(no_graph)
    out_base = model(pose, root, beta=beta, sample=False)
    assert torch.allclose(out_graph["actions"], out_base["actions"], atol=1e-5)

def test_encoder_z_is_independent_of_target_fit_graph():
    model = tiny_model(morph_fit_graph="smplh")
    pose, root, beta = tiny_motion()
    model.set_robot_batch(tiny_robot_batch_with_fit_graph(scale=1.0))
    z1 = model(pose, root, beta=beta, sample=False)["z"]
    model.set_robot_batch(tiny_robot_batch_with_fit_graph(scale=1.3))
    z2 = model(pose, root, beta=beta, sample=False)["z"]
    assert torch.equal(z1, z2)
```

Expected: fail until `MotionLatentRetargeter` builds and consumes fit graph fields.

- [x] **Step 2: Implement zero-init fusion**

Add constructor args:

```python
morph_fit_graph: str = "none"
morph_fit_graph_feature_dim: int = SMPLH_FIT_GRAPH_FEATURE_DIM
morph_fit_graph_layers: int = 1
root_height_source: str = "raw"
```

Add `encode_morphology_full()` returning `m_per_dof`, `m_cls`, `fit_cls`, and `fit_per_slot`; keep `encode_morphology()` as the old two-tensor wrapper.

- [x] **Step 3: Verify green**

Run:

```bash
PYTHONPATH=/cpfs/user/yaoyifei/codespace/gbc/GBC /cpfs/user/yaoyifei/miniconda3/envs/GBC/bin/python -m pytest GBC/tests/test_smplh_fit_graph_morphology.py GBC/tests/test_morphology_conditioned_retargeter.py GBC/tests/test_root_kinematics.py -q
```

Expected: new tests and existing morphology/root tests pass.

### Task 3: Learned Fit-Graph Root Height

**Files:**
- Modify: `GBC/GBC/utils/data_preparation/morphology_conditioned_retargeter.py`
- Modify: `GBC_MUJICA/GBC_MUJICA/data_preparation/gbc_retargeting/train_graph_latent.py`
- Test: `GBC/tests/test_smplh_fit_graph_morphology.py`

- [x] **Step 1: Write failing tests**

```python
def test_learned_fit_graph_root_height_runs_without_rest_root_height():
    model = tiny_model(morph_fit_graph="smplh", root_height_source="learned_fit_graph")
    rb = tiny_robot_batch_with_fit_graph(scale=1.0)
    assert "rest_root_height" not in rb
    model.set_robot_batch(rb)
    out = model(*tiny_motion(), sample=False)
    assert torch.isfinite(out["root_height"]).all()
    assert "fit_stand_height" in out
```

Expected: fail until the learned root-height source is implemented.

- [x] **Step 2: Implement root-height source**

Add `fit_stand_height_head(fit_cls) -> positive z_stand_hat`, pass that as the factorized height to `RootDecoder`, and include `fit_stand_height` in the output dict. Preserve compatibility:

```python
if root_height_factorized and root_height_source == "raw":
    root_height_source = "legacy_rest_height"
```

Add `--root-height-source` and `--fit-stand-height-weight` to `train_graph_latent.py`. Add loss:

```python
fit_stand_height_loss = mse(output["fit_stand_height"], analytic_rest_height)
total += args.fit_stand_height_weight * fit_stand_height_loss
```

- [x] **Step 3: Verify green**

Run the same pytest command as Task 2. Expected: pass.

### Task 4: Training Context Wiring

**Files:**
- Modify: `GBC_MUJICA/GBC_MUJICA/data_preparation/gbc_retargeting/train_graph_latent.py`
- Modify: `GBC_MUJICA/GBC_MUJICA/data_preparation/gbc_retargeting/train_cotrain_motion_latent.py`
- Test: `GBC/tests/test_smplh_fit_graph_morphology.py`

- [x] **Step 1: Write failing tests**

```python
def test_deprecated_fit_path_is_rejected():
    with pytest.raises(ValueError, match="deprecated"):
        validate_smplh_fit_graph_path("/tmp/deprecated/smplh_fit_DEPRECATED.pt")
```

Expected: fail until validation helper is wired.

- [x] **Step 2: Build fit graph in robot contexts**

When `args.morph_fit_graph == "smplh"`, validate `base_cfg["smpl_fits_dir"]`, build a fit graph from the already loaded `fit` and `body_model`, batch it, and add these fields to `robot_batch`:

```python
fit_node_features
fit_slot_assignment
fit_hop_matrix
```

Pass new constructor args into `MotionLatentRetargeter`.

- [x] **Step 3: Verify green**

Run:

```bash
PYTHONPATH=/cpfs/user/yaoyifei/codespace/gbc/GBC:/cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA /cpfs/user/yaoyifei/miniconda3/envs/GBC/bin/python -m pytest GBC/tests/test_smplh_fit_graph_morphology.py -q
```

Expected: pass.

### Task 5: Local Smoke and DLC Submit

**Files:**
- Modify or create: `GBC_MUJICA/scripts_phaseA/submit_cotrain_fitgraph_dlc.sh` if no existing suitable submit script can be parameterized.

- [x] **Step 1: Run local unit suite**

```bash
PYTHONPATH=/cpfs/user/yaoyifei/codespace/gbc/GBC:/cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA /cpfs/user/yaoyifei/miniconda3/envs/GBC/bin/python -m pytest GBC/tests/test_smplh_fit_graph_morphology.py GBC/tests/test_morph_descriptor_injection.py GBC/tests/test_root_kinematics.py -q
```

Expected: pass.

- [x] **Step 2: Run import/build smoke**

```bash
PYTHONPATH=/cpfs/user/yaoyifei/codespace/gbc/GBC:/cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA /cpfs/user/yaoyifei/miniconda3/envs/GBC/bin/python -m GBC_MUJICA.data_preparation.gbc_retargeting.train_cotrain_motion_latent --help >/tmp/fitgraph_help.txt
```

Expected: exit 0 and help includes `--morph-fit-graph`.

- [ ] **Step 3: Commit implementation**

```bash
git -C GBC add GBC/utils/data_preparation/morphology_conditioned_retargeter.py tests/test_smplh_fit_graph_morphology.py docs/superpowers/plans/2026-06-29-smplh-fit-graph-morphology-plan.md
git -C GBC commit -m "feat: add SMPL-H fit graph morphology conditioning"
git -C GBC_MUJICA add GBC_MUJICA/data_preparation/gbc_retargeting/train_graph_latent.py GBC_MUJICA/data_preparation/gbc_retargeting/train_cotrain_motion_latent.py
git -C GBC_MUJICA commit -m "feat: wire SMPL-H fit graph morphology training"
```

- [ ] **Step 4: Submit DLC**

Use the existing co-training submit pattern with these new flags:

```bash
--morph-fit-graph smplh
--root-height-source learned_fit_graph
--fit-stand-height-weight 10.0
--morph-descriptor-kind none
```

Record the DLC job id, save dir, and submit command in the final response.
