# Retarget GMR SFT RLFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the current GBC retarget model by first offline-computing a GMR-retargeted SFT dataset, then training on that prepared dataset with explicit short-sequence smoothness/contact losses and a staged RLFT loop.

**Architecture:** Keep `PoseTransformer.forward(pose) -> actions` unchanged for deployment. GMR is used only in an offline data-preparation stage that produces a paired dataset of AMASS pose windows and retargeted robot actions; the trainer loads this dataset directly. RLFT is a separate second-stage pipeline that produces refined retargeted datasets or fine-tuned checkpoints with KL anchoring to the SFT model.

**Tech Stack:** PyTorch, existing `RobotKinematics`, existing AMASS loaders, existing IsaacLab reference pickle format, optional external GMR export files, pytest unit tests with tensor fixtures.

---

## Scope

This plan covers five deliverables:

1. Offline GMR retargeted dataset generation and validation.
2. SFT training from the prepared GMR-retargeted dataset while preserving FK, limit, symmetry, and disturbance losses.
3. Optional short-sequence training for velocity, acceleration, jerk, and contact consistency.
4. Offline metrics and conversion validation for comparing current, GMR-direct, SFT, sequence-SFT, and RLFT outputs.
5. RLFT loop design that starts from clip-level offline advantage-weighted regression before any expensive online PPO-style retarget training.

The plan intentionally does not replace the existing retarget network or downstream reference pickle schema. It adds optional paths that can be enabled per experiment.

## Modified Interfaces

### Interfaces Kept Stable

These interfaces must remain backward compatible:

- `GBC.utils.data_preparation.pose_transformer.PoseTransformer.forward(x: torch.Tensor) -> torch.Tensor`
- `GBC.utils.data_preparation.amass_action_converter.AMASSActionConverter.pose_to_action(pose: torch.Tensor) -> torch.Tensor`
- Existing converted reference pickle keys consumed by `RefObservationManager`: `trans`, `root_orient`, `actions`, `fps`, `cyclic_subseq`
- Existing `DATA_PATHS.pose_transformer_path` usage in `AMASSActionConverter`

### New Config Interfaces

Modify `GBC/utils/data_preparation/data_preparation_cfg.py`:

```python
@configclass
class RetargetSFTDatasetCfg(BaseCfg):
    retargeted_dataset_path: str = ""
    dataset_source: str = "none"  # one of: "none", "gmr"
    robot_name: str = ""
    require_dof_name_match: bool = True
    valid_only: bool = True

@configclass
class RetargetSequenceCfg(BaseCfg):
    window_size: int = 1
    stride: int = 1
    enable_temporal_losses: bool = False
```

Modify `LossCoefficients` in `GBC/utils/data_preparation/pose_transformer_trainer.py`:

```python
sft_action_loss: float = 0.0
sft_velocity_loss: float = 0.0
sft_acceleration_loss: float = 0.0
sft_jerk_loss: float = 0.0
temporal_velocity_loss: float = 0.0
temporal_acceleration_loss: float = 0.0
temporal_jerk_loss: float = 0.0
contact_consistency_loss: float = 0.0
sft_kl_anchor_loss: float = 0.0
```

Modify `PoseFormerTrainer.__init__` in `GBC/utils/data_preparation/pose_transformer_trainer.py`:

```python
sft_dataset_cfg: Optional[RetargetSFTDatasetCfg] = None
sequence_cfg: Optional[RetargetSequenceCfg] = None
```

Modify `PoseFormerTrainer.train(...)`:

```python
sft_weight_schedule: Optional[dict[str, float]] = None
```

The default behavior must match current training when `sft_dataset_cfg is None` and `sequence_cfg is None`.

### New Offline Retargeted SFT Dataset Format

Create a `.pt` or shard directory dataset with this schema. This file is produced before training; the trainer does not call GMR and does not perform late frame matching against a separate side table.

```python
{
    "schema_version": 1,
    "source": "gmr",
    "robot_name": "unitree_g1_29dof",
    "dof_names": list[str],
    "fps": int,
    "clips": {
        "relative/amass/path/file.npz": {
            "frame_indices": torch.Tensor,  # int64, shape (N,)
            "poses": torch.Tensor,          # float32, shape (N, 66 or 156), includes root orientation
            "actions": torch.Tensor,        # float32, shape (N, num_dofs)
            "valid_mask": torch.Tensor,     # bool, shape (N,)
            "root_trans": torch.Tensor,     # float32, shape (N, 3)
            "root_orient": torch.Tensor,    # float32 angle-axis, shape (N, 3)
            "feet_contact": torch.Tensor,   # float32 or bool, shape (N, num_feet)
            "metrics": dict[str, float],
        }
    }
}
```

The dataset is read-only during training. Regeneration belongs to the offline GMR dataset builder, not the trainer.

### New CLI Entry Points

Add scripts:

- `scripts/data_preparation/build_gmr_sft_dataset.py`
- `scripts/data_preparation/evaluate_retarget_outputs.py`
- `scripts/data_preparation/run_retarget_rlft_iteration.py`

All scripts must support `--help` and must accept paths rather than relying only on `DATA_PATHS`.

## File Structure

- Create `GBC/utils/data_preparation/retarget_sft_dataset.py`
  - Loads offline retargeted SFT datasets.
  - Validates schema, DOF names, tensor shapes, masks.
  - Returns frame-level and sequence-level paired samples directly from the prepared dataset.

- Create `GBC/utils/data_preparation/retarget_sequence_dataset.py`
  - Provides contiguous windows over the prepared retargeted SFT dataset.
  - Keeps frame-level SFT dataset behavior available.

- Create `GBC/utils/data_preparation/retarget_losses.py`
  - Implements SFT action loss and temporal finite-difference losses.
  - Keeps loss code out of the large trainer file.

- Create `GBC/utils/data_preparation/retarget_metrics.py`
  - Computes FK MPJPE, action velocity/acceleration/jerk, limit violation, foot slip, penetration/floating, contact consistency.

- Create `GBC/utils/data_preparation/gmr_adapter.py`
  - Converts GMR exports and AMASS pose files into the offline retargeted SFT dataset schema.
  - Handles DOF order remapping and root orientation conversion.

- Modify `GBC/utils/data_preparation/amass_loader.py`
  - Add optional metadata-preserving dataset mode.
  - Do not change default return type.

- Modify `GBC/utils/data_preparation/pose_transformer_trainer.py`
  - Wire SFT datasets, SFT action loss, sequence loss, metrics logging.
  - Keep current training path unchanged when new configs are disabled.

- Modify `GBC/utils/data_preparation/data_preparation_cfg.py`
  - Add SFT dataset and sequence config classes.

- Create tests under `tests/data_preparation/`.

## Task 1: Offline Retargeted SFT Dataset Contract

**Files:**
- Create: `GBC/utils/data_preparation/retarget_sft_dataset.py`
- Create: `tests/data_preparation/test_retarget_sft_dataset.py`

- [ ] **Step 1: Write failing dataset schema tests**

Create `tests/data_preparation/test_retarget_sft_dataset.py` with tests that build a minimal prepared dataset in memory and verify:

```python
def test_retarget_sft_dataset_validates_dof_names(tmp_path):
    dataset_path = tmp_path / "gmr_sft_dataset.pt"
    torch.save({
        "schema_version": 1,
        "source": "gmr",
        "robot_name": "unitree_g1_29dof",
        "dof_names": ["left_hip_pitch_joint", "right_hip_pitch_joint"],
        "fps": 50,
        "clips": {
            "ACCAD/example.npz": {
                "frame_indices": torch.tensor([0, 10]),
                "poses": torch.zeros(2, 66),
                "actions": torch.zeros(2, 2),
                "valid_mask": torch.tensor([True, False]),
                "root_trans": torch.zeros(2, 3),
                "root_orient": torch.zeros(2, 3),
                "feet_contact": torch.zeros(2, 2),
                "metrics": {"fk_mpjpe": 0.01},
            }
        },
    }, dataset_path)
    dataset = RetargetSFTDataset(dataset_path, expected_dof_names=["left_hip_pitch_joint", "right_hip_pitch_joint"])
    sample = dataset[0]
    assert sample["pose"].shape == (66,)
    assert sample["actions"].shape == (2,)
    assert bool(sample["valid_mask"])
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
PYTHONPATH=. pytest tests/data_preparation/test_retarget_sft_dataset.py -q
```

Expected: import failure for `RetargetSFTDataset`.

- [ ] **Step 3: Implement `RetargetSFTDataset`**

Implement:

```python
@dataclass(frozen=True)
class RetargetSFTSample:
    pose: torch.Tensor
    actions: torch.Tensor
    valid_mask: torch.Tensor
    root_trans: torch.Tensor
    root_orient: torch.Tensor
    feet_contact: torch.Tensor
    clip_key: str
    frame_idx: int

class RetargetSFTDataset(Dataset):
    def __init__(self, path: str | os.PathLike, expected_dof_names: Optional[list[str]] = None, valid_only: bool = True):
        ...
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        ...
    def get_sequence(self, clip_key: str, start_frame_idx: int, window_size: int, stride: int = 1) -> dict[str, torch.Tensor | str]:
        ...
```

The dataset builds its sample index at load time from the offline file. `valid_only=True` excludes invalid frames from `__getitem__`; sequence windows must stay inside one clip and preserve invalid-frame masks for loss masking.

- [ ] **Step 4: Run SFT dataset tests**

Run:

```bash
PYTHONPATH=. pytest tests/data_preparation/test_retarget_sft_dataset.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add GBC/utils/data_preparation/retarget_sft_dataset.py tests/data_preparation/test_retarget_sft_dataset.py
git commit -m "feat: add offline retarget SFT dataset contract"
```

## Task 2: Sequence Windows Over Prepared Retargeted Dataset

**Files:**
- Create: `GBC/utils/data_preparation/retarget_sequence_dataset.py`
- Create: `tests/data_preparation/test_retarget_sequence_dataset.py`

- [ ] **Step 1: Add tests for prepared sequence windows**

Use a fake prepared retargeted dataset with two clips. Test that windows never cross clip boundaries and return:

```python
{
    "poses": torch.Tensor,          # shape (T, pose_dim)
    "actions": torch.Tensor,        # shape (T, num_dofs)
    "valid_mask": torch.Tensor,     # shape (T,)
    "root_trans": torch.Tensor,     # shape (T, 3)
    "root_orient": torch.Tensor,    # shape (T, 3)
    "feet_contact": torch.Tensor,   # shape (T, num_feet)
    "clip_key": str,
    "frame_indices": torch.Tensor,
}
```

- [ ] **Step 2: Add `RetargetSFTSequenceDataset`**

Create `RetargetSFTSequenceDataset` with signature:

```python
class RetargetSFTSequenceDataset(Dataset):
    def __init__(self, sft_dataset: RetargetSFTDataset, window_size: int, stride: int):
        ...
    def __getitem__(self, idx: int) -> dict[str, Any]:
        ...
```

The wrapper must use the clip-level tensors already stored in `RetargetSFTDataset`; it must not reload AMASS files and must not call GMR.

- [ ] **Step 3: Run sequence dataset tests**

Run:

```bash
PYTHONPATH=. pytest tests/data_preparation/test_retarget_sequence_dataset.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add GBC/utils/data_preparation/retarget_sequence_dataset.py tests/data_preparation/test_retarget_sequence_dataset.py
git commit -m "feat: add sequence windows for retarget SFT datasets"
```

## Task 3: Offline GMR Dataset Builder

**Files:**
- Create: `GBC/utils/data_preparation/gmr_adapter.py`
- Create: `scripts/data_preparation/build_gmr_sft_dataset.py`
- Create: `tests/data_preparation/test_gmr_adapter.py`

- [ ] **Step 1: Write adapter tests**

Use fake AMASS poses and fake GMR retarget results with a different DOF order. Verify output action columns are reordered to GBC FK order and the saved dataset contains both `poses` and retargeted `actions`.

Expected public functions:

```python
def build_sft_dataset_from_gmr(
    amass_root: str,
    gmr_root: str,
    output_path: str,
    expected_dof_names: list[str],
    robot_name: str,
    fps: int,
) -> dict[str, Any]:
    ...

def remap_actions_by_dof_names(actions: torch.Tensor, source_dof_names: list[str], target_dof_names: list[str]) -> torch.Tensor:
    ...
```

- [ ] **Step 2: Implement adapter**

The adapter must:

- Load GMR outputs from `.pt`, `.pkl`, or `.npz`.
- Load the corresponding AMASS pose sequence once during dataset preparation.
- Convert root quaternions to angle-axis when needed.
- Validate action shape `(T, num_dofs)`.
- Create `valid_mask` from finite actions and joint-limit checks.
- Save exactly the offline retargeted SFT dataset schema defined above.

- [ ] **Step 3: Add CLI**

`scripts/data_preparation/build_gmr_sft_dataset.py` arguments:

```bash
--amass-root
--gmr-root
--output-path
--urdf-path
--robot-name
--fps
--strict-dof-names
```

The CLI loads `RobotKinematics(urdf_path).dof_names` as target order.

- [ ] **Step 4: Run adapter tests**

Run:

```bash
PYTHONPATH=. pytest tests/data_preparation/test_gmr_adapter.py -q
python scripts/data_preparation/build_gmr_sft_dataset.py --help
```

Expected: tests pass and help text lists all arguments.

- [ ] **Step 5: Commit**

```bash
git add GBC/utils/data_preparation/gmr_adapter.py scripts/data_preparation/build_gmr_sft_dataset.py tests/data_preparation/test_gmr_adapter.py
git commit -m "feat: add offline GMR SFT dataset builder"
```

## Task 4: SFT and Temporal Losses

**Files:**
- Create: `GBC/utils/data_preparation/retarget_losses.py`
- Modify: `GBC/utils/data_preparation/pose_transformer_trainer.py`
- Create: `tests/data_preparation/test_retarget_losses.py`

- [ ] **Step 1: Write loss tests**

Tests cover:

- `sft_action_loss` respects `valid_mask`.
- finite-difference velocity uses `a[t] - a[t-1]`.
- acceleration uses `v[t] - v[t-1]`.
- jerk uses `acc[t] - acc[t-1]`.
- all losses return zero for sequences shorter than the required order.

- [ ] **Step 2: Implement loss functions**

Public functions:

```python
def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 0.05) -> torch.Tensor: ...
def finite_difference(x: torch.Tensor, order: int, dt: float = 1.0) -> torch.Tensor: ...
def sft_action_loss(actions: torch.Tensor, target_actions: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor: ...
def temporal_smoothness_losses(actions: torch.Tensor, fps: float) -> dict[str, torch.Tensor]: ...
def sft_temporal_losses(actions: torch.Tensor, target_actions: torch.Tensor, valid_mask: torch.Tensor, fps: float) -> dict[str, torch.Tensor]: ...
```

- [ ] **Step 3: Wire trainer logging keys**

Add metrics:

```python
sft_action_loss
sft_velocity_loss
sft_acceleration_loss
sft_jerk_loss
temporal_velocity_loss
temporal_acceleration_loss
temporal_jerk_loss
sft_valid_ratio
```

All values must be logged in both train and validation when SFT-dataset or sequence mode is enabled.

- [ ] **Step 4: Run loss tests**

Run:

```bash
PYTHONPATH=. pytest tests/data_preparation/test_retarget_losses.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add GBC/utils/data_preparation/retarget_losses.py GBC/utils/data_preparation/pose_transformer_trainer.py tests/data_preparation/test_retarget_losses.py
git commit -m "feat: add SFT and temporal retarget losses"
```

## Task 5: Trainer Integration

**Files:**
- Modify: `GBC/utils/data_preparation/data_preparation_cfg.py`
- Modify: `GBC/utils/data_preparation/pose_transformer_trainer.py`
- Create: `tests/data_preparation/test_pose_former_trainer_sft.py`

- [ ] **Step 1: Write trainer wiring tests**

Use small fake datasets and monkeypatch `RobotKinematics` and `BodyModel` so tests do not require Isaac or SMPL assets. Verify:

- default constructor path still uses AMASS and does not require a prepared SFT dataset.
- SFT path instantiates `RetargetSFTDataset`.
- batch data includes `target_actions` and `valid_mask` from the prepared dataset.
- loss coefficient default for SFT losses is zero.

- [ ] **Step 2: Add config classes**

Add `RetargetSFTDatasetCfg` and `RetargetSequenceCfg` as listed in the interface section.

- [ ] **Step 3: Update trainer constructor**

Store:

```python
self.sft_dataset_cfg = sft_dataset_cfg
self.sequence_cfg = sequence_cfg
self.sft_dataset = None
```

When `sft_dataset_cfg.dataset_source == "gmr"`, load `RetargetSFTDataset` with expected DOF names from `self.fk.dof_names`.

- [ ] **Step 4: Update dataloader construction**

If `sft_dataset_cfg.retargeted_dataset_path` is set, instantiate `RetargetSFTDataset` instead of `InMemoryAMASSDatasetSingleFrame`. If sequence mode is enabled, wrap it with `RetargetSFTSequenceDataset`.

- [ ] **Step 5: Update train and validate loops**

Add helper:

```python
def _extract_training_batch(self, batch: Any) -> dict[str, Any]:
    ...
```

It normalizes current tensor batches, metadata batches, and sequence batches into:

```python
{
    "pose": torch.Tensor,
    "pose_body": torch.Tensor,
    "pose_hand": Optional[torch.Tensor],
    "target_actions": Optional[torch.Tensor],
    "valid_mask": Optional[torch.Tensor],
    "fps": Optional[float],
}
```

- [ ] **Step 6: Run trainer tests**

Run:

```bash
PYTHONPATH=. pytest tests/data_preparation/test_pose_former_trainer_sft.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add GBC/utils/data_preparation/data_preparation_cfg.py GBC/utils/data_preparation/pose_transformer_trainer.py tests/data_preparation/test_pose_former_trainer_sft.py
git commit -m "feat: integrate prepared SFT dataset in retarget trainer"
```

## Task 6: Retarget Metrics and Offline Evaluation

**Files:**
- Create: `GBC/utils/data_preparation/retarget_metrics.py`
- Create: `scripts/data_preparation/evaluate_retarget_outputs.py`
- Create: `tests/data_preparation/test_retarget_metrics.py`

- [ ] **Step 1: Write metric tests**

Test deterministic tensors for:

- mean and max FK position error.
- action velocity, acceleration, jerk norms.
- joint-limit violation ratio and magnitude.
- foot slip during contact.
- penetration and floating ratios.

- [ ] **Step 2: Implement metrics**

Public API:

```python
@dataclass
class RetargetMetrics:
    fk_mpjpe: float
    fk_max_error: float
    action_vel_mean: float
    action_acc_mean: float
    action_jerk_mean: float
    joint_limit_violation_ratio: float
    foot_slip_mean: float
    penetration_ratio: float
    floating_ratio: float
    contact_switch_rate: float

def compute_retarget_metrics(...)-> RetargetMetrics:
    ...
```

- [ ] **Step 3: Add evaluation CLI**

`scripts/data_preparation/evaluate_retarget_outputs.py` compares:

```bash
--current-model-path
--retargeted-dataset-path
--dataset-root
--secondary-dir
--urdf-path
--smplh-model-path
--dmpls-model-path
--smpl-fits-path
--output-json
```

It writes one JSON with per-clip and aggregate metrics.

- [ ] **Step 4: Run tests and CLI help**

Run:

```bash
PYTHONPATH=. pytest tests/data_preparation/test_retarget_metrics.py -q
python scripts/data_preparation/evaluate_retarget_outputs.py --help
```

Expected: tests pass and CLI help prints the listed arguments.

- [ ] **Step 5: Commit**

```bash
git add GBC/utils/data_preparation/retarget_metrics.py scripts/data_preparation/evaluate_retarget_outputs.py tests/data_preparation/test_retarget_metrics.py
git commit -m "feat: add retarget offline metrics"
```

## Task 7: Sequence-SFT Experiment Entrypoint

**Files:**
- Create: `scripts/data_preparation/train_retarget_gmr_sft.py`
- Modify: `GBC/utils/data_preparation/pose_transformer_trainer.py`

- [ ] **Step 1: Add train script**

CLI arguments:

```bash
--dataset-root
--secondary-dir
--retargeted-dataset-path
--urdf-path
--smplh-model-path
--dmpls-model-path
--smpl-fits-path
--save-dir
--device
--batch-size
--sample-steps
--window-size
--sft-action-weight
--sft-velocity-weight
--sft-acceleration-weight
--sft-jerk-weight
--temporal-velocity-weight
--temporal-acceleration-weight
--temporal-jerk-weight
--epochs
--lr
--min-lr
```

- [ ] **Step 2: Add SFT loss schedule support**

Implement `sft_weight_schedule` as a simple epoch multiplier:

```python
def _sft_schedule_multiplier(epoch: int, total_epochs: int, schedule: Optional[dict[str, float]]) -> float:
    ...
```

Supported keys:

```python
{"start": 1.0, "end": 0.3, "warmup_epochs": 0}
```

- [ ] **Step 3: Smoke-run help**

Run:

```bash
python scripts/data_preparation/train_retarget_gmr_sft.py --help
```

Expected: all arguments listed.

- [ ] **Step 4: Commit**

```bash
git add scripts/data_preparation/train_retarget_gmr_sft.py GBC/utils/data_preparation/pose_transformer_trainer.py
git commit -m "feat: add GMR SFT retarget training entrypoint"
```

## Task 8: RLFT Iteration Loop

**Files:**
- Create: `GBC/utils/data_preparation/retarget_rlft.py`
- Create: `scripts/data_preparation/run_retarget_rlft_iteration.py`
- Create: `tests/data_preparation/test_retarget_rlft.py`

- [ ] **Step 1: Write RLFT unit tests**

Test:

- clip reward normalization.
- advantage-weighted sample weights.
- KL anchor loss against frozen SFT actions.
- invalid candidates receive zero weight.

- [ ] **Step 2: Implement offline RLFT primitives**

Public API:

```python
@dataclass
class RetargetCandidate:
    clip_key: str
    actions: torch.Tensor
    log_prob: torch.Tensor
    valid_mask: torch.Tensor
    metrics: dict[str, float]

def compute_clip_reward(metrics: dict[str, float], weights: dict[str, float]) -> float: ...
def compute_advantage_weights(rewards: torch.Tensor, beta: float, max_weight: float) -> torch.Tensor: ...
def kl_anchor_loss(actions: torch.Tensor, anchor_actions: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor: ...
```

- [ ] **Step 3: Add RLFT CLI skeleton with real file IO**

`run_retarget_rlft_iteration.py` arguments:

```bash
--sft-model-path
--retargeted-dataset-path
--metrics-json
--output-dataset-path
--output-model-path
--beta
--max-weight
--kl-weight
```

The first implementation updates an output retargeted dataset by weighting candidate actions from evaluation metrics. It must not require live IsaacLab rollout in the unit-test path.

- [ ] **Step 4: Run RLFT tests and help**

Run:

```bash
PYTHONPATH=. pytest tests/data_preparation/test_retarget_rlft.py -q
python scripts/data_preparation/run_retarget_rlft_iteration.py --help
```

Expected: tests pass and CLI help prints the listed arguments.

- [ ] **Step 5: Commit**

```bash
git add GBC/utils/data_preparation/retarget_rlft.py scripts/data_preparation/run_retarget_rlft_iteration.py tests/data_preparation/test_retarget_rlft.py
git commit -m "feat: add offline retarget RLFT primitives"
```

## Task 9: End-to-End Experiment Protocol

**Files:**
- Create: `docs/retarget_gmr_rlft_protocol.md`
- Create: `scripts/data_preparation/run_retarget_ablation.sh`

- [ ] **Step 1: Document fixed experiment matrix**

Include these rows:

- M0 current PoseTransformer.
- M1 GMR-direct converted reference.
- M2 GMR-SFT.
- M3 GMR-SFT plus sequence losses.
- M4 M3 plus RLFT dataset/model iteration.

- [ ] **Step 2: Add ablation shell script**

The script must run commands in order and write outputs under one experiment root:

```bash
EXP_ROOT=${EXP_ROOT:-outputs/retarget_gmr_rlft}
```

The script stages:

```bash
python scripts/data_preparation/build_gmr_sft_dataset.py ...
python scripts/data_preparation/train_retarget_gmr_sft.py ...
python scripts/data_preparation/evaluate_retarget_outputs.py ...
python scripts/data_preparation/run_retarget_rlft_iteration.py ...
```

- [ ] **Step 3: Run shell syntax check**

Run:

```bash
bash -n scripts/data_preparation/run_retarget_ablation.sh
```

Expected: no output and exit code 0.

- [ ] **Step 4: Commit**

```bash
git add docs/retarget_gmr_rlft_protocol.md scripts/data_preparation/run_retarget_ablation.sh
git commit -m "docs: add retarget GMR RLFT experiment protocol"
```

## Validation Gates

Before starting full training:

```bash
PYTHONPATH=. pytest tests/data_preparation -q
python scripts/data_preparation/build_gmr_sft_dataset.py --help
python scripts/data_preparation/train_retarget_gmr_sft.py --help
python scripts/data_preparation/evaluate_retarget_outputs.py --help
python scripts/data_preparation/run_retarget_rlft_iteration.py --help
```

Expected:

- Unit tests pass.
- Help commands print argument lists.
- No import-time dependency on IsaacLab for pure data-preparation tests.

Before accepting M2:

- `sft_valid_ratio >= 0.95` on selected train split.
- Offline FK MPJPE not worse than M0 by more than 5 percent.
- action jerk lower than M0 or equal within 5 percent.
- joint-limit violation ratio lower than M0.

Before accepting M3:

- action acceleration and jerk lower than M2.
- cyclic boundary action jump lower than M2.
- no degradation in FK MPJPE beyond 5 percent.

Before accepting M4:

- IsaacLab tracking success/fall rate improves against M3 on the same reference set.
- KL-to-M3 stays below configured threshold.
- foot slip and penetration do not regress.

## Review Points

Please review these decisions before implementation:

1. Prepared SFT dataset schema path key: use AMASS relative `.npz` path as the canonical clip key.
2. Keep `PoseTransformer.forward` single-frame for deployment; use sequence windows only in training.
3. Use GMR only in the offline dataset builder; trainer loads the prepared SFT dataset and never calls GMR.
4. Start RLFT with offline advantage-weighted regression before online retarget PPO.
5. Add tests under a new `tests/data_preparation/` tree.
