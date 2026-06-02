# Experiment plan: validate + characterize the locality-fixed structured latent

**Date:** 2026-05-28
**Supersedes:** the `phaseA_20260528_ablation_v2/v3/v4` grids (those ran on the OLD
entangled global-`z` architecture and the foot/EE-weighting hypotheses, both now
understood — see `project_latent_collapse_diagnosis`).
**Depends on:** locality fix landed in `graph_retarget_latent.py` (see
`project_latent_locality_fix` and `2026-05-28-latent-locality-leak-fix-plan.md`).
**Runners:** `scripts_phaseA/launch_row.sh` (rewritten), `scripts_phaseA/submit_locality_grid.sh` (new).

---

## 0. What changed since the old grids (why these are new experiments)

The structured path now has, gated on `--latent-architecture structured`:
per-slot `z_slots` bottleneck (`z_dim` DERIVED = `7 * slot_z` = 336 @ slot_z 48,
incl. an **independent pelvis-only `root_state` slot**), per-slot temporal
encoder (`per_slot_temporal`, `root_global_hub=False`), routed+masked decoder
(`build_dof_slot_correspondence`, `--decoder-context`, `--decoder-root-slot-into-arms`),
root clique off (`--no-robot-dof-root-edges`), and the input-concat tokenizer fix.
New CLI: `--slot-z`, `--graph-layers`, `--decoder-context {masked_graph_hidden,none}`,
`--decoder-root-slot-into-arms`, `--robot-dof-root-edges`.

The old grids' headline questions (foot physics, EE-weighting, surface-match) are
**closed**: the twist-DOF collapse is conditional-mean collapse on an
input-unrecoverable target, not a leverage/physics-weight problem. The open
questions are now (1) does the new arch regress quality, (2) does restoring
locality recover any collapsed DOFs, (3) is the new latent good for downstream
(VLA/WAM) and FPS/window-robust.

## 1. Scientific questions

- **Q1 (no-regression):** does the locality-fixed structured arch retain tracked-DOF quality (hip/knee/shoulder/elbow `corr`, `mean_l2_m`) vs the old runs?
- **Q2 (the payoff):** does restoring locality move per-DOF `corr` for the previously-collapsed twist DOFs (ankle pitch/roll, wrists, shoulder-yaw)? This partitions Phase-2 scope: recovered ⇒ leak was the cause; still-collapsed ⇒ needs a learnable target.
- **Q3 (downstream-clean latent):** does `--decoder-context none` (pure `z_slots`, no skip around the bottleneck) keep `mean_l2_m` while improving latent quality (linearity R², active units)? This is the form a VLA/WAM policy would consume.
- **Q4 (locality strictness):** `--graph-layers 1` vs `2` — quality vs leakage (memory: 2 is already ~0 leak with `root_state`; 1 marginally cleaner).
- **Q5 (FPS/window robustness):** `--temporal-module causal_conv` vs `transformer` — window-placement consistency (E1e) and FPS-invariance (E1d). The constraints memory requires a per-frame, FPS-robust latent.

## 2. Validation ladder

### Phase 0 — pre-train gates (no GPU-hours; run FIRST)
| Gate | Command | Pass |
|---|---|---|
| Locality (Jacobian, random init + encoder+decoder) | `cd GBC && python -m pytest tests/test_latent_locality.py -q` | all pass; arm→hip & leg→shoulder ratio < 0.05 |
| Baseline untouched (golden/determinism) | same suite (`test_baseline_*`) | pass |
| Forward smoke (with & without geometry) | `python -m pytest GBC/tests/test_graph_retarget_latent.py -q` | 35/35 |

If any fail, STOP — the arch is broken; do not spend Phase-1 GPU-hours.

### Phase 1 — architecture ablation grid (the new experiments)
Launch with `scripts_phaseA/submit_locality_grid.sh "G0 G1 G2 G3 [G4 G5]"`.
All cells: 8 epochs, identical scaffolding from `launch_row.sh`; only the knob under test differs.

| Cell | Knob (EXTRA_FLAGS) | Question |
|---|---|---|
| `T0_control` | (defaults) | Q1 anchor + Q2 corr re-measure |
| `T1_glayers1` | `--graph-layers 1` | Q4 |
| `T2_ctx_none` | `--decoder-context none` | Q3 (downstream-clean latent) |
| `T3_causalconv` | `--temporal-module causal_conv` | Q5 |
| `T4_baseline` (opt) | `--latent-architecture baseline` | regression reference (OLD arch) |
| `T5_rootarms_off` (opt) | `--no-decoder-root-slot-into-arms` | root-into-arms ablation |

### Phase 1 — per-cell measurement (no extra training)
Run on each cell's best checkpoint (`.../T*/models/*best*epoch_8*.pt`):
1. **Training curves** (from `*.log` / `metrics.jsonl`): `val mean_l2_m`, `foot_mean_l2_m`, `weighted_mse` per epoch (plateau + level).
2. **Per-DOF corr/ratio** (Q2): `python scripts_phaseA/analyze_dof_predictability.py --cfg <cfg> --run-dir <cell-dir> --n-clips 12`.
3. **Latent quality** (Q3/Q5): `python -m GBC_MUJICA.data_preparation.gbc_retargeting.analyses.latent_diagnostics --ckpt <best.pt> --reference-actions-dir <ref> --output diag_<cell>.json` → E1a linearity R², E1b active units, E1d fps-invariance, E1e window-consistency.
4. **Locality holds after training**: re-run the Jacobian test loading the trained ckpt (not just random init).

## 3. Decision rules

- **G1 (no-regression, gate on T0):** `mean_l2_m ≤ 0.072` (old 0.0685, slack), tracked-DOF `corr ≥ 0.6`, no NaN. Fail ⇒ debug arch before interpreting anything else.
- **G2 (locality payoff, T0 vs old v3 per-DOF corr):** for each twist DOF, `corr` recovered (`>0.4`) ⇒ leak was the cause, done; still-collapsed (`<0.3`) ⇒ Phase-2 target. Old v3 reference: ankle pitch ≈ −0.14, wrist roll ≈ 0.04, shoulder-yaw ≈ 0.18.
- **Q3:** adopt `--decoder-context none` if `T2.mean_l2_m` within ~5% of T0 AND E1a/E1b improve — that's the latent we hand to VLA/WAM.
- **Q5:** adopt `causal_conv` if `T3` E1e window-consistency ≪ T0 with acceptable `mean_l2_m`.
- **Q4:** keep `graph_layers 2` unless T1 shows a real quality/leak win.

## 4. Phase 2 — objective fixes (gated on G2; one cell per surviving DOF class)
Only for DOFs that are STILL collapsed after the locality fix, built on the best Phase-1 config:
- **Ankle** (task #48): sole-orientation-vs-ground during stance — observable, determined target (`--latent-foot-surface-match-weight` already exists; reframe as orientation, not just position).
- **Wrist:** analogous hand-orientation target.
- **Elbow:** it tracks but under-shoots — anneal `--sample-latent` noise via KL schedule (`--kl-weight-target`) or add a variance-matching term.
- **Arm-1 recon:** raise `--latent-human-recon-weight` off 100.
Re-measure per-DOF `corr` after each; keep only changes that move the targeted DOF without regressing G1.

## 5. Notes / caveats
- `--init-from-teacher` only warm-starts the (unused-in-latent-output) `frame_model`; harmless. FK/physics losses stay on, so the no-FK⇒no-teacher rule does not trigger here.
- EE-weighting set to uniform (`--ref-action-ee-weight 1`) in the control to read locality's effect cleanly (EE-weighting is a known confound).
- Structured ckpts are NOT resumable from old runs (latent heads + decoder reshaped); all Phase-1 cells train from scratch.
- Pre-existing edge case (not introduced here): structured tokenizer NaNs on a fully-dropped limb; training uses the full body so it never triggers — do not enable part-dropout in these runs.
- GPU hygiene: the unrelated `gbc_temporal_first_pass_v2` jobs may occupy some GPUs; pass only free GPU ids to the submitter.

---

## 6. Execution update (2026-05-29): window 60 + DLC submission

The 2026-05-28 launch at **window 120** OOM'd. Root cause (confirmed by
`scripts_phaseA/mem_probe.py`): activation memory scales with `B*window` and
`compute_losses` retains ~3 full forward graphs/step (main + symmetry + noise);
params (14M=56MB) and the teacher (frame path off) are not the cause. Measured
peak CUDA mem at `slot_z48 / graph_layers2 / temporal_layers4`:

| config | bs16 k=3 | bs32 k=3 |
|---|---|---|
| **window 120** | 25.4 GB | **OOM** (>46) |
| **window 60**  | 13.0 GB | 25.8 GB |

Real (end-to-end) peak matches the probe: the window-60 bs32 smoke ran at
**25.4 GB**. So:

- **Window 60 is the new context length** for every cell (3 s @ 20 Hz — ample;
  motion is low-frequency, no up/downsample). `launch_row.sh` already pins
  `--window 60 --stride 30`.
- **DLC workers are `pj4090` = 24 GB.** window-60/bs32 (25.8 GB) OOMs a 4090, so
  the DLC grid uses **`--batch-size 16`** (~13 GB probe / ~14–15 GB real → safe).
  The local 46 GB card can still run bs32; only DLC needs bs16. This is the OOM
  fix carried onto the cluster: shrink window 120→60 **and** batch 32→16 for DLC.

### Runner: `scripts_phaseA/submit_locality_grid_dlc.sh`
DLC analogue of the local `submit_locality_grid.sh`. One single-GPU
`dlc submit pytorchjob` per cell; the cluster schedules them (no free-GPU-id
juggling). Differences from the local launcher, all forced by DLC:
1. `--batch-size 16` (24 GB budget).
2. Outputs → `/cpfs/user/yaoyifei/gbc_runs/phaseA_20260528_locality_dlc/T*` —
   **persistent**. The container's `/mnt/workspace` is ephemeral and is wiped
   when the job ends; checkpoints there would be lost.
3. Activates the cpfs `GBC` conda env (`/cpfs/user/yaoyifei/miniconda3`, torch
   2.9.0+cu128) over the mount — same env used locally.

Usage: `bash scripts_phaseA/submit_locality_grid_dlc.sh "0 1 2 3"` (core grid;
add `4 5` for the optional baseline + rootarms cells). `DRY_RUN=1` prints the
exact `dlc` command without submitting. Checkpoint footprint per run is small
(initial + per-improving-epoch best + final, ~0.2 GB each, ≈1 GB/run), so the
6-cell grid is ≈6 GB on cpfs.

### Pre-flight done (2026-05-29, do not redo)
- Phase 0 gates GREEN: `tests/test_latent_locality.py` 6/6, `tests/test_graph_retarget_latent.py` 2/2 (run via `python -m unittest`; the env has no pytest and is offline).
- Window-60 smoke (bs32, local): exit 0, finite losses, `reference_action_coverage=1.0`, peak 25.4 GB.
- DLC-config smoke (bs16, window 60, local): see run log; confirms the 24 GB budget before spending cluster hours.
