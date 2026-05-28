# Plan: fix latent locality leak in UnifiedLatentRetargeter (structured path)

**Date:** 2026-05-28
**Target file (model):** `GBC/GBC/utils/data_preparation/graph_retarget_latent.py`
**Target file (trainer):** `GBC_MUJICA/GBC_MUJICA/data_preparation/gbc_retargeting/train_graph_latent.py`
**Scope:** `latent_architecture == "structured"` only. The `baseline` path is left byte-for-byte unchanged.
**Status:** D (input concat) LANDED 2026-05-28; A0 (encoder slot-locality, codex #1/#5) added as prerequisite; A–C (per-slot bottleneck + routed/masked decoder + root-clique flag) designed, reviewer questions resolved (§8), KL scale fixed (codex #2), unknown-DOF routing specified (codex #3), stale flagged-question rewritten (codex #4); ready to implement A0→A→C→B; E (window consistency) metric specced.

---

## 1. Problem (evidence-backed)

Two diagnoses from this session:

- **Leak (this plan's primary target):** a disturbance on a human shoulder changes robot **hip pitch/roll** output. Root cause is structural, not training:
  1. The VAE bottleneck is **global**: `encode_latent` does `pooled = _masked_mean(graph_hidden, dim=2)` (`:1247`) then `mu/logvar` from `pooled` → `z` is one entangled `(B,T,256)` vector over all 6 body slots. The per-slot `slot_latents` (`slot_projector`, `:1251`) are computed but **never decoded**.
  2. The structured decoder re-injects that global signal into **every** DOF token: `x = x + self.z_embed(z).unsqueeze(2)` (`RobotGraphActionDecoder.forward:722`) and an **unmasked** cross-attention `cross_attn(flat, ctx, ctx)` with `ctx = graph_hidden` over all slots (`:737`, called with `context=graph_hidden` at `:1301-1302`).
  3. The kinematic-tree self-attention mask (`attn_mask_template` from `robot_dof_edges`, `:740-744`) is applied **after** (1) and (2), so it cannot un-mix; it also includes a **root clique** (`fk.get_dof_edges(include_root_edges=True)`, trainer `:1500`) wiring hip↔waist↔hip.

- **Collapse (separate, Phase 2):** twist/null-space DOFs (ankle pitch/roll, wrists, shoulder-yaw) have `corr(pred,GMR) ≈ 0` even under a dominant 5000-weight action-L2 (they sit at the conditional mean); the elbow is tracked but amplitude-attenuated (~47%). See `project_latent_collapse_diagnosis` memory and `scripts_phaseA/analyze_dof_predictability.py`.

This plan fixes the **leak/locality** defect first because it contaminates every DOF (even well-tracked ones) and is a prerequisite for cleanly attributing residual collapse.

## 2. Goals / non-goals

**Goals**
- A robot DOF's output depends only on its kinematically-relevant human region (+ shared root), not on unrelated regions. Verified by a Jacobian acceptance test.
- The bottleneck `z` becomes **structured per-slot**; the discarded `slot_latents` idea becomes the actual latent.
- Preserve current G1 retargeting quality on tracked DOFs (no regression in `mean_l2_m`, tracked-DOF `corr`).

**Non-goals (this plan)**
- The collapse fix (sole-orientation / hand-orientation objectives, elbow de-smoothing, recon-weight) — specified as Phase 2, gated on re-measurement.
- Multi-robot / FSQ / universality refactors (separate effort; see `graph_retarget_latent_REVIEW.md`).

## 3. Design overview

One structured latent, three changes, all gated on `latent_architecture == "structured"`:

- **A. Per-slot bottleneck.** `mu/logvar` per slot token → `z_slots (B,T,S,slot_z)`. For global consumers (human recon, root-height, residual, spectral, world, KL) expose `z_global = z_slots.reshape(B,T,S*slot_z)`; set `z_dim = S * slot_z`. KL is computed per-slot.
- **B. Routed + masked action decoder.** `RobotGraphActionDecoder` consumes `z_slots`: each DOF token is seeded only from its **region slot latent + root slot latent** (no global broadcast), and its cross-attention to `graph_hidden` is **masked** so a leg DOF can only attend to {its leg slot, root} and an arm DOF to {its arm slot, trunk, root}.
- **C. Drop the root clique.** Default `include_root_edges=False` for the structured decoder graph (new CLI flag), keeping only true parent-child chain edges.

`S = 6` slots: `global_root, trunk_head, left_arm, right_arm, left_leg, right_leg` (`DEFAULT_BODY_SLOTS`).

### Why global consumers keep a flattened global `z`
Human-pose reconstruction, root-height, residual blend, spectral/world aux are **whole-body** by definition — they *should* see all slots. The leak only matters for the **per-DOF robot action** decoder, so only that decoder gets per-slot routing+masking. This scopes the blast radius.

## 4. Detailed changes

### A0. Encoder slot-locality — PREREQUISITE (codex review #1, #5)

**The per-slot latent is local only if the ENCODER is slot-local.** `mu/logvar`
read from `graph_hidden`; the A0 `GraphTemporalEncoder` does `GraphFuseBlock×L`
(edge-masked slot mixing, per frame) **then flattens to `(B, T·S, d)` and runs a
FULL bidirectional Transformer over all `T·S` tokens** (`:489-501`). That full
attention mixes `leg@t'` into `arm@t`, so `graph_hidden[arm]` — hence
`z_slots[arm]` AND the masked-`graph_hidden` context — already carry leg info.
**Decoder routing (B) cannot un-mix what the encoder blended**, so the
input→output Jacobian gate (§6.1) would still fail. This was the plan's biggest
internal inconsistency: it treated window/full-attention mixing as separate from
spatial locality, but they are the same leak entering one step earlier.

Required change (structured path): make temporal mixing **per-slot**. Reshape to
`(B·S, T, d)` and run the temporal attention/conv WITHIN each slot; cross-slot
exchange happens ONLY through the edge-masked `GraphFuseBlock`.
- `CausalConvTemporalEncoder` (A1/A2) already does per-slot temporal conv, so it
  is structurally closer to slot-local; the A0 transformer needs the per-slot
  reshape (or a block-diagonal-by-slot mask on the `T·S` attention).
- Also bound cross-slot reach in the GraphFuse stack: with `graph_layers` hops a
  leaf slot (arm) can reach a distant slot (leg) via trunk→root→leg. Keep
  `graph_layers` small and/or use a tree-structured slot edge set so arm's
  encoder receptive field is {arm, trunk, root}, not leg.
- This also removes the cross-time slot mixing that breaks window-placement
  consistency (§4E) — per-slot temporal attention is the shared fix.

### A. Per-slot bottleneck (`graph_retarget_latent.py`)

Add structured heads (keep the existing `d_model→z_dim` heads for the baseline path untouched):
```python
# __init__ (structured branch only)
self.slot_z = int(slot_z)                       # new param, default 48
self.mu_head_slot     = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.slot_z))
self.logvar_head_slot = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.slot_z))
# z_dim becomes derived for structured: z_dim = num_slots * slot_z   (assert / log both)
```
`encode_latent` (structured branch): stop pooling for the latent.
```python
graph_hidden = self.encoder(slot_tokens, part_mask)        # (B,T,S,d_model)
mu     = self.mu_head_slot(graph_hidden)                    # (B,T,S,slot_z)
logvar = self.logvar_head_slot(graph_hidden)               # (B,T,S,slot_z)
return {"mu": mu, "logvar": logvar, "graph_hidden": graph_hidden,
        "part_mask": part_mask, "slot_latents": self.slot_projector(graph_hidden)}
```
`forward`:
```python
z_slots  = self.reparameterize(mu, logvar, sample)         # (B,T,S,slot_z)
z_global = z_slots.reshape(B, T, S * self.slot_z)          # for global consumers
```
- Global consumers (`human_decoder`, `root_height_head`, `residual_head`, `spectral_aux_head`, `world_head` in trainer) take `z_global`. Their input dims already equal `z_dim = S*slot_z`, so no signature change beyond the value passed.
- `latent_action_decoder` (structured) takes `z_slots` (see B).

KL: `kl_divergence(mu, logvar)` (trainer `:112`, called `:628`). With per-slot
`(B,T,S,slot_z)`, **keep `.mean()` over ALL of `(B,T,S,slot_z)`**. Because
`z_dim = S·slot_z`, this is the *same scale* as the current flat-z KL (it is the
same elements, reshaped), so the existing `--kl-weight` schedule transfers
unchanged and is already per-element balanced across slots. **Do NOT "sum over
slot_z, mean over (B,T,S)"** — that multiplies KL pressure by `slot_z` (=48×) and
silently changes the regularization scale (codex review #2). If a different scale
is ever wanted, log it explicitly as a changed `kl_weight`.

### B. Routed + masked `RobotGraphActionDecoder`

Build correspondence once in `__init__` from `robot_dof_names` + human slot names:
```python
# helper: build_dof_slot_correspondence(robot_dof_names, human_slots)
#   region by keyword: hip|knee|ankle -> {left,right}_leg ; shoulder|elbow|wrist -> {left,right}_arm
#   waist|torso|spine -> trunk_head ; side from 'left'/'right' prefix
#   returns: primary_slot[num_dof] (long)  and  allowed_mask[num_dof, S] (bool)
#   allowed = {region_slot, global_root} ; arms also add trunk_head
#   UNKNOWN DOF (codex review #3): allowed = all-True (safe) AND
#       primary_slot = global_root  (deterministic — primary_slot is never left
#       undefined, since the routed-z gather in lines below indexes it for every DOF)
self.register_buffer("dof_primary_slot", primary_slot, persistent=False)   # (num_dof,)
self.register_buffer("dof_slot_attn_mask", ~allowed_mask, persistent=False) # (num_dof,S) True=block
self.z_slot_embed = nn.Sequential(nn.Linear(slot_z, d_model), nn.GELU(), nn.LayerNorm(d_model))
self.root_slot_embed = nn.Sequential(nn.Linear(slot_z, d_model), nn.GELU(), nn.LayerNorm(d_model))
```
`forward(z_slots, beta, robot_id, context=graph_hidden, context_mask=part_mask)`:
```python
# (1) routed z injection — replace global z_embed broadcast (:722)
region = z_slots.gather(2, self.dof_primary_slot.view(1,1,-1,1).expand(B,T,-1,slot_z))  # (B,T,num_dof,slot_z)
root   = z_slots[:, :, ROOT_SLOT_IDX, :]                                                  # (B,T,slot_z)
x = dof_embed + self.z_slot_embed(region) + self.root_slot_embed(root).unsqueeze(2) + beta_token (+ dof_features)
# (2) MASKED cross-attention to human slots (:737) — add attn_mask
attn_out, _ = self.cross_attn(flat_dof, ctx_slots, ctx_slots,
                              attn_mask=self.dof_slot_attn_mask,      # (num_dof,S) True=block
                              key_padding_mask=ctx_padding)
# (3) self-attention graph blocks unchanged (now downstream of clean, region-local tokens)
```
Notes:
- `attn_mask` for `nn.MultiheadAttention` cross-attn accepts `(L,S)=(num_dof,S)` bool, broadcast over batch — verify against the installed torch version; if it requires `(B*nheads,L,S)`, expand.
- Root slot is shared across all DOFs intentionally (pelvis motion is physically global; this is root→all, not shoulder→hip). Make root-sharing a flag if reviewer prefers strict isolation.

### C. Root clique off (trainer)

`train_graph_latent.py:1500`:
```python
dof_edges = tuple(fk.get_dof_edges(include_root_edges=args.robot_dof_root_edges)) if hasattr(fk,"get_dof_edges") else ()
```
Add CLI `--robot-dof-root-edges` (`BooleanOptionalAction`, default **False**). Tradeoff: with root edges off, left-leg / right-leg / torso chains are disconnected components in the DOF *self*-attention; bilateral coordination then flows only through the temporal stack and the shared root slot. Acceptable for a locality fix; revisit if bilateral balance regresses.

### Decoder context — RESOLVED (was a reviewer question; see §8.1)
The decoder cross-attends to `graph_hidden` (pre-bottleneck features) — a skip
around the VAE bottleneck. **Resolved default: option (i)** — keep the
cross-attention but **mask it by region** (a DOF attends only to {its region
slot, root; + trunk for arms}). CLI `--decoder-context {masked_graph_hidden,
none}` (default `masked_graph_hidden`) exposes option (ii) — drop the skip, feed
only routed `z_slots` (cleaner bottleneck for the "latent feeds VLA/WAM" goal,
lower capacity) — for an A/B once locality is verified. **Caveat:** the masked
skip only helps locality if the encoder is slot-local (§A0); otherwise the
masked context still carries pre-mixed leg info.

### D. Input concatenation fix — `SkeletalJointTokenizer`  (LANDED 2026-05-28)

`SkeletalJointTokenizer` previously **summed** separate nonlinear per-modality
embeds: `pose_embed(aa) + joint_pos_embed(pos) + joint_vel_embed(vel) +
joint_contact_embed(contact)`. Summing GELU/LayerNorm blocks entangles the
channels — the encoder cannot reliably distinguish angle-axis from velocity
from contact (a learned mix of "3 rad pose, 0 vel" can collide with "0 pose,
3 rad/s vel"). Replaced with a **single embed over the concatenation**:

```python
self.joint_input_dim = 3 + joint_geometry_dim + joint_velocity_dim + joint_contact_dim
self.joint_input_embed = nn.Sequential(nn.Linear(self.joint_input_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))
# forward: joint_tokens = joint_input_embed(cat([pose_aa, joint_pos, joint_vel, joint_contact], -1))
```

Configured-but-absent modalities are zero-filled to keep the width fixed.
Status: implemented + smoke-tested (forward runs with and without geometry;
35/35 unit tests pass). State_dict changes (removes 4 embeds, adds
`joint_input_embed`) → not resumable from old structured ckpts; retrain anyway
for the leak fix. **Independent of A–C; already landed.**

### E. Window-placement consistency (metric + encoder note — separate from spatial locality)

Distinct from spatial leakage: the latent is not consistent across temporal
window placement. Encoding `[0,120]` and `[80,200]` should give the same
`z`/action on the `[80,120]` overlap, but the A0 `GraphTemporalEncoder` uses
**hardcoded absolute sinusoidal PE** + bidirectional full attention over the
flattened `T·S` tokens, so a frame's latent depends on its absolute index AND
the window bounds → the overlap differs by construction. (This also feeds the
spatial blend: full `T·S` attention mixes `leg@t'` into `arm@t`.)

- A1/A2 `CausalConvTemporalEncoder` is window-invariant within its receptive
  field (causal/bounded) — it does **not** have this defect. A real point in
  its favor for the temporal-module ablation.
- Encoder-level fix (separate effort from this spatial-locality PR): RoPE /
  relative PE, or prefer the causal-conv encoder. RoPE alone does not fully fix
  bidirectional attention (the neighbor set still differs across windows);
  causal + bounded receptive field does.
- **Add a cheap diagnostic regardless of which fix lands** (`latent_diagnostics.py`
  E1e `window_placement_consistency`): encode a clip in two overlapping
  windows, report L2 of `(z, actions)` on the overlap, relative to `z`/action
  norm. This is the missing measurement; it costs one extra encode per clip.

## 5. Phase 2 — objective fixes (gated on re-measurement, NOT in this PR)

After A–C land and we re-run `analyze_dof_predictability.py`:
- DOFs whose `corr` recovers → leak was the cause; done.
- DOFs that still collapse (expected: ankle/wrist twist) → add learnable targets: sole-orientation-vs-ground during stance (ankle, task #48), hand-orientation (wrist).
- Elbow amplitude → anneal `--sample-latent` noise via KL schedule, or variance-matching term.
- Arm-1 recon → raise `--latent-human-recon-weight` off 100.

## 6. Validation / acceptance

1. **Leak unit test (primary gate)** — new `tests/test_latent_locality.py`. At random init *and* a trained ckpt: build a batch, autograd-Jacobian of `‖{left_hip_pitch,left_hip_roll} output‖` w.r.t. (a) left-arm input joints, (b) left-leg input joints. **Assert** `‖∂/∂arm‖ / ‖∂/∂leg‖ < 0.05` (arm→hip leakage negligible vs leg→hip). Symmetric check shoulder/elbow output should depend on arm, not leg. **This Jacobian is end-to-end (input joints → output DOFs, through encoder AND decoder), so it catches ENCODER-side leakage too (codex review #5): A0 will FAIL it until §A0 lands. Run it after §A0+A–C, not after decoder-only changes — a passing decoder-only build is not evidence of locality.**
2. **No-regression** — from-scratch short run of the `both` cell (8 ep): tracked-DOF `corr` (hip/knee/shoulder/elbow) ≥ 0.6 (≈ current), `mean_l2_m` ≤ ~0.072 (current 0.0685, allow small slack), no NaN.
3. **Re-measure** `corr` per DOF; record which twist DOFs recover (decides Phase 2 scope).
4. Baseline path unchanged: run a `--latent-architecture baseline` smoke forward and confirm identical output to pre-change (golden test).

## 7. Migration / risk / compatibility

- **Checkpoints:** latent heads + decoder change shape → structured latent ckpts are **not resumable**; retrain from scratch. `--init-from-teacher` still works (only `frame_model` is loaded; `load_teacher` `:1359` untouched).
- **`z_dim` semantics:** for structured, `z_dim = num_slots * slot_z`. Add assertion + log both `slot_z` and derived `z_dim`. `slot_z` default 48 → `z_dim=288` (≈ current 256). Global-consumer Linear `in_features` follow `z_dim` automatically.
- **Enumerated `z` consumers to update to `z_global`:** action decoder→`z_slots`; `residual_head` (`:1310`), `root_height_head` (`:1317`), `human_decoder` (`:1323`), `spectral_aux_head` (`:1331`), trainer `world_head` (`:904/909`) → `z_global`; KL via `mu/logvar` (per-slot).
- **`run_config` logging** (`:1764+`, `:1805`): add `slot_z`, `z_dim_derived`, `robot_dof_root_edges`, `dof_slot_correspondence` (as name→slot lists). Ensure JSON-serializable.
- **Capacity:** masking cross-attn + dropping the clique reduces decoder capacity; mitigated by retaining root-slot sharing and the temporal stack.
- **torch cross-attn mask shape:** verify `attn_mask=(L,S)` broadcasts; otherwise expand to `(B*nheads,L,S)`.

## 8. Open questions for the reviewer — RESOLVED (2026-05-28)
1. **Masked cross-attn (i) vs drop `graph_hidden` skip (ii)** → **(i) masked cross-attn now.** Keeps capacity, smaller quality risk, and gets the locality win. Add a CLI `--decoder-context {masked_graph_hidden, none}` so (ii) (pure-`z_slots` bottleneck, the cleaner "latent feeds VLA/WAM" form) can be A/B-tested once locality is verified. Default `masked_graph_hidden`.
2. **Root-slot sharing** → **shared root→all DOFs, with a flag.** Pelvis motion is physically global (root→all, not shoulder→hip), so it does not reintroduce the leak. Add `--decoder-root-slot-into-arms` (BooleanOptionalAction, default **True**); set False for strict arm isolation (root only into legs+waist) if the Jacobian test still flags root-mediated arm↔leg coupling.
3. **`slot_z`** → **48** (z_dim_derived = 6 × 48 = 288 ≈ current 256). **KL: sum over `slot_z`, mean over `(B,T,S)`** so every slot is balanced regardless of capacity. Assert + log `slot_z` and `z_dim_derived`.
4. **Root clique** → **drop it** (default `--no-robot-dof-root-edges`). Keep only true parent-child chain edges. Bilateral coordination flows through the temporal stack + the shared root slot; revisit only if the no-regression run shows a bilateral-balance regression, in which case add a single torso↔hip bridge edge (not the full clique).

## 9. Implementation order (for writing-plans)
1. **D — input concat** ✅ landed.
2. **E — window-consistency metric** (E1e in `latent_diagnostics.py`) — cheap, do alongside.
3. **A0 — encoder slot-locality** (per-slot temporal attention/conv; bound GraphFuse hop reach). **MUST precede A/B** — without it the per-slot latent is pre-mixed and the Jacobian gate fails regardless of decoder routing (codex #1/#5).
4. **A — per-slot bottleneck** (`encode_latent`, `reparameterize`, `z_global` for global consumers, KL = mean over all `(B,T,S,slot_z)`).
5. **C — root-clique flag** (trainer `--robot-dof-root-edges`, default off).
6. **B — routed + masked decoder** (`dof_primary_slot` with root fallback for unknown DOFs, `dof_slot_attn_mask`, `z_slot_embed`/`root_slot_embed`, masked cross-attn; `--decoder-context`, `--decoder-root-slot-into-arms`).
7. **Validation** — Jacobian leak test (primary gate, catches encoder+decoder leakage; run after A0+A–C), no-regression short run, baseline golden test, per-DOF `corr` re-measure, window-consistency metric.
Dependency order: A0 → A → C → B (B consumes z_slots from A; A's locality is only real once A0 lands; C is a one-line trainer flag).
