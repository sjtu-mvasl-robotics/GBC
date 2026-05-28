# Proposal: Morphology-Conditioned Unified Motion Latent for GBC

**Companion to:** `graph_retarget_latent_REVIEW.md`
**Target:** convert `UnifiedLatentRetargeter` from a G1-only retargeter into a true cross-morphology motion-prior + robot-conditional decoder.
**Anchor decision:** Interpretation B (cross-morphology) from the review. The GNN intuition is correct — kinematic trees are graphs, and a GNN over the URDF is the right vehicle for morphology conditioning.

---

## 1. Architecture at a glance

```
                +------------------------+
SMPL-H pose --> | Morphology-Free Motion |  z_t  (per-frame, optional FSQ tokens)
root traj   --> |   Encoder  (MFME)      |---------------------+
                +------------------------+                     |
                                                               |
                +------------------------+                     v
URDF (any   --> | Morphology Graph       | m_1..m_K   +-------------------+
  robot)        |   Encoder  (MGE, GNN)  |----------> | Cross-Morphology  |
                +------------------------+            |   Action Decoder  | --> a_t (K-dim, any robot)
                                                      |   (CMAD)          |
                                                      +-------------------+

Stages: (S1) pretrain MFME on humans only.  (S2) train MGE+CMAD on multi-robot retargeting with MFME frozen.
        (S3) joint fine-tune MFME+MGE+CMAD with cycle-consistency.
```

Three core principles:

1. **MFME knows nothing about robots.** Its inputs and outputs are defined entirely in human / canonical motion space. This is what makes the latent universal.
2. **MGE turns a URDF into per-DoF tokens.** The GNN reads the URDF graph and produces per-DoF embeddings that capture *what each DoF can do*, not just *what its name is*.
3. **CMAD is one decoder, not one per robot.** It cross-attends from URDF tokens (queries) into the motion latent (keys/values) and emits one action per DoF. K is variable across robots.

---

## 2. Module specifications

### 2.1 Morphology-Free Motion Encoder (MFME)

**Keep most of what's already there.** The existing `SkeletalJointTokenizer` + `GraphTemporalEncoder` is already morphology-free on the human side — just sever it from any robot action target.

**Required changes vs current code:**

| Change | Reason | File / line |
|---|---|---|
| Move root-frame canonicalization to a preprocessing step | A motion prior should be invariant to world heading. Encode body-local pose; carry root trajectory as a parallel stream. | new `_canonicalize_motion(pose_aa, root_pose)` upstream of `:1097` |
| Replace per-frame VAE bottleneck with **FSQ** (Mentzer 2023) — *optional but recommended* | Discrete tokens scale better, no posterior-collapse failure, natural interface for cross-source/cross-robot alignment. | new `FSQBottleneck`; replaces `mu_head` / `logvar_head` at `:994-995` |
| Add masked-pose-modeling pretraining objective | Robot-free self-supervision; the canonical MAE setup. | new `pretrain_mfme.py` |
| Add foot-contact prediction head | Cheapest documented foot-skating mitigation; morphology-free. | new `FootContactHead`, sibling to `FutureSlotHead` |
| Remove teacher residual from MFME training | Teacher is robot-specific and biases the latent. Use only at decoder stage. | gate `_zero_init_residual` behind a `use_teacher_residual` flag |

**Output of MFME (per frame):**
- `z_t ∈ ℝ^{z_dim}` (continuous) or `z_t ∈ {0,…,L}^{T_token}` (FSQ codes)
- `slot_latents_t ∈ ℝ^{6 × slot_latent_dim}` — kept; useful as a side-channel for CMAD
- `graph_hidden_t ∈ ℝ^{6 × d_model}` — full slot feature map for cross-attention

### 2.2 Morphology Graph Encoder (MGE) — the GNN

This is the new component. **Inputs: a URDF. Outputs: K per-DoF embeddings + a graph-level robot embedding.** Trained alongside CMAD; never sees motion.

#### 2.2.1 Node features per DoF (from URDF parsing)

For each actuated DoF `i`, build a feature vector `f_i ∈ ℝ^{F}`:

```
f_i = [
    one_hot(joint_type),          # revolute / prismatic / fixed (fixed nodes still carry structure)
    joint_axis_xyz,               # unit vector in parent frame
    log(link_length),             # parent-to-child translation magnitude
    parent_translation_xyz / L,   # normalized
    joint_lower_limit,
    joint_upper_limit,
    velocity_limit,
    effort_limit,
    one_hot(canonical_body_slot), # left_arm / right_arm / left_leg / right_leg / trunk_head / global_root
    depth_in_chain,               # integer depth from root
    one_hot(end_effector_role),   # hand / foot / head / none
]
```

The **canonical_body_slot** field is the bridge between the URDF and the existing 6-slot body decomposition — this is how MGE talks to the slot-structured motion latent.

Practical: write a small URDF parser (`urdfpy` already used elsewhere in the repo via `RK` wrapper at `viewer/isaacsim_viewer.py`). One pass produces `f_i`, the parent-index list, and the per-DoF body-slot assignment.

#### 2.2.2 Edge structure

- **Kinematic edges:** parent → child along the URDF tree. Directed; supplement with reverse edges for bidirectional message passing.
- **Slot-co-membership edges:** every DoF connects to every other DoF in the same canonical body slot. Cheap, captures "these are the wrist DoFs."
- **Root-broadcast edges:** root (pelvis) ↔ every DoF. Mirrors the slot-graph's global root special case (`:92-96`).

Store as `edge_index ∈ ℤ^{2 × E}` plus per-edge type one-hots.

#### 2.2.3 Architecture choice — transformer over the joint graph, not classic message-passing GNN

**Recommendation:** a small transformer with **graph-structural attention bias**. Concretely:

```python
class MorphologyGraphEncoder(nn.Module):
    def __init__(self, node_feat_dim, d_model, num_heads, num_layers, max_hops=8):
        ...
        self.feature_proj = nn.Sequential(
            nn.Linear(node_feat_dim, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.hop_bias = nn.Embedding(max_hops + 2, num_heads)   # +1 for self, +1 for "far"
        self.edge_type_bias = nn.Embedding(num_edge_types, num_heads)
        self.layers = nn.ModuleList(GraphormerLayer(d_model, num_heads) for _ in range(num_layers))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))   # robot-level summary

    def forward(self, node_features, hop_matrix, edge_type_matrix):
        # hop_matrix[k,i,j] = bucketized graph distance from i to j (within batch element k)
        # attention_bias[h,i,j] = hop_bias[hop_matrix][..., h] + edge_type_bias[edge_type_matrix][..., h]
        x = self.feature_proj(node_features)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        for layer in self.layers:
            x = layer(x, attention_bias=...)
        m_cls, m_per_dof = x[:, 0], x[:, 1:]
        return m_per_dof, m_cls   # K embeddings + 1 robot summary
```

Why a Graphormer-style transformer over GAT / GraphSAGE:

- **Style consistency.** The existing code already implements graph attention with attention masks (`_attention_mask_from_edges`, `:109-115`). Move from *hard masks* to *additive biases*; same primitives.
- **Scale.** SONIC's lesson is that learned attention with structural bias beats hard message-passing once data grows.
- **No extra deps.** PyG is fine but optional. The structural attention you need is a tiny patch on `nn.MultiheadAttention`.
- **Handles variable K cleanly.** K-DoF differs per robot; padding + key_padding_mask is already used throughout (`:411`, `:411`).

Keep `num_layers=4–6`, `d_model=128–256`. The morphology graph is small (~30 nodes for G1, ~50 for hand-equipped robots), so depth ≫ 6 is wasteful.

#### 2.2.4 Self-supervised pretraining of MGE

Before plugging into CMAD, pretrain the MGE on URDFs alone with three cheap losses:

1. **Masked joint feature reconstruction** — mask 15% of `f_i`, predict from neighbors. Standard graph-MAE.
2. **Parent prediction** — given a node, predict which neighbor is the parent. Forces topological awareness.
3. **Body-slot classification** — predict the canonical slot from node features alone; sanity check.

Train on a heterogeneous URDF mix: Unitree G1 / H1 / H1-2, Fourier GR1, Tien Kung, Booster, Atlas if available. Even 5–10 URDFs gives the GNN useful diversity since features are continuous, not just enum IDs.

### 2.3 Cross-Morphology Action Decoder (CMAD)

Refactor of `RobotGraphActionDecoder` (`:620-710`). One decoder, variable K.

**Inputs:**
- `z` (motion latent), shape `(B, T, z_dim)` or token codes
- `m_per_dof` (from MGE), shape `(B, K, d_model)`
- `m_cls` (from MGE), shape `(B, d_model)`
- `graph_hidden` (slot context from MFME), shape `(B, T, 6, d_model)`
- `slot_assignment[i]` — which canonical slot each DoF belongs to (from MGE parsing)

**Forward:**

```python
# Per-frame, per-DoF query token:
q_{b,t,i} = m_per_dof[b, i] + z_proj(z[b, t]) + dof_pos_embed(i)

# Structured cross-attention into the matching slot's hidden state:
#   each DoF attends primarily to graph_hidden of its assigned body slot,
#   plus the root slot, plus m_cls.
ctx_{b,t,i} = slot_aware_cross_attn(q_{b,t,i}, graph_hidden[b, t])

# Robot-graph self-attention with structural bias (reuse MGE biases here, frozen or shared):
ctx = morphology_self_attn(ctx, hop_bias=..., edge_type_bias=...)

# Temporal conv per DoF token (kept from current TemporalConvBlock at :599-617):
ctx = temporal_smooth(ctx)

# Action head — independent scalar per DoF:
a_{b,t,i} = action_head(ctx_{b,t,i})
```

**Key differences from current `RobotGraphActionDecoder`:**

| Current | New |
|---|---|
| `dof_embed = nn.Embedding(output_dim, d_model)` (`:644`) | `dof_embed = m_per_dof from MGE` (URDF-derived, robot-agnostic) |
| `robot_embed = nn.Embedding(num_robot_types, d_model)` (`:646`) | `m_cls from MGE` (graph-level robot summary) |
| Hard kinematic mask from hardcoded joint names (`:524-596`) | Soft additive bias from MGE's hop matrix |
| Fixed `output_dim` | Variable K per batch; mask via `key_padding_mask` |
| Slot context via single cross-attention (`:697`) | **Slot-aware** cross-attention: each DoF only attends to its assigned slot's hidden state (plus root). Sharper signal, fewer wasted FLOPs. |

### 2.4 Optional per-robot bootstrap teacher

Keep `PoseTransformer` and the residual scheme **as a per-robot warmup**, not as the architecture's identity. Concretely:

- Train one `PoseTransformer` teacher per robot (G1 first, then H1, GR1).
- During CMAD training, allow `actions = blend * teacher(pose) + (1-blend) * cmad(z, morph) + residual`, with `blend` annealed `1.0 → 0.0` over 30% of training.
- At inference, `blend = 0`: CMAD is fully on its own. This is the path to a truly universal decoder.

---

## 3. Training curriculum

### Stage S1 — MFME pretraining (humans only, no robots)
- Dataset: AMASS + your retargeted SMPL-H corpus. Just the humans.
- Losses: masked-pose reconstruction (15% mask ratio), future-slot prediction (existing `FutureSlotHead`), foot-contact BCE, KL or FSQ commitment.
- **No robot in the loss.** This guarantees morphology-freeness.
- Validate by encoding two clips of the same motion at different world rotations and checking latent similarity (canonicalization test).

### Stage S2 — MGE pretraining (URDFs only, no motion)
- Dataset: 5–10 robot URDFs (G1, H1, H1-2, GR1, Tien Kung, plus any others).
- Losses: masked feature recovery, parent prediction, slot classification.
- Output: per-DoF and per-robot embeddings that cluster sensibly (visualize with t-SNE; legs should separate from arms).

### Stage S3 — CMAD with frozen MFME + MGE
- Multi-robot retargeting. For each batch, sample a robot, retarget the human clip to that robot (use existing teacher pipeline as ground truth), train CMAD to match.
- Curriculum across robots: start with G1 only → add H1 (very similar) → add GR1 (different proportions) → hold out one robot for zero-shot eval.
- Optionally use per-robot teacher residual as bootstrap (Section 2.4).

### Stage S4 — joint fine-tune with cycle consistency
- Unfreeze everything.
- Add **cycle-consistency loss**: encode human → decode robot A → re-target back to human (via a small inverse model or shared latent re-encoding) → should match. Forces the latent to be the *common* representation across morphologies.
- Add **cross-morphology consistency** if you have paired data: same human motion → robot A and robot B → both retargets should map back to similar `z`.

---

## 4. Loss design

```
L_total = λ_recon * L_action_recon                # CMAD vs teacher (or vs ground truth retarget)
        + λ_mfme_mask * L_masked_pose             # MFME pretraining objective, kept in joint stage
        + λ_future * L_future_slot                # existing FutureSlotHead
        + λ_contact * L_foot_contact_bce          # new
        + λ_kl_or_fsq * L_bottleneck              # KL (if VAE) or commitment (if FSQ)
        + λ_residual * ||residual||^2             # if teacher residual is enabled
        + λ_dz * ||z_t - z_{t-1}||^2              # temporal smoothness of latent
        + λ_cycle * L_cycle                       # stage S4 only
        + λ_morph_aux * L_mge_pretraining         # mask-feature etc., kept warm in joint stage
```

Suggested initial coefficients (per-loss-magnitude scaled): all `λ` start at 1.0 except `λ_residual = 0.01`, `λ_kl_or_fsq = 0.1`, `λ_cycle = 0.1`. Tune via a sweep but start conservative on regularizers.

---

## 5. Code refactor checklist

Mapped to file:line in current code.

### Decoupling (no behavior change yet)
- [ ] Add `RobotSpec` dataclass: `{node_features: Tensor[K,F], parent_index: Tensor[K], slot_assignment: Tensor[K], edge_index: Tensor[2,E], edge_types: Tensor[E], hop_matrix: Tensor[K,K]}`. New file `robot_spec.py`.
- [ ] Write `urdf_to_robot_spec(urdf_path) -> RobotSpec`. Use existing `RK` wrapper at `viewer/isaacsim_viewer.py` as the URDF loader.
- [ ] Move encoder construction out of `UnifiedLatentRetargeter.__init__` into a `build_mfme(...)` factory; encoder must compile with `num_actions=0`.
- [ ] Verify MFME forward pass works without any decoder by writing a unit test `tests/test_mfme_standalone.py`.

### MGE
- [ ] New module `morphology_graph_encoder.py` with `MorphologyGraphEncoder` (Section 2.2.3).
- [ ] Write `pretrain_mge.py` for Stage S2.

### CMAD refactor
- [ ] In `RobotGraphActionDecoder` (`:620-710`):
  - [ ] Replace `dof_embed = nn.Embedding(output_dim, d_model)` (`:644`) with `m_per_dof` argument from MGE.
  - [ ] Replace `robot_embed = nn.Embedding(num_robot_types, d_model)` (`:646`) with `m_cls` argument from MGE.
  - [ ] Replace `attn_mask_template` (`:657-660`) with additive bias from `hop_matrix` and `edge_types`.
  - [ ] Change `output_dim` to be passed per-call (variable K). Drop `head` shape coupling.
  - [ ] Implement slot-aware cross-attention (each DoF attends only to its slot's `graph_hidden` + root).

### Hardcoded G1 names — remove
- [ ] Delete `_build_robot_dof_edges` (`:524-596`) entirely. Its job moves into the URDF parser.
- [ ] Delete `DEFAULT_SMPLH_BODY_PARENTS` if not used outside this file — or move to a `humans/smplh.py` config.

### Top-level `UnifiedLatentRetargeter`
- [ ] Accept `robot_spec: RobotSpec` in `forward(...)`.
- [ ] Gate the teacher path behind `use_teacher_residual: bool` (default False). If True, the teacher checkpoint per robot is looked up by `robot_spec.name`.
- [ ] Rename file to `motion_latent.py` and rename class to `MotionLatentRetargeter`. The file's current name doubles down on the wrong framing.

### Data pipeline
- [ ] Multi-robot dataloader: each batch carries `(pose_aa, root_pose, robot_spec)`. For Stage S3, sample robots uniformly; for fine-tune, oversample harder ones.

---

## 6. Validation: how to prove the latent is morphology-conditioned

Three concrete tests. If you can pass all three, the universality claim is real.

1. **Latent invariance to target robot.** Encode the same human clip; the latent `z` must not depend on which robot the decoder will target. Implementation: run MFME alone with no `robot_spec`; verify `z` is computable. This is structural, not statistical.

2. **Decoder generalization to held-out robot.** Train CMAD on {G1, H1, GR1}, hold out H1-2. Zero-shot retargeting on H1-2 should be within ~20% of an oracle CMAD trained including H1-2 (since H1-2 is close to H1 in morphology).

3. **Latent reusability for downstream policy.** Train a tracking policy on G1 conditioned on `z`. Train a second tracking policy on H1 conditioned on the *same* `z` from the same clip. If `z` is universal, both policies should achieve high tracking success without re-encoding.

Standard metrics on top of these (apply per robot): global / root-relative MPJPE, foot skating, joint-limit violation rate, success rate, smoothness.

---

## 7. The GNN, in more detail — what to actually implement first

Since you flagged the GNN as the part you're confident about, here is a minimal but production-quality starting recipe.

### 7.1 Single `GraphormerLayer` (drop-in replacement for `GraphTokenBlock` at `:146-170`)

```python
class GraphormerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        # x: (B, N, d_model); attn_bias: (B, num_heads, N, N) additive
        B, N, D = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / self.head_dim ** 0.5
        if attn_bias is not None:
            scores = scores + attn_bias
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(out)
        x = x + self.ffn(self.norm2(x))
        return x
```

### 7.2 Structural bias computation

```python
def shortest_path_hops(parent_index, num_nodes, max_hops=8):
    # Floyd-Warshall on an undirected adjacency built from parent_index.
    INF = max_hops + 1
    D = torch.full((num_nodes, num_nodes), INF, dtype=torch.long)
    for i in range(num_nodes):
        D[i, i] = 0
    for child, parent in enumerate(parent_index):
        if parent >= 0:
            D[child, parent] = 1
            D[parent, child] = 1
    for k in range(num_nodes):
        D = torch.minimum(D, D[:, k:k+1] + D[k:k+1, :])
    return D.clamp_max(max_hops + 1)  # bucket "far" into one bin
```

Cache this per URDF — it's static. The MGE looks up `hop_bias_embedding(D)` once per robot.

### 7.3 Why this is preferable to PyG GAT for your case

- **Variable K with masking** is native (already the codebase idiom).
- **Hop bias** captures multi-hop relations in one attention layer, vs needing K message-passing steps in GAT.
- **Same primitives as `GraphFuseBlock`** (`:383-417`) — easy code review and migration.
- **End-to-end differentiable, no graph batching gymnastics** (PyG batching across heterogeneous URDFs is messy).

---

## 8. Risks and open questions

| Risk | Mitigation |
|---|---|
| Not enough multi-robot retargeted data to learn cross-morphology generalization | Bootstrap with per-robot `PoseTransformer` teachers (Section 2.4). Generate retargets per-robot using your existing pipeline; the data multiplier is K × current dataset. |
| Cycle consistency in S4 is hard to make work | Start without it; introduce only after S3 converges. Treat as a stretch goal, not a gating requirement. |
| FSQ replacement is invasive | Optional. The continuous VAE works for S1; FSQ pays off mostly when adding multi-source inputs (SONIC-style A path). Defer to a separate experiment. |
| MGE pretraining (S2) lacks signal with only ~5 URDFs | Augment with feature noising and graph rewiring. Or skip S2 and let MGE co-train with CMAD; gradients from CMAD will shape MGE adequately given a few thousand updates. |
| Slot-aware cross-attention loses information when a DoF spans slots (e.g. waist) | Allow each DoF to attend to its slot **plus root** (already in plan). For genuine cross-slot DoFs, mark them in `slot_assignment` as `trunk_head` and let the body slot graph route the signal. |
| Different robots have different action conventions (delta-PD vs absolute joint angle vs torque) | Carry a `control_mode` per-DoF into `node_features`. CMAD's action head can branch on it. |

---

## 9. Recommended next two weeks

1. **Days 1–3:** write `RobotSpec` + URDF parser. Generate specs for G1, H1, H1-2, GR1. Visualize per-DoF features as a sanity check.
2. **Days 4–6:** stub out `MorphologyGraphEncoder` and `MotionLatentRetargeter` (renamed). Wire MGE outputs into a copy of `RobotGraphActionDecoder` that uses them. Verify shapes end-to-end on G1; G1 results should match the current model within noise (it's the same architecture, just refactored).
3. **Days 7–10:** add H1 to the training mix. Train CMAD multi-robot with per-robot teacher warmup. Measure MPJPE on each robot. If H1 results are within 1.5× G1's, the abstraction is working.
4. **Days 11–14:** hold out GR1, run zero-shot retargeting through the trained CMAD. This is the first real test of morphology conditioning. Report MPJPE, foot-skating, joint-limit violation rate.

If GR1 zero-shot is meaningfully better than a baseline (e.g., a CMAD trained on G1 only and forced to consume GR1 specs at inference), the architecture is sound and you proceed to S4. If not, the failure mode is almost always either (a) insufficient morphology diversity in S2/S3, or (b) the slot-aware cross-attention is over-routing — both diagnosable from per-DoF gradient norms and attention visualizations.

---

## 10. What to keep from the current file

Final note on what survives the refactor — useful as a checklist when you start cutting code:

| Component | Verdict |
|---|---|
| `HumanSlotTokenizer` (`:173-254`) | Keep. Morphology-free on human side. |
| `SkeletalJointTokenizer` (`:257-380`) | Keep, with structural attention bias upgrade. |
| `GraphTokenBlock`, `GraphFuseBlock` (`:146-170`, `:383-417`) | Replace with `GraphormerLayer` (Section 7.1) for additive-bias attention. |
| `GraphTemporalEncoder` (`:420-459`) | Keep. |
| `FutureSlotHead` (`:713-744`) | Keep. |
| `RetargetDecoder` (`:462-512`) | Delete. Replaced by CMAD. |
| `RobotGraphActionDecoder` (`:620-710`) | Refactor into CMAD (Section 2.3). |
| `_build_robot_dof_edges` (`:524-596`) | Delete. Replaced by URDF parser → `RobotSpec`. |
| Teacher residual (`:1029-1060`, `:1153-1160`) | Keep behind a feature flag; use as per-robot bootstrap only. |
| `slot_latents` projector (`:990-993`) | Keep — useful auxiliary output. |
| Per-frame VAE heads (`:994-995`) | Keep for now; FSQ is an optional later swap. |
