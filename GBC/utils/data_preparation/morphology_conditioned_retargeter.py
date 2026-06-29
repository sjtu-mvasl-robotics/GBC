"""Morphology-conditioned unified motion-latent retargeter.

Alternate design to ``graph_retarget_latent.py``. Three decoupled modules:

* ``MorphologyFreeMotionEncoder`` (MFME) — encodes SMPL-H motion into a latent
  ``z`` plus per-slot hidden states. **Never sees a robot.**
* ``MorphologyGraphEncoder`` (MGE) — converts a URDF-derived per-DoF feature
  table into per-DoF tokens and a graph-level summary. **Never sees motion.**
* ``CrossMorphologyActionDecoder`` (CMAD) — variable-K decoder that cross-attends
  each DoF token into the motion latent's slot states (soft slot routing) and
  produces one action per DoF. **Works for any robot whose spec MGE accepts.**

Design choices (after Codex review of the proposal):

* MGE defaults to a 2-layer MLP over the rich 21-D URDF feature vector
  (``RobotKinematics.get_dof_features``) followed by 1-2 graph-attention layers
  with additive hop-distance bias. The deeper Graphormer is opt-in.
* Slot cross-attention is **soft**: each DoF gets a learned (slot_assignment,
  target_slot) bias added to attention scores, so it preferentially routes to
  its slot + root but can still pull from neighbors when needed.
* Variable K within a batch is supported via padding + ``dof_padding_mask``.
* No teacher in the architecture itself; the teacher-residual scheme stays out
  of the universal-latent path (can be added per-robot at training time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from GBC.utils.data_preparation.graph_retarget_latent import (
    DEFAULT_BODY_EDGES,
    DEFAULT_BODY_SLOTS,
    DEFAULT_SLOT_JOINTS,
    DEFAULT_SMPLH_BODY_PARENTS,
    FutureSlotHead,
    GraphTemporalEncoder,
    GraphTokenBlock,
    HumanSlotTokenizer,
    SkeletalJointTokenizer,
    TemporalConvBlock,
    _attention_mask_from_edges,
    _sinusoidal_position_encoding,
    _masked_mean,
)


URDF_FEATURE_DIM = 21  # matches RobotKinematics.get_dof_features


# ---------------------------------------------------------------------------
# Robot spec + builders
# ---------------------------------------------------------------------------

@dataclass
class RobotSpec:
    """All static, motion-free morphology data the model needs about a robot.

    Tensors are unbatched (per-robot). Wrap in ``batch_robot_specs`` to pad
    across heterogeneous robots in a single batch.
    """

    name: str
    node_features: torch.Tensor          # (K, F)
    parent_index: torch.Tensor           # (K,) long; -1 = root
    slot_assignment: torch.Tensor        # (K,) long, values in [0, len(slots))
    hop_matrix: torch.Tensor             # (K, K) long, bucketized
    control_mode: torch.Tensor           # (K,) long; 0=angle 1=delta_pd 2=torque
    dof_names: Tuple[str, ...]

    @property
    def num_dofs(self) -> int:
        return int(self.node_features.shape[0])

    def to(self, device: torch.device | str) -> "RobotSpec":
        return RobotSpec(
            name=self.name,
            node_features=self.node_features.to(device),
            parent_index=self.parent_index.to(device),
            slot_assignment=self.slot_assignment.to(device),
            hop_matrix=self.hop_matrix.to(device),
            control_mode=self.control_mode.to(device),
            dof_names=self.dof_names,
        )


_SLOT_NAME_TO_INDEX = {name: i for i, name in enumerate(DEFAULT_BODY_SLOTS)}


def _slot_for_joint_name(name: str) -> int:
    """Map a robot joint name to a canonical body slot index.

    Heuristic and intentionally simple. Override per robot when a URDF uses
    nonstandard naming.
    """
    n = name.lower()
    if "left" in n and any(k in n for k in ("hip", "knee", "ankle", "leg", "foot", "toe")):
        return _SLOT_NAME_TO_INDEX["left_leg"]
    if "right" in n and any(k in n for k in ("hip", "knee", "ankle", "leg", "foot", "toe")):
        return _SLOT_NAME_TO_INDEX["right_leg"]
    if "left" in n and any(k in n for k in ("shoulder", "arm", "elbow", "wrist", "hand", "finger")):
        return _SLOT_NAME_TO_INDEX["left_arm"]
    if "right" in n and any(k in n for k in ("shoulder", "arm", "elbow", "wrist", "hand", "finger")):
        return _SLOT_NAME_TO_INDEX["right_arm"]
    if any(k in n for k in ("waist", "torso", "spine", "chest", "neck", "head")):
        return _SLOT_NAME_TO_INDEX["trunk_head"]
    return _SLOT_NAME_TO_INDEX["global_root"]


def _hop_matrix_from_parents(
    parent_index: torch.Tensor,
    max_hops: int = 8,
) -> torch.Tensor:
    """Floyd-Warshall over an undirected tree; bucket distances at ``max_hops``."""
    k = int(parent_index.shape[0])
    far = max_hops + 1
    D = torch.full((k, k), far, dtype=torch.long)
    D.fill_diagonal_(0)
    for child in range(k):
        p = int(parent_index[child].item())
        if p >= 0:
            D[child, p] = 1
            D[p, child] = 1
    for via in range(k):
        via_row = D[via:via + 1, :]
        via_col = D[:, via:via + 1]
        D = torch.minimum(D, via_col + via_row)
    return D.clamp_max(far)


def _hop_matrix_from_edges(
    edges: Iterable[Tuple[int, int]],
    num_nodes: int,
    max_hops: int = 8,
) -> torch.Tensor:
    far = max_hops + 1
    D = torch.full((num_nodes, num_nodes), far, dtype=torch.long)
    D.fill_diagonal_(0)
    for i, j in edges:
        if 0 <= i < num_nodes and 0 <= j < num_nodes:
            D[i, j] = 1
            D[j, i] = 1
    for via in range(num_nodes):
        D = torch.minimum(D, D[:, via:via + 1] + D[via:via + 1, :])
    return D.clamp_max(far)


def build_robot_spec_from_kinematics(
    rk,  # GBC.utils.base.base_fk.RobotKinematics
    name: str,
    slot_overrides: Optional[Dict[str, str]] = None,
    control_mode: str = "joint_angle",
    max_hops: int = 8,
    include_root_edges: bool = False,
) -> RobotSpec:
    """Build a :class:`RobotSpec` from a loaded ``RobotKinematics`` instance.

    ``include_root_edges`` mirrors the production ``--no-robot-dof-root-edges``
    flag (E2b/D4 default OFF): when False, the root clique (hip<->waist<->hip)
    is dropped so unrelated regions are not coupled in the decoder's
    self-attention. Default False to match E2b at the morph-free interface.
    """
    dof_names = tuple(rk.get_dof_names())
    node_features = rk.get_dof_features().detach().cpu()
    edges = rk.get_dof_edges(include_root_edges=include_root_edges)
    k = len(dof_names)

    parent_index = torch.full((k,), -1, dtype=torch.long)
    adj: Dict[int, List[int]] = {i: [] for i in range(k)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    # BFS from any node with no movable parent (degree-1 or root) to set parents
    visited = [False] * k
    queue: List[int] = []
    for i in range(k):
        if not visited[i]:
            queue.append(i)
            visited[i] = True
            while queue:
                node = queue.pop(0)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        parent_index[nb] = node
                        queue.append(nb)

    overrides = {k.lower(): v for k, v in (slot_overrides or {}).items()}
    slot_assignment = torch.zeros(k, dtype=torch.long)
    for idx, name_i in enumerate(dof_names):
        if name_i.lower() in overrides:
            slot_assignment[idx] = _SLOT_NAME_TO_INDEX[overrides[name_i.lower()]]
        else:
            slot_assignment[idx] = _slot_for_joint_name(name_i)

    hop_matrix = _hop_matrix_from_edges(edges, k, max_hops=max_hops)

    mode_id = {"joint_angle": 0, "delta_pd": 1, "torque": 2}.get(control_mode, 0)
    control = torch.full((k,), mode_id, dtype=torch.long)

    return RobotSpec(
        name=name,
        node_features=node_features,
        parent_index=parent_index,
        slot_assignment=slot_assignment,
        hop_matrix=hop_matrix,
        control_mode=control,
        dof_names=dof_names,
    )


def build_synthetic_robot_spec(
    name: str,
    dof_names: Sequence[str],
    parent_index: Sequence[int],
    *,
    feature_dim: int = URDF_FEATURE_DIM,
    seed: int = 0,
    control_mode: str = "joint_angle",
    max_hops: int = 8,
) -> RobotSpec:
    """Construct a fake :class:`RobotSpec` for tests without loading a URDF."""
    g = torch.Generator().manual_seed(seed)
    k = len(dof_names)
    parents = torch.tensor(list(parent_index), dtype=torch.long)
    node_features = torch.randn(k, feature_dim, generator=g)
    slot_assignment = torch.tensor(
        [_slot_for_joint_name(n) for n in dof_names], dtype=torch.long
    )
    hop = _hop_matrix_from_parents(parents, max_hops=max_hops)
    mode_id = {"joint_angle": 0, "delta_pd": 1, "torque": 2}.get(control_mode, 0)
    return RobotSpec(
        name=name,
        node_features=node_features,
        parent_index=parents,
        slot_assignment=slot_assignment,
        hop_matrix=hop,
        control_mode=torch.full((k,), mode_id, dtype=torch.long),
        dof_names=tuple(dof_names),
    )


def build_human_spec(
    name: str = "smplh_body",
    num_joints: int = 21,
    joint_parents: Sequence[int] = DEFAULT_SMPLH_BODY_PARENTS,
    slot_joints: Optional[Dict[str, Sequence[int]]] = None,
    slots: Sequence[str] = DEFAULT_BODY_SLOTS,
    feature_dim: int = URDF_FEATURE_DIM,
    max_hops: int = 8,
) -> RobotSpec:
    """Build a RobotSpec describing the SMPL-H body skeleton as an "embodiment".

    The point: the MFME-MGE-CMAD pipeline becomes morph-symmetric. The same
    decoder body that produces robot DoF actions also produces SMPL-H per-joint
    axis-angle rotations, fed by a per-joint feature table just like a URDF's
    per-DoF table. This forces the encoder/MGE/CMAD weights to handle ≥2
    embodiments before any cross-robot training is attempted.

    Feature layout matches RobotKinematics.get_dof_features (21 fields). Most
    URDF fields don't apply to SMPL-H ball joints, so we zero them and set
    `type_other = 1` plus the graph-structural fields (movable_depth_norm,
    has_movable_parent, graph_degree_norm) so MGE can still place each joint
    within the skeleton. `control_mode = 3` flags rotation/axis-angle outputs.
    """
    if feature_dim != URDF_FEATURE_DIM:
        raise ValueError(
            f"build_human_spec assumes URDF_FEATURE_DIM={URDF_FEATURE_DIM}; got {feature_dim}"
        )
    if len(joint_parents) != num_joints:
        raise ValueError(
            f"joint_parents has {len(joint_parents)} entries, num_joints={num_joints}"
        )

    parent_index = torch.tensor(list(joint_parents), dtype=torch.long)

    # Slot assignment from DEFAULT_SLOT_JOINTS (fallback to global_root).
    slot_joints_eff = slot_joints or DEFAULT_SLOT_JOINTS
    slot_to_idx = {s: i for i, s in enumerate(slots)}
    slot_assignment = torch.zeros(num_joints, dtype=torch.long)
    for slot_name, joints in slot_joints_eff.items():
        if slot_name not in slot_to_idx:
            continue
        for j in joints:
            if 0 <= int(j) < num_joints:
                slot_assignment[int(j)] = slot_to_idx[slot_name]

    # Build edges from parent links (bidirectional) for both hop matrix and
    # graph-degree feature.
    edges: List[Tuple[int, int]] = []
    deg = torch.zeros(num_joints, dtype=torch.long)
    for child, parent in enumerate(joint_parents):
        if parent < 0 or parent >= num_joints or child >= num_joints:
            continue
        edges.append((int(child), int(parent)))
        deg[int(child)] += 1
        deg[int(parent)] += 1

    # BFS depth from the root (parent_index == -1) for movable_depth_norm.
    depth = torch.zeros(num_joints, dtype=torch.long)
    visited = [False] * num_joints
    adj: Dict[int, List[int]] = {i: [] for i in range(num_joints)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    queue: List[Tuple[int, int]] = []
    for i in range(num_joints):
        if int(parent_index[i].item()) < 0 and not visited[i]:
            queue.append((i, 0))
            visited[i] = True
    while queue:
        node, d = queue.pop(0)
        depth[node] = d
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                queue.append((nb, d + 1))
    max_depth = float(max(int(depth.max().item()), 1))
    max_deg = float(max(int(deg.max().item()), 1))

    # Feature vector — see URDF schema. Most fields zero; flag type_other and
    # populate the three graph-structural fields.
    feats = torch.zeros(num_joints, feature_dim, dtype=torch.float32)
    type_other_idx = 6
    movable_depth_idx = 18
    has_movable_parent_idx = 19
    graph_degree_idx = 20
    feats[:, type_other_idx] = 1.0
    feats[:, movable_depth_idx] = depth.float() / max_depth
    feats[:, has_movable_parent_idx] = (parent_index >= 0).float()
    feats[:, graph_degree_idx] = deg.float() / max_deg

    hop_matrix = _hop_matrix_from_parents(parent_index, max_hops=max_hops)

    # control_mode == 3 marks "rotation / axis-angle" so MGE distinguishes
    # human ball joints from robot 1-DoF actuators. The MGE embedding must
    # accommodate at least 4 modes; see MorphologyGraphEncoder.__init__.
    control = torch.full((num_joints,), 3, dtype=torch.long)

    joint_names = tuple(f"smplh_joint_{i:02d}" for i in range(num_joints))

    return RobotSpec(
        name=name,
        node_features=feats,
        parent_index=parent_index,
        slot_assignment=slot_assignment,
        hop_matrix=hop_matrix,
        control_mode=control,
        dof_names=joint_names,
    )


def batch_robot_specs(specs: Sequence[RobotSpec]) -> Dict[str, torch.Tensor]:
    """Pad heterogeneous :class:`RobotSpec` instances to a fixed K_max batch.

    Returns dict with keys ``node_features`` ``slot_assignment`` ``hop_matrix``
    ``control_mode`` ``parent_index`` ``dof_padding_mask`` (True where padded).
    """
    k_max = max(s.num_dofs for s in specs)
    b = len(specs)
    f = specs[0].node_features.shape[1]
    node_features = torch.zeros(b, k_max, f)
    slot_assignment = torch.zeros(b, k_max, dtype=torch.long)
    hop_matrix = torch.full((b, k_max, k_max), 0, dtype=torch.long)
    control_mode = torch.zeros(b, k_max, dtype=torch.long)
    parent_index = torch.full((b, k_max), -1, dtype=torch.long)
    padding = torch.ones(b, k_max, dtype=torch.bool)
    for i, s in enumerate(specs):
        k = s.num_dofs
        node_features[i, :k] = s.node_features
        slot_assignment[i, :k] = s.slot_assignment
        hop_matrix[i, :k, :k] = s.hop_matrix
        control_mode[i, :k] = s.control_mode
        parent_index[i, :k] = s.parent_index
        padding[i, :k] = False
    return {
        "node_features": node_features,
        "slot_assignment": slot_assignment,
        "hop_matrix": hop_matrix,
        "control_mode": control_mode,
        "parent_index": parent_index,
        "dof_padding_mask": padding,
    }


# ---------------------------------------------------------------------------
# Graphormer-style layer (additive structural bias)
# ---------------------------------------------------------------------------

class GraphormerLayer(nn.Module):
    """Multi-head self-attention with optional additive per-pair bias."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-1, -2)) / self.head_dim ** 0.5
        if attn_bias is not None:
            scores = scores + attn_bias
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        x = x + self.proj(out)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Morphology-Free Motion Encoder (MFME)
# ---------------------------------------------------------------------------

class MorphologyFreeMotionEncoder(nn.Module):
    """Encodes SMPL-H motion into a per-frame latent ``z`` + slot context.

    Knows nothing about robots. Output keys (per dict):

    * ``mu`` / ``logvar`` / ``z`` — (B, T, z_dim). Continuous VAE bottleneck.
    * ``slot_latents``                    — (B, T, 6, slot_latent_dim).
    * ``graph_hidden``                    — (B, T, 6, d_model).
    * ``part_mask``                       — (B, 6).
    """

    def __init__(
        self,
        num_human_joints: int = 21,
        beta_dim: int = 16,
        root_dim: int = 9,
        d_model: int = 256,
        z_dim: int = 256,
        slot_z: Optional[int] = None,
        slot_latent_dim: int = 32,
        graph_layers: int = 2,
        temporal_layers: int = 4,
        num_heads: int = 8,
        slots: Sequence[str] = DEFAULT_BODY_SLOTS,
        edges: Iterable[Tuple[str, str]] = DEFAULT_BODY_EDGES,
        slot_joints: Optional[Dict[str, Sequence[int]]] = None,
        tokenizer: str = "structured",
        dropout: float = 0.1,
        latent_mode: str = "global",
        causal: bool = False,
    ):
        super().__init__()
        self.num_human_joints = int(num_human_joints)
        self.beta_dim = int(beta_dim)
        self.root_dim = int(root_dim)
        self.d_model = int(d_model)
        self.latent_mode = str(latent_mode)
        if self.latent_mode not in {"global", "per_slot"}:
            raise ValueError(f"Unknown latent_mode: {latent_mode!r}")
        self.causal = bool(causal)
        slots = tuple(slots)
        # In per_slot mode we mirror UnifiedLatentRetargeter's structured
        # layout: append an independent `root_state` slot if it is not already
        # present. This keeps E2b's per-slot heads shape-compatible at load time
        # (parameter shape is (slot_z, d_model), slot-agnostic).
        if self.latent_mode == "per_slot" and "root_state" not in slots:
            slots = slots + ("root_state",)
        self.slots = slots

        if self.latent_mode == "per_slot":
            if slot_z is None:
                raise ValueError("latent_mode='per_slot' requires slot_z")
            self.slot_z = int(slot_z)
            # Override z_dim to be derived; mirrors UnifiedLatentRetargeter.
            self.z_dim = len(self.slots) * self.slot_z
        else:
            self.slot_z = int(slot_z) if slot_z is not None else 0
            self.z_dim = int(z_dim)

        slot_joints_eff = slot_joints or DEFAULT_SLOT_JOINTS

        if tokenizer == "structured":
            self.tokenizer = SkeletalJointTokenizer(
                num_human_joints=num_human_joints,
                beta_dim=beta_dim,
                d_model=d_model,
                slots=self.slots,
                slot_joints=slot_joints_eff,
                root_dim=root_dim,
                joint_parents=DEFAULT_SMPLH_BODY_PARENTS,
                graph_layers=graph_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        elif tokenizer == "baseline":
            self.tokenizer = HumanSlotTokenizer(
                num_human_joints=num_human_joints,
                beta_dim=beta_dim,
                d_model=d_model,
                slots=self.slots,
                slot_joints=slot_joints_eff,
                root_dim=root_dim,
            )
        else:
            raise ValueError(f"Unknown tokenizer: {tokenizer}")

        # Pass `causal` AND mirror UnifiedLatentRetargeter's structured-only
        # per-slot temporal mixing for causal mode (the per-slot temporal flag
        # is required for causal=True per `GraphTemporalEncoder.forward`).
        self.encoder = GraphTemporalEncoder(
            d_model=d_model,
            num_heads=num_heads,
            slots=self.slots,
            edges=edges,
            graph_layers=graph_layers,
            temporal_layers=temporal_layers,
            dropout=dropout,
            per_slot_temporal=self.causal,
            causal=self.causal,
        )
        self.slot_projector = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, slot_latent_dim),
        )
        if self.latent_mode == "global":
            self.mu_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.z_dim))
            self.logvar_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.z_dim))
        else:
            # Per-slot heads — slot-agnostic Linear(d_model, slot_z), parameter
            # shape independent of slot count. This makes E2b warm-start
            # shape-match cleanly (Option B per design spec §4.1).
            self.mu_head_slot = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.slot_z))
            self.logvar_head_slot = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.slot_z))

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(
        self,
        pose_aa: torch.Tensor,
        root_pose: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
        sample: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        slot_tokens, part_mask = self.tokenizer(pose_aa, root_pose, beta, part_mask)
        graph_hidden = self.encoder(slot_tokens, part_mask)
        sample_eff = self.training if sample is None else sample
        slot_latents = self.slot_projector(graph_hidden)
        if self.latent_mode == "global":
            pooled = _masked_mean(graph_hidden, part_mask[:, None, :], dim=2)
            mu = self.mu_head(pooled)
            logvar = self.logvar_head(pooled)
            z = self.reparameterize(mu, logvar, sample_eff)
            return {
                "mu": mu,
                "logvar": logvar,
                "z": z,
                "slot_latents": slot_latents,
                "graph_hidden": graph_hidden,
                "part_mask": part_mask,
            }
        # latent_mode == "per_slot": emit per-slot mu/logvar so E2b warm-start
        # is shape-compatible (see design spec §4.1, Option B).
        mu_slots = self.mu_head_slot(graph_hidden)         # (B, T, S, slot_z)
        logvar_slots = self.logvar_head_slot(graph_hidden) # (B, T, S, slot_z)
        z_slots = self.reparameterize(mu_slots, logvar_slots, sample_eff)
        # Flatten across slots so global consumers (KL, smoothness, spectral
        # aux) keep a flat (B, T, z_dim) tensor.
        B, T, S, Sz = z_slots.shape
        z_flat       = z_slots.reshape(B, T, S * Sz)
        mu_flat      = mu_slots.reshape(B, T, S * Sz)
        logvar_flat  = logvar_slots.reshape(B, T, S * Sz)
        return {
            "mu": mu_flat,
            "logvar": logvar_flat,
            "z": z_flat,
            "mu_slots": mu_slots,
            "logvar_slots": logvar_slots,
            "z_slots": z_slots,
            "slot_latents": slot_latents,
            "graph_hidden": graph_hidden,
            "part_mask": part_mask,
        }


# ---------------------------------------------------------------------------
# Morphology Graph Encoder (MGE)
# ---------------------------------------------------------------------------

class MorphologyGraphEncoder(nn.Module):
    """Per-DoF + per-robot embeddings from URDF static features.

    Default: 2-layer MLP over the 21-D URDF feature vector + ``num_graph_layers``
    Graphormer layers with hop-distance + slot-membership additive bias.
    Set ``num_graph_layers=0`` to ablate to pure MLP.
    """

    def __init__(
        self,
        feature_dim: int = URDF_FEATURE_DIM,
        d_model: int = 256,
        num_heads: int = 8,
        num_graph_layers: int = 2,
        max_hops: int = 8,
        num_slots: int = len(DEFAULT_BODY_SLOTS),
        num_control_modes: int = 3,  # 0=angle 1=delta_pd 2=torque; pass 4 to add 3=rotation (unified human decoder)
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.max_hops = int(max_hops)

        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.slot_embed = nn.Embedding(num_slots, d_model)
        self.control_embed = nn.Embedding(num_control_modes, d_model)

        # +2 hop buckets: 0 = self, max_hops+1 = "far / disconnected"
        self.hop_bias = nn.Embedding(max_hops + 2, num_heads)
        self.same_slot_bias = nn.Embedding(2, num_heads)

        self.layers = nn.ModuleList(
            GraphormerLayer(d_model, num_heads, dropout) for _ in range(num_graph_layers)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        node_features: torch.Tensor,        # (B, K, F)
        slot_assignment: torch.Tensor,      # (B, K) long
        hop_matrix: torch.Tensor,           # (B, K, K) long
        control_mode: torch.Tensor,         # (B, K) long
        dof_padding_mask: Optional[torch.Tensor] = None,  # (B, K) bool, True where padded
        node_bias: Optional[torch.Tensor] = None,  # (B, d_model) added to every node pre-graph
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, k, _ = node_features.shape
        x = self.feature_proj(node_features)
        x = x + self.slot_embed(slot_assignment)
        x = x + self.control_embed(control_mode)
        if node_bias is not None:
            # Morphology-descriptor bias broadcast over DoFs BEFORE the graph layers,
            # so the kinematic-tree attention propagates it into both m_per_dof and CLS.
            x = x + node_bias.unsqueeze(1)

        # Prepend a CLS token, padded-aware.
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, K+1, d_model)
        if dof_padding_mask is not None:
            cls_pad = torch.zeros(b, 1, dtype=torch.bool, device=dof_padding_mask.device)
            key_padding_mask = torch.cat([cls_pad, dof_padding_mask], dim=1)
        else:
            key_padding_mask = None

        if self.layers:
            attn_bias = self._build_attn_bias(slot_assignment, hop_matrix)
            for layer in self.layers:
                x = layer(x, attn_bias=attn_bias, key_padding_mask=key_padding_mask)

        return x[:, 1:, :], x[:, 0, :]

    def _build_attn_bias(
        self,
        slot_assignment: torch.Tensor,  # (B, K)
        hop_matrix: torch.Tensor,       # (B, K, K)
    ) -> torch.Tensor:
        b, k = slot_assignment.shape
        # Build (B, K+1, K+1) bias by padding row/col for CLS (zero bias to CLS).
        hops_clamped = hop_matrix.clamp_max(self.max_hops + 1)
        hop_b = self.hop_bias(hops_clamped)  # (B, K, K, H)
        same_slot = (slot_assignment[:, :, None] == slot_assignment[:, None, :]).long()
        slot_b = self.same_slot_bias(same_slot)  # (B, K, K, H)
        pair_b = (hop_b + slot_b).permute(0, 3, 1, 2)  # (B, H, K, K)
        full = torch.zeros(
            b, self.num_heads, k + 1, k + 1, device=pair_b.device, dtype=pair_b.dtype
        )
        full[:, :, 1:, 1:] = pair_b
        return full


# ---------------------------------------------------------------------------
# Cross-Morphology Action Decoder (CMAD)
# ---------------------------------------------------------------------------

class _SoftSlotCrossAttention(nn.Module):
    """Per-DoF cross-attention into the 6 motion slot states with a learned
    (assigned_slot, target_slot) additive bias. Soft routing — every DoF can in
    principle see every slot, but the bias steers it to its own slot + root.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_slots: int,
        dropout: float,
        num_motion_slots: Optional[int] = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # `num_slots`         = number of distinct DoF-side slot ids (bias row dim)
        # `num_motion_slots`  = number of motion-side slots from the encoder
        #                       (bias column dim). Defaults to num_slots for
        #                       backward compatibility.
        self.num_slots = num_slots
        self.num_motion_slots = int(num_motion_slots) if num_motion_slots is not None else num_slots
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Learned bias table: (assigned_slot, target_slot) -> per-head bias.
        # Init: +1 on the diagonal (own slot, when the indices align) and the
        # root column (index 0 is `global_root`), 0 elsewhere.
        bias = torch.zeros(self.num_slots, self.num_motion_slots, num_heads)
        for s in range(min(self.num_slots, self.num_motion_slots)):
            bias[s, s] = 1.0
        for s in range(self.num_slots):
            bias[s, 0] = 1.0  # global_root is motion-slot index 0
        self.slot_pair_bias = nn.Parameter(bias)

    def forward(
        self,
        dof_tokens: torch.Tensor,         # (B, T, K, d_model)
        slot_hidden: torch.Tensor,        # (B, T, S, d_model)
        slot_assignment: torch.Tensor,    # (B, K) long
        slot_part_mask: torch.Tensor,     # (B, S) bool, True where slot is present
    ) -> torch.Tensor:
        b, t, k, d = dof_tokens.shape
        s = slot_hidden.shape[2]
        q = self.q_proj(dof_tokens).view(b, t, k, self.num_heads, self.head_dim)
        k_ = self.k_proj(slot_hidden).view(b, t, s, self.num_heads, self.head_dim)
        v = self.v_proj(slot_hidden).view(b, t, s, self.num_heads, self.head_dim)
        # scores: (B, T, K, S, H)
        scores = torch.einsum("btkhd,btshd->btksh", q, k_) / self.head_dim ** 0.5
        bias = self.slot_pair_bias[slot_assignment]  # (B, K, S, H)
        scores = scores + bias[:, None]
        # mask absent slots
        absent = (~slot_part_mask)[:, None, None, :, None]  # (B,1,1,S,1)
        scores = scores.masked_fill(absent, float("-inf"))
        attn = scores.softmax(dim=3)
        attn = self.dropout(attn)
        out = torch.einsum("btksh,btshd->btkhd", attn, v).reshape(b, t, k, d)
        return self.out_proj(out)


class CrossMorphologyActionDecoder(nn.Module):
    """Decode per-frame latent ``z`` + per-DoF morphology tokens into actions.

    Per-DoF query construction: ``m_per_dof + z_proj(z) + dof_pos_embed``. The
    query cross-attends into the motion slots with a soft routing bias, then a
    morphology self-attention layer mixes within-robot signal, then a temporal
    conv smooths along T. A per-DoF scalar head produces ``actions``.
    """

    def __init__(
        self,
        z_dim: int,
        d_model: int,
        num_heads: int,
        num_slots: int = len(DEFAULT_BODY_SLOTS),
        graph_layers: int = 2,
        temporal_layers: int = 2,
        max_hops: int = 8,
        dropout: float = 0.1,
        latent_mode: str = "global",
        slot_z: Optional[int] = None,
        num_motion_slots: Optional[int] = None,
        causal: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.max_hops = int(max_hops)
        self.latent_mode = str(latent_mode)
        if self.latent_mode not in {"global", "per_slot"}:
            raise ValueError(f"Unknown latent_mode: {latent_mode!r}")
        self.causal = bool(causal)
        self.num_slots = int(num_slots)
        self.num_motion_slots = (
            int(num_motion_slots) if num_motion_slots is not None else self.num_slots
        )
        if self.latent_mode == "global":
            self.z_proj = nn.Sequential(nn.Linear(z_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))
        else:
            if slot_z is None:
                raise ValueError("latent_mode='per_slot' requires slot_z")
            self.slot_z = int(slot_z)
            # Per-slot z is gathered per-DoF via slot_assignment, then projected
            # — analogous to UnifiedLatentRetargeter's z_slot_embed but without
            # baking the DoF count into the parameter shape.
            self.z_slot_proj = nn.Sequential(
                nn.Linear(self.slot_z, d_model), nn.GELU(), nn.LayerNorm(d_model)
            )
        self.dof_pos_embed = nn.Embedding(256, d_model)  # plenty of headroom
        self.cross = _SoftSlotCrossAttention(
            d_model, num_heads, num_slots, dropout, num_motion_slots=self.num_motion_slots,
        )
        self.cross_norm = nn.LayerNorm(d_model)

        self.hop_bias = nn.Embedding(max_hops + 2, num_heads)
        self.same_slot_bias = nn.Embedding(2, num_heads)
        self.graph_layers = nn.ModuleList(
            GraphormerLayer(d_model, num_heads, dropout) for _ in range(graph_layers)
        )
        self.temporal_blocks = nn.ModuleList(
            TemporalConvBlock(d_model, dropout, causal=self.causal) for _ in range(temporal_layers)
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def _build_attn_bias(
        self, slot_assignment: torch.Tensor, hop_matrix: torch.Tensor
    ) -> torch.Tensor:
        hops_clamped = hop_matrix.clamp_max(self.max_hops + 1)
        hop_b = self.hop_bias(hops_clamped)  # (B, K, K, H)
        same_slot = (slot_assignment[:, :, None] == slot_assignment[:, None, :]).long()
        slot_b = self.same_slot_bias(same_slot)
        return (hop_b + slot_b).permute(0, 3, 1, 2)  # (B, H, K, K)

    def decode_features(
        self,
        z: torch.Tensor,                  # (B, T, z_dim) — global mode
        slot_hidden: torch.Tensor,        # (B, T, S, d_model)
        slot_part_mask: torch.Tensor,     # (B, S) bool
        m_per_dof: torch.Tensor,          # (B, K, d_model)
        slot_assignment: torch.Tensor,    # (B, K) long
        hop_matrix: torch.Tensor,         # (B, K, K) long
        dof_padding_mask: Optional[torch.Tensor] = None,
        z_slots: Optional[torch.Tensor] = None,  # (B, T, S, slot_z) — per_slot mode
    ) -> torch.Tensor:
        """Run the CMAD body, returning per-node embeddings (B, T, K, d_model)
        before any output head. Useful when an external caller wants to apply
        a different head (e.g., 3-d axis-angle for the human AE arm) sharing
        the same processing pipeline.
        """
        k = m_per_dof.shape[1]
        if self.latent_mode == "global":
            b, t, _ = z.shape
            device = z.device
            z_tok = self.z_proj(z).unsqueeze(2)            # (B, T, 1, d)
            z_contrib = z_tok                              # broadcast over K
        else:
            assert z_slots is not None, "latent_mode='per_slot' requires z_slots"
            b, t, _, _ = z_slots.shape
            device = z_slots.device
            # Gather per-DoF slot z via slot_assignment, then project.
            # slot_assignment is (B, K); z_slots is (B, T, S, slot_z).
            # Resulting z_per_dof is (B, T, K, slot_z).
            sa = slot_assignment[:, None, :, None].expand(b, t, k, z_slots.shape[-1])
            z_per_dof = torch.gather(z_slots, 2, sa)        # (B, T, K, slot_z)
            z_contrib = self.z_slot_proj(z_per_dof)         # (B, T, K, d_model)

        dof_ids = torch.arange(k, device=device)
        pos = self.dof_pos_embed(dof_ids).view(1, 1, k, self.d_model)
        m_tok = m_per_dof.unsqueeze(1)                     # (B, 1, K, d)
        q = m_tok + z_contrib + pos                        # (B, T, K, d)

        # 1) soft slot-aware cross-attention into motion slots
        cross_out = self.cross(q, slot_hidden, slot_assignment, slot_part_mask)
        x = self.cross_norm(q + cross_out)

        # 2) per-frame within-robot self-attention with structural bias
        attn_bias = self._build_attn_bias(slot_assignment, hop_matrix)
        flat = x.reshape(b * t, k, self.d_model)
        bias_flat = attn_bias.unsqueeze(1).expand(b, t, self.num_heads, k, k).reshape(
            b * t, self.num_heads, k, k
        )
        if dof_padding_mask is not None:
            kpm = dof_padding_mask[:, None, :].expand(b, t, k).reshape(b * t, k)
        else:
            kpm = None
        for layer in self.graph_layers:
            flat = layer(flat, attn_bias=bias_flat, key_padding_mask=kpm)
        x = flat.reshape(b, t, k, self.d_model)

        # 3) temporal smoothing per DoF
        time_pos = _sinusoidal_position_encoding(t, self.d_model, device, x.dtype)
        x = x + time_pos.view(1, t, 1, self.d_model)
        for block in self.temporal_blocks:
            x = block(x)

        return x  # (B, T, K, d_model)

    def forward(
        self,
        z: torch.Tensor,
        slot_hidden: torch.Tensor,
        slot_part_mask: torch.Tensor,
        m_per_dof: torch.Tensor,
        slot_assignment: torch.Tensor,
        hop_matrix: torch.Tensor,
        dof_padding_mask: Optional[torch.Tensor] = None,
        z_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.decode_features(
            z=z,
            slot_hidden=slot_hidden,
            slot_part_mask=slot_part_mask,
            m_per_dof=m_per_dof,
            slot_assignment=slot_assignment,
            hop_matrix=hop_matrix,
            dof_padding_mask=dof_padding_mask,
            z_slots=z_slots,
        )
        actions = self.action_head(x).squeeze(-1)  # (B, T, K)
        if dof_padding_mask is not None:
            actions = actions.masked_fill(dof_padding_mask[:, None, :], 0.0)
        return actions


# ---------------------------------------------------------------------------
# Graph-based human motion decoder (AE arm)
# ---------------------------------------------------------------------------

class GraphHumanDecoder(nn.Module):
    """Inverse-of-encoder graph decoder: per-slot z + beta → per-joint pose_aa
    + root features. The AE arm of the morph-free design — without it the
    latent has no obligation to preserve human motion structure, and the
    universal-latent claim collapses.

    Architecture mirrors ``SkeletalJointTokenizer`` on the way back:
      * Per-joint slot routing: each SMPL-H joint reads its assigned body
        slot's z (from ``DEFAULT_SLOT_JOINTS``).
      * Per-joint token = ``Linear(slot_z → d_model) + joint_pos_embed +
        beta_embed`` (broadcast).
      * ``graph_layers`` × ``GraphTokenBlock`` with attention masked to the
        SMPL-H kinematic-tree edges (parent ↔ child).
      * Per-joint head emits axis-angle (3,); pooled head emits root features
        (root_dim).

    Output is FLAT, matching the existing ``RetargetDecoder`` interface so
    the rest of the trainer keeps working: ``(B, T, num_joints * 3 + root_dim)``.
    The trainer slices ``[..., : num_joints * 3]`` for ``human_pose_aa_recon``
    and ``[..., num_joints * 3 :]`` for ``human_root_recon``.
    """

    def __init__(
        self,
        slot_z: int,
        beta_dim: int,
        d_model: int,
        num_joints: int = 21,
        root_dim: int = 9,
        slots: Sequence[str] = DEFAULT_BODY_SLOTS,
        slot_joints: Optional[Dict[str, Sequence[int]]] = None,
        joint_parents: Sequence[int] = DEFAULT_SMPLH_BODY_PARENTS,
        graph_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_joints = int(num_joints)
        self.d_model = int(d_model)
        self.root_dim = int(root_dim)
        self.slots = tuple(slots)

        slot_joints_eff = slot_joints or DEFAULT_SLOT_JOINTS
        # Build per-joint slot index. Joints not listed in slot_joints fall
        # back to slot index 0 (global_root) — matches the encoder's fallback.
        per_joint_slot = torch.zeros(num_joints, dtype=torch.long)
        for slot_idx, slot_name in enumerate(self.slots):
            if slot_name not in slot_joints_eff:
                continue
            for j in slot_joints_eff[slot_name]:
                if 0 <= int(j) < num_joints:
                    per_joint_slot[int(j)] = int(slot_idx)
        self.register_buffer("per_joint_slot", per_joint_slot)

        # SMPL-H kinematic-tree mask: parent ↔ child edges (bidirectional).
        edges: list[tuple[int, int]] = []
        for child, parent in enumerate(joint_parents):
            if parent < 0 or parent >= num_joints or child >= num_joints:
                continue
            edges.append((child, parent))
            edges.append((parent, child))
        self.register_buffer(
            "joint_attn_mask",
            _attention_mask_from_edges(num_joints, edges, device=torch.device("cpu")),
        )

        self.z_to_joint = nn.Sequential(
            nn.Linear(slot_z, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.joint_pos_embed = nn.Embedding(num_joints, d_model)
        self.beta_embed = nn.Sequential(
            nn.Linear(beta_dim, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.graph_blocks = nn.ModuleList(
            GraphTokenBlock(d_model, num_heads, dropout) for _ in range(graph_layers)
        )

        # Per-joint pose_aa head + pooled root head.
        # IMPORTANT: do NOT zero-init the last linears. Zero-init blocks
        # gradient from flowing upstream through the decoder (any input x
        # multiplied by W=0 has no gradient to x, only to W). With lr=2e-5
        # the head crawls out of zero very slowly and the AE arm stays
        # effectively untrained — verified empirically: hpose_recon was
        # stuck at ~0.09 for 540 steps. Production's `RetargetDecoder` is
        # plain Kaiming-init MLP; same convention here.
        self.pose_norm = nn.LayerNorm(d_model)
        self.pose_aa_head = nn.Linear(d_model, 3)
        self.root_norm = nn.LayerNorm(d_model)
        self.root_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, root_dim)
        )

    def forward(self, z_slots: torch.Tensor, beta: Optional[torch.Tensor]) -> torch.Tensor:
        """z_slots: (B, T, S, slot_z); beta: (B, beta_dim) or None.

        Returns flat (B, T, num_joints*3 + root_dim) — mirrors RetargetDecoder.
        """
        B, T, S, sz = z_slots.shape
        J = self.num_joints
        # Per-joint gather. per_joint_slot may reference indices beyond the
        # available S if the caller passes a slot count smaller than expected;
        # clamp to S-1 as a safety net so we don't OOB.
        idx = self.per_joint_slot.clamp_max(S - 1)
        z_per_joint = z_slots.index_select(2, idx)            # (B, T, J, slot_z)
        tok = self.z_to_joint(z_per_joint)                    # (B, T, J, d_model)
        joint_ids = torch.arange(J, device=z_slots.device)
        tok = tok + self.joint_pos_embed(joint_ids).view(1, 1, J, self.d_model)
        if beta is not None:
            bt = self.beta_embed(beta).view(B, 1, 1, self.d_model)
            tok = tok + bt

        # Per-frame graph attention (flatten B, T into batch dim).
        # joint_attn_mask is (J, J) bool with True = forbidden, broadcasted by
        # MultiheadAttention across the (B*T) batch dim.
        x = tok.reshape(B * T, J, self.d_model)
        for blk in self.graph_blocks:
            x = blk(x, attn_mask=self.joint_attn_mask)
        x = x.reshape(B, T, J, self.d_model)

        pose_aa = self.pose_aa_head(self.pose_norm(x))        # (B, T, J, 3)
        root_pooled = self.root_norm(x.mean(dim=2))           # (B, T, d_model)
        root_recon = self.root_head(root_pooled)              # (B, T, root_dim)

        out = torch.cat([pose_aa.reshape(B, T, J * 3), root_recon], dim=-1)
        return out


# ---------------------------------------------------------------------------
# Robot Motion Encoder (robot-side latent, SONIC universal-token pattern)
# ---------------------------------------------------------------------------

class RobotMotionEncoder(nn.Module):
    """Encode robot proprioceptive motion into the SAME per-slot latent as MFME.

    Inputs per frame: ``dof_pos`` (q), ``dof_vel`` (q̇), ``root_features``, plus
    the static morphology embedding ``m_per_dof`` from MGE (so the encoder knows
    *what* each DoF is, exactly the same token CMAD consumes on the decode side).

    Pipeline (mirrors MFME on the robot side):
      1. Per-DoF motion token = ``Linear([q, q̇]) + m_per_dof``  → (B, T, K, d).
      2. Masked scatter-mean per body slot via ``slot_assignment``  → (B, T, S, d);
         padded DoFs are excluded so a fake DoF can't move a real slot's latent.
      3. Add the root token (root_features → d) to every slot — gives the
         otherwise-DoF-free ``root_state`` slot its signal and mirrors MFME's
         root fusion.
      4. ``GraphTemporalEncoder`` (same slot graph + edges, causal if requested).
      5. Per-slot mu/logvar heads — the robot encoder's OWN heads (NOT shared
         with MFME). The two latent spaces are tied by the alignment loss
         ``‖z_human.detach() − z_robot‖²``, which pulls z_robot onto the
         stop-grad human anchor (recipe 3 reward / recipe 4 target). Sharing
         the head would let the alignment loss drift the human anchor it claims
         to chase — the BYOL/JEPA stop-grad-target pitfall.

    Output dict: ``mu_slots`` / ``logvar_slots`` / ``z_slots`` (B, T, S, slot_z),
    plus ``graph_hidden`` (B, T, S, d) and ``part_mask`` (B, S).
    """

    def __init__(
        self,
        d_model: int,
        slot_z: int,
        root_dim: int = 9,
        slots: Sequence[str] = DEFAULT_BODY_SLOTS,
        edges: Iterable[Tuple[str, str]] = DEFAULT_BODY_EDGES,
        graph_layers: int = 2,
        temporal_layers: int = 4,
        num_heads: int = 8,
        causal: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.slot_z = int(slot_z)
        self.root_dim = int(root_dim)
        self.slots = tuple(slots)
        self.num_slots = len(self.slots)
        self.causal = bool(causal)

        # Per-DoF motion projection: (q, q̇) -> d_model.
        self.dof_motion_proj = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.root_proj = nn.Sequential(
            nn.Linear(root_dim, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.encoder = GraphTemporalEncoder(
            d_model=d_model,
            num_heads=num_heads,
            slots=self.slots,
            edges=edges,
            graph_layers=graph_layers,
            temporal_layers=temporal_layers,
            dropout=dropout,
            per_slot_temporal=self.causal,
            causal=self.causal,
        )
        # Per-slot mu/logvar heads — the robot encoder OWNS these; they are
        # deliberately NOT shared with MFME. Sharing the projection while
        # stop-gradding the human target is self-contradictory: the alignment
        # loss would update the very head that produces the human anchor
        # (BYOL/JEPA pitfall — the predictor must not share the projection with
        # the stop-grad target). The two latent spaces are tied solely by the
        # alignment loss, which pulls z_robot onto the detached z_human; the
        # human head stays shaped only by the retargeting/AE objectives.
        self.mu_head_slot = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, slot_z))
        self.logvar_head_slot = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, slot_z))

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(
        self,
        dof_pos: torch.Tensor,                    # (B, T, K)
        dof_vel: torch.Tensor,                    # (B, T, K)
        root_features: torch.Tensor,              # (B, T, root_dim)
        m_per_dof: torch.Tensor,                  # (B, K, d_model) from MGE
        slot_assignment: torch.Tensor,            # (B, K) long in [0, num_slots)
        dof_padding_mask: Optional[torch.Tensor] = None,  # (B, K) bool, True=pad
        sample: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, K = dof_pos.shape
        S = self.num_slots
        device = dof_pos.device
        sample_eff = self.training if sample is None else sample

        # 1) per-DoF motion token + morphology identity
        dof_feat = torch.stack([dof_pos, dof_vel], dim=-1)        # (B, T, K, 2)
        dof_tok = self.dof_motion_proj(dof_feat)                  # (B, T, K, d)
        dof_tok = dof_tok + m_per_dof.unsqueeze(1)               # broadcast over T

        # 2) masked scatter-mean per slot
        onehot = torch.nn.functional.one_hot(
            slot_assignment.clamp(0, S - 1), num_classes=S
        ).to(dof_tok.dtype)                                       # (B, K, S)
        if dof_padding_mask is not None:
            onehot = onehot * (~dof_padding_mask).to(dof_tok.dtype).unsqueeze(-1)
        slot_sum = torch.einsum("btkd,bks->btsd", dof_tok, onehot)  # (B, T, S, d)
        slot_cnt = onehot.sum(dim=1)                              # (B, S)
        slot_tok = slot_sum / slot_cnt.clamp_min(1.0).view(B, 1, S, 1)

        # 3) add root token to every slot (gives root_state slot its signal)
        root_tok = self.root_proj(root_features)                 # (B, T, d)
        slot_tok = slot_tok + root_tok.unsqueeze(2)              # (B, T, S, d)

        # 4) graph + temporal encode. part_mask all-true: root token makes every
        #    slot informative, matching MFME's full-body AMASS part_mask.
        part_mask = torch.ones(B, S, dtype=torch.bool, device=device)
        graph_hidden = self.encoder(slot_tok, part_mask)         # (B, T, S, d)

        # 5) shared per-slot bottleneck
        mu_slots = self.mu_head_slot(graph_hidden)               # (B, T, S, slot_z)
        logvar_slots = self.logvar_head_slot(graph_hidden)
        z_slots = self.reparameterize(mu_slots, logvar_slots, sample_eff)
        return {
            "mu_slots": mu_slots,
            "logvar_slots": logvar_slots,
            "z_slots": z_slots,
            "graph_hidden": graph_hidden,
            "part_mask": part_mask,
        }


# ---------------------------------------------------------------------------
# Morphology-conditioned root decoder
# ---------------------------------------------------------------------------

def rest_root_height_from_fk(
    fk,
    num_dofs: int,
    foot_link_indices: Optional[Sequence[int]] = None,
    ground_offset: float = 0.0,
    device: Optional[torch.device] = None,
) -> float:
    """Analytic standing root height from the kinematic tree alone.

    FK at the rest pose (q = 0) puts the feet under the root for every
    humanoid in this project (G1 / H1-2 / GR3 all have a standing zero pose):
    z_stand = ground_offset − min_z(foot link frames, root-relative). No data,
    no learning — this is the morphology→height mapping computed exactly,
    so the RootDecoder only has to learn the scale-free dynamics on top
    (see root_height_factorized).

    foot_link_indices indexes rows of fk's link output; None = all links
    (the lowest link frame at rest is the foot for these robots).
    """
    with torch.no_grad():
        q0 = torch.zeros(1, int(num_dofs), device=device)
        links = fk(q0)[0]  # (L, 3) root-relative
        z = links[:, 2] if foot_link_indices is None else links[list(foot_link_indices), 2]
        return float(ground_offset - z.min().item())


class RootDecoder(nn.Module):
    """Decode the robot's global root xyz from the motion latent's root_state
    slot, conditioned on the robot morphology token (MGE CLS = aggregate of the
    kinematic tree).

    Design (per user 2026-06-10): root translation is retargeted exactly like
    joints — human root motion is encoded into the root_state slot under the
    human's morphology; here we DECODE it onto the robot's graph, so the robot's
    own geometry (leg length etc., captured by m_cls) sets the scale. The
    standing height and the reachable xyz range emerge from morphology; there is
    NO fixed ref_base_height constant. Output is full xyz (B, T, 3).
    """

    def __init__(self, slot_z, d_model, temporal_layers=2, dropout=0.1, causal=False):
        super().__init__()
        self.z_proj = nn.Sequential(nn.Linear(slot_z, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.morph_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.temporal = nn.ModuleList(
            TemporalConvBlock(d_model, dropout, causal=causal) for _ in range(temporal_layers)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3)
        )

    def forward(
        self,
        root_slot_z: torch.Tensor,
        m_cls: torch.Tensor,
        rest_root_height: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """root_slot_z: (B, T, slot_z); m_cls: (B, d_model). Returns (B, T, 3).

        With ``rest_root_height`` (B,) given (root_height_factorized mode),
        the head's z channel becomes a scale-free residual h and
        z = z_stand · (1 + h): the morphology→height mapping is the ANALYTIC
        standing height from the URDF (rest_root_height_from_fk), only the
        dynamics modulation is learned — shared across robots, so every
        robot's data supervises the same normalized quantity and an unseen
        robot starts at its kinematically correct standing height (h≈0).
        """
        b, t, _ = root_slot_z.shape
        x = self.z_proj(root_slot_z) + self.morph_proj(m_cls).unsqueeze(1)  # (B, T, d_model)
        x = x.unsqueeze(2)  # (B, T, 1, d_model) for TemporalConvBlock
        time_pos = _sinusoidal_position_encoding(t, x.shape[-1], x.device, x.dtype)
        x = x + time_pos.view(1, t, 1, -1)
        for blk in self.temporal:
            x = blk(x)
        out = self.head(x.squeeze(2))  # (B, T, 3)
        if rest_root_height is None:
            return out  # legacy: raw root xyz
        rest = rest_root_height.to(dtype=out.dtype, device=out.device).view(-1, 1)  # (B, 1)
        z = rest * (1.0 + out[..., 2])
        return torch.cat([out[..., :2], z.unsqueeze(-1)], dim=-1)


class MorphDescriptorAdapter(nn.Module):
    """Inject a per-robot morphology descriptor into the decoder's morphology tokens.

    Zero-initialized at the descriptor->model boundary so a model warm-started from a
    morph-free ckpt starts IDENTICAL to the baseline and only learns to ADD signal.

    inject:
      "add"      -> bias the post-MGE tokens: m_cls += MLP_cls(d); m_per_dof += MLP_dof(d)
      "node_aug" -> bias the pre-graph node hidden (graph-propagated): MGE node_bias = MLP_node(d)
    target (add only): "both" (root m_cls + joints m_per_dof) | "cls_only" (root m_cls only)
    """

    def __init__(self, descriptor_dim: int, d_model: int, inject: str, target: str = "both"):
        super().__init__()
        self.inject = str(inject)
        self.target = str(target)
        if self.inject == "add":
            self.cls_mlp = nn.Sequential(
                nn.Linear(descriptor_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
            )
            nn.init.zeros_(self.cls_mlp[-1].weight)
            nn.init.zeros_(self.cls_mlp[-1].bias)
            if self.target == "both":
                self.dof_mlp = nn.Sequential(
                    nn.Linear(descriptor_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
                )
                nn.init.zeros_(self.dof_mlp[-1].weight)
                nn.init.zeros_(self.dof_mlp[-1].bias)
        elif self.inject == "node_aug":
            self.node_mlp = nn.Sequential(
                nn.Linear(descriptor_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
            )
            nn.init.zeros_(self.node_mlp[-1].weight)
            nn.init.zeros_(self.node_mlp[-1].bias)
        else:
            raise ValueError(f"unknown morph inject {self.inject!r}")

    def node_bias(self, descriptor: torch.Tensor) -> torch.Tensor:  # (B, d_model)
        return self.node_mlp(descriptor)

    def apply_tokens(
        self, m_per_dof: torch.Tensor, m_cls: torch.Tensor, descriptor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m_cls = m_cls + self.cls_mlp(descriptor)
        if self.target == "both":
            m_per_dof = m_per_dof + self.dof_mlp(descriptor).unsqueeze(1)
        return m_per_dof, m_cls


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

class MotionLatentRetargeter(nn.Module):
    """MFME + MGE + CMAD wired into a universal latent + per-robot decoder.

    Forward inputs:
        pose_aa             (B, T, 21, 3)
        root_pose           (B, T, root_dim)
        beta                (B, beta_dim) or None
        part_mask           (B, 6) bool or None
        robot_batch: dict containing
            node_features     (B, K, F)
            slot_assignment   (B, K) long
            hop_matrix        (B, K, K) long
            control_mode      (B, K) long
            dof_padding_mask  (B, K) bool, optional

    Output dict:
        z, mu, logvar           — motion latent
        slot_latents            — per-slot motion side-channel
        graph_hidden            — per-slot hidden states (consumed by CMAD)
        part_mask               — slot presence mask
        m_per_dof, m_cls        — morphology embeddings
        actions                 — (B, T, K) per-DoF actions, padding zeroed
    """

    def __init__(
        self,
        num_human_joints: int = 21,
        num_actions: int = 29,
        beta_dim: int = 16,
        root_dim: int = 9,
        d_model: int = 256,
        z_dim: int = 256,
        slot_z: Optional[int] = None,
        slot_latent_dim: int = 32,
        encoder_graph_layers: int = 2,
        encoder_temporal_layers: int = 4,
        encoder_num_heads: int = 8,
        morphology_feature_dim: int = URDF_FEATURE_DIM,
        morphology_graph_layers: int = 2,
        morphology_num_heads: int = 8,
        decoder_graph_layers: int = 2,
        decoder_temporal_layers: int = 2,
        decoder_num_heads: int = 8,
        max_hops: int = 8,
        num_slots: int = len(DEFAULT_BODY_SLOTS),
        slots: Sequence[str] = DEFAULT_BODY_SLOTS,
        edges: Iterable[Tuple[str, str]] = DEFAULT_BODY_EDGES,
        tokenizer: str = "structured",
        future_offsets: Sequence[int] = (1, 5, 10),
        future_slot_dim: int = 16,
        dropout: float = 0.1,
        latent_mode: str = "global",
        causal: bool = False,
        # Production-trainer compatibility ----------------------------------
        ref_base_height: float = 0.8,
        root_height_range: float = 0.6,
        human_decoder_graph_layers: int = 2,
        human_decoder_num_heads: int = 8,
        unified_human_decoder: bool = False,
        robot_encoder_enabled: bool = False,
        root_height_factorized: bool = False,
        morph_descriptor_dim: int = 0,
        morph_descriptor_inject: str = "add",     # "add" | "node_aug"
        morph_descriptor_target: str = "both",    # "both" | "cls_only"  (add only)
    ):
        super().__init__()
        self.latent_mode = str(latent_mode)
        if self.latent_mode not in {"global", "per_slot"}:
            raise ValueError(f"Unknown latent_mode: {latent_mode!r}")
        self.causal = bool(causal)
        self.num_human_joints = int(num_human_joints)
        self.num_joints = int(num_human_joints)
        self.num_actions = int(num_actions)
        self.beta_dim = int(beta_dim)
        self.root_dim = int(root_dim)
        self.d_model = int(d_model)
        self.root_height_range = float(root_height_range)
        # `latent_architecture` is exposed for production-trainer parity (its
        # compute_losses uses it to decide block_compressor / per-slot routing).
        self.latent_architecture = "structured"
        # Empty `slots` placeholder mirrors the production-trainer field used in
        # spectral aux / world aux setup; we never enable those heads here.
        self.slots = tuple(self.mfme_slots_inferred(slots))

        self.mfme = MorphologyFreeMotionEncoder(
            num_human_joints=num_human_joints,
            beta_dim=beta_dim,
            root_dim=root_dim,
            d_model=d_model,
            z_dim=z_dim,
            slot_z=slot_z,
            slot_latent_dim=slot_latent_dim,
            graph_layers=encoder_graph_layers,
            temporal_layers=encoder_temporal_layers,
            num_heads=encoder_num_heads,
            slots=slots,
            edges=edges,
            tokenizer=tokenizer,
            dropout=dropout,
            latent_mode=self.latent_mode,
            causal=self.causal,
        )
        self.z_dim = int(self.mfme.z_dim)
        self.slot_z = int(getattr(self.mfme, "slot_z", 0) or 0)
        self.slots = tuple(self.mfme.slots)
        # num_control_modes: 3 (robot only: angle/delta_pd/torque) by default to
        # preserve backward compat with existing G1 ckpts (control_embed weight
        # shape (3, d_model)). When unified_human_decoder is set we need a 4th
        # mode (rotation/axis-angle) for SMPL-H ball joints.
        mge_num_control_modes = 4 if unified_human_decoder else 3
        self.mge = MorphologyGraphEncoder(
            feature_dim=morphology_feature_dim,
            d_model=d_model,
            num_heads=morphology_num_heads,
            num_graph_layers=morphology_graph_layers,
            max_hops=max_hops,
            num_slots=num_slots,
            dropout=dropout,
            num_control_modes=mge_num_control_modes,
        )
        # The MFME may have grown its slot count (e.g. +1 for root_state in
        # per_slot mode). Mirror that into CMAD's cross-attention so the
        # motion-side keys/values count matches.
        num_motion_slots = len(self.mfme.slots)
        self.cmad = CrossMorphologyActionDecoder(
            z_dim=self.mfme.z_dim,
            d_model=d_model,
            num_heads=decoder_num_heads,
            num_slots=num_slots,
            graph_layers=decoder_graph_layers,
            temporal_layers=decoder_temporal_layers,
            max_hops=max_hops,
            dropout=dropout,
            latent_mode=self.latent_mode,
            slot_z=getattr(self.mfme, "slot_z", None) or None,
            num_motion_slots=num_motion_slots,
            causal=self.causal,
        )
        self.future_head = FutureSlotHead(
            z_dim=self.mfme.z_dim,
            d_model=d_model,
            num_slots=num_slots,
            future_offsets=future_offsets,
            future_slot_dim=future_slot_dim,
        )

        # ----- AE arm + production-trainer heads ---------------------------
        # `root_embed`, `root_height_head`, `human_decoder` mirror
        # UnifiedLatentRetargeter so the production compute_losses works
        # unchanged. The human decoder is GRAPH-BASED here (per design
        # direction): per-slot z → per-joint pose_aa via SMPL-H kinematic-tree
        # graph attention, vs production's MLP RetargetDecoder.
        residual_hidden = int(d_model)
        if self.latent_mode != "per_slot" or self.slot_z == 0:
            raise ValueError(
                "MotionLatentRetargeter requires latent_mode='per_slot' with slot_z>0 "
                "for the graph human decoder. Set --latent-mode per_slot --slot-z 12."
            )
        self.human_decoder = GraphHumanDecoder(
            slot_z=self.slot_z,
            beta_dim=beta_dim,
            d_model=d_model,
            num_joints=num_human_joints,
            root_dim=root_dim,
            slots=self.mfme.slots,
            slot_joints=DEFAULT_SLOT_JOINTS,
            joint_parents=DEFAULT_SMPLH_BODY_PARENTS,
            graph_layers=int(human_decoder_graph_layers),
            num_heads=int(human_decoder_num_heads),
            dropout=dropout,
        )
        # Unified human decoder: route human reconstruction through the SAME
        # CMAD body as robot actions, using a SMPL-H "embodiment spec" through
        # MGE. Heads diverge (3-d axis-angle for human, 1-d action for robot).
        # See build_human_spec().
        self.unified_human_decoder = bool(unified_human_decoder)
        if self.unified_human_decoder:
            spec = build_human_spec(
                num_joints=num_human_joints,
                joint_parents=DEFAULT_SMPLH_BODY_PARENTS,
                slot_joints=DEFAULT_SLOT_JOINTS,
                slots=self.mfme.slots,
            )
            human_batch = batch_robot_specs([spec])
            for k, v in human_batch.items():
                if torch.is_tensor(v):
                    self.register_buffer(f"_human_spec_{k}", v, persistent=False)
            # Output heads on top of CMAD's per-node features.
            self.human_pose_aa_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 3),
            )
            self.human_root_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, root_dim),
            )
        # Morphology-conditioned root decoder: replaces the old fixed
        # ref_base_height + tanh-delta head. Root xyz is retargeted from the
        # motion latent's root_state slot, conditioned on the robot morphology
        # (m_cls). Standing height emerges from the robot's kinematics; no
        # per-cfg base-height constant. `ref_base_height` is accepted as a
        # constructor arg for signature compat but is NOT used.
        self.root_state_slot_idx = (
            self.mfme.slots.index("root_state")
            if "root_state" in self.mfme.slots
            else (self.mfme.slots.index("global_root") if "global_root" in self.mfme.slots else 0)
        )
        # Factorized z (z = analytic z_stand × learned modulation) requires the
        # robot_batch to carry "rest_root_height" — see RootDecoder.forward.
        self.root_height_factorized = bool(root_height_factorized)
        self.root_decoder = RootDecoder(
            slot_z=self.slot_z,
            d_model=d_model,
            temporal_layers=int(decoder_temporal_layers),
            dropout=dropout,
            causal=self.causal,
        )

        # ----- Morphology descriptor adapter (optional) --------------------
        # Injects a per-robot fit-derived morphology descriptor into the decoder's
        # morphology tokens. Zero-initialized -> a warm-started model starts identical
        # to the morph-free baseline. dim 0 (default) => no adapter, byte-identical.
        self.morph_descriptor_dim = int(morph_descriptor_dim)
        self.morph_descriptor_inject = str(morph_descriptor_inject)
        self.morph_adapter = None
        if self.morph_descriptor_dim > 0:
            self.morph_adapter = MorphDescriptorAdapter(
                descriptor_dim=self.morph_descriptor_dim,
                d_model=d_model,
                inject=self.morph_descriptor_inject,
                target=str(morph_descriptor_target),
            )

        # Robot-batch slot for the production-style forward (set externally).
        self._robot_batch: Optional[Dict[str, torch.Tensor]] = None

        # Production-trainer flags it may probe but we don't use.
        # `frame_model` is a stub Identity so .eval()/.parameters() are
        # silent no-ops — the trainer calls these unconditionally.
        self.spectral_aux_enabled = False
        self.spectral_aux_head = None
        self.world_aux_enabled = False
        self.world_head = None
        self.frame_model = nn.Identity()
        self.block_compressor = None
        self.block_upsampler = None

        # ----- Robot-side motion encoder (recipes 3 + 4) -------------------
        # Off by default so production retargeting runs are unaffected. It owns
        # its per-slot heads (NOT shared with MFME); the alignment loss ties
        # z_robot to the stop-grad z_human so z_robot and
        # z_human occupy the same latent space (alignment reward / world model).
        self.robot_encoder_enabled = bool(robot_encoder_enabled)
        if self.robot_encoder_enabled:
            if self.latent_mode != "per_slot" or self.slot_z == 0:
                raise ValueError(
                    "robot_encoder_enabled requires latent_mode='per_slot' with slot_z>0."
                )
            self.robot_encoder = RobotMotionEncoder(
                d_model=d_model,
                slot_z=self.slot_z,
                root_dim=root_dim,
                slots=self.mfme.slots,
                edges=edges,
                graph_layers=encoder_graph_layers,
                temporal_layers=encoder_temporal_layers,
                num_heads=encoder_num_heads,
                causal=self.causal,
                dropout=dropout,
            )
        else:
            self.robot_encoder = None

    @staticmethod
    def mfme_slots_inferred(slots):
        """Match SkeletalJointTokenizer's behaviour: tuple of strings."""
        return tuple(slots)

    # ------------------------------------------------------------------
    # Aliases for production-trainer attribute compatibility. The trainer's
    # logging / freeze / load paths probe these names directly on `model`;
    # routing them to our sub-modules keeps that code unchanged.
    # ------------------------------------------------------------------

    @property
    def encoder(self):
        return self.mfme.encoder

    @property
    def tokenizer(self):
        return self.mfme.tokenizer

    @property
    def latent_action_decoder(self):
        return self.cmad

    def load_teacher(self, checkpoint, map_location="cpu"):
        # No teacher path. Production trainer only calls this when
        # --init-from-teacher is set; raise so misuse is loud.
        raise RuntimeError(
            "MotionLatentRetargeter has no PoseTransformer frame_model; "
            "do not pass --init-from-teacher."
        )

    # ------------------------------------------------------------------
    # Robot-batch plumbing for the production-style forward
    # ------------------------------------------------------------------

    def set_robot_batch(self, robot_batch: Dict[str, torch.Tensor]) -> None:
        """Stash a B=1 robot_batch template. The production-style ``forward``
        expands it to the actual batch size every step (handles uneven last
        batches without rebuilding)."""
        device = next(self.parameters()).device
        rb = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in robot_batch.items()
        }
        # Squeeze to B=1 so expand_robot_batch can fan out cleanly.
        rb1 = {
            k: (v[:1] if torch.is_tensor(v) and v.shape[0] > 1 else v) for k, v in rb.items()
        }
        self._robot_batch = rb1

    def _expand_robot_batch(self, B: int, device: torch.device) -> Dict[str, torch.Tensor]:
        if self._robot_batch is None:
            raise RuntimeError(
                "MotionLatentRetargeter.forward called without set_robot_batch(rb). "
                "Call model.set_robot_batch(rb) once at trainer startup."
            )
        out: Dict[str, torch.Tensor] = {}
        for k, v in self._robot_batch.items():
            if not torch.is_tensor(v):
                out[k] = v
                continue
            v = v.to(device)
            out[k] = v.expand(B, *v.shape[1:]).contiguous() if v.shape[0] == 1 else v
        return out

    def _expand_human_spec(self, B: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Build a B-sized human spec batch from the cached B=1 buffers
        registered when unified_human_decoder=True."""
        keys = ("node_features", "slot_assignment", "hop_matrix",
                "control_mode", "parent_index", "dof_padding_mask")
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            buf_name = f"_human_spec_{k}"
            if not hasattr(self, buf_name):
                continue
            v = getattr(self, buf_name).to(device)
            out[k] = v.expand(B, *v.shape[1:]).contiguous() if v.shape[0] == 1 else v
        return out

    def encode(
        self,
        pose_aa: torch.Tensor,
        root_pose: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
        sample: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.mfme(pose_aa, root_pose, beta, part_mask, sample=sample)

    def encode_morphology(self, robot_batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Optional morphology-descriptor injection. `morph_descriptor` is absent for
        # the unified-human spec path -> desc is None -> neutral (the human IS the
        # neutral reference), so that path is unchanged.
        desc = robot_batch.get("morph_descriptor")
        node_bias = None
        if self.morph_adapter is not None and desc is not None and self.morph_adapter.inject == "node_aug":
            node_bias = self.morph_adapter.node_bias(desc)
        m_per_dof, m_cls = self.mge(
            node_features=robot_batch["node_features"],
            slot_assignment=robot_batch["slot_assignment"],
            hop_matrix=robot_batch["hop_matrix"],
            control_mode=robot_batch["control_mode"],
            dof_padding_mask=robot_batch.get("dof_padding_mask"),
            node_bias=node_bias,
        )
        if self.morph_adapter is not None and desc is not None and self.morph_adapter.inject == "add":
            m_per_dof, m_cls = self.morph_adapter.apply_tokens(m_per_dof, m_cls, desc)
        return m_per_dof, m_cls

    def encode_robot(
        self,
        dof_pos: torch.Tensor,                 # (B, T, K)
        dof_vel: torch.Tensor,                 # (B, T, K)
        root_features: torch.Tensor,           # (B, T, root_dim)
        sample: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode robot proprioceptive motion into the shared per-slot latent.

        Uses the pinned robot batch (``set_robot_batch``) to fetch the static
        morphology tokens from MGE, then runs the RobotMotionEncoder. Returns
        ``mu_slots`` / ``logvar_slots`` / ``z_slots`` (B, T, S, slot_z), directly
        comparable to MFME's ``z_slots`` for the alignment loss.
        """
        if self.robot_encoder is None:
            raise RuntimeError(
                "encode_robot called but robot_encoder_enabled=False at construction."
            )
        B = dof_pos.shape[0]
        device = dof_pos.device
        rb = self._expand_robot_batch(B, device)
        m_per_dof, _ = self.encode_morphology(rb)
        return self.robot_encoder(
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            root_features=root_features,
            m_per_dof=m_per_dof,
            slot_assignment=rb["slot_assignment"],
            dof_padding_mask=rb.get("dof_padding_mask"),
            sample=sample,
        )

    # ------------------------------------------------------------------
    # forward_morph: legacy 3-positional API (pose_aa, root_pose, robot_batch).
    # Kept for the existing tests + scripts_phaseA/render_morph_compare_to_gmr.py.
    # New code should use forward() (production-signature) below.
    # ------------------------------------------------------------------
    def forward_morph(
        self,
        pose_aa: torch.Tensor,
        root_pose: torch.Tensor,
        robot_batch: Dict[str, torch.Tensor],
        beta: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
        sample: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode(pose_aa, root_pose, beta=beta, part_mask=part_mask, sample=sample)
        m_per_dof, m_cls = self.encode_morphology(robot_batch)
        actions = self.cmad(
            z=encoded["z"],
            slot_hidden=encoded["graph_hidden"],
            slot_part_mask=encoded["part_mask"],
            m_per_dof=m_per_dof,
            slot_assignment=robot_batch["slot_assignment"],
            hop_matrix=robot_batch["hop_matrix"],
            dof_padding_mask=robot_batch.get("dof_padding_mask"),
            z_slots=encoded.get("z_slots"),
        )
        future_slots = self.future_head(encoded["z"])
        out = {
            **encoded,
            "m_per_dof": m_per_dof,
            "m_cls": m_cls,
            "actions": actions,
        }
        if future_slots is not None:
            out["future_slots"] = future_slots
        return out

    # ------------------------------------------------------------------
    # forward: PRODUCTION-COMPATIBLE signature so train_graph_latent.py's
    # compute_losses works UNCHANGED. Mirrors UnifiedLatentRetargeter.forward
    # field-for-field. The robot_batch is pulled from self._robot_batch
    # (call ``set_robot_batch`` once at trainer startup).
    # ------------------------------------------------------------------
    def forward(
        self,
        pose_body: torch.Tensor,                        # (B, T, num_joints*3) flat
        root_features: Optional[torch.Tensor] = None,   # (B, T, root_dim)
        beta: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
        robot_id: Optional[torch.Tensor] = None,        # ignored — morph-free
        sample: Optional[bool] = None,
        base_blend: Optional[float] = None,             # ignored — no teacher path
        use_frame_path: bool = False,                   # ignored — no PoseTransformer
        joint_pos: Optional[torch.Tensor] = None,       # encoder geometry (off in E2b)
        joint_vel: Optional[torch.Tensor] = None,
        joint_contact: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, window, dim = pose_body.shape
        if dim != self.num_joints * 3:
            raise ValueError(
                f"pose_body last dim must be {self.num_joints * 3}, got {dim}"
            )
        if root_features is None:
            root_features = pose_body.new_zeros(batch_size, window, self.root_dim)
        device = pose_body.device

        pose_aa = pose_body.reshape(batch_size, window, self.num_joints, 3)
        encoded = self.encode(pose_aa, root_features, beta=beta, part_mask=part_mask, sample=sample)

        rb = self._expand_robot_batch(batch_size, device)
        m_per_dof, _m_cls = self.encode_morphology(rb)
        actions = self.cmad(
            z=encoded["z"],
            slot_hidden=encoded["graph_hidden"],
            slot_part_mask=encoded["part_mask"],
            m_per_dof=m_per_dof,
            slot_assignment=rb["slot_assignment"],
            hop_matrix=rb["hop_matrix"],
            dof_padding_mask=rb.get("dof_padding_mask"),
            z_slots=encoded.get("z_slots"),
        )

        # Morphology-conditioned root xyz: decode the root_state slot latent
        # onto the robot graph, conditioned on the robot morphology token
        # (_m_cls = MGE CLS, aggregates the kinematic tree). Standing height +
        # reachable xyz range emerge from the robot's geometry; no fixed
        # ref_base_height. root_xyz: (B, T, 3). `root_height` (the z component)
        # is exposed for the trainer's physics foot loss + logging.
        root_slot_z = encoded["z_slots"][:, :, self.root_state_slot_idx, :]  # (B, T, slot_z)
        rest_h = rb.get("rest_root_height") if self.root_height_factorized else None
        if self.root_height_factorized and rest_h is None:
            raise RuntimeError(
                "root_height_factorized=True but robot_batch lacks 'rest_root_height' "
                "(compute it with rest_root_height_from_fk and add it to the batch "
                "passed to set_robot_batch) — without it the z head's residual "
                "output would be misread as an absolute height."
            )
        root_xyz = self.root_decoder(root_slot_z, _m_cls, rest_root_height=rest_h)  # (B, T, 3)
        root_height = root_xyz[..., 2:3]  # (B, T, 1)

        # AE arm — graph-based human pose reconstruction from z_slots.
        # Two paths supported:
        #   (a) legacy GraphHumanDecoder: per-slot z + beta → per-joint pose_aa
        #       via a fixed SMPL-H kinematic-tree graph attention.
        #   (b) unified decoder: SMPL-H spec runs through MGE just like a
        #       robot, and CMAD's body produces per-joint features that two
        #       small heads project to (pose_aa, root_recon). Forces the
        #       MGE+CMAD weights to handle ≥2 morphologies before any cross-
        #       robot training.
        z_slots = encoded["z_slots"]  # raises KeyError if not in per_slot mode
        if self.unified_human_decoder:
            human_batch = self._expand_human_spec(batch_size, device)
            m_human, _ = self.encode_morphology(human_batch)
            x_human = self.cmad.decode_features(
                z=encoded["z"],
                slot_hidden=encoded["graph_hidden"],
                slot_part_mask=encoded["part_mask"],
                m_per_dof=m_human,
                slot_assignment=human_batch["slot_assignment"],
                hop_matrix=human_batch["hop_matrix"],
                dof_padding_mask=human_batch.get("dof_padding_mask"),
                z_slots=z_slots,
            )  # (B, T, num_joints, d_model)
            human_pose_aa_recon = self.human_pose_aa_head(x_human)  # (B, T, J, 3)
            human_root_recon = self.human_root_head(x_human.mean(dim=2))  # (B, T, root_dim)
        else:
            human_out = self.human_decoder(z_slots, beta)
            human_pose_aa_recon = human_out[..., : self.num_joints * 3].view(
                batch_size, window, self.num_joints, 3
            )
            human_root_recon = human_out[..., self.num_joints * 3 :]

        # No teacher / no residual: actions == latent_actions, both zero-base.
        latent_actions = actions
        base_actions = torch.zeros_like(latent_actions)
        residual = torch.zeros_like(latent_actions)

        out = {
            "actions": actions,
            "base_actions": base_actions,
            "latent_actions": latent_actions,
            "residual": residual,
            "z": encoded["z"],
            "root_xyz": root_xyz,
            "latent_root_xyz": root_xyz,
            "root_height": root_height,
            "latent_root_height": root_height,
            "mu": encoded["mu"],
            "logvar": encoded["logvar"],
            "slot_latents": encoded["slot_latents"],
            "graph_hidden": encoded["graph_hidden"],
            "part_mask": encoded["part_mask"],
            "human_pose_aa_recon": human_pose_aa_recon,
            "human_root_recon": human_root_recon,
        }
        return out


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def masked_action_l2(
    pred: torch.Tensor,                # (B, T, K)
    target: torch.Tensor,              # (B, T, K)
    dof_padding_mask: Optional[torch.Tensor] = None,  # (B, K) bool, True=pad
) -> torch.Tensor:
    """Per-DoF L2 normalized by the number of *valid* DoFs in each row.

    Prevents high-K robots from dominating the gradient when mixing robots.
    """
    diff = (pred - target) ** 2
    if dof_padding_mask is not None:
        valid = (~dof_padding_mask).to(diff.dtype)  # (B, K)
        diff = diff * valid[:, None, :]
        denom = valid.sum(dim=1).clamp_min(1.0)
        per_sample = diff.sum(dim=(1, 2)) / (diff.shape[1] * denom)
        return per_sample.mean()
    return diff.mean()


def kl_divergence_loss(
    mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0
) -> torch.Tensor:
    """Standard VAE KL with optional per-dim free-bits floor.

    Free bits per dim prevents posterior collapse when a few high-frequency
    latent dims would otherwise be regularized to zero.
    """
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    if free_bits > 0:
        kl = torch.clamp(kl, min=free_bits)
    return kl.mean()


def latent_smoothness_loss(z: torch.Tensor) -> torch.Tensor:
    if z.shape[1] < 2:
        return z.new_zeros(())
    return (z[:, 1:] - z[:, :-1]).pow(2).mean()
