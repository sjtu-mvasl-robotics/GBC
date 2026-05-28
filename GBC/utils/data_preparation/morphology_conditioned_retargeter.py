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
    HumanSlotTokenizer,
    SkeletalJointTokenizer,
    TemporalConvBlock,
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
) -> RobotSpec:
    """Build a :class:`RobotSpec` from a loaded ``RobotKinematics`` instance."""
    dof_names = tuple(rk.get_dof_names())
    node_features = rk.get_dof_features().detach().cpu()
    edges = rk.get_dof_edges(include_root_edges=True)
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
        slot_latent_dim: int = 32,
        graph_layers: int = 2,
        temporal_layers: int = 4,
        num_heads: int = 8,
        slots: Sequence[str] = DEFAULT_BODY_SLOTS,
        edges: Iterable[Tuple[str, str]] = DEFAULT_BODY_EDGES,
        slot_joints: Optional[Dict[str, Sequence[int]]] = None,
        tokenizer: str = "structured",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_human_joints = int(num_human_joints)
        self.beta_dim = int(beta_dim)
        self.root_dim = int(root_dim)
        self.d_model = int(d_model)
        self.z_dim = int(z_dim)
        self.slots = tuple(slots)

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

        self.encoder = GraphTemporalEncoder(
            d_model=d_model,
            num_heads=num_heads,
            slots=self.slots,
            edges=edges,
            graph_layers=graph_layers,
            temporal_layers=temporal_layers,
            dropout=dropout,
        )
        self.slot_projector = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, slot_latent_dim),
        )
        self.mu_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, z_dim))
        self.logvar_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, z_dim))

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
        pooled = _masked_mean(graph_hidden, part_mask[:, None, :], dim=2)
        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled)
        sample_eff = self.training if sample is None else sample
        z = self.reparameterize(mu, logvar, sample_eff)
        slot_latents = self.slot_projector(graph_hidden)
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
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
        num_control_modes: int = 3,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, k, _ = node_features.shape
        x = self.feature_proj(node_features)
        x = x + self.slot_embed(slot_assignment)
        x = x + self.control_embed(control_mode)

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

    def __init__(self, d_model: int, num_heads: int, num_slots: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_slots = num_slots
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Learned bias table: (assigned_slot, target_slot) -> per-head bias.
        # Init: +1 on the diagonal (own slot) and the root column, 0 elsewhere.
        bias = torch.zeros(num_slots, num_slots, num_heads)
        for s in range(num_slots):
            bias[s, s] = 1.0
            bias[s, 0] = 1.0  # global_root is index 0
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
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.max_hops = int(max_hops)
        self.z_proj = nn.Sequential(nn.Linear(z_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.dof_pos_embed = nn.Embedding(256, d_model)  # plenty of headroom
        self.cross = _SoftSlotCrossAttention(d_model, num_heads, num_slots, dropout)
        self.cross_norm = nn.LayerNorm(d_model)

        self.hop_bias = nn.Embedding(max_hops + 2, num_heads)
        self.same_slot_bias = nn.Embedding(2, num_heads)
        self.graph_layers = nn.ModuleList(
            GraphormerLayer(d_model, num_heads, dropout) for _ in range(graph_layers)
        )
        self.temporal_blocks = nn.ModuleList(
            TemporalConvBlock(d_model, dropout) for _ in range(temporal_layers)
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

    def forward(
        self,
        z: torch.Tensor,                  # (B, T, z_dim)
        slot_hidden: torch.Tensor,        # (B, T, S, d_model)
        slot_part_mask: torch.Tensor,     # (B, S) bool
        m_per_dof: torch.Tensor,          # (B, K, d_model)
        slot_assignment: torch.Tensor,    # (B, K) long
        hop_matrix: torch.Tensor,         # (B, K, K) long
        dof_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, t, _ = z.shape
        k = m_per_dof.shape[1]
        device = z.device

        dof_ids = torch.arange(k, device=device)
        pos = self.dof_pos_embed(dof_ids).view(1, 1, k, self.d_model)
        z_tok = self.z_proj(z).unsqueeze(2)                # (B, T, 1, d)
        m_tok = m_per_dof.unsqueeze(1)                     # (B, 1, K, d)
        q = m_tok + z_tok + pos                            # (B, T, K, d)

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

        # 4) per-DoF scalar action
        actions = self.action_head(x).squeeze(-1)  # (B, T, K)
        if dof_padding_mask is not None:
            actions = actions.masked_fill(dof_padding_mask[:, None, :], 0.0)
        return actions


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
        beta_dim: int = 16,
        root_dim: int = 9,
        d_model: int = 256,
        z_dim: int = 256,
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
    ):
        super().__init__()
        self.mfme = MorphologyFreeMotionEncoder(
            num_human_joints=num_human_joints,
            beta_dim=beta_dim,
            root_dim=root_dim,
            d_model=d_model,
            z_dim=z_dim,
            slot_latent_dim=slot_latent_dim,
            graph_layers=encoder_graph_layers,
            temporal_layers=encoder_temporal_layers,
            num_heads=encoder_num_heads,
            slots=slots,
            edges=edges,
            tokenizer=tokenizer,
            dropout=dropout,
        )
        self.mge = MorphologyGraphEncoder(
            feature_dim=morphology_feature_dim,
            d_model=d_model,
            num_heads=morphology_num_heads,
            num_graph_layers=morphology_graph_layers,
            max_hops=max_hops,
            num_slots=num_slots,
            dropout=dropout,
        )
        self.cmad = CrossMorphologyActionDecoder(
            z_dim=z_dim,
            d_model=d_model,
            num_heads=decoder_num_heads,
            num_slots=num_slots,
            graph_layers=decoder_graph_layers,
            temporal_layers=decoder_temporal_layers,
            max_hops=max_hops,
            dropout=dropout,
        )
        self.future_head = FutureSlotHead(
            z_dim=z_dim,
            d_model=d_model,
            num_slots=num_slots,
            future_offsets=future_offsets,
            future_slot_dim=future_slot_dim,
        )

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
        return self.mge(
            node_features=robot_batch["node_features"],
            slot_assignment=robot_batch["slot_assignment"],
            hop_matrix=robot_batch["hop_matrix"],
            control_mode=robot_batch["control_mode"],
            dof_padding_mask=robot_batch.get("dof_padding_mask"),
        )

    def forward(
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
