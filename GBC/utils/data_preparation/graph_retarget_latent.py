from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from GBC.utils.data_preparation.pose_transformer import PoseTransformer
from GBC.utils.data_preparation.temporal_modules import CausalConvTemporalEncoder
from GBC.utils.data_preparation.block_causal_compress import (
    BlockCausalRoPELayer,
    BlockCausalUpsampler,
    BlockCompressor,
    build_block_causal_mask,
    pick_num_heads,
)
from GBC.utils.data_preparation.spectral_modules import (
    MultiStreamSpectralAuxiliaryHead,
)
from GBC.utils.data_preparation.world_modules import LatentForwardHead


DEFAULT_BODY_SLOTS: Tuple[str, ...] = (
    "global_root",
    "trunk_head",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
)


DEFAULT_BODY_EDGES: Tuple[Tuple[str, str], ...] = (
    ("global_root", "trunk_head"),
    ("global_root", "left_leg"),
    ("global_root", "right_leg"),
    ("trunk_head", "left_arm"),
    ("trunk_head", "right_arm"),
    ("left_arm", "right_arm"),
    ("left_leg", "right_leg"),
)


DEFAULT_SLOT_JOINTS: Dict[str, Tuple[int, ...]] = {
    # pose_aa indexes the 21 SMPL-H body joints after the root Pelvis joint.
    "trunk_head": (2, 5, 8, 11, 12, 13, 14),
    "left_arm": (15, 17, 19),
    "right_arm": (16, 18, 20),
    "left_leg": (0, 3, 6, 9),
    "right_leg": (1, 4, 7, 10),
}

DEFAULT_SMPLH_BODY_PARENTS: Tuple[int, ...] = (
    -1,  # L_Hip
    -1,  # R_Hip
    -1,  # Torso
    0,  # L_Knee
    1,  # R_Knee
    2,  # Spine
    3,  # L_Ankle
    4,  # R_Ankle
    5,  # Chest
    6,  # L_Toe
    7,  # R_Toe
    8,  # Neck
    8,  # L_Thorax
    8,  # R_Thorax
    11,  # Head
    12,  # L_Shoulder
    13,  # R_Shoulder
    15,  # L_Elbow
    16,  # R_Elbow
    17,  # L_Wrist
    18,  # R_Wrist
)


def _sinusoidal_position_encoding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype)
        * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / dim)
    )
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
    return pe


def _edge_attention_mask(
    slots: Sequence[str],
    edges: Iterable[Tuple[str, str]],
    device: torch.device,
    root_global_hub: bool = True,
) -> torch.Tensor:
    slot_to_idx = {name: i for i, name in enumerate(slots)}
    allowed = torch.eye(len(slots), dtype=torch.bool, device=device)

    for source, target in edges:
        if source not in slot_to_idx or target not in slot_to_idx:
            continue
        i = slot_to_idx[source]
        j = slot_to_idx[target]
        allowed[i, j] = True
        allowed[j, i] = True

    # root_global_hub=True (legacy/baseline): global_root attends to ALL slots,
    # turning the slot graph into a star. That makes the root token aggregate
    # arm+leg+trunk in one hop, which leaks unrelated regions into each other
    # through root (arm -> root -> leg). root_global_hub=False (structured
    # locality fix, plan A0) keeps ONLY the provided tree edges, so a leaf slot's
    # receptive field stays {itself, its tree neighbours} and cross-region
    # coupling needs multiple hops.
    if root_global_hub and "global_root" in slot_to_idx:
        root_idx = slot_to_idx["global_root"]
        allowed[root_idx, :] = True
        allowed[:, root_idx] = True

    return ~allowed


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype)
    while weight.ndim < x.ndim:
        weight = weight.unsqueeze(-1)
    total = (x * weight).sum(dim=dim)
    denom = weight.sum(dim=dim).clamp_min(1.0)
    return total / denom


def _attention_mask_from_edges(num_tokens: int, edges: Iterable[Tuple[int, int]], device: torch.device) -> torch.Tensor:
    allowed = torch.eye(num_tokens, dtype=torch.bool, device=device)
    for source, target in edges:
        if 0 <= source < num_tokens and 0 <= target < num_tokens:
            allowed[source, target] = True
            allowed[target, source] = True
    return ~allowed


def _slot_index_for_joints(
    slots: Sequence[str],
    slot_joints: Dict[str, Tuple[int, ...]],
    num_human_joints: int,
) -> torch.Tensor:
    slot_to_idx = {slot: idx for idx, slot in enumerate(slots)}
    slot_ids = torch.zeros(num_human_joints + 1, dtype=torch.long)
    root_idx = slot_to_idx.get("global_root", 0)
    slot_ids[0] = root_idx
    for slot, joints in slot_joints.items():
        if slot not in slot_to_idx:
            continue
        for joint_idx in joints:
            if 0 <= joint_idx < num_human_joints:
                slot_ids[joint_idx + 1] = slot_to_idx[slot]
    return slot_ids


def _part_mask_to_token_mask(
    part_mask: torch.Tensor,
    slot_ids: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    slot_ids = slot_ids.to(device=device)
    return part_mask.to(device=device, dtype=torch.bool)[:, slot_ids].reshape(batch_size, -1)


class GraphTokenBlock(nn.Module):
    """Self-attention block with a fixed sparse graph mask."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
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
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ffn(x))


class HumanSlotTokenizer(nn.Module):
    """Maps SMPL-style pose/root inputs into canonical body-slot tokens."""

    def __init__(
        self,
        num_human_joints: int,
        beta_dim: int,
        d_model: int,
        slots: Sequence[str] = DEFAULT_BODY_SLOTS,
        slot_joints: Optional[Dict[str, Sequence[int]]] = None,
        root_dim: int = 9,
    ):
        super().__init__()
        self.num_human_joints = num_human_joints
        self.beta_dim = beta_dim
        self.d_model = d_model
        self.slots = tuple(slots)
        self.root_dim = root_dim

        slot_joints = slot_joints or DEFAULT_SLOT_JOINTS
        self.slot_joints: Dict[str, Tuple[int, ...]] = {}
        self.slot_mlps = nn.ModuleDict()
        for slot in self.slots:
            if slot in ("global_root", "root_state"):
                input_dim = root_dim
                self.slot_joints[slot] = tuple()
            else:
                joints = tuple(j for j in slot_joints.get(slot, ()) if j < num_human_joints)
                self.slot_joints[slot] = joints
                input_dim = max(1, len(joints) * 3)
            self.slot_mlps[slot] = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )

        self.beta_embed = nn.Sequential(
            nn.Linear(beta_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.missing_part_tokens = nn.Parameter(torch.zeros(len(self.slots), d_model))

    def forward(
        self,
        pose_aa: torch.Tensor,
        root_pose: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pose_aa.ndim != 4:
            raise ValueError("pose_aa must have shape (B, T, J, 3)")
        if root_pose.ndim != 3:
            raise ValueError("root_pose must have shape (B, T, root_dim)")

        batch_size, window, _, _ = pose_aa.shape
        tokens = []
        for slot in self.slots:
            if slot in ("global_root", "root_state"):
                feature = root_pose
            else:
                joints = self.slot_joints[slot]
                if joints:
                    feature = pose_aa[:, :, joints, :].reshape(batch_size, window, -1)
                else:
                    feature = pose_aa.new_zeros(batch_size, window, 1)
            tokens.append(self.slot_mlps[slot](feature))

        slot_tokens = torch.stack(tokens, dim=2)
        if beta is None:
            beta = pose_aa.new_zeros(batch_size, self.beta_dim)
        beta_token = self.beta_embed(beta).view(batch_size, 1, 1, self.d_model)
        slot_tokens = slot_tokens + beta_token

        if part_mask is None:
            part_mask = torch.ones(batch_size, len(self.slots), dtype=torch.bool, device=pose_aa.device)
        else:
            part_mask = part_mask.to(device=pose_aa.device, dtype=torch.bool)

        missing = self.missing_part_tokens.view(1, 1, len(self.slots), self.d_model)
        slot_tokens = torch.where(part_mask.view(batch_size, 1, len(self.slots), 1), slot_tokens, missing)
        return slot_tokens, part_mask


class SkeletalJointTokenizer(nn.Module):
    """Encode SMPL-H joints first, then pool them into body slots.

    This keeps limb-internal structure visible to the encoder before body-slot
    fusion. It is intended as the structured alternative to HumanSlotTokenizer's
    flatten-then-MLP tokenization.
    """

    def __init__(
        self,
        num_human_joints: int,
        beta_dim: int,
        d_model: int,
        slots: Sequence[str] = DEFAULT_BODY_SLOTS,
        slot_joints: Optional[Dict[str, Sequence[int]]] = None,
        root_dim: int = 9,
        joint_parents: Sequence[int] = DEFAULT_SMPLH_BODY_PARENTS,
        graph_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        joint_geometry_dim: int = 0,
        joint_velocity_dim: int = 0,
        joint_contact_dim: int = 0,
    ):
        super().__init__()
        self.num_human_joints = int(num_human_joints)
        self.beta_dim = int(beta_dim)
        self.d_model = int(d_model)
        self.slots = tuple(slots)
        self.root_dim = int(root_dim)
        self.joint_geometry_dim = int(joint_geometry_dim)
        self.joint_velocity_dim = int(joint_velocity_dim)
        self.joint_contact_dim = int(joint_contact_dim)

        raw_slot_joints = slot_joints or DEFAULT_SLOT_JOINTS
        self.slot_joints: Dict[str, Tuple[int, ...]] = {
            slot: tuple(j for j in raw_slot_joints.get(slot, ()) if j < num_human_joints)
            for slot in self.slots
        }
        slot_ids = _slot_index_for_joints(self.slots, self.slot_joints, self.num_human_joints)
        self.register_buffer("slot_ids", slot_ids, persistent=False)

        # Per-joint modalities (angle-axis + optional FK pos / vel / contact) are
        # CONCATENATED along the feature axis and embedded once. Summing separate
        # nonlinear per-modality embeds entangles channels — a learned mix of
        # "3 rad pose, 0 vel" can collide with "0 pose, 3 rad/s vel" — so the
        # encoder cannot tell which modality is which. Concat keeps them distinct.
        self.joint_input_dim = (
            3 + self.joint_geometry_dim + self.joint_velocity_dim + self.joint_contact_dim
        )
        self.joint_input_embed = nn.Sequential(
            nn.Linear(self.joint_input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.root_embed = nn.Sequential(
            nn.Linear(root_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        # Independent "root_state" slot: a pelvis-only token built straight from
        # root features, BYPASSING the joint-tree attention, so it never absorbs
        # limb info. It is also isolated in the slot graph (no edges), so it stays
        # a pure root-state latent that the decoder can share into every DoF
        # without leaking one limb into another.
        self.root_state_embed = nn.Sequential(
            nn.Linear(root_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.beta_embed = nn.Sequential(
            nn.Linear(beta_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.joint_embed = nn.Embedding(num_human_joints + 1, d_model)
        self.slot_embed = nn.Embedding(len(self.slots), d_model)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.graph_blocks = nn.ModuleList(GraphTokenBlock(d_model, num_heads, dropout) for _ in range(graph_layers))

        edges: list[Tuple[int, int]] = []
        for joint_idx, parent_idx in enumerate(joint_parents[:num_human_joints]):
            token_idx = joint_idx + 1
            parent_token_idx = 0 if parent_idx < 0 else parent_idx + 1
            edges.append((parent_token_idx, token_idx))
        self.register_buffer(
            "attn_mask_template",
            _attention_mask_from_edges(num_human_joints + 1, edges, torch.device("cpu")),
            persistent=False,
        )

    def forward(
        self,
        pose_aa: torch.Tensor,
        root_pose: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
        joint_pos: Optional[torch.Tensor] = None,
        joint_vel: Optional[torch.Tensor] = None,
        joint_contact: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pose_aa.ndim != 4:
            raise ValueError("pose_aa must have shape (B, T, J, 3)")
        if root_pose.ndim != 3:
            raise ValueError("root_pose must have shape (B, T, root_dim)")

        batch_size, window, num_joints, _ = pose_aa.shape
        if num_joints != self.num_human_joints:
            raise ValueError(f"Expected {self.num_human_joints} human joints, got {num_joints}")
        if beta is None:
            beta = pose_aa.new_zeros(batch_size, self.beta_dim)
        if part_mask is None:
            part_mask = torch.ones(batch_size, len(self.slots), dtype=torch.bool, device=pose_aa.device)
        else:
            part_mask = part_mask.to(device=pose_aa.device, dtype=torch.bool)

        root_token = self.root_embed(root_pose).unsqueeze(2)
        # Concatenate per-joint modalities along the feature axis. Any configured
        # modality that is absent at call time is zero-filled so the input width
        # stays fixed at self.joint_input_dim.
        parts = [pose_aa]
        if self.joint_geometry_dim > 0:
            if joint_pos is not None:
                if joint_pos.shape[:3] != pose_aa.shape[:3] or joint_pos.shape[-1] != self.joint_geometry_dim:
                    raise ValueError(
                        f"joint_pos must be (B,T,{num_joints},{self.joint_geometry_dim}); got {tuple(joint_pos.shape)}"
                    )
                parts.append(joint_pos)
            else:
                parts.append(pose_aa.new_zeros(batch_size, window, num_joints, self.joint_geometry_dim))
        if self.joint_velocity_dim > 0:
            if joint_vel is not None:
                if joint_vel.shape[:3] != pose_aa.shape[:3] or joint_vel.shape[-1] != self.joint_velocity_dim:
                    raise ValueError(
                        f"joint_vel must be (B,T,{num_joints},{self.joint_velocity_dim}); got {tuple(joint_vel.shape)}"
                    )
                parts.append(joint_vel)
            else:
                parts.append(pose_aa.new_zeros(batch_size, window, num_joints, self.joint_velocity_dim))
        if self.joint_contact_dim > 0:
            if joint_contact is not None:
                if joint_contact.shape[:3] != pose_aa.shape[:3] or joint_contact.shape[-1] != self.joint_contact_dim:
                    raise ValueError(
                        f"joint_contact must be (B,T,{num_joints},{self.joint_contact_dim}); got {tuple(joint_contact.shape)}"
                    )
                parts.append(joint_contact)
            else:
                parts.append(pose_aa.new_zeros(batch_size, window, num_joints, self.joint_contact_dim))
        joint_input = torch.cat(parts, dim=-1)  # (B, T, num_joints, joint_input_dim)
        joint_tokens = self.joint_input_embed(joint_input)
        tokens = torch.cat([root_token, joint_tokens], dim=2)

        token_count = self.num_human_joints + 1
        joint_ids = torch.arange(token_count, device=pose_aa.device)
        slot_ids = self.slot_ids.to(device=pose_aa.device)
        tokens = tokens + self.joint_embed(joint_ids).view(1, 1, token_count, -1)
        tokens = tokens + self.slot_embed(slot_ids).view(1, 1, token_count, -1)
        tokens = tokens + self.beta_embed(beta).view(batch_size, 1, 1, self.d_model)

        token_mask = _part_mask_to_token_mask(part_mask, self.slot_ids, batch_size, pose_aa.device)
        tokens = torch.where(token_mask.view(batch_size, 1, token_count, 1), tokens, self.missing_token)

        flat = tokens.reshape(batch_size * window, token_count, self.d_model)
        key_padding_mask = ~token_mask[:, None, :].expand(batch_size, window, token_count).reshape(
            batch_size * window, token_count
        )
        attn_mask = self.attn_mask_template.to(device=pose_aa.device)
        for block in self.graph_blocks:
            flat = block(flat, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        tokens = flat.reshape(batch_size, window, token_count, self.d_model)

        slot_tokens = []
        for slot_idx, slot in enumerate(self.slots):
            if slot == "global_root":
                slot_tokens.append(tokens[:, :, 0, :])
                continue
            if slot == "root_state":
                # Pure pelvis token: built from root features only, NOT from the
                # joint-tree-attended tokens (which carry limb info).
                slot_tokens.append(self.root_state_embed(root_pose))
                continue
            joints = self.slot_joints.get(slot, ())
            if joints:
                token_indices = torch.tensor([j + 1 for j in joints], device=pose_aa.device, dtype=torch.long)
                slot_tokens.append(tokens.index_select(2, token_indices).mean(dim=2))
            else:
                slot_tokens.append(tokens.new_zeros(batch_size, window, self.d_model))
        return torch.stack(slot_tokens, dim=2), part_mask


class GraphFuseBlock(nn.Module):
    """Masked graph attention over canonical body slots for each frame."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        slots: Sequence[str] = DEFAULT_BODY_SLOTS,
        edges: Iterable[Tuple[str, str]] = DEFAULT_BODY_EDGES,
        dropout: float = 0.1,
        root_global_hub: bool = True,
    ):
        super().__init__()
        self.slots = tuple(slots)
        self.edges = tuple(edges)
        self.root_global_hub = bool(root_global_hub)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, part_mask: torch.Tensor) -> torch.Tensor:
        batch_size, window, num_slots, d_model = x.shape
        flat = x.reshape(batch_size * window, num_slots, d_model)
        key_padding_mask = ~part_mask[:, None, :].expand(batch_size, window, num_slots).reshape(batch_size * window, num_slots)
        attn_mask = _edge_attention_mask(self.slots, self.edges, x.device, root_global_hub=self.root_global_hub)

        attn_out, _ = self.attn(flat, flat, flat, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        flat = self.norm1(flat + attn_out)
        flat = self.norm2(flat + self.ffn(flat))
        return flat.reshape(batch_size, window, num_slots, d_model)


class GraphTemporalEncoder(nn.Module):
    # Bidirectional transformer over the time axis — acausal.
    # These class-level attributes exist so the trainer's metric logging
    # (getattr(model.encoder, "is_causal", default)) reports the correct
    # value without falling back to a default.
    is_causal: bool = False
    total_lookahead_frames: int = -1  # sentinel: "full window" (set per-call below)

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        slots: Sequence[str] = DEFAULT_BODY_SLOTS,
        edges: Iterable[Tuple[str, str]] = DEFAULT_BODY_EDGES,
        graph_layers: int = 2,
        temporal_layers: int = 2,
        dropout: float = 0.1,
        per_slot_temporal: bool = False,
        root_global_hub: bool = True,
        causal: bool = False,
        block_size: int = 1,
    ):
        super().__init__()
        self.slots = tuple(slots)
        # per_slot_temporal=False (default): legacy full (T*num_slots) bidirectional
        # attention — preserved byte-for-byte for the baseline path and the
        # standalone GraphRetargetVAE. per_slot_temporal=True (structured path):
        # temporal attention runs WITHIN each slot, so cross-slot mixing happens
        # only through the edge-masked GraphFuseBlock. Without this, the full
        # T*num_slots attention blends e.g. leg@t' into arm@t, defeating the
        # per-slot latent locality the structured decoder relies on (plan A0).
        self.per_slot_temporal = bool(per_slot_temporal)
        # Per-instance override of the class-level sentinels so the trainer's
        # metric logger (getattr(encoder, "is_causal"/"total_lookahead_frames"))
        # reports the truth: causal => 0 frames of lookahead; bidirectional => -1
        # ("full window").
        self.is_causal = bool(causal)
        self.total_lookahead_frames = 0 if causal else -1
        self.slot_embed = nn.Parameter(torch.randn(1, 1, len(self.slots), d_model) * 0.02)
        self.graph_blocks = nn.ModuleList(
            GraphFuseBlock(d_model, num_heads, self.slots, edges, dropout, root_global_hub=root_global_hub)
            for _ in range(graph_layers)
        )
        # block_size>1 => block-causal RoPE temporal (the clip-compression path):
        # bidirectional WITHIN a block, causal ACROSS blocks. RoPE supplies the
        # position signal inside attention, so this path skips the additive
        # sinusoidal PE. Graph fusion above is per-frame (no temporal mixing), so
        # block-causality of the whole encoder is preserved.
        self.block_size = int(block_size)
        if self.block_size > 1:
            if not self.per_slot_temporal:
                raise ValueError(
                    "block_size>1 (block-causal compression) requires "
                    "per_slot_temporal=True (latent_architecture='structured')."
                )
            self.is_causal = True
            self.total_lookahead_frames = self.block_size - 1
            self.block_temporal_layers = nn.ModuleList(
                BlockCausalRoPELayer(d_model, num_heads, dropout) for _ in range(temporal_layers)
            )
            self.temporal_encoder = None
        else:
            self.block_temporal_layers = None
            temporal_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)

    def forward(self, slot_tokens: torch.Tensor, part_mask: torch.Tensor) -> torch.Tensor:
        batch_size, window, num_slots, d_model = slot_tokens.shape
        x = slot_tokens + self.slot_embed

        for block in self.graph_blocks:
            x = block(x, part_mask)

        if self.block_size > 1:
            # Block-causal RoPE temporal, per slot. RoPE is applied inside
            # attention => no additive PE. Mask: within-block bidirectional,
            # across-block causal. window must be divisible by block_size.
            flat = x.permute(0, 2, 1, 3).reshape(batch_size * num_slots, window, d_model)
            positions = torch.arange(window, device=x.device)
            allowed = build_block_causal_mask(window, self.block_size, x.device, allowed_true=True)
            for layer in self.block_temporal_layers:
                flat = layer(flat, positions, allowed)
            return flat.reshape(batch_size, num_slots, window, d_model).permute(0, 2, 1, 3).contiguous()

        time_pos = _sinusoidal_position_encoding(window, d_model, x.device, x.dtype)
        x = x + time_pos.view(1, window, 1, d_model)
        if self.per_slot_temporal:
            # (B, T, S, D) -> (B*S, T, D): temporal attention is confined to each
            # slot's own time series. Missing slots carry placeholder tokens; we do
            # not mask them here (a fully-masked sequence would NaN the softmax) —
            # cross-slot leakage is already blocked by running attention per slot.
            flat = x.permute(0, 2, 1, 3).reshape(batch_size * num_slots, window, d_model)
            if self.is_causal:
                # z_t may only attend to frames <= t. Pass BOTH the explicit
                # float (-inf/0) mask (hits the SDPA fast path) AND is_causal=True;
                # per the PyTorch docs, is_causal=True without a mask is a footgun,
                # and a boolean mask would force the slow path.
                causal_mask = nn.Transformer.generate_square_subsequent_mask(
                    window, device=flat.device, dtype=flat.dtype
                )
                flat = self.temporal_encoder(flat, mask=causal_mask, is_causal=True)
            else:
                flat = self.temporal_encoder(flat)
            return flat.reshape(batch_size, num_slots, window, d_model).permute(0, 2, 1, 3).contiguous()
        if self.is_causal:
            raise ValueError(
                "causal_transformer requires per_slot_temporal=True "
                "(latent_architecture='structured'). Block-causal masking over the "
                "flattened (T*num_slots) sequence is not supported."
            )
        flat = x.reshape(batch_size, window * num_slots, d_model)
        temporal_mask = ~part_mask[:, None, :].expand(batch_size, window, num_slots).reshape(batch_size, window * num_slots)
        flat = self.temporal_encoder(flat, src_key_padding_mask=temporal_mask)
        return flat.reshape(batch_size, window, num_slots, d_model)


class RetargetDecoder(nn.Module):
    def __init__(
        self,
        z_dim: int,
        beta_dim: int,
        output_dim: int,
        hidden_dim: int,
        robot_embedding_dim: int = 0,
        num_robot_types: int = 1,
    ):
        super().__init__()
        self.robot_embedding_dim = robot_embedding_dim
        self.beta_embed = nn.Sequential(
            nn.Linear(beta_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        if robot_embedding_dim > 0:
            self.robot_embed = nn.Embedding(num_robot_types, robot_embedding_dim)
        else:
            self.robot_embed = None

        input_dim = z_dim + hidden_dim + robot_embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        beta: Optional[torch.Tensor],
        robot_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, window, _ = z.shape
        if beta is None:
            beta = z.new_zeros(batch_size, self.beta_embed[0].in_features)
        beta_token = self.beta_embed(beta).unsqueeze(1).expand(batch_size, window, -1)
        features = [z, beta_token]

        if self.robot_embed is not None:
            if robot_id is None:
                robot_id = torch.zeros(batch_size, dtype=torch.long, device=z.device)
            robot_token = self.robot_embed(robot_id).unsqueeze(1).expand(batch_size, window, -1)
            features.append(robot_token)

        return self.net(torch.cat(features, dim=-1))


def _build_robot_dof_edges(
    robot_dof_edges: Optional[Sequence[Tuple[int, int]]],
    num_actions: int,
) -> Tuple[Tuple[int, int], ...]:
    if robot_dof_edges is None:
        raise ValueError("robot_dof_edges must be provided for the structured robot graph decoder.")

    edge_set = set()
    for source, target in robot_dof_edges:
        source_idx = int(source)
        target_idx = int(target)
        if source_idx == target_idx:
            continue
        if not (0 <= source_idx < num_actions and 0 <= target_idx < num_actions):
            raise ValueError(
                f"robot_dof_edges contains out-of-range edge {(source_idx, target_idx)} "
                f"for {num_actions} actions"
            )
        edge_set.add(tuple(sorted((source_idx, target_idx))))
    if num_actions > 1 and not edge_set:
        raise ValueError("robot_dof_edges produced an empty graph for a multi-DoF robot.")
    return tuple(sorted(edge_set))


class TemporalConvBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, kernel_size: int = 5, causal: bool = False):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.causal = bool(causal)
        # Causal: left-pad K-1, conv padding 0 -> output[t] depends only on inputs
        #   [t-(K-1), ..., t]; the following 1x1 conv preserves causality.
        # Acausal (default): symmetric padding K//2 -> output[t] sees [t-K//2, t+K//2].
        # Both keep the time length at T.
        self.left_pad = (self.kernel_size - 1) if self.causal else 0
        conv_padding = 0 if self.causal else (self.kernel_size // 2)
        self.norm = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=self.kernel_size, padding=conv_padding)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(d_model * 2, d_model, kernel_size=1)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, window, num_tokens, d_model = x.shape
        flat = self.norm(x).permute(0, 2, 3, 1).reshape(batch_size * num_tokens, d_model, window)
        if self.causal:
            flat = F.pad(flat, (self.left_pad, 0))
        flat = self.drop2(self.conv2(self.drop1(self.act(self.conv1(flat)))))
        flat = flat.reshape(batch_size, num_tokens, d_model, window).permute(0, 3, 1, 2)
        return x + flat


class RobotActionAdapter(nn.Module):
    """Base interface for robot-conditioned action decoders."""

    robot_dof_graph_source: str = "unknown"
    robot_dof_edge_count: int = 0
    robot_dof_feature_dim: int = 0

    def forward(
        self,
        z: torch.Tensor,
        beta: Optional[torch.Tensor],
        robot_id: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


def build_dof_slot_correspondence(
    robot_dof_names: Sequence[str],
    human_slots: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Map each robot DoF to its kinematically-relevant human body slot(s).

    Region by keyword on the (lower-cased) DoF name:
        hip|knee|ankle      -> {left,right}_leg
        shoulder|elbow|wrist-> {left,right}_arm
        waist|torso|spine   -> trunk_head
    Side is taken from a 'left'/'right' substring. Returns:
        primary_slot[num_dof] (long)  — region slot; the routed-z gather indexes
            this for EVERY DoF, so it is always defined (unknown -> global_root).
        allowed_mask[num_dof, S] (bool) — slots a DoF may attend to:
            {region_slot, global_root}; arms also add trunk_head. UNKNOWN DoFs
            get all-True (safe) and primary_slot = global_root.
        log — list of (dof_name, slot-or-tag) for run_config.
    """
    slot_to_idx = {name: i for i, name in enumerate(human_slots)}
    num_slots = len(human_slots)
    root_idx = slot_to_idx.get("global_root", 0)
    # The shared root context/injection uses the pure 'root_state' slot when it
    # exists (it carries no limb info), else falls back to 'global_root'.
    shared_root = "root_state" if "root_state" in slot_to_idx else "global_root"
    num_dof = len(robot_dof_names)

    primary = torch.full((num_dof,), root_idx, dtype=torch.long)
    allowed = torch.zeros((num_dof, num_slots), dtype=torch.bool)
    log: list = []

    for d, raw in enumerate(robot_dof_names):
        name = str(raw).lower()
        side = "left" if "left" in name else ("right" if "right" in name else None)
        region: Optional[str] = None
        allow_names: Tuple[str, ...] = ()
        if any(k in name for k in ("hip", "knee", "ankle")) and side is not None:
            region = f"{side}_leg"
            allow_names = (region, shared_root)
        elif any(k in name for k in ("shoulder", "elbow", "wrist")) and side is not None:
            region = f"{side}_arm"
            allow_names = (region, "trunk_head", shared_root)
        elif any(k in name for k in ("waist", "torso", "spine")):
            region = "trunk_head"
            allow_names = (region, shared_root)

        if region is not None and region in slot_to_idx:
            primary[d] = slot_to_idx[region]
            for s in allow_names:
                if s in slot_to_idx:
                    allowed[d, slot_to_idx[s]] = True
            log.append((str(raw), region))
        else:
            # Unknown / side-less DoF: route to root, allow ALL slots (safe).
            primary[d] = root_idx
            allowed[d, :] = True
            log.append((str(raw), "UNKNOWN->global_root(all-allowed)"))

    return primary, allowed, log


class RobotGraphActionDecoder(RobotActionAdapter):
    """Decode a PER-SLOT latent through robot dof tokens and a robot graph.

    Each DoF token is seeded only from its region slot latent (+ the shared root
    slot), never from a global broadcast, and its cross-attention to the encoder
    context is masked to its allowed human slots. This keeps a robot DoF's output
    dependent only on its kinematically-relevant human region (+ root), which is
    the spatial-locality fix (plan B). Structured path only.
    """

    def __init__(
        self,
        slot_z: int,
        beta_dim: int,
        output_dim: int,
        d_model: int,
        num_heads: int,
        robot_dof_names: Optional[Sequence[str]] = None,
        human_slots: Sequence[str] = DEFAULT_BODY_SLOTS,
        robot_dof_edges: Optional[Sequence[Tuple[int, int]]] = None,
        robot_dof_features: Optional[torch.Tensor] = None,
        robot_embedding_dim: int = 0,
        num_robot_types: int = 1,
        graph_layers: int = 2,
        temporal_layers: int = 2,
        context_dim: Optional[int] = None,
        decoder_context: str = "masked_graph_hidden",
        root_slot_into_arms: bool = True,
        dropout: float = 0.1,
        temporal_decoder_causal: bool = False,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.d_model = int(d_model)
        self.slot_z = int(slot_z)
        self.robot_embedding_dim = int(robot_embedding_dim)
        self.robot_dof_names = tuple(robot_dof_names or ())
        self.human_slots = tuple(human_slots)
        if decoder_context not in {"masked_graph_hidden", "none"}:
            raise ValueError(f"Unknown decoder_context: {decoder_context}")
        self.decoder_context = str(decoder_context)
        self.root_slot_into_arms = bool(root_slot_into_arms)
        # Shared-root injection uses the pure 'root_state' slot when present (no
        # limb info), else the kinematic 'global_root' hub.
        if "root_state" in self.human_slots:
            self.root_slot_idx = self.human_slots.index("root_state")
        elif "global_root" in self.human_slots:
            self.root_slot_idx = self.human_slots.index("global_root")
        else:
            self.root_slot_idx = 0

        # DoF -> human-slot correspondence (routing + cross-attn mask).
        names_for_corr = self.robot_dof_names or tuple(str(i) for i in range(self.output_dim))
        primary_slot, allowed_mask, corr_log = build_dof_slot_correspondence(names_for_corr, self.human_slots)
        self.dof_slot_correspondence_log = [list(item) for item in corr_log]
        # Per-DoF gate for the shared root-slot contribution. Off for arm DoFs when
        # root_slot_into_arms is False (strict arm isolation).
        arm_idxs = {
            self.human_slots.index(s)
            for s in ("left_arm", "right_arm")
            if s in self.human_slots
        }
        root_gate = torch.ones(self.output_dim, dtype=torch.float32)
        if not self.root_slot_into_arms:
            for d in range(self.output_dim):
                if int(primary_slot[d].item()) in arm_idxs:
                    root_gate[d] = 0.0
                    # Consistency: if root is NOT injected into an arm DoF, the arm
                    # must also not cross-attend to the root slot's context, which
                    # carries leg info (root <-> legs in the slot tree). Otherwise
                    # the masked-context path silently re-leaks leg -> arm.
                    allowed_mask[d, self.root_slot_idx] = False
        self.register_buffer("dof_primary_slot", primary_slot, persistent=False)        # (num_dof,)
        self.register_buffer("dof_slot_attn_mask", ~allowed_mask, persistent=False)     # (num_dof, S) True=block
        self.register_buffer("dof_root_gate", root_gate.view(1, 1, self.output_dim, 1), persistent=False)

        # Routed-z injectors (replace the global z_embed broadcast).
        self.z_slot_embed = nn.Sequential(nn.Linear(self.slot_z, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.root_slot_embed = nn.Sequential(nn.Linear(self.slot_z, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.beta_embed = nn.Sequential(nn.Linear(beta_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.dof_embed = nn.Embedding(output_dim, d_model)
        if robot_dof_features is None:
            feature_tensor = torch.zeros(output_dim, 0, dtype=torch.float32)
        else:
            feature_tensor = torch.as_tensor(robot_dof_features, dtype=torch.float32).detach().clone()
            if feature_tensor.ndim != 2 or feature_tensor.shape[0] != output_dim:
                raise ValueError(
                    "robot_dof_features must have shape "
                    f"({output_dim}, F), got {tuple(feature_tensor.shape)}"
                )
        self.robot_dof_feature_dim = int(feature_tensor.shape[-1])
        self.register_buffer("robot_dof_features", feature_tensor, persistent=False)
        if self.robot_dof_feature_dim > 0:
            self.robot_dof_feature_proj = nn.Sequential(
                nn.Linear(self.robot_dof_feature_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        else:
            self.robot_dof_feature_proj = None
        if robot_embedding_dim > 0:
            self.robot_embed = nn.Embedding(num_robot_types, d_model)
        else:
            self.robot_embed = None
        self.context_proj = nn.Linear(context_dim or d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)

        self.graph_blocks = nn.ModuleList(GraphTokenBlock(d_model, num_heads, dropout) for _ in range(graph_layers))
        self.temporal_blocks = nn.ModuleList(
            TemporalConvBlock(d_model, dropout, causal=bool(temporal_decoder_causal))
            for _ in range(temporal_layers)
        )
        edges = _build_robot_dof_edges(robot_dof_edges, output_dim)
        self.robot_dof_graph_source = "kinematic_tree"
        self.robot_dof_edge_count = len(edges)
        self.register_buffer(
            "attn_mask_template",
            _attention_mask_from_edges(output_dim, edges, torch.device("cpu")),
            persistent=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        beta: Optional[torch.Tensor],
        robot_id: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # z is per-slot z_slots: (B, T, S, slot_z).
        if z.ndim != 4:
            raise ValueError(f"RobotGraphActionDecoder expects z_slots (B,T,S,slot_z); got {tuple(z.shape)}")
        batch_size, window, num_slots, slot_z = z.shape
        if beta is None:
            beta = z.new_zeros(batch_size, self.beta_embed[0].in_features)

        dof_ids = torch.arange(self.output_dim, device=z.device)
        x = self.dof_embed(dof_ids).view(1, 1, self.output_dim, self.d_model)
        if self.robot_dof_feature_proj is not None:
            dof_features = self.robot_dof_features.to(device=z.device, dtype=self.dof_embed.weight.dtype)
            feature_token = self.robot_dof_feature_proj(dof_features).to(dtype=x.dtype)
            x = x + feature_token.view(1, 1, self.output_dim, self.d_model)
        x = x.expand(batch_size, window, self.output_dim, self.d_model).contiguous()

        # Routed z injection: each DoF token gets ONLY its region slot latent +
        # the shared root slot latent (gated per DoF). No global broadcast.
        primary = self.dof_primary_slot.to(device=z.device)
        region = z.gather(
            2, primary.view(1, 1, self.output_dim, 1).expand(batch_size, window, self.output_dim, slot_z)
        )  # (B, T, num_dof, slot_z)
        root = z[:, :, self.root_slot_idx, :]  # (B, T, slot_z)
        x = x + self.z_slot_embed(region)
        root_contrib = self.root_slot_embed(root).unsqueeze(2) * self.dof_root_gate.to(dtype=x.dtype)
        x = x + root_contrib
        x = x + self.beta_embed(beta).view(batch_size, 1, 1, self.d_model)
        if self.robot_embed is not None:
            if robot_id is None:
                robot_id = torch.zeros(batch_size, dtype=torch.long, device=z.device)
            x = x + self.robot_embed(robot_id).view(batch_size, 1, 1, self.d_model)

        if self.decoder_context == "masked_graph_hidden" and context is not None:
            ctx = self.context_proj(context).reshape(batch_size * window, context.shape[2], self.d_model)
            flat = x.reshape(batch_size * window, self.output_dim, self.d_model)
            key_padding_mask = None
            if context_mask is not None:
                key_padding_mask = ~context_mask[:, None, :].expand(batch_size, window, context.shape[2]).reshape(
                    batch_size * window, context.shape[2]
                )
            # Region mask: DoF d may only attend to its allowed human slots.
            attn_mask = self.dof_slot_attn_mask.to(device=z.device)
            attn_out, _ = self.cross_attn(flat, ctx, ctx, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            x = self.cross_norm(flat + attn_out).reshape(batch_size, window, self.output_dim, self.d_model)

        flat = x.reshape(batch_size * window, self.output_dim, self.d_model)
        attn_mask = self.attn_mask_template.to(device=z.device)
        for block in self.graph_blocks:
            flat = block(flat, attn_mask=attn_mask)
        x = flat.reshape(batch_size, window, self.output_dim, self.d_model)

        time_pos = _sinusoidal_position_encoding(window, self.d_model, z.device, z.dtype)
        x = x + time_pos.view(1, window, 1, self.d_model)
        for block in self.temporal_blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


class FutureSlotHead(nn.Module):
    def __init__(
        self,
        z_dim: int,
        d_model: int,
        num_slots: int,
        future_offsets: Sequence[int],
        future_slot_dim: int,
    ):
        super().__init__()
        self.future_offsets = tuple(future_offsets)
        self.num_slots = num_slots
        self.future_slot_dim = future_slot_dim
        self.offset_embed = nn.Embedding(len(self.future_offsets), d_model)
        self.net = nn.Sequential(
            nn.Linear(z_dim + d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_slots * future_slot_dim),
        )

    def forward(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.future_offsets:
            return None

        batch_size, window, _ = z.shape
        outputs = []
        for offset_idx in range(len(self.future_offsets)):
            offset = self.offset_embed.weight[offset_idx].view(1, 1, -1).expand(batch_size, window, -1)
            pred = self.net(torch.cat([z, offset], dim=-1))
            outputs.append(pred.view(batch_size, window, self.num_slots, self.future_slot_dim))
        return torch.stack(outputs, dim=2)


class GraphRetargetVAE(nn.Module):
    """UPP-style canonical body-slot graph fuse network for GMR-SFT retargeting."""

    def __init__(
        self,
        num_human_joints: int = 21,
        robot_action_dim: int = 29,
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
        future_offsets: Sequence[int] = (1, 5, 10),
        future_slot_dim: int = 16,
        num_robot_types: int = 1,
        robot_embedding_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_human_joints = num_human_joints
        self.robot_action_dim = robot_action_dim
        self.beta_dim = beta_dim
        self.root_dim = root_dim
        self.d_model = d_model
        self.z_dim = z_dim
        self.slots = tuple(slots)

        self.tokenizer = HumanSlotTokenizer(
            num_human_joints=num_human_joints,
            beta_dim=beta_dim,
            d_model=d_model,
            slots=self.slots,
            root_dim=root_dim,
        )
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

        hidden_dim = max(d_model, z_dim)
        self.human_decoder = RetargetDecoder(
            z_dim=z_dim,
            beta_dim=beta_dim,
            output_dim=num_human_joints * 3 + root_dim,
            hidden_dim=hidden_dim,
        )
        self.robot_decoder = RetargetDecoder(
            z_dim=z_dim,
            beta_dim=beta_dim,
            output_dim=root_dim + robot_action_dim,
            hidden_dim=hidden_dim,
            robot_embedding_dim=robot_embedding_dim,
            num_robot_types=num_robot_types,
        )
        self.future_head = FutureSlotHead(
            z_dim=z_dim,
            d_model=d_model,
            num_slots=len(self.slots),
            future_offsets=future_offsets,
            future_slot_dim=future_slot_dim,
        )

    def encode(
        self,
        pose_aa: torch.Tensor,
        root_pose: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        slot_tokens, part_mask = self.tokenizer(pose_aa, root_pose, beta, part_mask)
        graph_hidden = self.encoder(slot_tokens, part_mask)
        pooled = _masked_mean(graph_hidden, part_mask[:, None, :], dim=2)
        slot_latents = self.slot_projector(graph_hidden)
        return {
            "mu": self.mu_head(pooled),
            "logvar": self.logvar_head(pooled),
            "slot_latents": slot_latents,
            "graph_hidden": graph_hidden,
            "part_mask": part_mask,
        }

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        pose_aa: torch.Tensor,
        root_pose: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
        robot_id: Optional[torch.Tensor] = None,
        sample: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode(pose_aa=pose_aa, root_pose=root_pose, beta=beta, part_mask=part_mask)
        if sample is None:
            sample = self.training
        z = self.reparameterize(encoded["mu"], encoded["logvar"], sample=sample)

        batch_size, window, _ = pose_aa.shape[:3]
        human_out = self.human_decoder(z, beta)
        robot_out = self.robot_decoder(z, beta, robot_id=robot_id)
        human_pose = human_out[..., : self.num_human_joints * 3].view(batch_size, window, self.num_human_joints, 3)
        human_root = human_out[..., self.num_human_joints * 3 :]
        robot_root = robot_out[..., : self.root_dim]
        robot_actions = robot_out[..., self.root_dim :]

        output = {
            "z": z,
            "mu": encoded["mu"],
            "logvar": encoded["logvar"],
            "slot_latents": encoded["slot_latents"],
            "graph_hidden": encoded["graph_hidden"],
            "part_mask": encoded["part_mask"],
            "human_pose_aa": human_pose,
            "human_root_pose": human_root,
            "robot_root_pose": robot_root,
            "robot_actions": robot_actions,
        }
        future_slots = self.future_head(z)
        if future_slots is not None:
            output["future_slots"] = future_slots
        return output


class UnifiedLatentRetargeter(nn.Module):
    """Canonical body-slot sequence latent retargeter.

    The graph encoder produces a per-frame shared latent from body-part slots.
    The primary path decodes that latent sequence directly to robot actions.
    The legacy frame/base path is retained only for base/blend compatibility and
    can be skipped in latent-only training.
    """

    def __init__(
        self,
        load_hands: bool = False,
        num_actions: int = 29,
        beta_dim: int = 16,
        root_dim: int = 9,
        d_model: int = 256,
        z_dim: int = 256,
        slot_z: int = 48,
        slot_latent_dim: int = 32,
        graph_layers: int = 2,
        temporal_layers: int = 4,
        num_heads: int = 8,
        slots: Sequence[str] = DEFAULT_BODY_SLOTS,
        edges: Iterable[Tuple[str, str]] = DEFAULT_BODY_EDGES,
        frame_embedding_dim: int = 32,
        frame_joint_embedding_dim: int = 128,
        frame_num_heads: int = 4,
        frame_num_layers: int = 4,
        robot_embedding_dim: int = 0,
        num_robot_types: int = 1,
        residual_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        residual_scale: float = 0.25,
        teacher_base_blend: float = 1.0,
        ref_base_height: float = 0.8,
        root_height_range: float = 0.6,
        latent_architecture: str = "structured",
        robot_dof_names: Optional[Sequence[str]] = None,
        robot_dof_edges: Optional[Sequence[Tuple[int, int]]] = None,
        robot_dof_features: Optional[torch.Tensor] = None,
        decoder_graph_layers: int = 2,
        decoder_temporal_layers: int = 2,
        decoder_context: str = "masked_graph_hidden",
        decoder_root_slot_into_arms: bool = True,
        frame_chunk_size: int = 4096,
        joint_geometry_dim: int = 0,
        joint_velocity_dim: int = 0,
        joint_contact_dim: int = 0,
        temporal_module: str = "transformer",
        temporal_kernel: int = 5,
        temporal_dilations: Tuple[int, ...] = (1, 2, 4, 8),
        temporal_num_layers: Optional[int] = None,
        temporal_lookahead_frames: int = 0,
        block_size: int = 1,
        spectral_aux_enabled: bool = False,
        spectral_aux_substream_dims: Optional[Dict[str, int]] = None,
        spectral_aux_substream_M: Optional[Dict[str, int]] = None,
        spectral_aux_hidden: int = 256,
        world_aux_enabled: bool = False,
        world_aux_hidden: int = 512,
        world_aux_num_layers: int = 3,
    ):
        super().__init__()
        self.load_hands = bool(load_hands)
        self.num_joints = 21 if not load_hands else 51
        if self.num_joints != 21:
            raise ValueError("UnifiedLatentRetargeter currently supports SMPL-H body joints without hands")
        self.joint_geometry_dim = int(joint_geometry_dim)
        self.joint_velocity_dim = int(joint_velocity_dim)
        self.joint_contact_dim = int(joint_contact_dim)
        self.num_actions = int(num_actions)
        self.beta_dim = int(beta_dim)
        self.root_dim = int(root_dim)
        self.d_model = int(d_model)
        self.slots = tuple(slots)
        self.latent_architecture = str(latent_architecture)
        if self.latent_architecture not in {"baseline", "structured"}:
            raise ValueError(f"Unknown latent_architecture: {latent_architecture}")
        # Structured path appends an INDEPENDENT 'root_state' slot: a pelvis-only
        # latent that is isolated in the slot graph (no edges) and bypasses the
        # joint-tree attention, so it carries pure root state with no limb info.
        # The decoder shares this slot into every DoF, giving each DoF physical
        # pelvis context with zero cross-limb leakage. The 'global_root' slot
        # stays the kinematic-tree hub (limbs attach to it) and the unknown-DoF
        # fallback. Baseline keeps the 6-slot default unchanged.
        if self.latent_architecture == "structured" and "root_state" not in self.slots:
            self.slots = self.slots + ("root_state",)
        # Structured path uses a PER-SLOT latent: each body slot owns `slot_z`
        # dims, and the global latent width is DERIVED as num_slots * slot_z
        # (overriding any passed z_dim). Global consumers (human recon, residual,
        # root-height, spectral, world, KL) see the flattened z_global of width
        # self.z_dim; only the routed action decoder consumes the per-slot
        # z_slots. The baseline path keeps the passed z_dim unchanged.
        self.slot_z = int(slot_z)
        if self.latent_architecture == "structured":
            z_dim = len(self.slots) * self.slot_z
        self.z_dim = int(z_dim)
        self.residual_scale = float(residual_scale)
        self.teacher_base_blend = float(teacher_base_blend)
        self.root_height_range = float(root_height_range)
        self.frame_chunk_size = int(frame_chunk_size)
        self.register_buffer("ref_base_height", torch.tensor(float(ref_base_height), dtype=torch.float32))

        self.frame_model = PoseTransformer(
            load_hands=load_hands,
            num_actions=num_actions,
            embedding_dim=frame_embedding_dim,
            joint_embedding_dim=frame_joint_embedding_dim,
            num_heads=frame_num_heads,
            num_layers=frame_num_layers,
        )
        if self.latent_architecture == "structured":
            self.tokenizer = SkeletalJointTokenizer(
                num_human_joints=self.num_joints,
                beta_dim=beta_dim,
                d_model=d_model,
                slots=self.slots,
                root_dim=root_dim,
                graph_layers=graph_layers,
                num_heads=num_heads,
                dropout=dropout,
                joint_geometry_dim=self.joint_geometry_dim,
                joint_velocity_dim=self.joint_velocity_dim,
                joint_contact_dim=self.joint_contact_dim,
            )
        else:
            self.tokenizer = HumanSlotTokenizer(
                num_human_joints=self.num_joints,
                beta_dim=beta_dim,
                d_model=d_model,
                slots=self.slots,
                root_dim=root_dim,
            )
        self.temporal_module = str(temporal_module)
        self.block_size = int(block_size)
        # Only set for temporal_module='block_causal_compress' (clip-compression);
        # None for every other module so the forward can branch on them.
        self.block_compressor = None
        self.block_upsampler = None
        encoder_temporal_layers = (
            temporal_layers if temporal_num_layers is None else int(temporal_num_layers)
        )
        # Structured path drops the root-as-global-hub slot edge so the encoder's
        # per-slot receptive field stays {itself, tree neighbours} — the second
        # half of the A0 locality fix (the first is per_slot_temporal). Baseline
        # keeps the legacy star graph (root attends to all slots) unchanged.
        encoder_root_global_hub = self.latent_architecture != "structured"
        if self.temporal_module in {"transformer", "causal_transformer"}:
            self.encoder = GraphTemporalEncoder(
                d_model=d_model,
                num_heads=num_heads,
                slots=self.slots,
                edges=edges,
                graph_layers=graph_layers,
                temporal_layers=encoder_temporal_layers,
                dropout=dropout,
                # Structured path needs slot-local temporal mixing so the per-slot
                # latent (and the masked decoder context) stay region-local.
                per_slot_temporal=(self.latent_architecture == "structured"),
                root_global_hub=encoder_root_global_hub,
                # E1: causal_transformer keeps the same architecture but masks the
                # per-slot temporal attention so z_t only sees frames <= t.
                causal=(self.temporal_module == "causal_transformer"),
            )
        elif self.temporal_module in {"causal_conv", "lookahead_conv"}:
            if self.temporal_module == "causal_conv" and int(temporal_lookahead_frames) != 0:
                raise ValueError(
                    "temporal_module='causal_conv' requires temporal_lookahead_frames == 0; "
                    f"got {temporal_lookahead_frames}. Use 'lookahead_conv' for non-zero lookahead."
                )
            self.encoder = CausalConvTemporalEncoder(
                d_model=d_model,
                num_heads=num_heads,
                slots=self.slots,
                edges=edges,
                num_layers=encoder_temporal_layers,
                kernel_size=int(temporal_kernel),
                dilations=tuple(int(d) for d in temporal_dilations),
                total_lookahead_frames=int(temporal_lookahead_frames),
                dropout=dropout,
                root_global_hub=encoder_root_global_hub,
            )
        elif self.temporal_module == "block_causal_compress":
            # SONIC-style temporal compaction: the latent unit is a CLIP of
            # block_size frames, not a frame. Encoder is block-causal (bidir
            # within block, causal across); the K=T/block_size clip tokens are
            # the VAE bottleneck; an upsampler expands z back to per-frame.
            if self.latent_architecture != "structured":
                raise ValueError("block_causal_compress requires latent_architecture='structured'.")
            if self.block_size <= 1:
                raise ValueError("block_causal_compress requires block_size > 1.")
            self.encoder = GraphTemporalEncoder(
                d_model=d_model,
                num_heads=num_heads,
                slots=self.slots,
                edges=edges,
                graph_layers=graph_layers,
                temporal_layers=encoder_temporal_layers,
                dropout=dropout,
                per_slot_temporal=True,
                root_global_hub=encoder_root_global_hub,
                block_size=self.block_size,
            )
            # Compress per-frame hidden (B,T,S,D) -> clip tokens (B,K,S,D) in
            # d_model space; mu/logvar then produce the K-token latent z_clip.
            self.block_compressor = BlockCompressor(d_model, self.block_size)
            # Decompress sampled z_clip (B,K,S,slot_z) -> per-frame z (B,T,S,slot_z),
            # block-causally. Heads sized so RoPE head_dim stays even for slot_z.
            up_heads = pick_num_heads(self.slot_z, prefer=num_heads)
            self.block_upsampler = BlockCausalUpsampler(
                self.slot_z, up_heads, self.block_size, dropout=dropout
            )
        else:
            raise ValueError(
                f"Unknown temporal_module: {self.temporal_module}. Choose from: "
                "transformer, causal_transformer, causal_conv, lookahead_conv, block_causal_compress."
            )
        self.slot_projector = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, slot_latent_dim),
        )
        if self.latent_architecture == "structured":
            # Per-slot bottleneck: each slot token gets its own mu/logvar of
            # width slot_z, read directly from graph_hidden (NO masked-mean pool).
            # z_global = z_slots.reshape(B, T, num_slots * slot_z) is what global
            # consumers see; the routed decoder consumes z_slots.
            self.mu_head_slot = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.slot_z))
            self.logvar_head_slot = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.slot_z))
        else:
            self.mu_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, z_dim))
            self.logvar_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, z_dim))

        decoder_hidden = max(d_model, z_dim)
        if self.latent_architecture == "structured":
            self.latent_action_decoder = RobotGraphActionDecoder(
                slot_z=self.slot_z,
                beta_dim=beta_dim,
                output_dim=num_actions,
                d_model=decoder_hidden,
                num_heads=num_heads,
                robot_dof_names=robot_dof_names,
                human_slots=self.slots,
                robot_dof_edges=robot_dof_edges,
                robot_dof_features=robot_dof_features,
                robot_embedding_dim=robot_embedding_dim,
                num_robot_types=num_robot_types,
                graph_layers=decoder_graph_layers,
                temporal_layers=decoder_temporal_layers,
                context_dim=d_model,
                decoder_context=str(decoder_context),
                root_slot_into_arms=bool(decoder_root_slot_into_arms),
                dropout=dropout,
                # E1: decoder temporal Conv1d becomes left-pad causal so the
                # decoder cannot peek at future frames either.
                # E1 forces the decoder Conv1d left-pad causal too. T-series
                # (block_causal_compress) defaults FALSE here so the existing
                # T4/T5 checkpoints (trained 2026-06-02 with symmetric padding)
                # load and run with the SAME padding they were optimized for.
                # NOTE: a fully streaming-causal T-series needs a retrain with
                # this flag flipped to True — see [[project-nan-rootcause]]'s
                # sibling streaming TODO. Adds ~±4 frames decoder lookahead now.
                temporal_decoder_causal=(self.temporal_module == "causal_transformer"),
            )
        else:
            self.latent_action_decoder = RetargetDecoder(
                z_dim=z_dim,
                beta_dim=beta_dim,
                output_dim=num_actions,
                hidden_dim=decoder_hidden,
                robot_embedding_dim=robot_embedding_dim,
                num_robot_types=num_robot_types,
            )

        residual_hidden = int(residual_hidden_dim or decoder_hidden)
        self.root_embed = nn.Sequential(
            nn.Linear(root_dim, residual_hidden),
            nn.GELU(),
            nn.LayerNorm(residual_hidden),
        )
        # Human reconstruction decoder — makes this a true VAE on the human side.
        # Output = (num_joints * 3) pose_aa + root_dim root features. Without this
        # decoder the latent has no incentive to encode motion semantics; it's only
        # judged by whether the robot actions it produces are good.
        self.human_decoder = RetargetDecoder(
            z_dim=z_dim,
            beta_dim=beta_dim,
            output_dim=self.num_joints * 3 + root_dim,
            hidden_dim=decoder_hidden,
        )
        # Optional auxiliary heads (spectral aux, world aux) are constructed
        # inside fork_rng so they do not consume RNG from the global stream
        # and shift initialization of downstream modules (residual_head,
        # root_height_head). Without this, enabling/disabling an aux head
        # would silently change the initialization of every parameter
        # constructed afterwards — invalidating any "controlled ablation"
        # comparison between aux-on and aux-off runs at the same seed.
        self.spectral_aux_enabled = bool(spectral_aux_enabled)
        if self.spectral_aux_enabled:
            if spectral_aux_substream_dims is None or spectral_aux_substream_M is None:
                raise ValueError(
                    "spectral_aux_enabled=True requires substream_dims and substream_M dicts"
                )
            with torch.random.fork_rng(devices=[]):
                self.spectral_aux_head = MultiStreamSpectralAuxiliaryHead(
                    z_dim=z_dim,
                    substream_dims=dict(spectral_aux_substream_dims),
                    substream_M=dict(spectral_aux_substream_M),
                    hidden=int(spectral_aux_hidden),
                    dropout=float(dropout),
                )
        else:
            self.spectral_aux_head = None
        self.world_aux_enabled = bool(world_aux_enabled)
        if self.world_aux_enabled:
            with torch.random.fork_rng(devices=[]):
                self.world_head = LatentForwardHead(
                    z_dim=z_dim,
                    action_dim=int(num_actions),
                    hidden=int(world_aux_hidden),
                    num_layers=int(world_aux_num_layers),
                    dropout=float(dropout),
                )
        else:
            self.world_head = None
        self.residual_head = nn.Sequential(
            nn.Linear(z_dim + num_actions * 2 + residual_hidden, residual_hidden),
            nn.GELU(),
            nn.LayerNorm(residual_hidden),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden, residual_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden, num_actions),
        )
        self.root_height_head = nn.Sequential(
            nn.Linear(z_dim + residual_hidden, residual_hidden),
            nn.GELU(),
            nn.LayerNorm(residual_hidden),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden, 1),
        )
        self._zero_init_residual()
        self._zero_init_root_height()

    def _zero_init_residual(self) -> None:
        last = self.residual_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _zero_init_root_height(self) -> None:
        last = self.root_height_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def encode_frames(self, pose_body: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if pose_body.ndim != 3:
            raise ValueError(f"pose_body must have shape (B, T, D), got {tuple(pose_body.shape)}")
        batch_size, window, dim = pose_body.shape
        expected = self.num_joints * 3
        if dim != expected:
            raise ValueError(f"pose_body last dim must be {expected}, got {dim}")

        flat = pose_body.reshape(batch_size * window, self.num_joints, 3)
        frame_features_out = []
        base_actions_out = []
        chunk_size = self.frame_chunk_size if self.frame_chunk_size > 0 else flat.shape[0]
        for start in range(0, flat.shape[0], chunk_size):
            chunk = flat[start : start + chunk_size]
            x = chunk.permute(0, 2, 1)
            x = self.frame_model.joint_embedding(x)
            x = x.permute(0, 2, 1)
            x = self.frame_model.embedding(x)
            x = x + self.frame_model.positional_encoding
            x = self.frame_model.transformer(x)
            frame_features = x.mean(dim=1)
            frame_features_out.append(frame_features)
            base_actions_out.append(self.frame_model.decoder(frame_features))
        frame_features = torch.cat(frame_features_out, dim=0)
        base_actions = torch.cat(base_actions_out, dim=0)
        return (
            frame_features.reshape(batch_size, window, -1),
            base_actions.reshape(batch_size, window, -1),
        )

    def encode_latent(
        self,
        pose_aa: torch.Tensor,
        root_features: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
        joint_pos: Optional[torch.Tensor] = None,
        joint_vel: Optional[torch.Tensor] = None,
        joint_contact: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(self.tokenizer, SkeletalJointTokenizer):
            slot_tokens, part_mask = self.tokenizer(
                pose_aa, root_features, beta, part_mask,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                joint_contact=joint_contact,
            )
        else:
            slot_tokens, part_mask = self.tokenizer(pose_aa, root_features, beta, part_mask)
        graph_hidden = self.encoder(slot_tokens, part_mask)
        if self.latent_architecture == "structured":
            if self.block_compressor is not None:
                # block_causal_compress: compress per-frame hidden (B,T,S,D) into
                # K=T/block_size clip tokens, then bottleneck on the CLIP tokens.
                # mu/logvar are (B, K, S, slot_z) — the compressed latent. The
                # upsampler in forward() expands sampled z back to per-frame.
                hidden_for_latent = self.block_compressor(graph_hidden)
            else:
                # Per-slot mu/logvar straight off graph_hidden: (B, T, S, slot_z).
                hidden_for_latent = graph_hidden
            mu = self.mu_head_slot(hidden_for_latent)
            logvar = self.logvar_head_slot(hidden_for_latent)
        else:
            pooled = _masked_mean(graph_hidden, part_mask[:, None, :], dim=2)
            mu = self.mu_head(pooled)
            logvar = self.logvar_head(pooled)
        return {
            "mu": mu,
            "logvar": logvar,
            "slot_latents": self.slot_projector(graph_hidden),
            "graph_hidden": graph_hidden,
            "part_mask": part_mask,
        }

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(
        self,
        pose_body: torch.Tensor,
        root_features: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
        robot_id: Optional[torch.Tensor] = None,
        sample: Optional[bool] = None,
        base_blend: Optional[float] = None,
        use_frame_path: bool = True,
        joint_pos: Optional[torch.Tensor] = None,
        joint_vel: Optional[torch.Tensor] = None,
        joint_contact: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, window, dim = pose_body.shape
        if dim != self.num_joints * 3:
            raise ValueError(f"pose_body last dim must be {self.num_joints * 3}, got {dim}")
        if root_features is None:
            root_features = pose_body.new_zeros(batch_size, window, self.root_dim)
        if root_features.shape[:2] != (batch_size, window) or root_features.shape[-1] != self.root_dim:
            raise ValueError(
                "root_features must have shape "
                f"({batch_size}, {window}, {self.root_dim}), got {tuple(root_features.shape)}"
            )

        pose_aa = pose_body.reshape(batch_size, window, self.num_joints, 3)
        encoded = self.encode_latent(
            pose_aa, root_features, beta=beta, part_mask=part_mask,
            joint_pos=joint_pos, joint_vel=joint_vel, joint_contact=joint_contact,
        )
        if sample is None:
            sample = self.training
        z_full = self.reparameterize(encoded["mu"], encoded["logvar"], sample=sample)
        if self.latent_architecture == "structured":
            if self.block_upsampler is not None:
                # block_causal_compress: z_full is the CLIP latent (B, K, S, slot_z).
                # Decompress block-causally to per-frame z_slots (B, T, S, slot_z).
                # Option (a): the K clip tokens are the SOLE information channel, so
                # the decoder gets NO per-frame context — feeding the per-frame
                # graph_hidden would bypass the bottleneck and void the test.
                z_slots = self.block_upsampler(z_full)
                decoder_context = None
                decoder_context_mask = None
            else:
                # z_full is per-slot (B, T, S, slot_z). The routed decoder consumes
                # the per-slot tensor; output["z"] uses the flattened z_global.
                z_slots = z_full
                decoder_context = encoded["graph_hidden"]
                decoder_context_mask = encoded["part_mask"]
            z = z_slots.reshape(batch_size, window, -1)
            latent_actions = self.latent_action_decoder(
                z_slots,
                beta,
                robot_id=robot_id,
                context=decoder_context,
                context_mask=decoder_context_mask,
            )
        else:
            z = z_full
            latent_actions = self.latent_action_decoder(z, beta, robot_id=robot_id)
        root_token = self.root_embed(root_features)
        blend = self.teacher_base_blend if base_blend is None else float(base_blend)
        if use_frame_path:
            _, base_actions = self.encode_frames(pose_body)
            residual_input = torch.cat([z, latent_actions, base_actions, root_token], dim=-1)
            residual = self.residual_head(residual_input) * self.residual_scale
            actions = blend * base_actions + (1.0 - blend) * latent_actions + residual
        else:
            base_actions = torch.zeros_like(latent_actions)
            residual = torch.zeros_like(latent_actions)
            actions = latent_actions
        root_height_delta = self.root_height_head(torch.cat([z, root_token], dim=-1))
        root_height = self.ref_base_height.to(device=z.device, dtype=z.dtype).view(1, 1, 1)
        root_height = root_height + self.root_height_range * torch.tanh(root_height_delta)

        # Human reconstruction (decodes z back into SMPL pose + root). Required for
        # the latent to be a real motion-prior VAE rather than a translator.
        human_out = self.human_decoder(z, beta)
        human_pose_aa_recon = human_out[..., : self.num_joints * 3].view(
            batch_size, window, self.num_joints, 3
        )
        human_root_recon = human_out[..., self.num_joints * 3 :]

        spectral_aux_pred: Optional[Dict[str, torch.Tensor]] = None
        if self.spectral_aux_head is not None:
            spectral_aux_pred = self.spectral_aux_head(z)

        # NOTE: world_head is intentionally NOT called inside forward(). It
        # lives as an attribute (`self.world_head`) so the trainer can pick
        # the action source explicitly (model-predicted vs. GMR reference).
        # See compute_losses() in train_graph_latent.py and the
        # `--world-aux-action-source` CLI flag.

        out = {
            "actions": actions,
            "base_actions": base_actions,
            "latent_actions": latent_actions,
            "residual": residual,
            "z": z,
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
        if spectral_aux_pred is not None:
            out["spectral_aux_pred"] = spectral_aux_pred
        return out

    def load_teacher(self, checkpoint: str, map_location: str | torch.device = "cpu") -> None:
        state = torch.load(checkpoint, map_location=map_location, weights_only=False)
        self.frame_model.load_state_dict(state)
