"""Temporal encoder variants for the unified-latent retargeter.

Implements A1 (causal conv) and A2 (limited-lookahead conv) backbones with
alternating GraphFuseBlock + CausalConv1DBlock cells. Preserves the
(B, T, num_slots, d_model) -> (B, T, num_slots, d_model) contract of the
original GraphTemporalEncoder.

Causality contract per module:
    is_causal: bool
    total_lookahead_frames: int   # future frames > t that may influence output at t

`right_taps` in CausalConv1DBlock counts FUTURE KERNEL TAPS (not future
frames); effective future frames at a single layer = right_taps * dilation.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_graph_fuse_block_cls():
    """Lazy import to break the circular dependency with graph_retarget_latent."""
    from GBC.utils.data_preparation.graph_retarget_latent import GraphFuseBlock
    return GraphFuseBlock


class CausalConv1DBlock(nn.Module):
    """Per-slot 1D causal/lookahead conv over the time axis.

    Forward shape: (B, T, S, D) -> (B, T, S, D).
    Time padding is asymmetric. With kernel k, dilation d, right_taps r:
        left_pad  = (k - 1 - r) * d
        right_pad = r * d
    so the receptive field at output time t is
        [t - (k - 1 - r) * d, t + r * d]
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dilation: int,
        right_taps: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1; got {kernel_size}")
        if dilation < 1:
            raise ValueError(f"dilation must be >= 1; got {dilation}")
        if not (0 <= right_taps <= kernel_size - 1):
            raise ValueError(
                f"right_taps must satisfy 0 <= right_taps <= kernel_size - 1; "
                f"got right_taps={right_taps}, kernel_size={kernel_size}"
            )
        self.d_model = int(d_model)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.right_taps = int(right_taps)
        self.left_pad = (kernel_size - 1 - right_taps) * dilation
        self.right_pad = right_taps * dilation
        if self.left_pad < 0:
            raise ValueError(
                f"left_pad computed negative ({self.left_pad}); kernel/right_taps inconsistent"
            )
        self.norm = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=True,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    @property
    def future_frames(self) -> int:
        return self.right_taps * self.dilation

    @property
    def past_frames(self) -> int:
        return (self.kernel_size - 1 - self.right_taps) * self.dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"CausalConv1DBlock expects (B, T, S, D); got {tuple(x.shape)}")
        B, T, S, D = x.shape
        if D != self.d_model:
            raise ValueError(f"expected D={self.d_model}, got {D}")
        residual = x
        y = self.norm(x)
        # (B, T, S, D) -> (B*S, D, T) for Conv1d.
        y = y.permute(0, 2, 3, 1).contiguous().reshape(B * S, D, T)
        y = F.pad(y, (self.left_pad, self.right_pad))
        y = self.conv(y)
        if y.shape[-1] != T:
            raise RuntimeError(
                f"Conv output length {y.shape[-1]} != T {T}; padding logic broken "
                f"(kernel={self.kernel_size}, dilation={self.dilation}, "
                f"right_taps={self.right_taps})"
            )
        y = y.reshape(B, S, D, T).permute(0, 3, 1, 2).contiguous()
        y = self.act(y)
        y = self.dropout(y)
        return residual + y


class SlotTemporalCell(nn.Module):
    """One alternating cell: GraphFuseBlock (slot mixing per frame), then
    CausalConv1DBlock (time mixing per slot).

    After N cells, both slot and time mixing have happened N times. This is the
    structural fix that prevents per-slot-only temporal modeling.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        slots: Sequence[str],
        edges: Iterable[Tuple[str, str]],
        kernel_size: int,
        dilation: int,
        right_taps: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        GraphFuseBlock = _get_graph_fuse_block_cls()
        self.graph = GraphFuseBlock(d_model, num_heads, slots, edges, dropout)
        self.conv = CausalConv1DBlock(
            d_model=d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            right_taps=right_taps,
            dropout=dropout,
        )

    def forward(self, slot_tokens: torch.Tensor, part_mask: torch.Tensor) -> torch.Tensor:
        slot_tokens = self.graph(slot_tokens, part_mask)
        slot_tokens = self.conv(slot_tokens)
        return slot_tokens


class CausalConvTemporalEncoder(nn.Module):
    """Stack of SlotTemporalCells.

    `total_lookahead_frames` is the total number of future frames any output at
    time t may depend on. We allocate the entire lookahead budget to the LAST
    cell's right_taps, rounded DOWN to a multiple of the last cell's dilation
    and capped at kernel_size - 1 taps. effective_lookahead_frames is the
    achieved value and is logged separately.

    Forward shape: (B, T, S, D) -> (B, T, S, D).
    """

    is_causal: bool
    total_lookahead_frames: int
    requested_lookahead_frames: int

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        slots: Sequence[str],
        edges: Iterable[Tuple[str, str]],
        num_layers: int = 4,
        kernel_size: int = 5,
        dilations: Tuple[int, ...] = (1, 2, 4, 8),
        total_lookahead_frames: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1; got {num_layers}")
        if total_lookahead_frames < 0:
            raise ValueError(
                f"total_lookahead_frames must be >= 0; got {total_lookahead_frames}"
            )
        if len(dilations) != num_layers:
            dilations = tuple(dilations[i % len(dilations)] for i in range(num_layers))
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(d) for d in dilations)
        self.requested_lookahead_frames = int(total_lookahead_frames)

        last_dilation = self.dilations[-1]
        last_right_taps = min(
            self.requested_lookahead_frames // last_dilation,
            self.kernel_size - 1,
        )
        effective = last_right_taps * last_dilation
        right_taps_per_layer = [0] * (num_layers - 1) + [last_right_taps]

        self.layers = nn.ModuleList(
            SlotTemporalCell(
                d_model=d_model,
                num_heads=num_heads,
                slots=slots,
                edges=edges,
                kernel_size=self.kernel_size,
                dilation=self.dilations[i],
                right_taps=right_taps_per_layer[i],
                dropout=dropout,
            )
            for i in range(num_layers)
        )
        self.total_lookahead_frames = int(effective)
        self.is_causal = self.total_lookahead_frames == 0

    @property
    def effective_lookahead_frames(self) -> int:
        return self.total_lookahead_frames

    def forward(self, slot_tokens: torch.Tensor, part_mask: torch.Tensor) -> torch.Tensor:
        for cell in self.layers:
            slot_tokens = cell(slot_tokens, part_mask)
        return slot_tokens
