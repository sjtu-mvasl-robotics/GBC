"""Block-causal clip-compression for the unified-latent retargeter.

Ported/adapted from the user's rsl_rl block-causal toolkit
(`rsl_rl/modules/actor_critic_mm_gpt.py::generate_block_causal_mask`,
`Conv1dCompressor`, `ConvTranspose1dCompressor`/`PerceiverUpsampler`,
`rsl_rl/networks/rope*.py`) into the data-prep package so the retargeter does
NOT take a hard dependency on the RL trainer. Tensors here are per-slot 4-D
`(B, T, S, D)` (the structured retargeter layout), not the RL `(T, B, D)`.

Design (SONIC-style temporal compaction, made streaming-safe):
  * A motion **clip** of `block_size` frames is the latent unit, not a frame.
  * **Block-causal**: bidirectional attention WITHIN a block (the whole clip
    pattern is visible — bounded lookahead = block_size frames) but causal
    ACROSS blocks (no future block leaks into a past one). This is the
    `kron(tril(num_blocks), ones(k, k))` mask generalized from the user's k=2.
  * **RoPE** positional encoding (relative; small offsets within a block, clean
    across blocks). Encoder uses frame positions; the upsampler places each
    block token at its block-center frame position so query/key geometry is
    consistent.

Pipeline a caller assembles:
    enc  = BlockCausalRoPEEncoder(...)      # (B,T,S,D) -> (B,T,S,D), block-causal
    comp = BlockCompressor(block_size)      # (B,T,S,D) -> (B,K,S,D)  K=T/k
    # ... VAE bottleneck on the K clip tokens -> z_clip (B,K,S,slot_z) ...
    up   = BlockCausalUpsampler(...)        # (B,K,S,Dz) -> (B,T,S,Dz), block-causal
    # ... existing per-frame DOF-graph decoder on the upsampled per-frame z ...
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Block-causal mask
# --------------------------------------------------------------------------- #
def build_block_causal_mask(
    num_frames: int,
    block_size: int,
    device: torch.device,
    allowed_true: bool = True,
) -> torch.Tensor:
    """`[T, T]` bool mask: bidirectional within a block, causal across blocks.

    Generalizes `generate_block_causal_mask` (rsl_rl, k=2) to arbitrary
    `block_size`. With `allowed_true=True`, ``mask[i, j] == True`` means query
    frame ``i`` is ALLOWED to attend to key frame ``j``.

    Requires ``num_frames % block_size == 0``.
    """
    if num_frames % block_size != 0:
        raise ValueError(
            f"num_frames ({num_frames}) must be divisible by block_size ({block_size})."
        )
    n_blocks = num_frames // block_size
    inter = torch.tril(torch.ones(n_blocks, n_blocks, device=device, dtype=torch.bool))
    intra = torch.ones(block_size, block_size, device=device, dtype=torch.bool)
    allowed = torch.kron(inter, intra)  # [T, T] bool
    return allowed if allowed_true else ~allowed


def _additive_mask(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Bool 'allowed' mask -> additive float mask (0 where allowed, -inf else)."""
    return torch.zeros_like(allowed, dtype=dtype).masked_fill(~allowed, float("-inf"))


def pick_num_heads(dim: int, prefer: int = 8) -> int:
    """Largest head count <= prefer that divides `dim` AND yields an EVEN head_dim.

    RoPE requires an even per-head dimension. Used to size attention over the
    slot_z latent (e.g. slot_z=48 -> 8 heads (dh=6); 24 -> 6 (dh=4); 12 -> 6 (dh=2)).
    """
    for h in (prefer, 8, 6, 4, 3, 2, 1):
        if h <= prefer and dim % h == 0 and (dim // h) % 2 == 0:
            return h
    return 1


# --------------------------------------------------------------------------- #
# RoPE
# --------------------------------------------------------------------------- #
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., H, L, dh); cos/sin: (L, dh)."""
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


class RotaryPE(nn.Module):
    """Standard rotary positional embedding (split-half convention)."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}.")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim

    def cos_sin(self, positions: torch.Tensor, dtype: torch.dtype):
        """positions: (L,) -> cos, sin each (L, head_dim)."""
        freqs = torch.outer(positions.to(self.inv_freq.dtype), self.inv_freq)  # (L, dh/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (L, dh)
        return emb.cos().to(dtype), emb.sin().to(dtype)


# --------------------------------------------------------------------------- #
# Block-causal self-attention encoder (per-slot over time)
# --------------------------------------------------------------------------- #
class _RoPESelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}.")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.rope = RotaryPE(self.head_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, positions: torch.Tensor, allowed_mask: torch.Tensor) -> torch.Tensor:
        # x: (N, T, D); positions: (T,); allowed_mask: (T, T) bool (True = allowed)
        n, t, d = x.shape
        qkv = self.qkv(x).reshape(n, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (N, H, T, dh)
        cos, sin = self.rope.cos_sin(positions, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn_mask = _additive_mask(allowed_mask, x.dtype)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(n, t, d)
        return self.proj(out)


class BlockCausalRoPELayer(nn.Module):
    """One pre-norm RoPE attention + GELU-FFN block. Caller supplies the
    `(T, T)` block-causal `allowed_mask` and frame `positions`; reused both by
    the standalone encoder here and by `GraphTemporalEncoder` (graph fusion +
    this temporal block) in `graph_retarget_latent.py`."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _RoPESelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, allowed_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), positions, allowed_mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class BlockCausalRoPEEncoder(nn.Module):
    """Per-slot temporal encoder: bidirectional within block, causal across blocks.

    Input/output: `(B, T, S, D)`. Attention runs per slot over the time axis
    (slots folded into batch), so cross-slot mixing stays the job of the graph
    blocks upstream — matching the structured retargeter's `per_slot_temporal`.
    """

    is_causal: bool = True  # across blocks; trainer metric logger reads this

    def __init__(self, d_model: int, num_heads: int, num_layers: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.block_size = int(block_size)
        self.layers = nn.ModuleList(
            BlockCausalRoPELayer(d_model, num_heads, dropout) for _ in range(num_layers)
        )
        # Across-block lookahead is 0 (causal); within-block lookahead is bounded
        # by block_size-1. Report the bounded value for the trainer's logger.
        self.total_lookahead_frames = self.block_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, s, d = x.shape
        if t % self.block_size != 0:
            raise ValueError(f"window T={t} not divisible by block_size={self.block_size}.")
        flat = x.permute(0, 2, 1, 3).reshape(b * s, t, d)
        positions = torch.arange(t, device=x.device)
        allowed = build_block_causal_mask(t, self.block_size, x.device, allowed_true=True)
        for layer in self.layers:
            flat = layer(flat, positions, allowed)
        return flat.reshape(b, s, t, d).permute(0, 2, 1, 3).contiguous()


# --------------------------------------------------------------------------- #
# Compressor: frames -> clip tokens (non-overlapping, so block-causality holds)
# --------------------------------------------------------------------------- #
class BlockCompressor(nn.Module):
    """`(B, T, S, D) -> (B, K, S, D)`, K = T / block_size.

    Non-overlapping strided conv (kernel = stride = block_size): each output
    token depends ONLY on its own block's frames, so compression introduces no
    cross-block leakage on top of the block-causal encoder.
    """

    def __init__(self, d_model: int, block_size: int):
        super().__init__()
        self.block_size = int(block_size)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=self.block_size, stride=self.block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, s, d = x.shape
        if t % self.block_size != 0:
            raise ValueError(f"window T={t} not divisible by block_size={self.block_size}.")
        flat = x.permute(0, 2, 3, 1).reshape(b * s, d, t)  # (B*S, D, T)
        comp = self.conv(flat)  # (B*S, D, K)
        k = comp.shape[-1]
        return comp.reshape(b, s, d, k).permute(0, 3, 1, 2).contiguous()  # (B, K, S, D)


# --------------------------------------------------------------------------- #
# Upsampler: clip tokens -> frames (block-causal cross-attention, RoPE)
# --------------------------------------------------------------------------- #
class BlockCausalUpsampler(nn.Module):
    """`(B, K, S, D) -> (B, T, S, D)`, T = K * block_size.

    Perceiver-style cross-attention (anti-checkerboard vs transposed conv): a
    per-frame query (learned intra-block phase embedding + RoPE frame position)
    attends to clip tokens, masked block-causally so a frame in block b only
    sees clip tokens 0..b. A repeat-interleave of the nearest clip token is
    added as the residual for a stable identity-ish init.
    """

    def __init__(self, d_model: int, num_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}.")
        self.block_size = int(block_size)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.phase = nn.Parameter(torch.randn(self.block_size, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.rope = RotaryPE(self.head_dim)
        self.dropout = dropout

    def _heads(self, t: torch.Tensor, n: int, length: int) -> torch.Tensor:
        return t.reshape(n, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, k, s, d = tokens.shape
        bs = b * s
        ksz = self.block_size
        t = k * ksz
        kv_in = tokens.permute(0, 2, 1, 3).reshape(bs, k, d)  # (B*S, K, D)

        # Per-frame queries: shared intra-block phase embedding, tiled over blocks.
        q_emb = self.phase.unsqueeze(0).expand(k, ksz, d).reshape(t, d)  # (T, D)
        q = self.q_proj(q_emb).unsqueeze(0).expand(bs, t, d)
        # Residual: repeat each clip token block_size times (nearest-token identity).
        resid = kv_in.repeat_interleave(ksz, dim=1)  # (B*S, T, D)

        kv = self.kv_proj(kv_in)
        key, val = kv[..., :d], kv[..., d:]
        qh = self._heads(q, bs, t)
        kh = self._heads(key, bs, k)
        vh = self._heads(val, bs, k)

        # Two-level RoPE geometry: queries at frame positions, keys at block-centers.
        q_pos = torch.arange(t, device=tokens.device)
        k_pos = torch.arange(k, device=tokens.device) * ksz + (ksz - 1) / 2.0
        qc, qs = self.rope.cos_sin(q_pos, tokens.dtype)
        kc, ks = self.rope.cos_sin(k_pos, tokens.dtype)
        qh = apply_rope(qh, qc, qs)
        kh = apply_rope(kh, kc, ks)

        # Block-causal cross mask [T, K]: frame's block index >= clip token index.
        block_of_frame = (torch.arange(t, device=tokens.device) // ksz).unsqueeze(1)  # (T,1)
        token_block = torch.arange(k, device=tokens.device).unsqueeze(0)  # (1,K)
        allowed = token_block <= block_of_frame  # (T, K) bool
        attn_mask = _additive_mask(allowed, tokens.dtype)

        out = F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(bs, t, d)
        out = self.norm(self.proj(out) + resid)
        return out.reshape(b, s, t, d).permute(0, 2, 1, 3).contiguous()
