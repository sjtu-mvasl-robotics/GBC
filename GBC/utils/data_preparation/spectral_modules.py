"""Spectral modules for the temporal-ablations suite.

Provides:
    - DCT-II helpers (dependency-free)
    - SubstreamNormalizer: applies frozen train-set-global mean/std
    - SubstreamStats: dataclass that owns the frozen statistics
    - MultiStreamSpectralAuxiliaryHead: per-frame z -> predicted DCT
      coefficients of forward K-frame windows across motion-feature substreams
      (positions, velocities, root, contact, robot actions).

Used by A4 (causal-backbone-only) and A5 (any-backbone) auxiliary heads.
Both share this class; CLI flags gate which one is active at training time.

The frozen-stats helpers DO NOT normalize per-window. The whole point of
freezing stats train-set-global is to keep DC and rare-event semantics
(root height, contact bits) intact across windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# DCT-II helpers
# ----------------------------------------------------------------------------


def _build_dct_ii_matrix(K: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Orthonormal DCT-II matrix of shape (K, K).

    Row k, column n:
        Phi[0, n]   = 1 / sqrt(K)
        Phi[k, n]   = sqrt(2 / K) * cos(pi * (2n + 1) * k / (2 * K))   for k >= 1
    """
    if K < 1:
        raise ValueError(f"K must be >= 1; got {K}")
    n = torch.arange(K, dtype=dtype).unsqueeze(0)         # (1, K)
    k = torch.arange(K, dtype=dtype).unsqueeze(1)         # (K, 1)
    pi = torch.acos(torch.tensor(-1.0, dtype=dtype))
    phi = torch.cos(pi * (2.0 * n + 1.0) * k / (2.0 * K))
    scale = torch.full((K,), (2.0 / K) ** 0.5, dtype=dtype)
    scale[0] = (1.0 / K) ** 0.5
    return scale.unsqueeze(1) * phi                       # (K, K)


def dct_ii(x: torch.Tensor, dim: int = -2, basis: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply orthonormal DCT-II along `dim` (default: second-to-last axis).

    Args:
        x: arbitrary tensor.
        dim: the axis treated as time.
        basis: optional precomputed (K, K) DCT basis on the right device/dtype.

    Returns:
        Same shape as x; values along `dim` are DCT-II coefficients.
    """
    K = x.shape[dim]
    if basis is None:
        basis = _build_dct_ii_matrix(K, dtype=x.dtype).to(device=x.device)
    elif basis.shape != (K, K):
        raise ValueError(f"basis shape {tuple(basis.shape)} != ({K}, {K})")
    moved = x.movedim(dim, -1)                            # (..., K)
    # (..., K) @ (K, K)^T  -> (..., K) along last; basis row indexes freq k.
    out = moved @ basis.transpose(0, 1)
    return out.movedim(-1, dim)


# ----------------------------------------------------------------------------
# Substream definitions + frozen stats
# ----------------------------------------------------------------------------


# Default substreams. Each substream is identified by name and has a fixed
# feature dim. Trainer builds the dict at runtime; this default is used by the
# aux head's defaulting only when no override is provided.
DEFAULT_SUBSTREAM_DIMS: Dict[str, int] = {
    "pos": 63,        # 21 joints * 3
    "vel": 63,
    "root": 9,
    "contact": 2,     # left foot, right foot contact bits
    "action": 29,     # G1 default; configurable
}


@dataclass
class SubstreamStats:
    """Frozen train-set-global mean/std per substream.

    Shapes:
        mean[name]: (feat_dim,)
        std [name]: (feat_dim,)
    """

    mean: Dict[str, torch.Tensor]
    std: Dict[str, torch.Tensor]

    def to(self, device: torch.device) -> "SubstreamStats":
        return SubstreamStats(
            mean={k: v.to(device=device) for k, v in self.mean.items()},
            std={k: v.to(device=device) for k, v in self.std.items()},
        )

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {"mean": dict(self.mean), "std": dict(self.std)}

    @classmethod
    def from_state_dict(cls, sd: Dict[str, Dict[str, torch.Tensor]]) -> "SubstreamStats":
        return cls(mean=dict(sd["mean"]), std=dict(sd["std"]))

    def normalize(self, name: str, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Apply (x - mean) / (std + eps) along the last axis.

        x: (..., feat_dim). Returns the same shape.
        """
        mean = self.mean[name].to(device=x.device, dtype=x.dtype)
        std = self.std[name].to(device=x.device, dtype=x.dtype)
        return (x - mean) / (std + eps)


def make_dummy_stats(substream_dims: Dict[str, int]) -> SubstreamStats:
    """Construct a no-op SubstreamStats (mean=0, std=1) for tests and bootstraps.

    Real runs replace this with the result of compute_substream_stats() from
    the trainer.
    """
    return SubstreamStats(
        mean={name: torch.zeros(dim) for name, dim in substream_dims.items()},
        std={name: torch.ones(dim) for name, dim in substream_dims.items()},
    )


# ----------------------------------------------------------------------------
# Multi-stream spectral auxiliary head
# ----------------------------------------------------------------------------


class _PerSubstreamHead(nn.Module):
    """One MLP that maps z (B, T, z_dim) -> (B, T, M * feat_dim) coefficients.

    Output is reshaped to (B, T, M, feat_dim) by the parent.
    """

    def __init__(self, z_dim: int, feat_dim: int, M: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.M = int(M)
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, M * feat_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, T, _ = z.shape
        out = self.net(z)                                   # (B, T, M*feat_dim)
        return out.view(B, T, self.M, self.feat_dim)


class MultiStreamSpectralAuxiliaryHead(nn.Module):
    """Per-frame z -> predicted DCT-II coefficients of forward K-frame windows
    across motion-feature substreams.

    Substream dims and per-substream M can be tuned independently. Targets are
    computed in the trainer (after frozen-stats normalization, then DCT) and
    passed in for loss; the head itself only knows about input z and output
    shapes.

    Input  : z (B, T, z_dim)
    Output : dict[str, Tensor], each (B, T, M_substream, feat_dim_substream)
    """

    def __init__(
        self,
        z_dim: int,
        substream_dims: Dict[str, int],
        substream_M: Dict[str, int],
        hidden: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        if substream_dims.keys() != substream_M.keys():
            raise ValueError(
                "substream_dims and substream_M must have the same keys; "
                f"dims={list(substream_dims)}, M={list(substream_M)}"
            )
        self.z_dim = int(z_dim)
        self.substream_dims = dict(substream_dims)
        self.substream_M = dict(substream_M)
        self.heads = nn.ModuleDict(
            {
                name: _PerSubstreamHead(
                    z_dim=z_dim,
                    feat_dim=self.substream_dims[name],
                    M=self.substream_M[name],
                    hidden=hidden,
                    dropout=dropout,
                )
                for name in self.substream_dims
            }
        )

    @property
    def substream_names(self) -> Tuple[str, ...]:
        return tuple(self.substream_dims.keys())

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z.ndim != 3:
            raise ValueError(f"z must be (B, T, z_dim); got {tuple(z.shape)}")
        if z.shape[-1] != self.z_dim:
            raise ValueError(f"expected z_dim={self.z_dim}, got {z.shape[-1]}")
        return {name: head(z) for name, head in self.heads.items()}


# ----------------------------------------------------------------------------
# Target builder (callable used by compute_losses)
# ----------------------------------------------------------------------------


def build_spectral_targets(
    raw_substreams: Dict[str, torch.Tensor],
    stats: SubstreamStats,
    window_K: int,
    substream_M: Dict[str, int],
    basis_cache: Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """For each substream, build the forward-window DCT-II target.

    Args:
        raw_substreams: dict name -> (B, T, feat_dim) raw motion-feature stream.
        stats: frozen train-set statistics.
        window_K: forward window length in frames.
        substream_M: dict name -> M (number of low-freq coeffs to keep).
        basis_cache: optional dict K -> (K, K) DCT basis on the right device.

    Returns:
        targets: dict name -> (B, T - K + 1, M, feat_dim) (DCT-II of forward windows).
        valid_mask: dict name -> (B, T - K + 1) bool, True where window has K
            real frames within the clip. (Right now: all True; future: support
            clip-end masking via a per-batch length tensor.)
    """
    targets: Dict[str, torch.Tensor] = {}
    valid_mask: Dict[str, torch.Tensor] = {}
    cache = basis_cache if basis_cache is not None else {}
    for name, raw in raw_substreams.items():
        if raw.ndim != 3:
            raise ValueError(
                f"raw_substreams['{name}'] must be (B, T, feat_dim); got {tuple(raw.shape)}"
            )
        B, T, F = raw.shape
        if T < window_K:
            raise ValueError(
                f"clip length T={T} < window_K={window_K}; cannot build forward windows"
            )
        # Normalize with frozen stats.
        normed = stats.normalize(name, raw)
        # Unfold forward windows: (B, T - K + 1, K, F).
        windows = normed.unfold(dimension=1, size=window_K, step=1)  # (B, T-K+1, F, K)
        windows = windows.permute(0, 1, 3, 2).contiguous()             # (B, T-K+1, K, F)
        # DCT along K axis.
        K = windows.shape[-2]
        if K not in cache:
            cache[K] = _build_dct_ii_matrix(K, dtype=windows.dtype).to(device=windows.device)
        basis = cache[K]
        coeffs = dct_ii(windows, dim=-2, basis=basis)                 # (B, T-K+1, K, F)
        M = substream_M[name]
        if M > K:
            raise ValueError(f"substream '{name}': M={M} > K={K}")
        targets[name] = coeffs[..., :M, :]                             # (B, T-K+1, M, F)
        valid_mask[name] = torch.ones(
            (B, T - window_K + 1), dtype=torch.bool, device=raw.device
        )
    return targets, valid_mask


def spectral_substream_loss(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    valid_mask: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    enabled: Dict[str, bool],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Sum of per-substream weighted spectral MSE losses.

    Args:
        pred: name -> (B, T, M, F)  (head output sliced to T - K + 1 by caller).
        target: name -> (B, T - K + 1, M, F).
        valid_mask: name -> (B, T - K + 1) bool.
        weights: name -> scalar weight.
        enabled: name -> bool. Disabled substreams contribute zero.

    Returns:
        total_loss: scalar tensor, sum_w * mse over enabled substreams.
        per_term: dict name -> scalar tensor (the per-substream component).
    """
    per_term: Dict[str, torch.Tensor] = {}
    total = None
    for name, target_t in target.items():
        if not enabled.get(name, True):
            zero = target_t.new_zeros(())
            per_term[name] = zero
            continue
        pred_t = pred[name]
        if pred_t.shape != target_t.shape:
            raise ValueError(
                f"shape mismatch for '{name}': pred {tuple(pred_t.shape)} vs target {tuple(target_t.shape)}"
            )
        mask = valid_mask[name]
        # mask is (B, T-K+1); broadcast to (B, T-K+1, M, F).
        m = mask.unsqueeze(-1).unsqueeze(-1).to(dtype=target_t.dtype)
        diff_sq = (pred_t - target_t).square() * m
        denom = m.sum() * pred_t.shape[-2] * pred_t.shape[-1]
        if float(denom.item()) <= 0.0:
            term = pred_t.sum() * 0.0
        else:
            term = diff_sq.sum() / denom
        weighted = weights.get(name, 1.0) * term
        per_term[name] = term.detach()
        total = weighted if total is None else total + weighted
    if total is None:
        # No substreams; return a zero connected to params via a dummy add.
        dummy = next(iter(pred.values()))
        total = dummy.sum() * 0.0
    return total, per_term
