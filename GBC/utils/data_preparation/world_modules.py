"""Latent-space world-model auxiliary modules.

Implements the E3a-det "world-model bridge" head:

    LatentForwardHead: (z_t, a_t) -> z_hat_{t+1}

Trained alongside the supervised retargeter as a stop-gradient probe on the
encoder. Both input and target are detached from the encoder graph, so the
head learns to track the encoder's output without affecting any supervised
loss. The point is to shape a forward map in latent space that is ready to
be picked up by a downstream RL / FB-style controller (see
``2026-05-27-phase2-isaacsim-fb-pretraining-sketch.md``); it does not
co-shape the encoder itself.

Forward shape contract: z (B, T, z_dim), a (B, T, action_dim) ->
output (B, T-1, z_dim). The output frame for index t corresponds to a
prediction of z_{t+1} given (z_t, a_t).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LatentForwardHead(nn.Module):
    """MLP that predicts z_{t+1} from (z_t, a_t).

    Inputs are intended to be DETACHED from the supervised graph at call site;
    see ``UnifiedLatentRetargeter.forward`` for the canonical wiring. Detaching
    is the difference between "frozen-encoder forward probe" (this module's
    purpose) and "joint training" (which would risk collapsing the encoder to
    a trivial trajectory that is easy to predict).
    """

    def __init__(
        self,
        z_dim: int,
        action_dim: int,
        hidden: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1; got {num_layers}")
        self.z_dim = int(z_dim)
        self.action_dim = int(action_dim)
        in_dim = self.z_dim + self.action_dim
        layers: list[nn.Module] = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, self.z_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict z_{t+1} for each t in [0, T-2].

        Args:
            z: (B, T, z_dim).  Caller is responsible for detach() if desired.
            a: (B, T, action_dim).

        Returns:
            (B, T-1, z_dim). Entry [b, t] is the prediction of z[b, t+1].
        """
        if z.ndim != 3:
            raise ValueError(f"z must be (B, T, z_dim); got {tuple(z.shape)}")
        if a.ndim != 3:
            raise ValueError(f"a must be (B, T, action_dim); got {tuple(a.shape)}")
        if z.shape[:2] != a.shape[:2]:
            raise ValueError(
                f"z and a must share (B, T) prefix; got {z.shape[:2]} vs {a.shape[:2]}"
            )
        if z.shape[-1] != self.z_dim:
            raise ValueError(f"expected z_dim={self.z_dim}, got {z.shape[-1]}")
        if a.shape[-1] != self.action_dim:
            raise ValueError(f"expected action_dim={self.action_dim}, got {a.shape[-1]}")
        if z.shape[1] < 2:
            # No frames to predict against — return empty tensor with right shape.
            return z.new_zeros(z.shape[0], 0, self.z_dim)
        x = torch.cat([z[:, :-1], a[:, :-1]], dim=-1)
        return self.net(x)


def world_aux_loss(
    pred_next: torch.Tensor,
    z_target: torch.Tensor,
) -> torch.Tensor:
    """MSE between predicted next-z and (detached) encoder next-z.

    Args:
        pred_next: (B, T-1, z_dim) from LatentForwardHead.
        z_target:  (B, T-1, z_dim) -- typically z[:, 1:].detach().

    Returns:
        scalar tensor (mean MSE).
    """
    if pred_next.shape != z_target.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(pred_next.shape)} vs target {tuple(z_target.shape)}"
        )
    if pred_next.shape[1] == 0:
        return pred_next.sum() * 0.0
    return (pred_next - z_target).square().mean()


@torch.no_grad()
def rollout_in_latent(
    head: LatentForwardHead,
    z0: torch.Tensor,                       # (B, z_dim)
    actions: torch.Tensor,                  # (B, H, action_dim)
) -> torch.Tensor:
    """Autoregressive rollout in latent space, no encoder involvement.

    Useful for evaluation: feed an initial z and a sequence of actions, get
    back the predicted z trajectory of length H+1. Not used during training.

    Returns:
        (B, H+1, z_dim) with index 0 == z0.
    """
    B, H, action_dim = actions.shape
    z_traj = [z0]
    z_t = z0
    for t in range(H):
        z_in = z_t.unsqueeze(1)                          # (B, 1, z_dim)
        a_in = actions[:, t : t + 1]                     # (B, 1, action_dim)
        # Pad to length 2 so the head returns one prediction.
        z_pair = torch.cat([z_in, z_in], dim=1)
        a_pair = torch.cat([a_in, a_in], dim=1)
        z_next = head(z_pair, a_pair)[:, 0]              # (B, z_dim)
        z_traj.append(z_next)
        z_t = z_next
    return torch.stack(z_traj, dim=1)
