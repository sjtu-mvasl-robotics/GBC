from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from GBC.utils.data_preparation.pose_transformer import PoseTransformer


def sinusoidal_time_encoding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype)
        * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / dim)
    )
    enc = torch.zeros(length, dim, device=device, dtype=dtype)
    enc[:, 0::2] = torch.sin(positions * div_term)
    enc[:, 1::2] = torch.cos(positions * div_term[: enc[:, 1::2].shape[1]])
    return enc


class SequenceRetargeter(nn.Module):
    """Frame teacher backbone plus a temporal residual head.

    The frame backbone is structurally the existing PoseTransformer, so a trained
    retargeting checkpoint can be loaded directly into `frame_model`. The temporal
    residual head is zero-initialized; after loading the teacher weights the model
    initially reproduces the original retargeter and can then learn sequence-level
    corrections from differentiable FK losses.
    """

    def __init__(
        self,
        load_hands: bool = False,
        num_actions: int = 29,
        frame_embedding_dim: int = 32,
        frame_joint_embedding_dim: int = 128,
        frame_num_heads: int = 4,
        frame_num_layers: int = 4,
        root_dim: int = 9,
        temporal_dim: int = 256,
        temporal_heads: int = 8,
        temporal_layers: int = 4,
        temporal_ff_mult: int = 4,
        dropout: float = 0.1,
        residual_scale: float = 0.25,
        causal: bool = False,
    ):
        super().__init__()
        self.load_hands = load_hands
        self.num_joints = 21 if not load_hands else 51
        self.input_dim = 3
        self.num_actions = num_actions
        self.root_dim = root_dim
        self.frame_embedding_dim = frame_embedding_dim
        self.residual_scale = float(residual_scale)
        self.causal = bool(causal)

        self.frame_model = PoseTransformer(
            load_hands=load_hands,
            num_actions=num_actions,
            embedding_dim=frame_embedding_dim,
            joint_embedding_dim=frame_joint_embedding_dim,
            num_heads=frame_num_heads,
            num_layers=frame_num_layers,
        )

        temporal_in = frame_embedding_dim + num_actions + root_dim
        self.temporal_input = nn.Sequential(
            nn.Linear(temporal_in, temporal_dim),
            nn.GELU(),
            nn.LayerNorm(temporal_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=temporal_heads,
            dim_feedforward=temporal_dim * temporal_ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Linear(temporal_dim, temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, num_actions),
        )
        self._zero_init_residual()

    def _zero_init_residual(self) -> None:
        last = self.residual_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def encode_frames(self, pose_body: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pose_body.ndim != 3:
            raise ValueError(f"pose_body must have shape (B, T, D), got {tuple(pose_body.shape)}")
        batch_size, window, dim = pose_body.shape
        expected = self.num_joints * self.input_dim
        if dim != expected:
            raise ValueError(f"pose_body last dim must be {expected}, got {dim}")

        flat = pose_body.reshape(batch_size * window, self.num_joints, self.input_dim)
        x = flat.permute(0, 2, 1)
        x = self.frame_model.joint_embedding(x)
        x = x.permute(0, 2, 1)
        x = self.frame_model.embedding(x)
        x = x + self.frame_model.positional_encoding
        x = self.frame_model.transformer(x)
        frame_features = x.mean(dim=1)
        base_actions = self.frame_model.decoder(frame_features)
        return (
            frame_features.reshape(batch_size, window, -1),
            base_actions.reshape(batch_size, window, -1),
        )

    def forward(
        self,
        pose_body: torch.Tensor,
        root_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, window, _ = pose_body.shape
        frame_features, base_actions = self.encode_frames(pose_body)
        if root_features is None:
            root_features = pose_body.new_zeros(batch_size, window, self.root_dim)
        if root_features.shape[:2] != (batch_size, window) or root_features.shape[-1] != self.root_dim:
            raise ValueError(
                "root_features must have shape "
                f"({batch_size}, {window}, {self.root_dim}), got {tuple(root_features.shape)}"
            )

        temporal = torch.cat([frame_features, base_actions, root_features], dim=-1)
        temporal = self.temporal_input(temporal)
        temporal = temporal + sinusoidal_time_encoding(window, temporal.shape[-1], temporal.device, temporal.dtype).view(
            1, window, -1
        )
        mask = None
        if self.causal:
            mask = torch.triu(
                torch.ones(window, window, dtype=torch.bool, device=temporal.device),
                diagonal=1,
            )
        temporal = self.temporal_encoder(temporal, mask=mask)
        residual = self.residual_head(temporal) * self.residual_scale
        actions = base_actions + residual
        return {
            "actions": actions,
            "base_actions": base_actions,
            "residual": residual,
            "frame_features": frame_features,
            "temporal_features": temporal,
        }

    def load_teacher(self, checkpoint: str, map_location: str | torch.device = "cpu") -> None:
        state = torch.load(checkpoint, map_location=map_location, weights_only=False)
        self.frame_model.load_state_dict(state)
