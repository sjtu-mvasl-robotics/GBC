from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


def encoder_body_joint_indices() -> Tuple[int, ...]:
    """SMPL-H joint indices the structured encoder consumes.

    These are the 21 body joints AFTER Pelvis, matching the ``pose_aa`` order
    used by ``SkeletalJointTokenizer.forward``. Pelvis is handled separately
    via the root token.
    """
    return tuple(range(1, 22))


@dataclass
class EncoderGeometryInputs:
    """Triple of optional geometry tensors fed to the structured encoder."""

    joint_pos: Optional[torch.Tensor] = None
    joint_vel: Optional[torch.Tensor] = None
    joint_contact: Optional[torch.Tensor] = None


# Constant rotvec equal to R.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec().
# Hardcoded to avoid scipy import. Verified numerically: axis = (1,1,1)/sqrt(3),
# angle = 2*pi/3, so rotvec = (1,1,1)/sqrt(3) * 2*pi/3 ≈ (1.2091996,)*3.
_CANONICAL_ROOT_ROTVEC: Tuple[float, float, float] = (1.2091996, 1.2091996, 1.2091996)


def compute_encoder_body_joints(
    body_model,
    poses: torch.Tensor,
    fit: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Return Pelvis-relative SMPL-H body joint positions for the encoder.

    Mirrors ``compute_gt_joints`` but uses the 21-joint body subset (no
    Pelvis, no robot mapping) and is reserved for the encoder.

    Args:
        body_model: instance of ``human_body_prior.body_model.BodyModel``.
        poses: (B, T, >=66) SMPL-H pose array; pose_body in [3:66] is
               passed through to ``body_model``.
        fit: dict with ``"beta"``, ``"dmpls"``, optionally ``"scale"``.
        device: torch device.

    Returns:
        (B, T, 21, 3) joint positions, Pelvis-relative, scaled by
        ``fit["scale"]`` when present.
    """
    batch_size, window, _ = poses.shape
    pose_body = poses[..., 3:66].reshape(batch_size * window, 63)
    root_orient = torch.tensor(
        _CANONICAL_ROOT_ROTVEC,
        dtype=torch.float32,
        device=device,
    ).view(1, 3).expand(batch_size * window, -1)
    with torch.no_grad():
        jtr = body_model(
            root_orient=root_orient,
            pose_body=pose_body,
            betas=fit["beta"].expand(batch_size * window, -1),
            dmpls=fit["dmpls"].expand(batch_size * window, -1),
        ).Jtr
    idx = list(encoder_body_joint_indices())
    body = jtr[:, idx, :]
    pelvis = jtr[:, :1, :]
    body = body - pelvis
    scale = fit.get("scale", None)
    if scale is not None:
        if torch.is_tensor(scale):
            body = body * float(scale.view(-1).mean().item())
        else:
            body = body * float(scale)
    return body.reshape(batch_size, window, 21, 3)
