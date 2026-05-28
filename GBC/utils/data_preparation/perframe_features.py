from __future__ import annotations

import torch


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula. Vectorised over arbitrary leading dims.

    Args:
        aa: (..., 3) axis-angle.

    Returns:
        (..., 3, 3) rotation matrix.
    """
    if aa.shape[-1] != 3:
        raise ValueError(f"axis_angle last dim must be 3, got {aa.shape[-1]}")
    angle = aa.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis = aa / angle
    angle = angle.squeeze(-1)
    cos = angle.cos()
    sin = angle.sin()
    one_minus_cos = 1.0 - cos

    x, y, z = axis.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, zx = x * y, y * z, z * x

    r00 = cos + xx * one_minus_cos
    r01 = xy * one_minus_cos - z * sin
    r02 = zx * one_minus_cos + y * sin
    r10 = xy * one_minus_cos + z * sin
    r11 = cos + yy * one_minus_cos
    r12 = yz * one_minus_cos - x * sin
    r20 = zx * one_minus_cos - y * sin
    r21 = yz * one_minus_cos + x * sin
    r22 = cos + zz * one_minus_cos

    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def axis_angle_to_rot6d(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to 6D rotation rep (Zhou et al. 2019).

    Takes the first two rows of the rotation matrix; equivalent up to a
    transpose to the original column-stacking formulation. Both forms are
    continuous and bijective with SO(3) via Gram-Schmidt; we use rows for
    simpler row-major flattening.
    """
    if aa.shape[-1] != 3:
        raise ValueError(f"axis_angle last dim must be 3, got {aa.shape[-1]}")
    matrix = axis_angle_to_matrix(aa)
    return matrix[..., :2, :].reshape(*aa.shape[:-1], 6)


def finite_diff_velocity(positions: torch.Tensor, fps: float) -> torch.Tensor:
    """Centered finite difference velocities along the time axis (dim -3)."""
    if positions.shape[-3] < 2:
        return torch.zeros_like(positions)
    velocity = torch.zeros_like(positions)
    fwd = positions[..., 1:, :, :] - positions[..., :-1, :, :]
    velocity[..., 0, :, :] = fwd[..., 0, :, :]
    velocity[..., -1, :, :] = fwd[..., -1, :, :]
    if positions.shape[-3] > 2:
        centered = (positions[..., 2:, :, :] - positions[..., :-2, :, :]) * 0.5
        velocity[..., 1:-1, :, :] = centered
    return velocity * float(fps)
