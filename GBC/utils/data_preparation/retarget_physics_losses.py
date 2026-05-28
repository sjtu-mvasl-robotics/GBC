from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import torch


Reduction = Literal["mean", "sum", "none"]


@dataclass(frozen=True)
class MeshAnchorFitReference:
    beta: torch.Tensor
    dmpls: torch.Tensor
    scale: torch.Tensor
    metrics: Mapping[str, float]
    fit_poses: tuple[str, ...]


@dataclass(frozen=True)
class RetargetPhysicsLossConfig:
    fps: float = 50.0
    ground_height: float = 0.0
    foot_ground_offset: float = 0.0
    contact_clearance: float = 0.03
    min_feet_distance: float = 0.20
    max_feet_distance: float = 0.65
    joint_limit_margin: float = 0.0
    w_foot_slide: float = 1.0
    w_ground_penetration: float = 1.0
    w_contact_height: float = 1.0
    w_feet_distance: float = 1.0
    w_joint_limit: float = 1.0
    w_body_shape: float = 1.0


def _check_reduction(reduction: Reduction) -> None:
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")


def _as_tensor_like(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _reduce(value: torch.Tensor, reduction: Reduction = "mean", mask: torch.Tensor | None = None) -> torch.Tensor:
    _check_reduction(reduction)
    if reduction == "none":
        return value if mask is None else value * mask
    if mask is not None:
        mask = mask.to(device=value.device, dtype=value.dtype)
        while mask.dim() < value.dim():
            mask = mask.unsqueeze(-1)
        if mask.shape != value.shape:
            mask = mask.expand_as(value)
        value = value * mask
        if reduction == "mean":
            denom = mask.sum().clamp_min(1.0)
            return value.sum() / denom
        return value.sum()
    if reduction == "mean":
        return value.mean()
    return value.sum()


def finite_difference_velocity(positions: torch.Tensor, fps: float = 50.0) -> torch.Tensor:
    """Compute sequence velocity along the time axis.

    Args:
        positions: `(B, T, ..., C)` positions.
        fps: Frame rate used to convert finite differences to velocity.

    Returns:
        Tensor with the same shape as `positions`. The first frame velocity is zero.
    """
    if positions.dim() < 3:
        raise ValueError(
            f"positions must include batch, time, and coordinate dims, got {tuple(positions.shape)}"
        )
    previous = torch.cat([positions[:, :1], positions[:, :-1]], dim=1)
    return (positions - previous) * float(fps)


def foot_sliding_loss(
    foot_positions: torch.Tensor,
    feet_contact: torch.Tensor,
    fps: float = 50.0,
    xy_only: bool = True,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Penalize foot velocity during contact.

    This is the differentiable tensor form of the existing contact-no-velocity rule.
    `foot_positions` is `(B, T, F, 3)` and `feet_contact` is `(B, T, F)` or
    `(B, T, F, 1)`.
    """
    if foot_positions.dim() != 4 or foot_positions.shape[-1] < 3:
        raise ValueError(
            f"foot_positions must have shape (B, T, F, 3+), got {tuple(foot_positions.shape)}"
        )
    velocity = finite_difference_velocity(foot_positions, fps=fps)
    if xy_only:
        velocity = velocity[..., :2]
    squared_speed = velocity.square().sum(dim=-1)
    mask = feet_contact.squeeze(-1) if feet_contact.dim() == 4 and feet_contact.shape[-1] == 1 else feet_contact
    return _reduce(squared_speed, reduction=reduction, mask=mask)


def foot_ground_penetration_loss(
    foot_positions: torch.Tensor,
    ground_height: torch.Tensor | float = 0.0,
    foot_ground_offset: torch.Tensor | float = 0.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Penalize feet below the ground plane."""
    if foot_positions.shape[-1] < 3:
        raise ValueError(f"foot_positions last dim must contain xyz, got {tuple(foot_positions.shape)}")
    z = foot_positions[..., 2] - _as_tensor_like(foot_ground_offset, foot_positions)
    ground = _as_tensor_like(ground_height, foot_positions)
    penetration = torch.relu(ground - z).square()
    return _reduce(penetration, reduction=reduction)


def contact_height_loss(
    foot_positions: torch.Tensor,
    feet_contact: torch.Tensor,
    ground_height: torch.Tensor | float = 0.0,
    foot_ground_offset: torch.Tensor | float = 0.0,
    clearance: float = 0.03,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Keep contact feet close to the ground plane with a small dead zone."""
    if foot_positions.shape[-1] < 3:
        raise ValueError(f"foot_positions last dim must contain xyz, got {tuple(foot_positions.shape)}")
    z = foot_positions[..., 2] - _as_tensor_like(foot_ground_offset, foot_positions)
    ground = _as_tensor_like(ground_height, foot_positions)
    error = torch.relu(torch.abs(z - ground) - float(clearance)).square()
    mask = feet_contact.squeeze(-1) if feet_contact.dim() == 4 and feet_contact.shape[-1] == 1 else feet_contact
    return _reduce(error, reduction=reduction, mask=mask)


def feet_distance_barrier_loss(
    foot_positions: torch.Tensor,
    left_index: int = 0,
    right_index: int = 1,
    min_distance: float = 0.20,
    max_distance: float = 0.65,
    eps: float = 1e-8,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Soft xy-distance barrier between left and right feet."""
    if foot_positions.dim() != 4 or foot_positions.shape[-2] <= max(left_index, right_index):
        raise ValueError(
            "foot_positions must have shape (B, T, F, 3+) with both foot indices, "
            f"got {tuple(foot_positions.shape)}"
        )
    left = foot_positions[..., left_index, :2]
    right = foot_positions[..., right_index, :2]
    distance = torch.sqrt((left - right).square().sum(dim=-1) + eps)
    too_close = torch.relu(float(min_distance) - distance).square()
    too_far = torch.relu(distance - float(max_distance)).square()
    return _reduce(too_close + too_far, reduction=reduction)


def joint_limit_barrier_loss(
    actions: torch.Tensor,
    min_limits: torch.Tensor,
    max_limits: torch.Tensor,
    margin: float = 0.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Soft joint-limit barrier for retargeted robot actions."""
    min_limits = _as_tensor_like(min_limits, actions)
    max_limits = _as_tensor_like(max_limits, actions)
    lower = torch.relu(min_limits + float(margin) - actions).square()
    upper = torch.relu(actions - (max_limits - float(margin))).square()
    return _reduce(lower + upper, reduction=reduction)


def body_shape_prior_loss(
    pred_beta: torch.Tensor,
    reference_beta: torch.Tensor,
    pred_dmpls: torch.Tensor | None = None,
    reference_dmpls: torch.Tensor | None = None,
    pred_scale: torch.Tensor | None = None,
    reference_scale: torch.Tensor | None = None,
    beta_weight: float = 1.0,
    dmpl_weight: float = 0.25,
    scale_weight: float = 1.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Keep a predicted SMPLH shape near a fitted robot body reference."""
    reference_beta = _as_tensor_like(reference_beta, pred_beta)
    losses = [float(beta_weight) * _reduce((pred_beta - reference_beta).square(), reduction=reduction)]

    if pred_dmpls is not None and reference_dmpls is not None:
        reference_dmpls = _as_tensor_like(reference_dmpls, pred_dmpls)
        losses.append(float(dmpl_weight) * _reduce((pred_dmpls - reference_dmpls).square(), reduction=reduction))
    if pred_scale is not None and reference_scale is not None:
        reference_scale = _as_tensor_like(reference_scale, pred_scale)
        losses.append(float(scale_weight) * _reduce((pred_scale - reference_scale).square(), reduction=reduction))

    return torch.stack(losses).sum()


def load_mesh_anchor_fit(
    path: str | Path,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = torch.float32,
) -> MeshAnchorFitReference:
    """Load `mesh_anchor_smplh_fit.py` output as tensors for training-time priors."""
    path = Path(path)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "beta" not in obj:
        raise ValueError(f"Expected mesh-anchor fit dict with a 'beta' tensor in {path}")

    beta = torch.as_tensor(obj["beta"])
    if dtype is not None and beta.is_floating_point():
        beta = beta.to(dtype=dtype)
    if device is not None:
        beta = beta.to(device=device)

    dmpls = torch.as_tensor(obj.get("dmpls", torch.zeros(beta.shape[:-1] + (8,), dtype=beta.dtype)))
    scale = torch.as_tensor(obj.get("scale", torch.ones(beta.shape[:-1], dtype=beta.dtype)))
    if dtype is not None:
        if dmpls.is_floating_point():
            dmpls = dmpls.to(dtype=dtype)
        if scale.is_floating_point():
            scale = scale.to(dtype=dtype)
    if device is not None:
        dmpls = dmpls.to(device=device)
        scale = scale.to(device=device)

    fit_poses = obj.get("fit_poses", ())
    if isinstance(fit_poses, str):
        fit_poses = (fit_poses,)
    metrics = obj.get("metrics", {})
    return MeshAnchorFitReference(
        beta=beta,
        dmpls=dmpls,
        scale=scale,
        metrics=metrics if isinstance(metrics, Mapping) else {},
        fit_poses=tuple(str(name) for name in fit_poses),
    )


def retarget_physics_loss(
    foot_positions: torch.Tensor,
    feet_contact: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    action_min: torch.Tensor | None = None,
    action_max: torch.Tensor | None = None,
    body_beta: torch.Tensor | None = None,
    body_dmpls: torch.Tensor | None = None,
    body_scale: torch.Tensor | None = None,
    body_reference: MeshAnchorFitReference | None = None,
    config: RetargetPhysicsLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Aggregate differentiable retargeting losses.

    Sensor-only rules from the RL environment are intentionally not included here;
    this module expects tensors produced by a retargeting network and FK.
    """
    if config is None:
        config = RetargetPhysicsLossConfig()

    total = foot_positions.sum() * 0.0
    losses: dict[str, torch.Tensor] = {}

    losses["ground_penetration"] = foot_ground_penetration_loss(
        foot_positions,
        ground_height=config.ground_height,
        foot_ground_offset=config.foot_ground_offset,
    )
    total = total + config.w_ground_penetration * losses["ground_penetration"]

    losses["feet_distance"] = feet_distance_barrier_loss(
        foot_positions,
        min_distance=config.min_feet_distance,
        max_distance=config.max_feet_distance,
    )
    total = total + config.w_feet_distance * losses["feet_distance"]

    if feet_contact is not None:
        losses["foot_slide"] = foot_sliding_loss(foot_positions, feet_contact, fps=config.fps)
        total = total + config.w_foot_slide * losses["foot_slide"]
        losses["contact_height"] = contact_height_loss(
            foot_positions,
            feet_contact,
            ground_height=config.ground_height,
            foot_ground_offset=config.foot_ground_offset,
            clearance=config.contact_clearance,
        )
        total = total + config.w_contact_height * losses["contact_height"]

    if actions is not None and action_min is not None and action_max is not None:
        losses["joint_limit"] = joint_limit_barrier_loss(
            actions,
            action_min,
            action_max,
            margin=config.joint_limit_margin,
        )
        total = total + config.w_joint_limit * losses["joint_limit"]

    if body_beta is not None and body_reference is not None:
        losses["body_shape"] = body_shape_prior_loss(
            body_beta,
            body_reference.beta,
            pred_dmpls=body_dmpls,
            reference_dmpls=body_reference.dmpls,
            pred_scale=body_scale,
            reference_scale=body_reference.scale,
        )
        total = total + config.w_body_shape * losses["body_shape"]

    losses["total"] = total
    return losses
