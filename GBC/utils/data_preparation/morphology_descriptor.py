"""Per-robot morphology descriptor derived from the SMPL-H mesh-anchor fit.

Fixed-dim, robot-INDEPENDENT (same SMPL-H kintree for every robot incl. unseen GR3),
so one shared MLP conditions the decoder. This is the per-limb morphology the fit
captures (beta/dmpls/scale) but currently discards. Encoder stays morph-free; only the
decoder sees this. Spec: docs/superpowers/specs/2026-06-25-morph-descriptor-ablation-spec.md
"""
from __future__ import annotations

import torch

# SMPL-H body joint order (matches MeshAnchorSMPLHFitter.smplh_body_names).
_SMPLH_BODY_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine", "L_Ankle",
    "R_Ankle", "Chest", "L_Toe", "R_Toe", "Neck", "L_Thorax", "R_Thorax", "Head",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]
_J = {n: i for i, n in enumerate(_SMPLH_BODY_NAMES)}

# (name, a, b, mirror_a, mirror_b) — L/R-averaged bones average |a-b| with |ma-mb|;
# widths/torso/head set mirror to None.
_BONELEN_SPEC = [
    ("upper_leg", "L_Hip", "L_Knee", "R_Hip", "R_Knee"),
    ("lower_leg", "L_Knee", "L_Ankle", "R_Knee", "R_Ankle"),
    ("upper_arm", "L_Shoulder", "L_Elbow", "R_Shoulder", "R_Elbow"),
    ("lower_arm", "L_Elbow", "L_Wrist", "R_Elbow", "R_Wrist"),
    ("shoulder_width", "L_Shoulder", "R_Shoulder", None, None),
    ("hip_width", "L_Hip", "R_Hip", None, None),
    ("torso", "Pelvis", "Chest", None, None),
    ("head", "Neck", "Head", None, None),
]
BONELEN_DIM = len(_BONELEN_SPEC)  # 8


def descriptor_dim(kind: str, beta_dim: int = 16) -> int:
    if kind == "none":
        return 0
    if kind == "bonelen":
        return BONELEN_DIM
    if kind == "betascale":
        return int(beta_dim) + 1
    raise ValueError(f"unknown descriptor kind {kind!r}")


def _smplh_joints(body_model, beta, dmpls, scale, device) -> torch.Tensor:
    out = body_model(betas=beta.to(device), dmpls=dmpls.to(device))
    return (out.Jtr[0] - out.Jtr[0, 0:1, :]) * float(scale)   # (>=22, 3) root-relative, scaled


def _raw_bonelengths(joints: torch.Tensor) -> torch.Tensor:
    vals = []
    for _name, a, b, ma, mb in _BONELEN_SPEC:
        d = (joints[_J[a]] - joints[_J[b]]).norm()
        if ma is not None:
            d = 0.5 * (d + (joints[_J[ma]] - joints[_J[mb]]).norm())
        vals.append(d)
    return torch.stack(vals)   # (8,)


def neutral_bonelengths(body_model, device="cpu", num_betas=16, num_dmpls=8) -> torch.Tensor:
    beta = torch.zeros(1, num_betas)
    dmpls = torch.zeros(1, num_dmpls)
    return _raw_bonelengths(_smplh_joints(body_model, beta, dmpls, 1.0, device))


def morphology_descriptor_from_fit(fit, body_model, kind="bonelen", device="cpu") -> torch.Tensor:
    """Return the (D,) descriptor for one robot's fit as CPU float32."""
    if kind == "none":
        return torch.zeros(0)
    if kind == "betascale":
        beta = fit["beta"][0].detach().float().cpu()
        scale = torch.log(fit["scale"].detach().float().cpu().reshape(1).clamp_min(1e-3))
        return torch.cat([beta, scale])
    if kind == "bonelen":
        joints = _smplh_joints(body_model, fit["beta"], fit["dmpls"], float(fit["scale"]), device)
        raw = _raw_bonelengths(joints).detach().float().cpu()
        ref = neutral_bonelengths(body_model, device,
                                  num_betas=fit["beta"].shape[-1],
                                  num_dmpls=fit["dmpls"].shape[-1]).detach().float().cpu()
        return raw / ref.clamp_min(1e-6) - 1.0
    raise ValueError(f"unknown descriptor kind {kind!r}")
