"""Tests for the kinematic root-z machinery (2026-06-12):
  1. rest_root_height_from_fk — analytic standing height from FK at rest
  2. RootDecoder factorized mode — z = z_stand · (1 + h), xy untouched
  3. MotionLatentRetargeter(root_height_factorized=True) demands
     robot_batch["rest_root_height"] (loud error, no silent garbage)
  4. factorized forward keeps streaming causality (exact zero leak)

Run:
  PYTHONPATH=/cpfs/user/yaoyifei/codespace/gbc/GBC:/cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA \
    python GBC/tests/test_root_kinematics.py
"""
import torch

from GBC.utils.data_preparation.morphology_conditioned_retargeter import (
    MotionLatentRetargeter,
    RootDecoder,
    batch_robot_specs,
    build_synthetic_robot_spec,
    rest_root_height_from_fk,
)


class _StubFK:
    """fk(q) -> (N, L, 3) with fixed root-relative link positions."""

    def __init__(self, link_z):
        self.link_z = torch.tensor(link_z, dtype=torch.float32)
        self.num_dofs = 7

    def __call__(self, q):
        n = q.shape[0]
        out = torch.zeros(n, len(self.link_z), 3)
        out[..., 2] = self.link_z
        return out


def test_rest_root_height_from_fk():
    fk = _StubFK([0.0, -0.30, -0.78, -0.20])  # lowest frame = foot at -0.78
    h = rest_root_height_from_fk(fk, fk.num_dofs, None, ground_offset=0.011)
    assert abs(h - 0.791) < 1e-6, h
    h2 = rest_root_height_from_fk(fk, fk.num_dofs, [1, 3], ground_offset=0.0)
    assert abs(h2 - 0.30) < 1e-6, h2
    print("PASS test_rest_root_height_from_fk")


def test_root_decoder_factorized_math():
    torch.manual_seed(0)
    rd = RootDecoder(slot_z=8, d_model=64, temporal_layers=1, causal=True)
    rd.eval()
    z_slots = torch.randn(2, 6, 8)
    m_cls = torch.randn(2, 64)
    with torch.no_grad():
        raw = rd(z_slots, m_cls)
        rest = torch.tensor([0.9, 1.1])
        fac = rd(z_slots, m_cls, rest_root_height=rest)
    assert torch.allclose(fac[..., :2], raw[..., :2]), "xy must be untouched"
    expect_z = rest.view(-1, 1) * (1.0 + raw[..., 2])
    assert torch.allclose(fac[..., 2], expect_z, atol=1e-6), "z must be z_stand*(1+h)"
    # near-zero h ⇒ z ≈ analytic standing height (zero-shot default property)
    print(f"PASS test_root_decoder_factorized_math (raw h range "
          f"[{raw[..., 2].min():.3f},{raw[..., 2].max():.3f}])")


def _make_model(**kw):
    torch.manual_seed(0)
    return MotionLatentRetargeter(
        num_human_joints=21, num_actions=15, beta_dim=16, root_dim=9,
        d_model=64, z_dim=64, slot_z=8, slot_latent_dim=8,
        encoder_graph_layers=1, encoder_temporal_layers=1, encoder_num_heads=4,
        morphology_graph_layers=1, morphology_num_heads=4,
        decoder_graph_layers=1, decoder_temporal_layers=1, decoder_num_heads=4,
        future_offsets=(1,), future_slot_dim=4, latent_mode="per_slot",
        causal=True, unified_human_decoder=False, **kw)


def test_factorized_requires_rest_height_and_stays_causal():
    B, T = 2, 16
    pose = torch.randn(B, T, 63) * 0.1
    root = torch.randn(B, T, 9) * 0.1
    beta = torch.randn(B, 16) * 0.1
    parents = [-1] + list(range(14))
    spec = build_synthetic_robot_spec("rA", [f"j{i}" for i in range(15)], parents, seed=1)
    rb = batch_robot_specs([spec])

    m = _make_model(root_height_factorized=True)
    m.eval()
    m.set_robot_batch(rb)  # no rest_root_height on purpose
    try:
        m(pose_body=pose, root_features=root, beta=beta, sample=False)
        raise AssertionError("factorized forward without rest_root_height must raise")
    except RuntimeError as e:
        assert "rest_root_height" in str(e)
    rb2 = dict(rb)
    rb2["rest_root_height"] = torch.tensor([0.85])
    m.set_robot_batch(rb2)
    with torch.no_grad():
        out = m(pose_body=pose, root_features=root, beta=beta, sample=False)
        assert torch.isfinite(out["root_xyz"]).all()
        # causality: future perturbation must not change past root_xyz
        pose_p = pose.clone(); pose_p[:, T // 2:] += 10.0
        root_p = root.clone(); root_p[:, T // 2:] += 10.0
        out_p = m(pose_body=pose_p, root_features=root_p, beta=beta, sample=False)
        leak = (out["root_xyz"][:, : T // 2] - out_p["root_xyz"][:, : T // 2]).abs().max().item()
    assert leak == 0.0, f"causal leak {leak}"
    print("PASS test_factorized_requires_rest_height_and_stays_causal")


def test_root_ballistic_loss():
    """A real free-fall parabola over airborne frames → ~0; a flat trajectory
    (the saturated failure mode) → penalized; grounded frames are ignored."""
    from GBC_MUJICA.data_preparation.gbc_retargeting.train_graph_latent import root_ballistic_loss
    fps = 20.0
    dt = 1.0 / fps
    g = 9.81
    T = 20
    t = torch.arange(T, dtype=torch.float32) * dt
    # take off at frame 4, land at 16; apex parabola with v0 chosen for ~12-frame flight
    t0, t1 = 4, 16
    flight = (torch.arange(T) >= t0) & (torch.arange(T) <= t1)
    tau = (torch.arange(T, dtype=torch.float32) - t0) * dt
    Tf = (t1 - t0) * dt
    v0 = 0.5 * g * Tf  # so it lands at the same height it left
    z = torch.full((T,), 0.9)
    z_fly = 0.9 + v0 * tau - 0.5 * g * tau ** 2
    z = torch.where(flight, z_fly, z)
    z = z[None]  # (1, T)
    # contact: 2 feet, airborne (both 0) exactly on flight frames
    contact = torch.ones(1, T, 2)
    contact[0, flight] = 0.0
    loss_parab = root_ballistic_loss(z, contact, fps).item()
    assert loss_parab < 1e-5, f"true parabola must satisfy ballistic eq, got {loss_parab}"
    # flat (apex clipped to standing height) → large residual on flight frames
    z_flat = torch.full((1, T), 0.9)
    loss_flat = root_ballistic_loss(z_flat, contact, fps).item()
    assert loss_flat > 1e-4, f"flat trajectory must be penalized, got {loss_flat}"
    # all-grounded → no flight triplet → exactly 0 (no spurious penalty)
    contact_ground = torch.ones(1, T, 2)
    assert root_ballistic_loss(z, contact_ground, fps).item() == 0.0
    # T<3 guard
    assert root_ballistic_loss(z[:, :2], contact[:, :2], fps).item() == 0.0
    print(f"PASS test_root_ballistic_loss (parabola {loss_parab:.2e}, flat {loss_flat:.4f})")


if __name__ == "__main__":
    test_rest_root_height_from_fk()
    test_root_decoder_factorized_math()
    test_factorized_requires_rest_height_and_stays_causal()
    test_root_ballistic_loss()
    print("\nOK: analytic rest height, factorized z, loud missing-key, causality "
          "preserved, ballistic free-fall constraint.")
