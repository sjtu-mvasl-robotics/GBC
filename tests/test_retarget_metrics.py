"""Tests for the retargeting evaluation metrics in math_utils
(motion_jerk, foot_slide_metric, ground_penetration_metric,
contact_preservation_metric).

These judge a retarget by PHYSICAL plausibility + contact, the OmniRetarget way
— NOT by distance to a reference. Each test pins the metric's defining property
on a hand-built motion where the right answer is known.

Run:
  PYTHONPATH=/cpfs/user/yaoyifei/codespace/gbc/GBC:/cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA \
    python GBC/tests/test_retarget_metrics.py
"""
import numpy as np
import torch

from GBC.utils.base.math_utils import (
    motion_jerk,
    foot_slide_metric,
    ground_penetration_metric,
    contact_preservation_metric,
    foot_sticking_mask,
    center_of_mass,
    balance_support_metric,
)


def test_motion_jerk_smooth_vs_jittery():
    # constant-velocity ramp -> 3rd time-derivative analytically 0 (smooth);
    # float64 keeps the /dt^3 amplification of rounding negligible.
    T = 30
    t = torch.arange(T, dtype=torch.float64)[:, None]
    x = t * 0.1 + 0.3                      # (T, 1) linear in time
    js = motion_jerk(x, fps=20).item()
    g = torch.Generator().manual_seed(0)
    xj = x + torch.randn(T, 1, generator=g, dtype=torch.float64) * 0.05  # per-frame jitter
    jj = motion_jerk(xj, fps=20).item()
    # metric works in float32 internally; at fps=20 the /dt^3 amplifies float32
    # rounding to a ~2e-3 floor for perfectly smooth motion — still tiny vs jitter
    assert js < 1e-2, js                          # ~floor for smooth motion
    assert jj > 100 * max(js, 1e-9), (js, jj)     # jitter is orders worse
    # numpy input + (T,L,3) shape both accepted; <4 frames -> 0
    assert motion_jerk(np.zeros((10, 4, 3), dtype=np.float32), fps=20).item() == 0.0
    assert motion_jerk(torch.zeros(3, 2), fps=20).item() == 0.0
    print(f"PASS test_motion_jerk_smooth_vs_jittery (smooth {js:.2e} << jitter {jj:.1f})")


def test_foot_slide_uses_source_stance():
    # SOURCE (human) toes are planted the whole clip (xy fixed) -> stance=True.
    # ROBOT foot 0 honors it (fixed xy); robot foot 1 SLIDES while source says
    # planted -> foot-skating. Morph-aware: stance from source, slip from robot.
    T = 40
    src_toe = torch.zeros(T, 2, 2)                      # source toes fixed (planted)
    z = torch.full((T, 2, 1), 0.02)
    rxy0 = torch.zeros(T, 2)                            # robot foot 0 planted
    sx = torch.linspace(0, 1.0, T)                     # robot foot 1 drifts 1 m
    rxy1 = torch.stack([sx, torch.zeros(T)], dim=-1)
    robot = torch.cat([torch.stack([rxy0, rxy1], dim=1), z], dim=-1)  # (T,2,3)
    m = foot_slide_metric(robot, source_toe_xy=src_toe, fps=20)
    assert m["max_slip"].item() > 0.3, m["max_slip"].item()         # foot 1 slides
    assert m["slip_frac"].item() > 0.4, m["slip_frac"].item()
    # robot honors the planted source on BOTH feet -> ~0 slip
    robot_ok = robot.clone(); robot_ok[:, 1, 0] = 0.0
    mp = foot_slide_metric(robot_ok, source_toe_xy=src_toe, fps=20)
    assert mp["mean_slip"].item() < 1e-4 and mp["max_slip"].item() < 1e-4
    # when the SOURCE foot itself moves (swing), those frames are NOT stance,
    # so robot motion there is NOT counted as slip
    src_moving = torch.stack([torch.stack([torch.linspace(0, 2, T), torch.zeros(T)], -1)] * 2, dim=1)
    m2 = foot_slide_metric(robot, source_toe_xy=src_moving, fps=20)
    assert m2["slip_frac"].item() < 1e-6, "source-swing frames must not count as slip"
    # must refuse robot-only stance (not morph-aware)
    try:
        foot_slide_metric(robot, fps=20); raise AssertionError("should require source stance")
    except ValueError:
        pass
    print(f"PASS test_foot_slide_uses_source_stance (slide max {m['max_slip']:.3f}, swing slip {m2['slip_frac']:.1e})")


def test_ground_penetration():
    T, L = 20, 3
    pos = torch.zeros(T, L, 3)
    pos[..., 2] = 0.1                     # all 10 cm above floor
    clean = ground_penetration_metric(pos, ground_z=0.0)
    assert clean["mean_depth"].item() == 0.0 and clean["pen_frac"].item() == 0.0
    pos[5:10, 0, 2] = -0.03              # link 0 dips 3 cm below floor on 5 frames
    pen = ground_penetration_metric(pos, ground_z=0.0)
    assert abs(pen["max_depth"].item() - 0.03) < 1e-6, pen["max_depth"].item()
    assert abs(pen["pen_frac"].item() - 5 / T) < 1e-6, pen["pen_frac"].item()
    # clearance shifts the threshold (a sole radius)
    pen2 = ground_penetration_metric(pos, ground_z=0.0, clearance=0.05)
    assert pen2["max_depth"].item() > pen["max_depth"].item()
    print("PASS test_ground_penetration")


def test_contact_preservation():
    T, K = 30, 2
    h = torch.zeros(T, K, dtype=torch.bool)
    h[10:20, 0] = True; h[15:25, 1] = True   # human contacts
    # identical robot -> preserved 1, false 0
    same = contact_preservation_metric(h, h)
    assert abs(same["preserved"].item() - 1.0) < 1e-6 and same["false_contact"].item() < 1e-6
    # robot drops all -> preserved 0
    dropped = contact_preservation_metric(h, torch.zeros_like(h))
    assert dropped["preserved"].item() == 0.0
    # robot invents contacts where human has none -> false_contact > 0
    r = h.clone(); r[0:5, 0] = True
    inv = contact_preservation_metric(h, r)
    assert inv["preserved"].item() > 0.99 and inv["false_contact"].item() > 0.0
    print("PASS test_contact_preservation")


def test_foot_sticking_mask_source_velocity():
    # source toe: planted (xy fixed) frames 5-15, swinging (xy moving) elsewhere
    T = 20
    xy = torch.zeros(T, 1, 2)
    move = torch.linspace(0, 1.0, T)            # 1 m over clip -> ~0.05 m/frame (swing)
    xy[:5, 0, 0] = move[:5]; xy[15:, 0, 0] = move[15:] - move[15] + 0.0  # moving ends
    # frames 5-15 fixed at move[5]
    xy[5:15, 0, 0] = move[5]
    mask = foot_sticking_mask(xy, vel_threshold=0.01)
    assert mask[6:15, 0].all(), "fixed-xy source frames are stance"
    assert not mask[0, 0], "frame 0 never stance"
    assert not mask[2, 0], "moving (swing) source frames are not stance"
    # height is IRRELEVANT (morph-independent): same xy at any z gives same mask
    print("PASS test_foot_sticking_mask_source_velocity")


def test_center_of_mass_mass_weighted():
    # two links, masses 1 and 3 -> CoM is 3/4 of the way to the heavy link
    pos = torch.zeros(5, 2, 3)
    pos[:, 1, 0] = 4.0                       # heavy link at x=4
    mass = torch.tensor([1.0, 3.0])
    com = center_of_mass(pos, mass)
    assert torch.allclose(com[:, 0], torch.full((5,), 3.0), atol=1e-6), com[0]
    print("PASS test_center_of_mass_mass_weighted")


def test_balance_support_static():
    T = 10
    feet = torch.zeros(T, 2, 2)
    feet[:, 0, 0] = -0.15; feet[:, 1, 0] = 0.15   # left foot x=-0.15, right x=+0.15
    stance = torch.ones(T, 2, dtype=torch.bool)   # double support
    # CoM centered between feet -> inside the segment -> stable (margin < 0)
    com_in = torch.zeros(T, 2)
    m_in = balance_support_metric(com_in, feet, stance, foot_radius=0.08)
    assert m_in["unstable_frac"].item() == 0.0 and m_in["mean_margin"].item() < 0
    # CoM far to the side (x=1) beyond feet+radius -> unstable
    com_out = torch.zeros(T, 2); com_out[:, 0] = 1.0
    m_out = balance_support_metric(com_out, feet, stance, foot_radius=0.08)
    assert m_out["unstable_frac"].item() == 1.0 and m_out["mean_margin"].item() > 0
    # single support: only left foot planted; CoM over left foot -> stable
    st1 = torch.zeros(T, 2, dtype=torch.bool); st1[:, 0] = True
    com_l = torch.zeros(T, 2); com_l[:, 0] = -0.15
    m1 = balance_support_metric(com_l, feet, st1, foot_radius=0.08)
    assert m1["unstable_frac"].item() == 0.0
    # flight (no planted foot) excluded from contact_frac
    st0 = torch.zeros(T, 2, dtype=torch.bool)
    m0 = balance_support_metric(com_in, feet, st0, foot_radius=0.08)
    assert m0["contact_frac"].item() == 0.0
    print(f"PASS test_balance_support_static (centered margin {m_in['mean_margin']:.3f} < 0 < side {m_out['mean_margin']:.3f})")


def test_length_scale_normalization():
    """ROBOT_HEIGHT (length_scale) normalization makes magnitudes cross-morph
    comparable: scaling length_scale by k divides reported lengths by k, while
    sign-based rate metrics (slip_frac/unstable_frac) are unchanged."""
    T = 30
    # foot-slide: source planted, robot foot 1 slides
    src = torch.zeros(T, 2, 2)
    z = torch.full((T, 2, 1), 0.02)
    sx = torch.linspace(0, 1.0, T)
    rxy = torch.stack([torch.zeros(T, 2), torch.stack([sx, torch.zeros(T)], -1)], dim=1)
    robot = torch.cat([rxy, z], dim=-1)
    m1 = foot_slide_metric(robot, source_toe_xy=src, fps=20, length_scale=1.0)
    m2 = foot_slide_metric(robot, source_toe_xy=src, fps=20, speed_tol=0.05 / 2, length_scale=2.0)
    assert abs(m2["max_slip"].item() - m1["max_slip"].item() / 2) < 1e-5
    assert abs(m2["slip_frac"].item() - m1["slip_frac"].item()) < 1e-6  # threshold scaled too
    # penetration depth halves
    pos = torch.zeros(10, 1, 3); pos[:, 0, 2] = -0.04
    p1 = ground_penetration_metric(pos, ground_z=0.0, length_scale=1.0)
    p2 = ground_penetration_metric(pos, ground_z=0.0, length_scale=2.0)
    assert abs(p2["max_depth"].item() - p1["max_depth"].item() / 2) < 1e-6
    assert p2["pen_frac"].item() == p1["pen_frac"].item()  # rate unchanged
    # balance margin halves, unstable_frac unchanged
    feet = torch.zeros(10, 2, 2); feet[:, 0, 0] = -0.15; feet[:, 1, 0] = 0.15
    st = torch.ones(10, 2, dtype=torch.bool); com = torch.zeros(10, 2)
    b1 = balance_support_metric(com, feet, st, foot_radius=0.08, length_scale=1.0)
    b2 = balance_support_metric(com, feet, st, foot_radius=0.08, length_scale=2.0)
    assert abs(b2["mean_margin"].item() - b1["mean_margin"].item() / 2) < 1e-6
    assert b2["unstable_frac"].item() == b1["unstable_frac"].item()
    print("PASS test_length_scale_normalization (magnitudes /k, rates unchanged)")


if __name__ == "__main__":
    test_motion_jerk_smooth_vs_jittery()
    test_foot_slide_uses_source_stance()
    test_ground_penetration()
    test_contact_preservation()
    test_foot_sticking_mask_source_velocity()
    test_center_of_mass_mass_weighted()
    test_balance_support_static()
    test_length_scale_normalization()
    print("\nOK: retarget eval metrics — jerk(smoothness), foot-slide, ground "
          "penetration, contact preservation, stance mask, CoM, static balance.")
