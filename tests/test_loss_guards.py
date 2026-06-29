"""Regression tests for the NaN-prevention loss guards (2026-06-03 rebalance).

Both guarded losses have a 1/||.|| gradient that, unguarded, explodes to inf/NaN at
degenerate geometry and (scaled by their large weights) corrupted training (see
scripts_phaseA/NAN_DEBUG_PROMPT.md). The guards must keep the gradient FINITE and
BOUNDED there.

Run:
  PYTHONPATH=/cpfs/user/yaoyifei/codespace/gbc/GBC:/cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA \
    python GBC/tests/test_loss_guards.py
"""

import torch


def test_body_plane_forward_vector_degenerate_grad_finite():
    from GBC_MUJICA.data_preparation.gbc_retargeting.retarget_loss_utils import (
        body_plane_forward_vector,
    )

    # Degenerate frontal plane: lateral=(1,0,0), vertical=(0,0,0) -> forward=cross=0.
    # order: [sh_L, sh_R, hp_L, hp_R]
    pos = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    n = body_plane_forward_vector(pos)
    loss = n.pow(2).sum()
    loss.backward()
    g = pos.grad
    assert torch.isfinite(g).all(), f"body_plane grad not finite at degeneracy: {g}"
    # Bounded by ~1/floor (floor=1e-2 -> ~1e2), allow margin.
    assert g.abs().max().item() < 1e4, f"body_plane grad too large: {g.abs().max().item()}"


def test_body_plane_forward_vector_normal_is_unit():
    from GBC_MUJICA.data_preparation.gbc_retargeting.retarget_loss_utils import (
        body_plane_forward_vector,
    )

    # Non-degenerate pose -> output should be (near) unit length.
    pos = torch.tensor(
        [[[-0.2, 0.5, 0.0], [0.2, 0.5, 0.0], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]]
    )
    n = body_plane_forward_vector(pos)
    assert abs(n.norm().item() - 1.0) < 1e-3, f"expected unit forward, got norm {n.norm().item()}"


def test_foot_l2_norm_grad_is_bounded():
    # foot_l2 uses the L2 NORM (magnitude), whose gradient is diff/||diff|| — a unit
    # vector, bounded everywhere (and 0 at an exact match, special-cased by PyTorch).
    # So foot_l2 is NOT a NaN source and needs no guard. This test documents that
    # (it was wrongly suspected during the NaN investigation).
    d = torch.zeros(2, 3, requires_grad=True)  # exact match -> distance 0
    torch.linalg.norm(d, dim=-1).mean().backward()
    assert torch.isfinite(d.grad).all(), "norm grad must be finite at exact zero"

    d2 = (torch.randn(64, 3) * 1e-6).clone().requires_grad_(True)  # near-zero distances
    torch.linalg.norm(d2, dim=-1).mean().backward()
    assert torch.isfinite(d2.grad).all()
    assert d2.grad.abs().max().item() <= 1.0 + 1e-5, "norm grad is a unit vector -> <=1"


if __name__ == "__main__":
    for fn in [
        test_body_plane_forward_vector_degenerate_grad_finite,
        test_body_plane_forward_vector_normal_is_unit,
        test_foot_l2_norm_grad_is_bounded,
    ]:
        fn()
        print(f"PASS {fn.__name__}")
    print("ALL LOSS-GUARD TESTS PASSED")
