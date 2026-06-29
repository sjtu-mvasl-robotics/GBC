"""Tests for morphology_descriptor: bone-length descriptor from a SMPL-H fit.
Run: PYTHONPATH=GBC:GBC_MUJICA python GBC/tests/test_morphology_descriptor.py
"""
import types

import torch

from GBC.utils.data_preparation.morphology_descriptor import (
    descriptor_dim, BONELEN_DIM, _raw_bonelengths, _J,
    morphology_descriptor_from_fit,
)


class _StubBody:
    """Fake BodyModel: returns fixed Jtr regardless of beta/dmpls so the pure
    descriptor math is testable without the real SMPL-H model."""

    def __init__(self, jtr):  # jtr: (22,3)
        self._jtr = jtr

    def __call__(self, betas=None, dmpls=None):
        return types.SimpleNamespace(Jtr=self._jtr.unsqueeze(0))


def _skeleton(scale=1.0):
    J = torch.zeros(22, 3)
    J[_J["L_Hip"]]      = torch.tensor([0.1, 0.0, 0.0]) * scale
    J[_J["L_Knee"]]     = torch.tensor([0.1, 0.0, -0.4]) * scale
    J[_J["L_Ankle"]]    = torch.tensor([0.1, 0.0, -0.8]) * scale
    J[_J["R_Hip"]]      = torch.tensor([-0.1, 0.0, 0.0]) * scale
    J[_J["R_Knee"]]     = torch.tensor([-0.1, 0.0, -0.4]) * scale
    J[_J["R_Ankle"]]    = torch.tensor([-0.1, 0.0, -0.8]) * scale
    J[_J["L_Shoulder"]] = torch.tensor([0.2, 0.0, 0.5]) * scale
    J[_J["L_Elbow"]]    = torch.tensor([0.2, 0.0, 0.2]) * scale
    J[_J["L_Wrist"]]    = torch.tensor([0.2, 0.0, 0.0]) * scale
    J[_J["R_Shoulder"]] = torch.tensor([-0.2, 0.0, 0.5]) * scale
    J[_J["R_Elbow"]]    = torch.tensor([-0.2, 0.0, 0.2]) * scale
    J[_J["R_Wrist"]]    = torch.tensor([-0.2, 0.0, 0.0]) * scale
    J[_J["Pelvis"]]     = torch.tensor([0.0, 0.0, 0.0])
    J[_J["Chest"]]      = torch.tensor([0.0, 0.0, 0.4]) * scale
    J[_J["Neck"]]       = torch.tensor([0.0, 0.0, 0.55]) * scale
    J[_J["Head"]]       = torch.tensor([0.0, 0.0, 0.7]) * scale
    return J


def test_descriptor_dim():
    assert descriptor_dim("none") == 0
    assert descriptor_dim("bonelen") == 8 == BONELEN_DIM
    assert descriptor_dim("betascale", beta_dim=16) == 17
    print("PASS test_descriptor_dim")


def test_raw_bonelengths_known_skeleton():
    J = _skeleton(1.0)
    b = _raw_bonelengths(J)
    assert b.shape == (8,)
    assert abs(b[0].item() - 0.4) < 1e-5, b[0].item()    # upper_leg
    assert abs(b[1].item() - 0.4) < 1e-5, b[1].item()    # lower_leg
    assert abs(b[4].item() - 0.4) < 1e-5, b[4].item()    # shoulder_width = |0.2-(-0.2)|
    print("PASS test_raw_bonelengths_known_skeleton")


def test_bonelen_descriptor_centered_and_scales():
    neutral = _StubBody(_skeleton(1.0))
    fit_neutral = {"beta": torch.zeros(1, 16), "dmpls": torch.zeros(1, 8), "scale": torch.tensor([1.0])}
    d0 = morphology_descriptor_from_fit(fit_neutral, neutral, kind="bonelen")
    assert torch.allclose(d0, torch.zeros(8), atol=1e-5), d0   # ratio-to-self - 1 = 0
    big = _StubBody(_skeleton(1.0))                            # stub ignores beta; scale applied in fn
    fit_big = {"beta": torch.zeros(1, 16), "dmpls": torch.zeros(1, 8), "scale": torch.tensor([1.5])}
    d1 = morphology_descriptor_from_fit(fit_big, big, kind="bonelen")
    assert torch.all(d1 > 0.49) and torch.all(d1 < 0.51), d1   # scale 1.5 -> +0.5 everywhere
    print("PASS test_bonelen_descriptor_centered_and_scales")


def test_betascale_descriptor():
    fit = {"beta": torch.arange(16).float().unsqueeze(0), "dmpls": torch.zeros(1, 8), "scale": torch.tensor([2.0])}
    d = morphology_descriptor_from_fit(fit, None, kind="betascale")
    assert d.shape == (17,)
    assert abs(d[-1].item() - torch.log(torch.tensor(2.0)).item()) < 1e-5, d[-1].item()
    print("PASS test_betascale_descriptor")


if __name__ == "__main__":
    test_descriptor_dim()
    test_raw_bonelengths_known_skeleton()
    test_bonelen_descriptor_centered_and_scales()
    test_betascale_descriptor()
    print("OK morphology_descriptor")
