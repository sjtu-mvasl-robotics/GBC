import pytest
import torch

from GBC.utils.data_preparation.morphology_conditioned_retargeter import (
    SMPLH_FIT_GRAPH_FEATURE_DIM,
    MotionLatentRetargeter,
    batch_robot_specs,
    batch_smplh_fit_graph_specs,
    build_smplh_fit_graph_spec,
    build_synthetic_robot_spec,
    validate_smplh_fit_graph_path,
)


class _FakeBodyOutput:
    def __init__(self, joints: torch.Tensor):
        self.Jtr = joints


class _FakeBodyModel:
    def __call__(self, betas=None, dmpls=None, **_kwargs):
        device = betas.device if betas is not None else torch.device("cpu")
        dtype = betas.dtype if betas is not None else torch.float32
        base = torch.tensor(
            [
                [0.00, 0.00, 0.00],
                [-0.10, 0.00, -0.10],
                [0.10, 0.00, -0.10],
                [0.00, 0.00, 0.15],
                [-0.10, 0.00, -0.45],
                [0.10, 0.00, -0.45],
                [0.00, 0.00, 0.35],
                [-0.10, 0.02, -0.80],
                [0.10, 0.02, -0.80],
                [0.00, 0.00, 0.55],
                [-0.10, 0.16, -0.90],
                [0.10, 0.16, -0.90],
                [0.00, 0.00, 0.72],
                [-0.18, 0.00, 0.60],
                [0.18, 0.00, 0.60],
                [0.00, 0.00, 0.90],
                [-0.35, 0.00, 0.58],
                [0.35, 0.00, 0.58],
                [-0.62, 0.00, 0.45],
                [0.62, 0.00, 0.45],
                [-0.86, 0.00, 0.35],
                [0.86, 0.00, 0.35],
            ],
            device=device,
            dtype=dtype,
        )
        batch = int(betas.shape[0]) if betas is not None else 1
        joints = base.unsqueeze(0).expand(batch, -1, -1).clone()
        if betas is not None:
            joints[:, :, 2] += betas[:, :1] * torch.linspace(0.0, 0.05, 22, device=device, dtype=dtype)
        if dmpls is not None:
            joints[:, :, 0] += dmpls[:, :1] * 0.01
        return _FakeBodyOutput(joints)


_DOFS = [
    "waist_yaw_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
]
_PARENTS = [-1, 0, 1, 0, 3, 0]


def _fit(scale: float = 1.0, beta0: float = 0.0):
    beta = torch.zeros(1, 16)
    beta[0, 0] = beta0
    return {
        "beta": beta,
        "dmpls": torch.zeros(1, 8),
        "scale": torch.tensor([scale]),
    }


def _tiny_motion(batch_size: int = 2, window: int = 5):
    torch.manual_seed(123)
    pose = torch.randn(batch_size, window, 21 * 3) * 0.05
    root = torch.randn(batch_size, window, 9) * 0.05
    beta = torch.randn(batch_size, 16) * 0.01
    return pose, root, beta


def _tiny_model(**overrides):
    kwargs = dict(
        num_human_joints=21,
        num_actions=len(_DOFS),
        beta_dim=16,
        root_dim=9,
        d_model=48,
        z_dim=64,
        slot_z=8,
        slot_latent_dim=8,
        encoder_graph_layers=1,
        encoder_temporal_layers=1,
        encoder_num_heads=4,
        morphology_graph_layers=1,
        morphology_num_heads=4,
        decoder_graph_layers=1,
        decoder_temporal_layers=1,
        decoder_num_heads=4,
        human_decoder_graph_layers=1,
        human_decoder_num_heads=4,
        future_offsets=(1,),
        future_slot_dim=4,
        dropout=0.0,
        latent_mode="per_slot",
        causal=True,
    )
    kwargs.update(overrides)
    torch.manual_seed(7)
    return MotionLatentRetargeter(**kwargs).eval()


def _robot_batch_with_fit_graph(scale: float = 1.0):
    spec = build_synthetic_robot_spec("tiny", _DOFS, _PARENTS, seed=0)
    rb = batch_robot_specs([spec])
    fg = build_smplh_fit_graph_spec(_fit(scale=scale), _FakeBodyModel())
    rb.update(batch_smplh_fit_graph_specs([fg]))
    return rb


def _robot_batch_without_fit_graph():
    spec = build_synthetic_robot_spec("tiny", _DOFS, _PARENTS, seed=0)
    return batch_robot_specs([spec])


def test_build_smplh_fit_graph_spec_shapes_and_scale_sensitivity():
    body_model = _FakeBodyModel()
    spec = build_smplh_fit_graph_spec(_fit(scale=1.0), body_model)
    assert spec.node_features.shape == (22, SMPLH_FIT_GRAPH_FEATURE_DIM)
    assert spec.slot_assignment.shape == (22,)
    assert spec.hop_matrix.shape == (22, 22)
    assert torch.isfinite(spec.node_features).all()
    assert torch.allclose(spec.node_features[0, :3], torch.zeros(3))

    spec2 = build_smplh_fit_graph_spec(_fit(scale=1.2), body_model)
    assert not torch.allclose(spec.node_features, spec2.node_features)


def test_fit_graph_zero_init_does_not_change_outputs():
    model = _tiny_model(morph_fit_graph="smplh")
    pose, root, beta = _tiny_motion()
    model.set_robot_batch(_robot_batch_with_fit_graph(scale=1.0))
    out_graph = model(pose_body=pose, root_features=root, beta=beta, sample=False)

    model.morph_fit_graph = "none"
    model.set_robot_batch(_robot_batch_without_fit_graph())
    out_base = model(pose_body=pose, root_features=root, beta=beta, sample=False)

    assert torch.allclose(out_graph["actions"], out_base["actions"], atol=1e-5)
    assert torch.allclose(out_graph["root_xyz"], out_base["root_xyz"], atol=1e-5)


def test_encoder_z_is_independent_of_target_fit_graph():
    model = _tiny_model(morph_fit_graph="smplh")
    pose, root, beta = _tiny_motion()

    model.set_robot_batch(_robot_batch_with_fit_graph(scale=1.0))
    z1 = model(pose_body=pose, root_features=root, beta=beta, sample=False)["z"]

    model.set_robot_batch(_robot_batch_with_fit_graph(scale=1.3))
    z2 = model(pose_body=pose, root_features=root, beta=beta, sample=False)["z"]

    assert torch.equal(z1, z2)


def test_learned_fit_graph_root_height_runs_without_rest_root_height():
    model = _tiny_model(morph_fit_graph="smplh", root_height_source="learned_fit_graph")
    rb = _robot_batch_with_fit_graph(scale=1.0)
    assert "rest_root_height" not in rb
    model.set_robot_batch(rb)
    pose, root, beta = _tiny_motion()

    out = model(pose_body=pose, root_features=root, beta=beta, sample=False)

    assert "fit_stand_height" in out
    assert out["fit_stand_height"].shape == (pose.shape[0], 1)
    assert torch.isfinite(out["root_height"]).all()
    assert torch.isfinite(out["fit_stand_height"]).all()
    assert (out["fit_stand_height"] > 0).all()


def test_deprecated_fit_path_is_rejected():
    with pytest.raises(ValueError, match="deprecated"):
        validate_smplh_fit_graph_path("/tmp/deprecated/smplh_fit_DEPRECATED.pt")

