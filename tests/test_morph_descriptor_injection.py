"""MorphDescriptorAdapter + MGE node_bias: zero-init identity & descriptor flow.

Proves: (a) node_bias defaults to a no-op; (b) the adapter is zero-initialized so a
warm-started model starts IDENTICAL to the morph-free baseline; (c) once the adapter
weights are non-zero the descriptor actually changes the outputs.

Run: PYTHONPATH=GBC:GBC_MUJICA python GBC/tests/test_morph_descriptor_injection.py
"""
import torch

from GBC.utils.data_preparation.morphology_conditioned_retargeter import (
    MorphDescriptorAdapter, MorphologyGraphEncoder, MotionLatentRetargeter,
    URDF_FEATURE_DIM, build_synthetic_robot_spec, batch_robot_specs,
)

_DOFS = ["waist_yaw_joint", "left_hip_pitch_joint", "left_knee_joint",
         "right_hip_pitch_joint", "right_knee_joint", "left_shoulder_pitch_joint"]
_PARENTS = [-1, 0, 1, 0, 3, 0]


# ---- MGE node_bias hook ----------------------------------------------------

def _mge_inputs(b=2, k=5):
    nf = torch.randn(b, k, URDF_FEATURE_DIM)
    sa = torch.zeros(b, k, dtype=torch.long)
    hop = torch.zeros(b, k, k, dtype=torch.long)
    cm = torch.zeros(b, k, dtype=torch.long)
    return nf, sa, hop, cm


def test_node_bias_default_is_noop():
    torch.manual_seed(0)
    mge = MorphologyGraphEncoder(d_model=32, num_heads=4, num_graph_layers=1).eval()
    nf, sa, hop, cm = _mge_inputs()
    a1 = mge(nf, sa, hop, cm)
    a2 = mge(nf, sa, hop, cm, node_bias=None)
    assert torch.allclose(a1[0], a2[0]) and torch.allclose(a1[1], a2[1])
    print("PASS test_node_bias_default_is_noop")


# ---- adapter unit ----------------------------------------------------------

def test_add_adapter_zero_init_identity():
    torch.manual_seed(0)
    ad = MorphDescriptorAdapter(8, 32, inject="add", target="both").eval()
    m_per_dof = torch.randn(2, 5, 32); m_cls = torch.randn(2, 32)
    desc = torch.randn(2, 8)
    p2, c2 = ad.apply_tokens(m_per_dof, m_cls, desc)
    assert torch.allclose(p2, m_per_dof) and torch.allclose(c2, m_cls)   # zero-init -> identity
    torch.nn.init.normal_(ad.cls_mlp[-1].weight, std=0.5)
    _, c3 = ad.apply_tokens(m_per_dof, m_cls, desc)
    assert not torch.allclose(c3, m_cls)                                 # descriptor now flows
    print("PASS test_add_adapter_zero_init_identity")


def test_add_cls_only_leaves_per_dof():
    ad = MorphDescriptorAdapter(8, 32, inject="add", target="cls_only").eval()
    assert not hasattr(ad, "dof_mlp")
    m_per_dof = torch.randn(2, 5, 32); m_cls = torch.randn(2, 32)
    p2, _ = ad.apply_tokens(m_per_dof, m_cls, torch.randn(2, 8))
    assert torch.allclose(p2, m_per_dof)                                 # per-dof untouched
    print("PASS test_add_cls_only_leaves_per_dof")


def test_node_aug_zero_init_identity():
    torch.manual_seed(0)
    ad = MorphDescriptorAdapter(8, 32, inject="node_aug").eval()
    nb = ad.node_bias(torch.randn(2, 8))
    assert torch.allclose(nb, torch.zeros(2, 32))                        # zero-init -> zero bias
    torch.nn.init.normal_(ad.node_mlp[-1].weight, std=0.5)
    assert not torch.allclose(ad.node_bias(torch.randn(2, 8)), torch.zeros(2, 32))
    print("PASS test_node_aug_zero_init_identity")


# ---- full model integration ------------------------------------------------

def _tiny_model(morph_dim=0, inject="add", target="both"):
    m = MotionLatentRetargeter(
        num_human_joints=21, num_actions=len(_DOFS), d_model=48, z_dim=32,
        latent_mode="per_slot", slot_z=12, dropout=0.0,
        encoder_graph_layers=1, encoder_temporal_layers=1, encoder_num_heads=4,
        morphology_graph_layers=1, morphology_num_heads=4,
        decoder_graph_layers=1, decoder_temporal_layers=1, decoder_num_heads=4,
        human_decoder_graph_layers=1, human_decoder_num_heads=4,
        morph_descriptor_dim=morph_dim, morph_descriptor_inject=inject,
        morph_descriptor_target=target,
    ).eval()
    spec = build_synthetic_robot_spec("tiny", _DOFS, _PARENTS, seed=0)
    rb = batch_robot_specs([spec])
    return m, rb


def test_default_dim0_has_no_adapter():
    m, _ = _tiny_model(morph_dim=0)
    assert m.morph_adapter is None                                       # default => byte-identical path
    print("PASS test_default_dim0_has_no_adapter")


def test_model_zero_init_matches_baseline_at_step0():
    torch.manual_seed(0)
    m, rb = _tiny_model(morph_dim=8, inject="add", target="both")
    B, T = 2, 8
    pose = torch.randn(B, T, 21 * 3); root = torch.randn(B, T, m.root_dim)
    rb_d = {**rb, "morph_descriptor": torch.randn(1, 8)}
    m.set_robot_batch(rb_d); out_d = m(pose, root, sample=False)
    m.set_robot_batch(rb);   out_0 = m(pose, root, sample=False)
    assert torch.allclose(out_d["actions"], out_0["actions"], atol=1e-5)
    assert torch.allclose(out_d["root_xyz"], out_0["root_xyz"], atol=1e-5)   # zero-init => baseline
    torch.nn.init.normal_(m.morph_adapter.cls_mlp[-1].weight, std=0.5)
    m.set_robot_batch(rb_d); out_p = m(pose, root, sample=False)
    assert not torch.allclose(out_p["root_xyz"], out_0["root_xyz"], atol=1e-5)  # descriptor flows to root
    print("PASS test_model_zero_init_matches_baseline_at_step0")


def test_node_aug_model_runs_and_flows():
    torch.manual_seed(0)
    m, rb = _tiny_model(morph_dim=8, inject="node_aug")
    B, T = 2, 8
    pose = torch.randn(B, T, 63); root = torch.randn(B, T, m.root_dim)
    rb_d = {**rb, "morph_descriptor": torch.randn(1, 8)}
    m.set_robot_batch(rb_d); out0 = m(pose, root, sample=False)
    assert out0["actions"].shape == (B, T, len(_DOFS))
    torch.nn.init.normal_(m.morph_adapter.node_mlp[-1].weight, std=0.5)
    m.set_robot_batch(rb_d); out1 = m(pose, root, sample=False)
    assert not torch.allclose(out1["actions"], out0["actions"], atol=1e-5)
    print("PASS test_node_aug_model_runs_and_flows")


if __name__ == "__main__":
    test_node_bias_default_is_noop()
    test_add_adapter_zero_init_identity()
    test_add_cls_only_leaves_per_dof()
    test_node_aug_zero_init_identity()
    test_default_dim0_has_no_adapter()
    test_model_zero_init_matches_baseline_at_step0()
    test_node_aug_model_runs_and_flows()
    print("OK morph descriptor injection")
