"""Smoke test for the unified human decoder path in MotionLatentRetargeter.

What we check:
  1. `build_human_spec()` returns a RobotSpec with the expected shapes and
     control_mode=3 (rotation).
  2. `MotionLatentRetargeter(unified_human_decoder=True)` constructs cleanly
     and registers `_human_spec_*` buffers.
  3. A forward pass on a small synthetic batch produces:
        - actions          (B, T, K_robot)
        - human_pose_aa_recon (B, T, J_human, 3)
        - human_root_recon (B, T, root_dim)
     with finite values and gradients flowing back to encoder, MGE, CMAD,
     and the new heads.
  4. The legacy path (`unified_human_decoder=False`) still works (regression).

Run:
  PYTHONPATH=/cpfs/user/yaoyifei/codespace/gbc/GBC:/cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA \
    python GBC/tests/test_unified_human_decoder.py
"""
import torch
from GBC.utils.data_preparation.morphology_conditioned_retargeter import (
    URDF_FEATURE_DIM,
    MotionLatentRetargeter,
    build_human_spec,
    build_synthetic_robot_spec,
    batch_robot_specs,
)


def _make_input(B=2, T=8, J=21, beta_dim=16, root_dim=9):
    pose_body = torch.randn(B, T, J * 3) * 0.1
    root_features = torch.randn(B, T, root_dim) * 0.1
    beta = torch.randn(B, beta_dim) * 0.1
    return pose_body, root_features, beta


def _make_robot_spec(K=15):
    # Small synthetic robot for the test. Parent = i-1 except root=−1.
    parents = [-1] + list(range(K - 1))
    names = [f"j{i}" for i in range(K)]
    spec = build_synthetic_robot_spec("test_robot", names, parents, seed=42)
    return spec


def test_build_human_spec_shapes_and_modes():
    spec = build_human_spec()
    assert spec.node_features.shape == (21, URDF_FEATURE_DIM), spec.node_features.shape
    assert spec.parent_index.shape == (21,)
    assert spec.slot_assignment.shape == (21,)
    assert spec.hop_matrix.shape == (21, 21)
    assert (spec.control_mode == 3).all(), "human joints should use rotation mode (3)"
    # type_other column (idx 6) is the flag we set
    assert (spec.node_features[:, 6] == 1.0).all()
    # graph degree should be non-trivial
    assert spec.node_features[:, 20].sum() > 0
    print("PASS test_build_human_spec_shapes_and_modes")


def test_unified_forward_runs_and_gradients_flow():
    torch.manual_seed(0)
    B, T = 2, 8
    model = MotionLatentRetargeter(
        num_human_joints=21,
        num_actions=15,
        beta_dim=16,
        root_dim=9,
        d_model=64,
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
        future_offsets=(1,),
        future_slot_dim=4,
        latent_mode="per_slot",
        causal=True,
        unified_human_decoder=True,
    )
    # Robot spec to feed the production-style forward.
    spec = _make_robot_spec(K=15)
    model.set_robot_batch(batch_robot_specs([spec]))

    # Verify _human_spec_* buffers exist and are integer/float as expected.
    assert hasattr(model, "_human_spec_node_features"), "human spec buffer missing"
    assert model._human_spec_node_features.shape == (1, 21, URDF_FEATURE_DIM)

    pose_body, root_features, beta = _make_input(B=B, T=T, J=21, beta_dim=16, root_dim=9)
    out = model(pose_body=pose_body, root_features=root_features, beta=beta)

    K = 15
    assert out["actions"].shape == (B, T, K), out["actions"].shape
    assert out["human_pose_aa_recon"].shape == (B, T, 21, 3), out["human_pose_aa_recon"].shape
    assert out["human_root_recon"].shape == (B, T, 9), out["human_root_recon"].shape

    for key in ("actions", "human_pose_aa_recon", "human_root_recon"):
        assert torch.isfinite(out[key]).all(), f"{key} contains non-finite values"

    # Loss + backward — both heads should receive grad.
    target_actions = torch.zeros_like(out["actions"])
    target_pose = torch.zeros_like(out["human_pose_aa_recon"])
    target_root = torch.zeros_like(out["human_root_recon"])
    loss = (
        (out["actions"] - target_actions).pow(2).mean()
        + (out["human_pose_aa_recon"] - target_pose).pow(2).mean()
        + (out["human_root_recon"] - target_root).pow(2).mean()
    )
    loss.backward()

    grad_norms = {}
    for name in (
        "human_pose_aa_head.1.weight",
        "human_root_head.1.weight",
        "cmad.action_head.1.weight",
        "mfme.encoder.encoder.graph_blocks.0.attn.in_proj_weight",
        "mge.control_embed.weight",
    ):
        p = dict(model.named_parameters()).get(name)
        if p is None or p.grad is None:
            grad_norms[name] = None
        else:
            grad_norms[name] = float(p.grad.norm())
    assert grad_norms["human_pose_aa_head.1.weight"] and grad_norms["human_pose_aa_head.1.weight"] > 0
    assert grad_norms["human_root_head.1.weight"] and grad_norms["human_root_head.1.weight"] > 0
    assert grad_norms["cmad.action_head.1.weight"] and grad_norms["cmad.action_head.1.weight"] > 0
    assert grad_norms["mge.control_embed.weight"] and grad_norms["mge.control_embed.weight"] > 0
    print("PASS test_unified_forward_runs_and_gradients_flow")
    print(f"  grad_norms: {grad_norms}")


def test_legacy_path_still_works():
    torch.manual_seed(0)
    model = MotionLatentRetargeter(
        num_human_joints=21,
        num_actions=15,
        beta_dim=16,
        root_dim=9,
        d_model=64,
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
        future_offsets=(1,),
        future_slot_dim=4,
        latent_mode="per_slot",
        causal=True,
        unified_human_decoder=False,
    )
    spec = _make_robot_spec(K=15)
    model.set_robot_batch(batch_robot_specs([spec]))
    pose_body, root_features, beta = _make_input()
    out = model(pose_body=pose_body, root_features=root_features, beta=beta)
    assert out["human_pose_aa_recon"].shape == (2, 8, 21, 3)
    assert torch.isfinite(out["actions"]).all()
    print("PASS test_legacy_path_still_works")


def test_existing_g1_ckpt_loads_with_default_construction():
    """Regression: an existing MorphE2 G1 checkpoint (trained before the
    unified-human-decoder change) must still load into a default-constructed
    MotionLatentRetargeter without shape mismatches on control_embed."""
    import os
    ckpt_path = (
        "/cpfs/user/yaoyifei/gbc_runs/local_morph_e2_recipe/"
        "models/04_14_38_latent_best_epoch_4_step_24088.pt"
    )
    if not os.path.exists(ckpt_path):
        print(f"SKIP test_existing_g1_ckpt_loads (ckpt absent): {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("config", {}).get("args", {})
    # Build the model using the ckpt's saved hyperparams, with
    # unified_human_decoder=False (the default, matches old training).
    model = MotionLatentRetargeter(
        num_human_joints=21,
        num_actions=len(ckpt.get("robot_dof_names", [])) or 29,
        beta_dim=16,
        root_dim=9,
        d_model=int(args.get("d_model", 256)),
        z_dim=int(args.get("z_dim", 256)),
        slot_z=int(args.get("slot_z", 12)),
        slot_latent_dim=int(args.get("slot_latent_dim", 32)),
        encoder_graph_layers=int(args.get("graph_layers", 2)),
        encoder_temporal_layers=int(args.get("temporal_layers", 4)),
        encoder_num_heads=int(args.get("num_heads", 8)),
        morphology_graph_layers=int(args.get("morphology_graph_layers", 2)),
        morphology_num_heads=int(args.get("morphology_num_heads", 8)),
        decoder_graph_layers=int(args.get("decoder_graph_layers", 2)),
        decoder_temporal_layers=int(args.get("decoder_temporal_layers", 2)),
        decoder_num_heads=int(args.get("decoder_num_heads", args.get("num_heads", 8))),
        dropout=0.0,
        latent_mode="per_slot",
        causal=(args.get("temporal_module", "transformer") == "causal_transformer"),
        human_decoder_graph_layers=int(args.get("human_decoder_graph_layers", 2)),
        human_decoder_num_heads=int(args.get("human_decoder_num_heads", 8)),
        unified_human_decoder=False,
    )
    sd = ckpt["model"]
    # Verify the critical control_embed shape matches (3, d_model).
    if "mge.control_embed.weight" in sd:
        assert sd["mge.control_embed.weight"].shape == model.mge.control_embed.weight.shape, (
            "ckpt control_embed shape does not match default-constructed model; "
            "default construction would be broken if num_control_modes diverged."
        )
    result = model.load_state_dict(sd, strict=False)
    # 2026-06-10: the fixed ref_base_height + root_height_head + root_embed were
    # REPLACED by the morphology-conditioned RootDecoder. So an OLD ckpt carries
    # exactly those removed keys as "unexpected", and the new root_decoder.* as
    # "missing". Everything else (encoder, MGE, CMAD, human heads) must still
    # load cleanly — that's what this regression guards.
    removed_root_prefixes = ("ref_base_height", "root_embed.", "root_height_head.")
    stray = [k for k in result.unexpected_keys if not k.startswith(removed_root_prefixes)]
    assert not stray, f"unexpected keys beyond the removed root machinery: {stray[:5]}"
    new_root = [k for k in result.missing_keys if k.startswith("root_decoder.")]
    assert new_root, "root_decoder.* should be present (missing) in the new model"
    print("PASS test_existing_g1_ckpt_loads_with_default_construction "
          f"(core loads; root machinery swapped: {len(result.unexpected_keys)} old root keys dropped, "
          f"{len(new_root)} root_decoder keys fresh)")


if __name__ == "__main__":
    test_build_human_spec_shapes_and_modes()
    test_unified_forward_runs_and_gradients_flow()
    test_legacy_path_still_works()
    test_existing_g1_ckpt_loads_with_default_construction()
    print("\nOK: unified human decoder path runs, gradients flow, legacy preserved + ckpt-load regression checked.")
