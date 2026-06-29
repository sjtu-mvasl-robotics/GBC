"""Tests for RobotMotionEncoder — the robot-side encoder that maps robot
proprioceptive motion (dof_pos, dof_vel, root) into the SAME per-slot latent
space as MFME, enabling encoder-as-reward (recipe 3) and world-model (recipe 4).

Design (SONIC universal-token pattern):
  * Separate per-DoF input tokenizer + graph-temporal encoder on the robot side.
  * SHARED per-slot mu/logvar heads with MFME — this is what forces z_robot and
    z_human into the same latent statistics so ||z_human - z_robot||^2 is a
    meaningful reward.
"""
from __future__ import annotations

import unittest

import torch

from GBC.utils.data_preparation.graph_retarget_latent import (
    DEFAULT_BODY_EDGES,
    DEFAULT_BODY_SLOTS,
)
from GBC.utils.data_preparation.morphology_conditioned_retargeter import (
    MorphologyFreeMotionEncoder,
    MorphologyGraphEncoder,
    RobotMotionEncoder,
    build_synthetic_robot_spec,
    batch_robot_specs,
)


SLOTS = tuple(DEFAULT_BODY_SLOTS) + ("root_state",)   # 7, mirrors MFME per_slot
NUM_SLOTS = len(SLOTS)
SLOT_Z = 12
D_MODEL = 48
ROOT_DIM = 9
BETA_DIM = 16


def _g1_like_spec(num_dofs=29, seed=0):
    return build_synthetic_robot_spec(
        name=f"g1_{num_dofs}",
        dof_names=tuple(f"j_{i}" for i in range(num_dofs)),
        parent_index=[-1] + list(range(num_dofs - 1)),
        seed=seed,
    )


def _make_encoder(causal=True):
    return RobotMotionEncoder(
        d_model=D_MODEL,
        slot_z=SLOT_Z,
        root_dim=ROOT_DIM,
        slots=SLOTS,
        edges=DEFAULT_BODY_EDGES,
        graph_layers=1,
        temporal_layers=1,
        num_heads=4,
        causal=causal,
        dropout=0.0,
    )


def _mge(num_control_modes=3):
    return MorphologyGraphEncoder(
        feature_dim=21, d_model=D_MODEL, num_heads=4,
        num_graph_layers=1, num_slots=len(DEFAULT_BODY_SLOTS),
        dropout=0.0, num_control_modes=num_control_modes,
    )


class TestRobotMotionEncoderShapes(unittest.TestCase):
    def test_output_z_slots_matches_mfme_layout(self):
        torch.manual_seed(0)
        enc = _make_encoder().eval()
        mge = _mge().eval()
        spec = _g1_like_spec(29)
        rb = batch_robot_specs([spec])
        B, T, K = 2, 8, 29
        rb = {k: (v.expand(B, *v.shape[1:]).contiguous() if torch.is_tensor(v) else v)
              for k, v in rb.items()}
        m_per_dof, _ = mge(rb["node_features"], rb["slot_assignment"],
                           rb["hop_matrix"], rb["control_mode"], rb.get("dof_padding_mask"))
        dof_pos = torch.randn(B, T, K)
        dof_vel = torch.randn(B, T, K)
        root = torch.randn(B, T, ROOT_DIM)
        with torch.inference_mode():
            out = enc(dof_pos, dof_vel, root, m_per_dof,
                      rb["slot_assignment"], dof_padding_mask=rb.get("dof_padding_mask"),
                      sample=False)
        self.assertEqual(tuple(out["z_slots"].shape), (B, T, NUM_SLOTS, SLOT_Z))
        self.assertEqual(tuple(out["mu_slots"].shape), (B, T, NUM_SLOTS, SLOT_Z))
        self.assertEqual(tuple(out["logvar_slots"].shape), (B, T, NUM_SLOTS, SLOT_Z))
        self.assertTrue(torch.isfinite(out["z_slots"]).all())


class TestRobotMotionEncoderOwnHeads(unittest.TestCase):
    def test_robot_heads_are_separate_from_mfme(self):
        torch.manual_seed(0)
        mfme = MorphologyFreeMotionEncoder(
            d_model=D_MODEL, slot_z=SLOT_Z, slot_latent_dim=16,
            graph_layers=1, temporal_layers=1, num_heads=4,
            latent_mode="per_slot", causal=True,
        )
        enc = _make_encoder()
        # The robot encoder MUST own its bottleneck heads. Sharing them with
        # MFME would let the alignment loss drift the stop-grad human anchor
        # (BYOL/JEPA pitfall). Same shape, different objects.
        self.assertIsNot(enc.mu_head_slot, mfme.mu_head_slot)
        self.assertIsNot(enc.logvar_head_slot, mfme.logvar_head_slot)
        self.assertEqual(enc.mu_head_slot[-1].weight.shape,
                         mfme.mu_head_slot[-1].weight.shape)


class TestRobotMotionEncoderVariableK(unittest.TestCase):
    def test_two_robots_different_K_same_latent_shape(self):
        torch.manual_seed(0)
        enc = _make_encoder().eval()
        mge = _mge().eval()
        B, T = 1, 6
        for K in (29, 27, 17):
            spec = _g1_like_spec(K, seed=K)
            rb = batch_robot_specs([spec])
            m_per_dof, _ = mge(rb["node_features"], rb["slot_assignment"],
                               rb["hop_matrix"], rb["control_mode"], rb.get("dof_padding_mask"))
            dof_pos = torch.randn(B, T, K)
            dof_vel = torch.randn(B, T, K)
            root = torch.randn(B, T, ROOT_DIM)
            with torch.inference_mode():
                out = enc(dof_pos, dof_vel, root, m_per_dof,
                          rb["slot_assignment"], dof_padding_mask=rb.get("dof_padding_mask"),
                          sample=False)
            self.assertEqual(tuple(out["z_slots"].shape), (B, T, NUM_SLOTS, SLOT_Z),
                             f"K={K} produced wrong latent shape")


class TestRobotMotionEncoderPaddingInvariance(unittest.TestCase):
    def test_padded_fake_dofs_do_not_change_real_latent(self):
        torch.manual_seed(0)
        enc = _make_encoder().eval()
        mge = _mge().eval()
        B, T, K = 1, 6, 23
        spec_real = _g1_like_spec(K, seed=3)
        spec_pad = _g1_like_spec(K + 9, seed=4)
        rb = batch_robot_specs([spec_real, spec_pad])  # pads spec_real to K+9
        # Use the FIRST sample (real K=23 padded to K_max=32).
        rb0 = {k: (v[:1] if torch.is_tensor(v) else v) for k, v in rb.items()}
        K_max = rb0["node_features"].shape[1]
        m_per_dof, _ = mge(rb0["node_features"], rb0["slot_assignment"],
                           rb0["hop_matrix"], rb0["control_mode"], rb0.get("dof_padding_mask"))
        dof_pos = torch.randn(B, T, K_max)
        dof_vel = torch.randn(B, T, K_max)
        root = torch.randn(B, T, ROOT_DIM)
        with torch.inference_mode():
            out_a = enc(dof_pos, dof_vel, root, m_per_dof,
                        rb0["slot_assignment"], dof_padding_mask=rb0["dof_padding_mask"],
                        sample=False)
            # Perturb ONLY the padded DoF channels; real-DoF latent must not move.
            pad = rb0["dof_padding_mask"][0]  # (K_max,)
            dof_pos_p = dof_pos.clone(); dof_vel_p = dof_vel.clone()
            dof_pos_p[:, :, pad] += torch.randn_like(dof_pos_p[:, :, pad]) * 10.0
            dof_vel_p[:, :, pad] += torch.randn_like(dof_vel_p[:, :, pad]) * 10.0
            out_b = enc(dof_pos_p, dof_vel_p, root, m_per_dof,
                        rb0["slot_assignment"], dof_padding_mask=rb0["dof_padding_mask"],
                        sample=False)
        diff = (out_a["mu_slots"] - out_b["mu_slots"]).abs().max().item()
        self.assertLess(diff, 1e-5,
                        f"padded DoFs leaked into the real-DoF latent: max|Δ|={diff:.2e}")


class TestRobotMotionEncoderGradFlow(unittest.TestCase):
    def test_grad_reaches_encoder_body_and_own_heads(self):
        torch.manual_seed(0)
        enc = _make_encoder().train()
        mge = _mge().train()
        spec = _g1_like_spec(29)
        B, T, K = 2, 8, 29
        rb = batch_robot_specs([spec])
        rb = {k: (v.expand(B, *v.shape[1:]).contiguous() if torch.is_tensor(v) else v)
              for k, v in rb.items()}
        m_per_dof, _ = mge(rb["node_features"], rb["slot_assignment"],
                           rb["hop_matrix"], rb["control_mode"], rb.get("dof_padding_mask"))
        out = enc(torch.randn(B, T, K), torch.randn(B, T, K), torch.randn(B, T, ROOT_DIM),
                  m_per_dof, rb["slot_assignment"], dof_padding_mask=rb.get("dof_padding_mask"),
                  sample=True)
        target = torch.randn(B, T, NUM_SLOTS, SLOT_Z)
        loss = (out["mu_slots"] - target).pow(2).mean()
        loss.backward()
        body_grad = [p for n, p in enc.named_parameters()
                     if not n.startswith(("mu_head_slot", "logvar_head_slot"))
                     and p.grad is not None and p.grad.abs().sum() > 0]
        self.assertGreater(len(body_grad), 0, "no grad reached the robot encoder body")
        head_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in enc.mu_head_slot.parameters())
        self.assertTrue(head_grad, "no grad reached the robot's own mu head")


class TestRobotMotionEncoderAlignmentFit(unittest.TestCase):
    def test_alignment_loss_decreases(self):
        torch.manual_seed(0)
        enc = _make_encoder().train()
        mge = _mge().eval()
        spec = _g1_like_spec(29)
        B, T, K = 2, 10, 29
        rb = batch_robot_specs([spec])
        rb = {k: (v.expand(B, *v.shape[1:]).contiguous() if torch.is_tensor(v) else v)
              for k, v in rb.items()}
        with torch.no_grad():
            m_per_dof, _ = mge(rb["node_features"], rb["slot_assignment"],
                               rb["hop_matrix"], rb["control_mode"], rb.get("dof_padding_mask"))
        dof_pos = torch.randn(B, T, K)
        dof_vel = torch.randn(B, T, K)
        root = torch.randn(B, T, ROOT_DIM)
        target = torch.randn(B, T, NUM_SLOTS, SLOT_Z)   # fixed z_human target
        opt = torch.optim.Adam(enc.parameters(), lr=1e-2)

        def step():
            out = enc(dof_pos, dof_vel, root, m_per_dof,
                      rb["slot_assignment"], dof_padding_mask=rb.get("dof_padding_mask"),
                      sample=False)
            return (out["mu_slots"] - target).pow(2).mean()

        first = float(step().item())
        for _ in range(80):
            opt.zero_grad(); l = step(); l.backward(); opt.step()
        last = float(step().item())
        self.assertLess(last, first * 0.5,
                        f"alignment loss did not halve: {first:.4f} -> {last:.4f}")


class TestMotionLatentRetargeterRobotEncoder(unittest.TestCase):
    """The top-level model exposes encode_robot() when the flag is set, sharing
    MFME's bottleneck heads, and produces a z_robot aligned with z_human."""

    def _model(self, robot_encoder_enabled):
        from GBC.utils.data_preparation.morphology_conditioned_retargeter import (
            MotionLatentRetargeter,
        )
        return MotionLatentRetargeter(
            num_actions=29, beta_dim=BETA_DIM, root_dim=ROOT_DIM,
            d_model=D_MODEL, slot_z=SLOT_Z, slot_latent_dim=16,
            encoder_graph_layers=1, encoder_temporal_layers=1, encoder_num_heads=4,
            morphology_graph_layers=1, morphology_num_heads=4,
            decoder_graph_layers=1, decoder_temporal_layers=1, decoder_num_heads=4,
            latent_mode="per_slot", causal=True,
            robot_encoder_enabled=robot_encoder_enabled,
        )

    def test_disabled_by_default(self):
        m = self._model(robot_encoder_enabled=False)
        self.assertIsNone(getattr(m, "robot_encoder", None))

    def test_encode_robot_uses_own_heads_and_shapes(self):
        torch.manual_seed(0)
        m = self._model(robot_encoder_enabled=True).eval()
        self.assertIsNotNone(m.robot_encoder)
        # Own heads, not MFME's — the alignment loss is the only tie.
        self.assertIsNot(m.robot_encoder.mu_head_slot, m.mfme.mu_head_slot)
        spec = _g1_like_spec(29)
        rb = batch_robot_specs([spec])
        m.set_robot_batch(rb)
        B, T, K = 2, 8, 29
        out = m.encode_robot(
            dof_pos=torch.randn(B, T, K),
            dof_vel=torch.randn(B, T, K),
            root_features=torch.randn(B, T, ROOT_DIM),
            sample=False,
        )
        S = len(m.mfme.slots)
        self.assertEqual(tuple(out["mu_slots"].shape), (B, T, S, SLOT_Z))
        self.assertEqual(tuple(out["z_slots"].shape), (B, T, S, SLOT_Z))


class TestAlignmentAnchorIntegrity(unittest.TestCase):
    """The alignment loss ‖z_human.detach() − z_robot‖² must NOT update any
    parameter on the human-latent path (MFME tokenizer / encoder / mu-logvar
    heads). If it does, the 'stop-gradient anchor' is a lie — the robot
    objective drifts the very target it claims to chase (BYOL/JEPA pitfall:
    a stop-grad target cannot share the projection head with the predictor).
    """

    def _model(self):
        from GBC.utils.data_preparation.morphology_conditioned_retargeter import (
            MotionLatentRetargeter,
        )
        return MotionLatentRetargeter(
            num_actions=29, beta_dim=BETA_DIM, root_dim=ROOT_DIM,
            d_model=D_MODEL, slot_z=SLOT_Z, slot_latent_dim=16,
            encoder_graph_layers=1, encoder_temporal_layers=1, encoder_num_heads=4,
            morphology_graph_layers=1, morphology_num_heads=4,
            decoder_graph_layers=1, decoder_temporal_layers=1, decoder_num_heads=4,
            latent_mode="per_slot", causal=True,
            robot_encoder_enabled=True,
        )

    def test_alignment_grad_does_not_touch_human_latent_path(self):
        torch.manual_seed(0)
        m = self._model().train()
        spec = _g1_like_spec(29)
        m.set_robot_batch(batch_robot_specs([spec]))
        B, T, K = 2, 8, 29
        pose_body = torch.randn(B, T, m.num_joints * 3)
        root = torch.randn(B, T, ROOT_DIM)
        beta = torch.randn(B, BETA_DIM)

        # Human forward (the anchor) — its latent must be shaped only by the
        # retargeting/AE losses, never by alignment.
        out = m(pose_body=pose_body, root_features=root, beta=beta, sample=False)
        robot_enc = m.encode_robot(
            dof_pos=torch.randn(B, T, K), dof_vel=torch.randn(B, T, K),
            root_features=root, sample=False,
        )
        S = len(m.mfme.slots)
        mu_h = out["mu"].view(B, T, S, SLOT_Z).detach()   # stop-grad anchor
        align = (mu_h - robot_enc["mu_slots"]).pow(2).mean()
        align.backward()

        leaked = [n for n, p in m.mfme.named_parameters()
                  if p.grad is not None and p.grad.abs().sum() > 0]
        self.assertEqual(
            leaked, [],
            f"alignment loss leaked gradient into the human-latent path (MFME): {leaked}",
        )
        # Sanity: the robot encoder body DID receive gradient.
        got = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in m.robot_encoder.parameters())
        self.assertTrue(got, "alignment produced no gradient on the robot encoder")


if __name__ == "__main__":
    unittest.main(verbosity=2)
