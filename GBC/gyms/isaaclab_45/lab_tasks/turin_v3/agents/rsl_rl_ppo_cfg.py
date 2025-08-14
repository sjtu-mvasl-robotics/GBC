# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from GBC.gyms.isaaclab_45.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlRefPpoActorCriticCfg,
    RslRlRefPpoAlgorithmCfg,
    RslRlRefOnPolicyRunnerCfg,
    RslRlPpoAmpCfg,
    RslRlRefPpoAmpNetCfg,
)
from typing import Literal

from GBC.gyms.isaaclab_45.lab_tasks.turin_v3.rough_env_cfg import get_actor_observation_symmetry, get_amp_ref_observations, get_amp_observations, GLOBAL_HISTORY_LENGTH, flipper
from GBC.gyms.isaaclab_45.lab_tasks.mdp import get_ref_observation_symmetry, actions_symmetry

import gym
import torch

def data_augmentation_func(
        obs: torch.Tensor | None = None,
        ref_obs: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        env: gym.Env | None = None,
        obs_type: Literal["policy", "state"] = "policy",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert obs_type == "policy", "Only policy mode is supported for now"
    sym_obs = None
    sym_ref_obs = None
    sym_actions = None
    if obs is not None:
        sym_obs = get_actor_observation_symmetry(env, obs)
    if ref_obs is not None:
        sym_ref_obs = get_ref_observation_symmetry(env, ref_obs)
    if actions is not None:
        sym_actions = actions_symmetry(env, actions, flipper=flipper)
    return sym_obs, sym_ref_obs, sym_actions

symmetry_cfg = {
    "data_augmentation_func": data_augmentation_func,
    "use_data_augmentation": False,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 1.0,
}

@configclass
class TurinV3RoughPPORunnerCfg(RslRlRefOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "turinv3_rough"
    empirical_normalization = False
    policy = RslRlRefPpoActorCriticCfg(
        # class_name="ActorCriticDebugMLP",
        class_name="ActorCriticMMTransformerV2",
        max_len=8,
        dim_model=256, # 96
        num_layers=2, # 4
        num_heads=8,
        init_noise_std=1.0,
        load_dagger=False,
        # load_dagger_path="/home/yifei/codespace/GBC_45/logs/rsl_rl/h1_dagger_V2/2025-06-02_10-42-19_h1_imitation/model_8100.pt",
        apply_mlp_residual=False,
        history_length=GLOBAL_HISTORY_LENGTH,
        concatenate_term_names={
            "policy":[["lft_sin_phase", "lft_cos_phase", "rht_sin_phase", "rht_cos_phase"], ["base_ang_vel", "projected_gravity"]],
            "critic":[["lft_sin_phase", "lft_cos_phase", "rht_sin_phase", "rht_cos_phase"], ["base_ang_vel", "projected_gravity"]],
        },
        concatenate_ref_term_names={
            "policy":[],
            "critic":[],
        },
        # load_actor_path="/home/yifei/dataset/save_models/model_stand_v1.pt"
    )
    algorithm = RslRlRefPpoAlgorithmCfg(
        class_name="MMPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.4,
        entropy_coef=1e-2,
        num_learning_epochs=4, # 4
        num_mini_batches=8, # 4
        learning_rate=1.0e-4,
        schedule="adaptive", # adaptive
        normalize_advantage_per_mini_batch=True,
        gamma=0.99,
        lam=0.95,
        desired_kl=0.05, # 0.01
        max_grad_norm=0.5, # 1.0
        # teacher_coef=0.995, #0.9,
        # teacher_coef_range=(0.2, 0.8),
        # teacher_coef_decay=0.8,
        # teacher_coef_decay_interval=100,
        # teacher_loss_coef=0.0001, #0.3,
        # teacher_loss_coef_range=(0.0001, 0.0025),
        # teacher_loss_coef_decay=0.9995,
        # teacher_loss_coef_decay_interval=100,
        # teacher_lr=5e-4,
        # teacher_update_interval=10,
        # teacher_only_interval=0,
        # teacher_supervising_intervals=40000,
        # teacher_updating_intervals=24000,
        # teacher_coef_mode="original_kl", # "kl" or "norm"
        rnd_cfg=None,
        symmetry_cfg=symmetry_cfg,
        amp_cfg=None,
    )
    run_name: str = "turinv3_rough"
    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    resume: bool = False


@configclass
class TurinV3FlatPPORunnerCfg(TurinV3RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "turinv3_flat"
        # self.policy.actor_hidden_dims = [128, 128, 128]
        # self.policy.critic_hidden_dims = [128, 128, 128]
