# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from typing import Literal

from isaaclab.utils import configclass

from GBC.gyms.isaaclab_45.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlRefPpoActorCriticCfg,
    RslRlRefPpoAlgorithmCfg,
    RslRlRefOnPolicyRunnerCfg,
    RslRlPpoAmpCfg,
    RslRlRefPpoAmpNetCfg,
)

#  rnd_cfg:
#       weight: 0.0  # initial weight of the RND reward

#       # note: This is a dictionary with a required key called "mode".
#       #   Please check the RND module for more information.
#       weight_schedule: null

#       reward_normalization: false  # whether to normalize RND reward
#       state_normalization: true  # whether to normalize RND state observations

#       # -- Learning parameters
#       learning_rate: 0.001  # learning rate for RND

#       # -- Network parameters
#       # note: if -1, then the network will use dimensions of the observation
#       num_outputs: 1  # number of outputs of RND network
#       predictor_hidden_dims: [-1] # hidden dimensions of predictor network
#       target_hidden_dims: [-1]  # hidden dimensions of target network


# rnd_cfg = {
#     "weight": 0.05,
#     "weight_schedule": None,
#     "reward_normalization": False,
#     "state_normalization": True,
#     "learning_rate": 0.001,
#     "num_outputs": 1,
#     "predictor_hidden_dims": [-1],
#     "target_hidden_dims": [-1],
# }
import torch
from GBC.gyms.isaaclab_45.lab_tasks.unitree_g1.rough_env_cfg import get_actor_observation_symmetry, get_amp_ref_observations, get_amp_observations, GLOBAL_HISTORY_LENGTH, flipper
from GBC.gyms.isaaclab_45.lab_tasks.mdp import get_ref_observation_symmetry, actions_symmetry
import gym
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
    "mirror_loss_coeff": 0.1,
}

amp_net_cfg = RslRlRefPpoAmpNetCfg(
    backbone_input_dim=146,
    backbone_output_dim=128,
    backbone="mlp",
    activation="relu",
    out_activation="sigmoid",
    net_kwargs={
        "hidden_dims": [512, 256],
    }
)

amp_cfg = RslRlPpoAmpCfg(
    net_cfg=amp_net_cfg,
    learning_rate=5e-4,
    amp_obs_extractor=get_amp_observations,
    amp_ref_obs_extractor=get_amp_ref_observations,
    amp_reward_scale=0.1,
    epsilon=1e-4,
    gradient_penalty_coeff=10.0,
    amp_update_interval=20,
    amp_pretrain_steps=6000,
)

@configclass
class UnitreeG1RoughRefPPORunnerCfg(RslRlRefOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 32 # 32
    max_iterations = 15000
    save_interval = 100
    experiment_name = "UnitreeG1_rough"
    empirical_normalization = True
    policy = RslRlRefPpoActorCriticCfg(
        # class_name="ActorCriticDebugMLP",
        class_name="ActorCriticMMGPT",
        max_len=8,
        dim_model=64, # 96
        num_layers=2, # 4
        num_heads=4,
        max_seq_len=12, # compress ratio: 3
        num_steps_per_env=32, 
        mlp_hidden_dims=[256, 128],  
        init_noise_std=1.0,
        load_dagger=False,
        apply_rope=True,
        # load_dagger_path="/home/yifei/codespace/GBC_45/logs/rsl_rl/h1_dagger_V2/2025-06-02_10-42-19_h1_imitation/model_8100.pt",
        apply_mlp_residual=False,
        history_length=GLOBAL_HISTORY_LENGTH,
        concatenate_term_names={
            "policy":[["lft_sin_phase", "lft_cos_phase", "rht_sin_phase", "rht_cos_phase"], ["base_ang_vel", "projected_gravity", "velocity_commands", "imu_orientation"]],
            "critic":[["lft_sin_phase", "lft_cos_phase", "rht_sin_phase", "rht_cos_phase"],["base_ang_vel", "projected_gravity", "velocity_commands", "imu_orientation"]],
        },
        concatenate_ref_term_names={
            "policy":[["lft_sin_phase", "lft_cos_phase", "rht_sin_phase", "rht_cos_phase"], ["target_actions", "target_actions_t1", "target_actions_t2", "target_actions_t3"], ["target_quaternion","target_quaternion_t1", "target_quaternion_t2", "target_quaternion_t3"]],
            "critic":[["lft_sin_phase", "lft_cos_phase", "rht_sin_phase", "rht_cos_phase"],["target_actions", "target_actions_t1", "target_actions_t2", "target_actions_t3"], ["target_quaternion","target_quaternion_t1", "target_quaternion_t2", "target_quaternion_t3"]],
        },
        # load_actor_path="/home/yifei/codespace/GBC_45/logs/rsl_rl/UnitreeG1_flat/2025-10-02_19-47-03_UnitreeG1_imitation/model_5500.pt"
    )
    algorithm = RslRlRefPpoAlgorithmCfg(
        class_name="MMPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.4,
        entropy_coef=1.0e-2, # 1e-2
        num_learning_epochs=4, # 4
        num_mini_batches=4, # 4
        learning_rate=1.0e-4,
        schedule="cosine", # adaptive
        normalize_advantage_per_mini_batch=True,
        gamma=0.99,
        lam=0.95, # 0.95
        desired_kl=0.01, # 0.01
        max_lr=3.0e-3,
        min_lr=1.0e-4,
        max_grad_norm=1.0, # 1.0
        # teacher_coef=0.995, #0.9,
        teacher_coef_range=(0.2, 0.8),
        teacher_coef_decay=0.8,
        teacher_coef_decay_interval=100,
        # teacher_loss_coef=0.0001, #0.3,
        teacher_loss_coef_range=(0.0001, 0.0025),
        teacher_loss_coef_decay=0.9995,
        teacher_loss_coef_decay_interval=100,
        teacher_lr=5e-4,
        teacher_update_interval=10,
        teacher_only_interval=0,
        teacher_supervising_intervals=40000,
        teacher_updating_intervals=24000,
        teacher_coef_mode="original_kl", # "kl" or "norm"
        rnd_cfg=None,
        # symmetry_cfg=symmetry_cfg,
        # amp_cfg=amp_cfg,
    )
    run_name: str = "UnitreeG1_imitation"
    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    resume: bool = False


@configclass
class UnitreeG1FlatRefPPORunnerCfg(UnitreeG1RoughRefPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 150000
        self.experiment_name = "UnitreeG1_flat"
        # self.policy.actor_hidden_dims = [128, 128, 128]
        # self.policy.critic_hidden_dims = [128, 128, 128]


@configclass
class UnitreeG1FlatRefNoDAggerPPORunnerCfg(UnitreeG1RoughRefPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 150000
        self.experiment_name = "UnitreeG1_flat_no_dagger"
        self.policy.load_dagger = False
        self.algorithm.teacher_coef = None
        self.algorithm.teacher_loss_coef = None
        self.add_experiment_name()

    def add_experiment_name(self):
        if "MLP" in self.policy.class_name:
            self.experiment_name += "_MLP"
        elif "V2" in self.policy.class_name:
            self.experiment_name += "_V2"


@configclass
class UnitreeG1FlatOrigRewardsPPORunnerCfg(UnitreeG1FlatRefNoDAggerPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "UnitreeG1_flat_orig_rewards"
        self.add_experiment_name()


@configclass
class UnitreeG1FlatFebRefPPORunnerCfg(UnitreeG1RoughRefPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 50000
        self.experiment_name = "UnitreeG1_flat_feb"
        # self.policy.actor_hidden_dims = [128, 128, 128]
        # self.policy.critic_hidden_dims = [128, 128, 128]

        
@configclass
class UnitreeG1TrainDAggerPPORunnerCfg(UnitreeG1FlatRefNoDAggerPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 50000
        self.experiment_name = "UnitreeG1_dagger"
        # self.algorithm.amp_cfg.amp_update_interval = 50
        # self.algorithm.amp_cfg.amp_pretrain_steps = 2000
        self.add_experiment_name()