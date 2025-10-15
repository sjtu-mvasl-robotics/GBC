# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, dagger_env_cfg, rough_env_cfg

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Velocity-Rough-G1-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.UnitreeG1RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-G1-Reference-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.UnitreeG1RoughRefEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:UnitreeG1RoughRefPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-Reference-Play-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.UnitreeG1RoughRefEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:UnitreeG1RoughRefPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-Reference-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.UnitreeG1FlatRefEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:UnitreeG1FlatRefPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-Reference-Play-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.UnitreeG1FlatRefEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:UnitreeG1FlatRefPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

# gym.register(
#     id="Isaac-Velocity-Flat-UnitreeG1-Reference-No-DAgger-v0",
#     entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": flat_env_cfg.UnitreeG1FlatRefEnvCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:UnitreeG1FlatRefNoDAggerPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )

gym.register(
    id="Isaac-Velocity-Rough-G1-Play-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.UnitreeG1RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)




gym.register(
    id="Isaac-Velocity-Flat-G1-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.UnitreeG1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


# gym.register(
#     id="Isaac-Velocity-Flat-UnitreeG1-Ref-Orig-Rewards-v0",
#     entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": flat_env_cfg.UnitreeG1FlatRefOrigRewardsEnvCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:UnitreeG1FlatOrigRewardsPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
#     },
# )


gym.register(
    id="Isaac-Velocity-Flat-G1-Play-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.UnitreeG1FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-DAgger-G1-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dagger_env_cfg.UnitreeG1DAggerEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:UnitreeG1TrainDAggerPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

# gym.register(
#     id="Isaac-Velocity-Dagger-UnitreeG1-Play-v0",
#     entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": dagger_env_cfg.UnitreeG1DaggerRefEnvCfg_PLAY,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:UnitreeG1TrainDAggerPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Velocity-Flat-UnitreeG1-Reference-Feb-v0",
#     entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": rough_env_feb_cfg.UnitreeG1FlatRefEnvCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:UnitreeG1FlatFebRefPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )