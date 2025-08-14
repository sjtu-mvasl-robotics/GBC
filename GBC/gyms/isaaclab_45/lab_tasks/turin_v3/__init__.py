# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg, dagger_env_cfg

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Velocity-Rough-TurinV3-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.TurinV3RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TurinV3RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-TurinV3-Reference-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.TurinV3RoughRefEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:TurinV3RoughRefPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-TurinV3-Reference-Play-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.TurinV3RoughRefEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:TurinV3RoughRefPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-TurinV3-Reference-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TurinV3FlatRefEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:TurinV3FlatRefPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-TurinV3-Reference-Play-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TurinV3FlatRefEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:TurinV3FlatRefPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

# gym.register(
#     id="Isaac-Velocity-Flat-TurinV3-Reference-No-DAgger-v0",
#     entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": flat_env_cfg.TurinV3FlatRefEnvCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:TurinV3FlatRefNoDAggerPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )

gym.register(
    id="Isaac-Velocity-Rough-TurinV3-Play-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.TurinV3RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TurinV3RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)




gym.register(
    id="Isaac-Velocity-Flat-TurinV3-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TurinV3FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TurinV3FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


# gym.register(
#     id="Isaac-Velocity-Flat-TurinV3-Ref-Orig-Rewards-v0",
#     entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": flat_env_cfg.TurinV3FlatRefOrigRewardsEnvCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:TurinV3FlatOrigRewardsPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
#     },
# )


gym.register(
    id="Isaac-Velocity-Flat-TurinV3-Play-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TurinV3FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TurinV3FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-DAgger-TurinV3-v0",
    entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dagger_env_cfg.TurinV3DAggerEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:TurinV3TrainDAggerPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

# gym.register(
#     id="Isaac-Velocity-Dagger-TurinV3-Play-v0",
#     entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": dagger_env_cfg.TurinV3DaggerRefEnvCfg_PLAY,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:TurinV3TrainDAggerPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Velocity-Flat-TurinV3-Reference-Feb-v0",
#     entry_point="GBC.gyms.isaaclab_45.envs:ManagerBasedRefRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": rough_env_feb_cfg.TurinV3FlatRefEnvCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ref_ppo_cfg:TurinV3FlatFebRefPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )