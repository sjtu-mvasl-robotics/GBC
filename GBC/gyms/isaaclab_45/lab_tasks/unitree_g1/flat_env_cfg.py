# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeG1RoughEnvCfg, UnitreeG1RoughRefEnvCfg


@configclass
class UnitreeG1FlatEnvCfg(UnitreeG1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 60.0

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.6

class UnitreeG1FlatEnvCfg_PLAY(UnitreeG1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 60.0

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

from GBC.gyms.isaaclab_45.envs import ManagerBasedRefRLEnvCfg
@configclass
class UnitreeG1FlatRefEnvCfg(UnitreeG1RoughRefEnvCfg):
    def __post_init__(self):
        # post init of parent
        self.episode_length_s = 60.0
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

@configclass
class UnitreeG1FlatRefEnvCfg_PLAY(UnitreeG1FlatRefEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 60.0
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.ref_observation.policy.enable_corruption = False

# @configclass
# class UnitreeG1FlatRefOrigRewardsEnvCfg(UnitreeG1RoughRefOrigRewardsEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()

#         # change terrain to flat
#         self.scene.terrain.terrain_type = "plane"
#         self.scene.terrain.terrain_generator = None
#         # no height scan
#         self.scene.height_scanner = None
#         self.observations.policy.height_scan = None
#         # no terrain curriculum
#         self.curriculum.terrain_levels = None
#         if self.rewards.feet_air_time:
#             self.rewards.feet_air_time.weight = 1.0
#             self.rewards.feet_air_time.params["threshold"] = 0.6

