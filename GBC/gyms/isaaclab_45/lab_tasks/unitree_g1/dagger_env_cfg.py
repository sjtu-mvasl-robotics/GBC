# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from GBC.gyms.isaaclab_45.lab_assets.unitree import G1_29DOF_CFG
from GBC.gyms.isaaclab_45.managers.ref_obs_term_cfg import ReferenceObservationTermCfg as RefObsTerm
from GBC.gyms.isaaclab_45.managers.ref_obs_term_cfg import ReferenceObservationCfg as RefObsCfg
from GBC.gyms.isaaclab_45.managers.ref_obs_term_cfg import ReferenceObservationGroupCfg as RefObsGroup
from GBC.utils.base import PROJECT_ROOT_DIR, DATA_PATHS
from ..mdp import *
from .rough_env_cfg import UnitreeG1RoughRefEnvCfg, UnitreeG1RoughRefEnvCfg_PLAY, UnitreeG1RefObservationCfg, UnitreeG1ObservationsCfg, GLOBAL_HISTORY_LENGTH, UnitreeG1EventCfg, UnitreeG1Rewards

@configclass
class UnitreeG1DAggerObservationsCfg(UnitreeG1ObservationsCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # replace observations with the ones for DAgger
        self.policy.base_ang_vel = ObsTerm(func=reference_base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        self.policy.projected_gravity = ObsTerm(
            func=reference_projected_gravity,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        self.critic.base_ang_vel = ObsTerm(func=reference_base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        self.critic.projected_gravity = ObsTerm(
            func=reference_projected_gravity,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        self.critic.base_lin_vel = ObsTerm(func=reference_base_lin_vel, noise=Unoise(n_min=-0.2, n_max=0.2))


@configclass
class UnitreeG1DAggerRewards(UnitreeG1Rewards):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # replace rewards with the ones for DAgger
        self.tracking_lin_vel = None
        self.tracking_ang_vel = None
        self.feet_contact = None
        self.feet_air_time = None
        self.feet_slide = None
        self.velocity_mismatch = None
        self.speed_matching = None
        self.feet_contact_forces = None
        self.oriention_non_violation = None
        self.base_lin_acc = None
        self.base_ang_acc_exp = None
        self.joint_deviation_arms = None

@configclass
class UnitreeG1DAggerRefRewards(UnitreeG1DAggerRewards):
    

    tracking_target_velocities_exp = RewTerm(
        func=tracking_target_actions_velocities_exp,
        weight=0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        }
    )
    
    tracking_target_actions_lower_body_diff = RewTerm(
        # func=tracking_target_actions_exp,
        func=tracking_target_actions_diff_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ ".*hip.*", ".*knee.*", ".*ankle.*roll.*"]),
            # "method": "mae",
            "std_updater_cfg": {
                "std_list": [0.65, 0.6, 0.58, 0.55, 0.5, 0.45, 0.42, 0.4],
                "reward_threshold": 0.35,
                "reward_key": "tracking_target_actions_lower_body_diff",
            }
        }
    )
    
    tracking_target_actions_lower_body = RewTerm(
        func=tracking_target_actions_exp,
        # func=tracking_target_actions_diff_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ ".*hip.*", ".*knee.*", ".*ankle.*roll.*"]),
            "method": "mae",
            "std_updater_cfg": {
                "std_list": [0.35, 0.3, 0.28, 0.25, 0.2, 0.15, 0.12, 0.1],
                "reward_threshold": 0.35,
                "reward_key": "tracking_target_actions_lower_body",
            }
        }
    )
    
    tracking_target_actions_diff = RewTerm(
        # func=tracking_target_actions_exp,
        func=tracking_target_actions_diff_exp,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ ".*"]),
            # "method": "mae",
            "std_updater_cfg": {
                "std_list": [0.65, 0.6, 0.58, 0.55, 0.5, 0.45, 0.42, 0.4],
                "reward_threshold": 0.35,
                "reward_key": "tracking_target_actions_diff",
            }
        }
    )
    
    tracking_target_actions = RewTerm(
        func=tracking_target_actions_exp,
        # func=tracking_target_actions_diff_exp,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ ".*"]),
            "method": "mae",
            "std_updater_cfg": {
                "std_list": [0.5, 0.45, 0.42, 0.4, 0.35, 0.3, 0.28, 0.25],
                "reward_threshold": 0.35,
                "reward_key": "tracking_target_actions",
            }
        }
    )

    # tracking_target_actions_knee = RewTerm(
    #     func=tracking_target_actions_exp,
    #     weight=2.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[ ".*knee.*"]),
    #         "method": "mae",
    #         "std_updater_cfg": {
    #             "std_list": [0.35, 0.3, 0.28, 0.25, 0.2, 0.15, 0.12, 0.1],
    #             "reward_threshold": 0.35,
    #             "reward_key": "tracking_target_actions_knee",
    #         }
    #     }
    # )
    
    tracking_target_actions_waist_diff = RewTerm(
        func=tracking_target_actions_diff_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ ".*waist.*"]),
            # "method": "mae",
            "std_updater_cfg": {
                "std_list": [0.65, 0.6, 0.58, 0.55, 0.5, 0.45, 0.42, 0.4],
                "reward_threshold": 0.35,
                "reward_key": "tracking_target_actions_waist_diff",
            }
        }
    )
    
    tracking_target_actions_waist = RewTerm(
        func=tracking_target_actions_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ ".*waist.*"]),
            "method": "mae",
            "std_updater_cfg": {
                "std_list": [0.35, 0.3, 0.28, 0.25, 0.2, 0.15, 0.12, 0.1],
                "reward_threshold": 0.35,
                "reward_key": "tracking_target_actions_waist",
            }
        }
    )

    tracking_target_actions_upper_body_diff = RewTerm(
        func=tracking_target_actions_diff_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ ".*shoulder.*", ".*elbow.*", ".*wrist.*"]),
            # "method": "mae",
            "std_updater_cfg": {
                "std_list": [0.65, 0.6, 0.58, 0.55, 0.5, 0.45, 0.42, 0.4],
                "reward_threshold": 0.35,
                "reward_key": "tracking_target_actions_upper_body_diff",
            }
        }
    )
    
    tracking_target_actions_upper_body = RewTerm(
        func=tracking_target_actions_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ ".*shoulder.*", ".*elbow.*", ".*wrist.*"]),
            "method": "mae",
            "std_updater_cfg": {
                "std_list": [0.35, 0.3, 0.28, 0.25, 0.2, 0.15, 0.12, 0.1],
                "reward_threshold": 0.35,
                "reward_key": "tracking_target_actions_upper_body",
            }
        }
    )
    
    tracking_target_joint_velocities_lower_body = RewTerm(
        func=tracking_target_actions_velocities_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ ".*hip.*", ".*knee.*", ".*ankle.*roll.*"]),
            "std": 0.5,
        }
    )
    
    tracking_target_joint_velocities_upper_body = RewTerm(
        weight= 1.0,
        func=tracking_target_actions_velocities_exp,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ ".*shoulder.*", ".*elbow.*", ".*wrist.*"]),
            "std": 0.5,
        }
    )


@configclass
class UnitreeG1DAggerEnvCfg(UnitreeG1RoughRefEnvCfg):
    observations: UnitreeG1DAggerObservationsCfg = UnitreeG1DAggerObservationsCfg()
    rewards: UnitreeG1DAggerRefRewards = UnitreeG1DAggerRefRewards()
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.scene.robot.spawn.articulation_props.fix_root_link = True
