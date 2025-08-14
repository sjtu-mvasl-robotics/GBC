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

from GBC.gyms.isaaclab_45.lab_assets.turin_v3 import TURIN_V3_CFG
from GBC.gyms.isaaclab_45.managers.ref_obs_term_cfg import ReferenceObservationTermCfg as RefObsTerm
from GBC.gyms.isaaclab_45.managers.ref_obs_term_cfg import ReferenceObservationCfg as RefObsCfg
from GBC.gyms.isaaclab_45.managers.ref_obs_term_cfg import ReferenceObservationGroupCfg as RefObsGroup
from GBC.utils.base import PROJECT_ROOT_DIR, DATA_PATHS
from ..mdp import *
from .rough_env_cfg import TurinV3RoughRefEnvCfg, TurinV3RoughRefEnvCfg_PLAY, TurinV3RefObservationCfg, TurinV3ObservationsCfg, GLOBAL_HISTORY_LENGTH, TurinV3EventCfg, TurinV3Rewards

@configclass
class TurinV3DAggerObservationsCfg(TurinV3ObservationsCfg):

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
class TurinV3DAggerRewards(TurinV3Rewards):
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
class TurinV3DAggerRefRewards(TurinV3DAggerRewards):
    tracking_target_actions_lower_body = RewTerm(
        func=tracking_target_actions_exp,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ ".*hip.*", ".*knee.*", ".*ankle.*"]),
            "method": "mae",
            "std_updater_cfg": {
                "std_list": [0.75, 0.6, 0.5, 0.45, 0.42, 0.4],
                "reward_threshold": 0.5,
                "reward_key": "tracking_target_actions_lower_body",
            }
        }
    )
    tracking_target_actions_lower_body_left = RewTerm(
        func=tracking_target_actions_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ "l.*hip.*", "l.*knee.*", "l.*ankle.*"]),
            "method": "mae",
            "std_updater_cfg": {
                "std_list": [0.75, 0.6, 0.5, 0.45, 0.42, 0.4],
                "reward_threshold": 0.5,
                "reward_key": "tracking_target_actions_lower_body",
            }
        }
    )

    tracking_target_actions_lower_body_right = RewTerm(
        func=tracking_target_actions_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ "r.*hip.*", "r.*knee.*", "r.*ankle.*"]),
            "method": "mae",
            "std_updater_cfg": {
                "std_list": [0.75, 0.6, 0.5, 0.45, 0.42, 0.4],
                "reward_threshold": 0.5,
                "reward_key": "tracking_target_actions_lower_body",
            }
        }
    )

    tracking_target_actions_upper_body = RewTerm(
        func=tracking_target_actions_exp,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ ".*arm.*", ".*elbow.*", ".*wrist.*"]),
            "method": "mae",
            "std_updater_cfg": {
                "std_list": [0.75, 0.6, 0.5, 0.45, 0.42, 0.4],
                "reward_threshold": 0.5,
                "reward_key": "tracking_target_actions_upper_body",
            }
        }
    )

    tracking_target_actions_upper_body_left = RewTerm(
        func=tracking_target_actions_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ "l.*arm.*", "l.*elbow.*", "l.*wrist.*"]),
            "method": "mae",
            "std_updater_cfg": {
                "std_list": [0.75, 0.6, 0.5, 0.45, 0.42, 0.4],
                "reward_threshold": 0.5,
                "reward_key": "tracking_target_actions_upper_body",
            }
        }
    )

    tracking_target_actions_upper_body_right = RewTerm(
        func=tracking_target_actions_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ "r.*arm.*", "r.*elbow.*", "r.*wrist.*"]),
            "method": "mae",
            "std_updater_cfg": {
                "std_list": [0.75, 0.6, 0.5, 0.45, 0.42, 0.4],
                "reward_threshold": 0.5,
                "reward_key": "tracking_target_actions_upper_body",
            }
        }
    )

    tracking_target_actions_normalized_lower_body = RewTerm(
        func=tracking_target_actions_normalized_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ ".*hip.*", ".*knee.*"]),
            "method": "mae",
            "std": 0.75,
            "actions_std_add_constant": 0.3,
        }
    )

    tracking_target_actions_normalized_upper_body = RewTerm(
        func=tracking_target_actions_normalized_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ ".*arm.*", ".*elbow.*",".*wrist.*"]),
        }
    )

    tracking_target_actions_exp = RewTerm(
        func=tracking_target_actions_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        }
    )

    tracking_target_velocities_exp = RewTerm(
        func=tracking_target_actions_velocities_exp,
        weight=0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        }
    )


@configclass
class TurinV3DAggerEnvCfg(TurinV3RoughRefEnvCfg):
    observations: TurinV3DAggerObservationsCfg = TurinV3DAggerObservationsCfg()
    rewards: TurinV3DAggerRefRewards = TurinV3DAggerRefRewards()
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.scene.robot.spawn.articulation_props.fix_root_link = True
