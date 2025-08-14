# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# from isaaclab.managers import RewardTermCfg as RewTerm
# from isaaclab.managers import SceneEntityCfg
# from isaaclab.managers import TerminationTermCfg as DoneTerm
# from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
# from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

# import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
# from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
#     LocomotionVelocityRoughEnvCfg,
#     RewardsCfg,
# )
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, ObservationsCfg, ObsTerm, RewardsCfg, EventCfg,EventTerm
from GBC.gyms.isaaclab_45.managers.physics_modifier_cfg import PhysicsModifierTermCfg as PhxModTerm

##
# Pre-defined configs
##
# from omni.isaac.lab_assets import H1_MINIMAL_CFG  # isort: skip
from GBC.gyms.isaaclab_45.lab_assets.turin_v3 import TURIN_V3_CFG
from GBC.gyms.isaaclab_45.managers.ref_obs_term_cfg import ReferenceObservationTermCfg as RefObsTerm
from GBC.gyms.isaaclab_45.managers.ref_obs_term_cfg import ReferenceObservationCfg as RefObsCfg
from GBC.gyms.isaaclab_45.managers.ref_obs_term_cfg import ReferenceObservationGroupCfg as RefObsGroup
from GBC.utils.base import PROJECT_ROOT_DIR, DATA_PATHS
from ..mdp import *
from GBC.utils.base.math_utils import list_expand
from GBC.utils.base.base_fk import RobotKinematics as FK
from GBC.utils.data_preparation.robot_flip_left_right import TurinV3FlipLeftRight

GLOBAL_HISTORY_LENGTH = 4
num_dof = 29
# flipper = TurinV3FlipLeftRight()

def get_flipper():
    """Get a flipper instance for left-right symmetry operations."""
    return TurinV3FlipLeftRight()

flipper = get_flipper()

def get_actor_observation_symmetry(env: ManagerBasedRLEnv, observations: torch.Tensor, history_length: int = GLOBAL_HISTORY_LENGTH) -> torch.Tensor:
    """Get the symmetry of the observations."""
    if history_length == 0:
        history_length = 1
    
    symmetry = torch.zeros_like(observations)
    symmetry[..., 0:3*history_length] = rot_symmetry(env, observations[..., 0:3*history_length], history_length=history_length) # base ang vel
    symmetry[..., 3*history_length:6*history_length] = pos_symmetry(env, observations[..., 3*history_length:6*history_length], history_length=history_length) # projected gravity
    symmetry[..., 6*history_length:9*history_length] = dim_symmetry(env, observations[..., 6*history_length:9*history_length], symmetry_dims=[1, 2], history_length=history_length) # velocity commands
    symmetry[..., 9*history_length:(9+num_dof)*history_length] = actions_symmetry(env, observations[..., 9*history_length:(9+num_dof)*history_length], history_length=history_length, flipper=flipper) # joint pos
    symmetry[..., (9+num_dof)*history_length:(9+2*num_dof)*history_length] = actions_symmetry(env, observations[..., (9+num_dof)*history_length:(9+2*num_dof)*history_length], history_length=history_length, flipper=flipper) # joint vel
    symmetry[..., (9+2*num_dof)*history_length:(9+3*num_dof)*history_length] = actions_symmetry(env, observations[..., (9+2*num_dof)*history_length:(9+3*num_dof)*history_length], history_length=history_length, flipper=flipper) # actions
    return symmetry

def get_critic_observation_symmetry(env: ManagerBasedRLEnv, observations: torch.Tensor, history_length: int = GLOBAL_HISTORY_LENGTH) -> torch.Tensor:
    """Get the symmetry of the observations."""
    if history_length == 0:
        history_length = 1

    symmetry = torch.zeros_like(observations)
    symmetry[..., 0:3*history_length] = rot_symmetry(env, observations[..., 0:3*history_length], history_length=history_length) # base ang vel
    symmetry[..., 3*history_length:6*history_length] = pos_symmetry(env, observations[..., 3*history_length:6*history_length], history_length=history_length) # projected gravity
    symmetry[..., 6*history_length:9*history_length] = dim_symmetry(env, observations[..., 6*history_length:9*history_length], symmetry_dims=[1, 2], history_length=history_length) # velocity commands
    symmetry[..., 9*history_length:(9+num_dof)*history_length] = actions_symmetry(env, observations[..., 9*history_length:(9+num_dof)*history_length], history_length=history_length, flipper=flipper) # joint pos
    symmetry[..., (9+num_dof)*history_length:(9+2*num_dof)*history_length] = actions_symmetry(env, observations[..., (9+num_dof)*history_length:(9+2*num_dof)*history_length], history_length=history_length, flipper=flipper) # joint vel
    symmetry[..., (9+2*num_dof)*history_length:(9+3*num_dof)*history_length] = actions_symmetry(env, observations[..., (9+2*num_dof)*history_length:(9+3*num_dof)*history_length], history_length=history_length, flipper=flipper) # actions

    symmetry[..., (9+3*num_dof)*history_length:(9+3*num_dof+3)*history_length] = pos_symmetry(env, observations[..., (9+3*num_dof)*history_length:(9+3*num_dof+3)*history_length], history_length=history_length) # base lin vel
    symmetry[..., (9+3*num_dof+3)*history_length:(9+3*num_dof+4)*history_length] = observations[..., (9+3*num_dof+5)*history_length:(9+3*num_dof+6)*history_length] # lft sin phase
    symmetry[..., (9+3*num_dof+4)*history_length:(9+3*num_dof+5)*history_length] = observations[..., (9+3*num_dof+6)*history_length:(9+3*num_dof+7)*history_length] # lft cos phase
    symmetry[..., (9+3*num_dof+5)*history_length:(9+3*num_dof+6)*history_length] = observations[..., (9+3*num_dof+3)*history_length:(9+3*num_dof+4)*history_length] # rht sin phase
    symmetry[..., (9+3*num_dof+6)*history_length:(9+3*num_dof+7)*history_length] = observations[..., (9+3*num_dof+4)*history_length:(9+3*num_dof+5)*history_length] # rht cos phase
    return symmetry

def get_amp_observations(observations: torch.Tensor, env: ManagerBasedRLEnv, history_length: int = GLOBAL_HISTORY_LENGTH) -> torch.Tensor:
    # base ang vel, projected gravity, joint pos
    base_ang_vel = observations[..., 3 * history_length - 3:3 * history_length]
    projected_gravity = observations[..., 6 * history_length - 3:6 * history_length]
    joint_pos = observations[..., (9+num_dof)*history_length - num_dof:(9+num_dof)*history_length]
    amp_observations = torch.cat(
        [
            base_ang_vel,
            projected_gravity,
            joint_pos, 
        ]
        , dim=-1
    )
    return amp_observations

def get_amp_ref_observations(ref_observations: tuple[torch.Tensor, torch.Tensor], env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    # base ang vel, projected gravity, joint pos
    ref_observation_terms = ["base_ang_vel", "target_projected_gravity", "target_actions"]
    ref_term_dims = env.unwrapped.ref_observation_manager.compute_amp_dims(ref_observation_terms)
    ref_observation_tensor, ref_observation_mask = ref_observations
    ref_terms = []
    for i in range(len(ref_term_dims)):
        ref_terms.append(ref_observation_tensor[..., ref_term_dims[i][0]:ref_term_dims[i][1]])
    ref_terms = torch.cat(ref_terms, dim=-1)
    return ref_terms, ref_observation_mask



@configclass
class TurinV3ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        '''Configuration for the policy observations.

        - base_ang_vel: (3,)
        - projected_gravity: (3,)
        - velocity_commands: (3,)
        - joint_pos: (num_dof,)
        - joint_vel: (num_dof,)
        - actions: (num_dof,)
        
        '''

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        lft_sin_phase = ObsTerm(func=get_phase, params={ "period": 0.8, "offset": 0.0, "ref_name": "lft_sin_phase" })
        lft_cos_phase = ObsTerm(func=get_phase, params={ "period": 0.8, "offset": 0.25, "ref_name": "lft_cos_phase" })
        rht_sin_phase = ObsTerm(func=get_phase, params={ "period": 0.8, "offset": 0.5, "ref_name": "rht_sin_phase" })
        rht_cos_phase = ObsTerm(func=get_phase, params={ "period": 0.8, "offset": 0.75, "ref_name": "rht_cos_phase" })

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = GLOBAL_HISTORY_LENGTH
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        '''Configuration for the critic observations (privileged observations).
        - base_ang_vel: (3,)
        - projected_gravity: (3,)
        - velocity_commands: (3,)
        - joint_pos: (num_dof,)
        - joint_vel: (num_dof,)
        - actions: (num_dof,)
        - base_lin_vel: (3,)        
        - lft_sin_phase: (1,)
        - lft_cos_phase: (1,)
        - rht_sin_phase: (1,)
        - rht_cos_phase: (1,)        
        '''
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        lft_sin_phase = ObsTerm(func=get_phase, params={ "period": 0.8, "offset": 0.0, "ref_name": "lft_sin_phase" })
        lft_cos_phase = ObsTerm(func=get_phase, params={ "period": 0.8, "offset": 0.25, "ref_name": "lft_cos_phase" })
        rht_sin_phase = ObsTerm(func=get_phase, params={ "period": 0.8, "offset": 0.5, "ref_name": "rht_sin_phase" })
        rht_cos_phase = ObsTerm(func=get_phase, params={ "period": 0.8, "offset": 0.75, "ref_name": "rht_cos_phase" })

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = GLOBAL_HISTORY_LENGTH
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class TurinV3RefObservationCfg(RefObsCfg):
    @configclass
    class PolicyCfg(RefObsGroup):
        target_actions = RefObsTerm(func=reference_action_reshape, name="actions", noise=None, symmetry=actions_symmetry, symmetry_params={"flipper": flipper}, params={
            "urdf_path": DATA_PATHS.urdf_path,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "add_default_joint_pos": False,
        }) # it is acctually the target joint positions, not joint actions

        target_projected_gravity = RefObsTerm(
            name="project_gravity", noise=None, symmetry=pos_symmetry,
        )

        base_ang_vel = RefObsTerm(name="ang_vel", noise=Unoise(n_min=-0.04, n_max=0.04), symmetry=rot_symmetry)

        lft_sin_phase = RefObsTerm(name="feet_contact", func=reference_feet_contact_phase, params={
            "index": 0,
            "offset": 0.0,
        }, symmetry=symmetry_by_ref_term_name, symmetry_params={"term_name": "rht_sin_phase"})
        lft_cos_phase = RefObsTerm(name="feet_contact", func=reference_feet_contact_phase, params={
            "index": 0,
            "offset": 0.25,
        }, symmetry=symmetry_by_ref_term_name, symmetry_params={"term_name": "rht_cos_phase"})
        rht_sin_phase = RefObsTerm(name="feet_contact", func=reference_feet_contact_phase, params={
            "index": 1,
            "offset": 0.0,
        }, symmetry=symmetry_by_ref_term_name, symmetry_params={"term_name": "lft_sin_phase"})
        rht_cos_phase = RefObsTerm(name="feet_contact", func=reference_feet_contact_phase, params={
            "index": 1,
            "offset": 0.25,
        }, symmetry=symmetry_by_ref_term_name, symmetry_params={"term_name": "lft_cos_phase"})


        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True
            self.load_seq_delay = 0.0
            self.history_length = GLOBAL_HISTORY_LENGTH

    @configclass
    class CriticCfg(RefObsGroup):
        target_actions = RefObsTerm(func=reference_action_reshape, name="actions", noise=None, symmetry=actions_symmetry, symmetry_params={"flipper": flipper}, params={
            "urdf_path": DATA_PATHS.urdf_path,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "add_default_joint_pos": False,
        })

        target_actions_std = RefObsTerm(func=reference_action_std, name="actions", noise=None, in_obs_tensor=False, is_constant=True, symmetry=actions_symmetry, symmetry_params={"flipper": flipper}, params={
            "urdf_path": DATA_PATHS.urdf_path,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "add_default_joint_pos": False,
        })

        target_joint_velocities = RefObsTerm(
            func=reference_action_velocity, name="actions", noise=None, in_obs_tensor=True, symmetry=actions_symmetry, symmetry_params={"flipper": flipper}, params={
                "urdf_path": DATA_PATHS.urdf_path,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            }
        )

        target_projected_gravity = RefObsTerm(
            name="project_gravity", noise=None, symmetry=pos_symmetry,
        )
        base_lin_vel = RefObsTerm(name="lin_vel", noise=Unoise(n_min=-0.04, n_max=0.04), symmetry=pos_symmetry)
        base_ang_vel = RefObsTerm(name="ang_vel", noise=Unoise(n_min=-0.04, n_max=0.04), symmetry=rot_symmetry)
        # feet_contact = RefObsTerm(name="feet_contact", noise=None)
        lft_sin_phase = RefObsTerm(name="feet_contact", func=reference_feet_contact_phase, params={
            "index": 0,
            "offset": 0.0,
        }, symmetry=symmetry_by_ref_term_name, symmetry_params={"term_name": "rht_sin_phase"})
        lft_cos_phase = RefObsTerm(name="feet_contact", func=reference_feet_contact_phase, params={
            "index": 0,
            "offset": 0.25,
        }, symmetry=symmetry_by_ref_term_name, symmetry_params={"term_name": "rht_cos_phase"})
        rht_sin_phase = RefObsTerm(name="feet_contact", func=reference_feet_contact_phase, params={
            "index": 1,
            "offset": 0.0,
        }, symmetry=symmetry_by_ref_term_name, symmetry_params={"term_name": "lft_sin_phase"})
        rht_cos_phase = RefObsTerm(name="feet_contact", func=reference_feet_contact_phase, params={
            "index": 1,
            "offset": 0.25,
        }, symmetry=symmetry_by_ref_term_name, symmetry_params={"term_name": "lft_cos_phase"})

        target_base_pose = RefObsTerm(
            name="base_pose",
            in_obs_tensor=True,
            is_base_pose=True,
            params={
                "lin_vel_name": "base_lin_vel",
                "ang_vel_name": "base_ang_vel"
            },
            symmetry=dim_symmetry,
            symmetry_params={
                "symmetry_dims": [1, 4, 6], # lin_y, quat_x, quat_z
            }
        )

        target_link_poses = RefObsTerm( # w.r.t. robot base frame
            name="actions",
            in_obs_tensor=False,
            func=reference_link_poses,
            params={
                "urdf_path": DATA_PATHS.urdf_path,
            },
            symmetry=dim_symmetry,
            symmetry_params={
                "symmetry_dims": list_expand([1, 4, 6], num_dof+1, 7), # lin_y, quat_x, quat_z
            }
        )

        target_link_velocities = RefObsTerm( # lin_vel in robot base frame, ang_vel in world frame
            name={
                "actions": "actions",
                "trans": "trans",
                "root_orient": "root_orient",
            },
            in_obs_tensor=False,
            func=reference_link_velocities,
            params={
                "urdf_path": DATA_PATHS.urdf_path,
            },
            symmetry=dim_symmetry,
            symmetry_params={
                "symmetry_dims": list_expand([1, 3, 5], num_dof+1, 6), # lin_x, ang_x, ang_z
            }
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True
            self.load_seq_delay = 0.0 #2.5
            self.history_length = GLOBAL_HISTORY_LENGTH

    
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    data_dir: list[str] = DATA_PATHS.ref_data_path
    working_mode: str = "recurrent"


@configclass
class TurinV3NoRefObservationCfg(RefObsCfg):
    @configclass
    class PolicyCfg(RefObsGroup):
        empty = RefObsTerm(make_empty=True, name="lin_vel")
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True
            self.load_seq_delay = 0.0
            self.history_length = GLOBAL_HISTORY_LENGTH
    
    policy: PolicyCfg = PolicyCfg()
    # critic: PolicyCfg = PolicyCfg()  # Same as policy, no critic in this case
    data_dir: list[str] = DATA_PATHS.ref_data_path
    working_mode: str = "recurrent"

    def __post_init__(self):
        self.data_dir = DATA_PATHS.ref_data_path


@configclass
class TurinV3Rewards:
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    tracking_lin_vel = RewTerm(
        func=track_lin_vel_xy_yaw_frame_exp_custom,
        weight=1.4,
        params={
            "std": 0.5,
            "command_name": "base_velocity",
            "std_updater_cfg": {
                "std_list": [0.5, 0.45, 0.42, 0.4, 0.35],
                "reward_threshold": 0.5, # when the reward is above this threshold * weight for certain number of steps, the std will be updated
                "reward_key": "tracking_lin_vel"
            }
        }
    )

    tracking_ang_vel = RewTerm(
        func=track_ang_vel_z_world_exp_custom,
        weight=1.1,
        params={
            "std": 0.5,
            "command_name": "base_velocity",
            "std_updater_cfg": {
                "std_list": [0.5, 0.45, 0.42, 0.4, 0.35],
                "reward_threshold": 0.5,
                "reward_key": "tracking_ang_vel"
            }
        }
    )

    feet_contact = RewTerm(
        func=tracking_feet_contact_phase,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["l.*ankle.*roll.*", "r.*ankle.*roll.*"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["l.*ankle.*roll.*", "r.*ankle.*roll.*"]),
        }
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle.*roll.*_link.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle.*roll.*_link.*"),
        },
    )

    feet_distances = RewTerm(
        func=feet_distance,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["l.*ankle.*roll.*", "r.*ankle.*roll.*"]),
            "distance_min": 0.2,
            "distance_max": 0.5,
        }
    )

    knee_distances = RewTerm(
        func=feet_distance,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["l.*knee.*", "r.*knee.*"]),
            "distance_min": 0.2,
            "distance_max": 0.5,
        }
    )

    velocity_mismatch = RewTerm(
        func=reward_speed_non_violation,
        params={},
        weight=0.5,
    )

    speed_matching = RewTerm(
        func=reward_speed_matching,
        params={
            "command_name": "base_velocity",
        },
        weight=0.2,
    )

    feet_contact_forces = RewTerm(
        func=feet_contact_forces,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["l.*ankle.*roll.*", "r.*ankle.*roll.*"]),
            "max_contact_force": 700.0,
        },
    )

    oriention_non_violation = RewTerm(
        func=reward_orientation_non_violation,
        weight=1.0,
    )

    base_height = RewTerm(
        func=reward_base_height,
        weight=0.2,
    )

    base_lin_acc = RewTerm(
        func=base_lin_acc_exp,
        weight=0.2,
        params={
            "std": 0.6,
        },
    )

    base_ang_acc_exp = RewTerm(
        func=base_ang_acc_exp,
        weight=0.1,
        params={
            "std": 0.4,
        },
    )

    action_scale = RewTerm(
        func=action_scale_l2,
        weight=-1.5e-4,
    )

    action_rate = RewTerm(
        func=action_rate_l2_dt,
        weight=-2.0e-5,
    )

    action_rate_acc = RewTerm(
        func=action_rate_acc_l2_dt,
        weight=-2.0e-9,
    )

    torques = RewTerm(func=mdp.joint_torques_l2, weight=-1e-10)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5.e-9)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-5)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle.*roll.*_link.*"),
            "threshold": 0.5,
        },
    )

    # joint_deviation_hip = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.2,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"])},
    # )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*arm.*", ".*elbow.*"])},
    )
    joint_deviation_wrist = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*wrist.*")}
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist.*"])},
    )
    survival = RewTerm(func=survival, weight=0.15)


@configclass
class TurinV3RefRewards(TurinV3Rewards):

    ###################################################################
    #
    # Basic joint tracking rewards
    #
    ###################################################################

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

    ###################################################################
    #
    # Joint position & velocity rewards
    #
    ###################################################################

    tracking_target_lower_body_xyz = RewTerm(
        func=tracking_body_pos_robot_frame,
        weight=0.5,
        params={
            "std": 0.2,
            "xyz_dim": (0, 1, 2),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*knee.*", ".*hip_roll.*", ".*ankle.*"]),
        }
    )

    tracking_target_upper_body_xyz = RewTerm(
        func=tracking_body_pos_robot_frame,
        weight=0.5,
        params={
            "std": 0.15,
            "xyz_dim": (0, 1, 2),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*arm.*", ".*wrist.*", ".*elbow.*"]),
        }
    )

    tracking_target_feet_height = RewTerm(
        func=tracking_body_pos_robot_frame,
        weight=1.0,
        params={
            "std": 0.2,
            "xyz_dim": (2),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
        }
    )

    tracking_target_upper_body_orn = RewTerm(
        func=tracking_body_quat_robot_frame,
        weight=0.3,
        params={
            "std": 0.3,
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*arm.*", ".*wrist.*", ".*elbow.*"]),
        }
    )
    tracking_target_feet_orn = RewTerm(
        func=tracking_body_quat_robot_frame,
        weight=1.0, # 1.0,
        params={
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
        }
    )

    tracking_target_lower_body_orn = RewTerm(
        func=tracking_body_quat_robot_frame,
        weight=0.3,
        params={
            "std": 0.3,
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*knee.*", ".*hip_roll.*"]),
        }
    )

    tracking_target_feet_lin_vel = RewTerm(
        func=tracking_body_lin_vel_robot_frame,
        weight=0.4,
        params={
            "std": 0.8,
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"])
        }
    )

    ###################################################################
    #
    # contact phase rewards & other rewards
    #
    ###################################################################

    tracking_feet_contact_left = RewTerm(
        func=tracking_feet_contact_phase,
        weight=0.1, # 15.0
        params={
            "ref_id": [0],
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["l.*ankle.*roll.*"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["l.*ankle.*roll.*"]),
        }
    )

    tracking_feet_contact_right = RewTerm(
        func=tracking_feet_contact_phase,
        weight=0.1, # 15.0
        params={
            "ref_id": [1],
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["r.*ankle.*roll.*"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["r.*ankle.*roll.*"]),
        }
    )

    feet_always_contact_penalty = RewTerm(
        func=penalize_one_foot_always_contact,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle.*_roll.*"),
        },
    )

    target_gravity_diff_l2 = RewTerm(
        func=tracking_gravity_diff_exp,
        params={
            "std": 0.25,
        },
        weight=2.0,
    )

    track_base_height = RewTerm(
        func=tracking_base_height,
        weight=2.0,
        params={
            "std": 0.4,
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )

    feet_dist_too_small = RewTerm(
        func=feet_distance_relative_to_target,
        weight=-2.0,
        params={ "asset_cfg": SceneEntityCfg("robot", body_names=["l.*ankle.*roll.*", "r.*ankle.*roll.*"]), },
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*arm.*", ".*elbow.*", ".*hip.*", ".*wrist.*"]), "threshold": 1.0},
    )



@configclass
class TurinV3EventCfg(EventCfg):

    # startup
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*base.*"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    

    # reset
    robot_joint_stiffness_and_damping = EventTerm(
      func=mdp.randomize_actuator_gains,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
          "stiffness_distribution_params": (0.8, 1.2),
          "damping_distribution_params": (0.8, 1.2),
          "operation": "scale",
          "distribution": "log_uniform",
      },
    )

    randomize_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    robot_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0, 0.1),
            "armature_distribution_params": (0.0, 0.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_start_time = EventTerm(
        func=randomize_initial_start_time,
        mode="reset",
        params={
            "sample_episode_ratio": 1.0,
        }
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class TurinV3RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: TurinV3ObservationsCfg = TurinV3ObservationsCfg()
    ref_observation: TurinV3NoRefObservationCfg = TurinV3NoRefObservationCfg()

    rewards: TurinV3Rewards = TurinV3Rewards()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.sim.dt = 0.001
        self.decimation = 10
        self.sim.render_interval = self.decimation
        # Scene
        self.scene.robot = TURIN_V3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # Randomization
        # self.events.push_robot = None
        # self.events.add_base_mass = None
        self.events.add_base_mass.params["asset_cfg"].body_names = [".*base_link"]
        self.events.physics_material.params["static_friction_range"] = (0.8, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.8)
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.1)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*base_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.4, 0.4),
                "y": (-0.3, 0.3),
                "z": (-0.5, 0.5),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        }


        # Terminations
        # self.terminations.base_contact.params["sensor_cfg"].body_names = [".*torso_link"]
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)

        self.observations.policy.history_length = GLOBAL_HISTORY_LENGTH


@configclass
class TurinV3RoughEnvCfg_PLAY(TurinV3RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class TurinV3RoughRefEnvCfg(TurinV3RoughEnvCfg):
    # observations: H1DAGObs = H1DAGObs()
    observations: TurinV3ObservationsCfg = TurinV3ObservationsCfg()
    ref_observation: TurinV3RefObservationCfg = TurinV3RefObservationCfg()
    rewards: TurinV3RefRewards = TurinV3RefRewards()
    events: TurinV3EventCfg = TurinV3EventCfg()
    # physics_modifiers: PhysicsModifiersCfg = PhysicsModifiersCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.events.reset_base = EventTerm(
            func=reset_root_state_by_start_time,
            mode="reset",
        )

        self.events.reset_robot_joints = EventTerm(
            func=reset_joints_by_start_time,
            mode="reset",
        )

        self.rewards.base_height = None  # base height is not used in the reference environment
        
        self.episode_length_s = 20.0
        # self.scene.robot.spawn.articulation_props.fix_root_link = True

@configclass
class TurinV3RoughRefEnvCfg_PLAY(TurinV3RoughEnvCfg_PLAY):
    observations: TurinV3ObservationsCfg = TurinV3ObservationsCfg()
    ref_observation: TurinV3RefObservationCfg = TurinV3RefObservationCfg()
    rewards: TurinV3RefRewards = TurinV3RefRewards()
    events: TurinV3EventCfg = TurinV3EventCfg()
    # events: H1EventCfg = H1EventCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()