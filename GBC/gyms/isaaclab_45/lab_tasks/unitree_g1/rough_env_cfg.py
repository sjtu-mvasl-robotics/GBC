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
from isaaclab.assets import Articulation, RigidObject

##
# Pre-defined configs
##
# from omni.isaac.lab_assets import H1_MINIMAL_CFG  # isort: skip
from GBC.gyms.isaaclab_45.lab_assets.unitree import G1_29DOF_CFG
from GBC.gyms.isaaclab_45.managers.ref_obs_term_cfg import ReferenceObservationTermCfg as RefObsTerm
from GBC.gyms.isaaclab_45.managers.ref_obs_term_cfg import ReferenceObservationCfg as RefObsCfg
from GBC.gyms.isaaclab_45.managers.ref_obs_term_cfg import ReferenceObservationGroupCfg as RefObsGroup
from GBC.utils.base import PROJECT_ROOT_DIR, DATA_PATHS
from ..mdp import *
from GBC.utils.base.math_utils import list_expand
from GBC.utils.base.base_fk import RobotKinematics as FK
from GBC.utils.data_preparation.robot_flip_left_right import UnitreeG1FlipLeftRight

GLOBAL_HISTORY_LENGTH = 1
ACTION_SCALE = 0.25
num_dof = 29
# flipper = UnitreeG1FlipLeftRight()

def get_flipper():
    """Get a flipper instance for left-right symmetry operations."""
    return UnitreeG1FlipLeftRight()

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
    symmetry[..., (9+3*num_dof+0)*history_length:(9+3*num_dof+1)*history_length] = observations[..., (9+3*num_dof+2)*history_length:(9+3*num_dof+3)*history_length] # lft sin phase
    symmetry[..., (9+3*num_dof+1)*history_length:(9+3*num_dof+2)*history_length] = observations[..., (9+3*num_dof+3)*history_length:(9+3*num_dof+4)*history_length] # lft cos phase
    symmetry[..., (9+3*num_dof+2)*history_length:(9+3*num_dof+3)*history_length] = observations[..., (9+3*num_dof+0)*history_length:(9+3*num_dof+1)*history_length] # rht sin phase
    symmetry[..., (9+3*num_dof+3)*history_length:(9+3*num_dof+4)*history_length] = observations[..., (9+3*num_dof+1)*history_length:(9+3*num_dof+2)*history_length] # rht cos phase
    # last 4 dims: imu orientation (quaternion)
    symmetry[..., (9+3*num_dof+4)*history_length:(9+3*num_dof+8)*history_length] = quat_symmetry(env, observations[..., (9+3*num_dof+4)*history_length:(9+3*num_dof+8)*history_length], history_length=history_length) # imu orientation
    
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
    
    # imu orientation (quaternion)
    symmetry[..., (9+3*num_dof+7)*history_length:(9+3*num_dof+11)*history_length] = quat_symmetry(env, observations[..., (9+3*num_dof+7)*history_length:(9+3*num_dof+11)*history_length], history_length=history_length) # imu orientation
    symmetry[..., (9+3*num_dof+11)*history_length:(9+3*num_dof+14)*history_length] = pos_symmetry(env, observations[..., (9+3*num_dof+11)*history_length:(9+3*num_dof+14)*history_length], history_length=history_length) # base pos w
    return symmetry

def get_amp_observations(env: ManagerBasedRLEnv, history_length: int = GLOBAL_HISTORY_LENGTH, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # base ang vel, projected gravity, joint pos
    # base_ang_vel = observations[..., 3 * history_length - 3:3 * history_length]
    # projected_gravity = observations[..., 6 * history_length - 3:6 * history_length]
    # joint_pos = observations[..., (9+num_dof)*history_length - num_dof:(9+num_dof)*history_length]
    # imu_orientation = observations[..., (9+3*num_dof+8)*history_length - 4:(9+3*num_dof+8)*history_length]
    # amp_observations = torch.cat(
    #     [
    #         base_ang_vel,
    #         projected_gravity,
    #         joint_pos, 
    #         imu_orientation,
    #     ]
    #     , dim=-1
    # )
    cur_env = env.unwrapped
    target_joints = [".*wrist.*yaw.*", ".*ankle.*roll.*"]
    if not hasattr(get_amp_observations, "joint_asset_cfg"):
        joint_asset_cfg = SceneEntityCfg("robot", body_names=target_joints)
        joint_asset_cfg.resolve(cur_env.scene)
        get_amp_observations.joint_asset_cfg = joint_asset_cfg
    joint_asset_cfg = get_amp_observations.joint_asset_cfg
    asset: Articulation = cur_env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_b # 3
    base_ang_vel = asset.data.root_ang_vel_b # 3
    base_projected_gravity = asset.data.projected_gravity_b # 3
    joint_pos = asset.data.joint_pos # num_dof
    joint_vel = asset.data.joint_vel # num_dof
    imu_orientation = asset.data.root_quat_w # 4
    root_xyz = asset.data.root_pos_w # 3
    joint_xyz = compute_actual_body_pose_robot_frame(cur_env, joint_asset_cfg) # (num_envs, len(target_joints), 7)
    joint_xyz = joint_xyz.view(joint_xyz.shape[0], -1) # (num_envs, len(target_joints)*7)
    
    amp_observations = torch.cat(
        [
            base_lin_vel,
            base_ang_vel,
            base_projected_gravity,
            joint_pos, 
            # joint_vel,
            root_xyz,
            imu_orientation,
            joint_xyz,
        ]
        , dim=-1
    )
    
    return amp_observations

def get_amp_ref_observations(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    # base ang vel, projected gravity, joint pos
    # ref_observation_terms = ["base_ang_vel", "target_projected_gravity", "target_actions", "target_quaternion"]
    ref_observation_terms = [
        "base_lin_vel",
        "base_ang_vel",
        "target_projected_gravity",
        "target_actions",
        # "target_joint_velocities",
        "target_base_pose", # this is already the concatenation of pos and quat
        "target_link_poses"
    ]
    target_joints = [".*wrist.*yaw.*", ".*ankle.*roll.*"]
    if not hasattr(get_amp_ref_observations, "joint_asset_cfg"):
        joint_asset_cfg = SceneEntityCfg("robot", body_names=target_joints)
        joint_asset_cfg.resolve(env.unwrapped.scene)
        get_amp_ref_observations.joint_asset_cfg = joint_asset_cfg
    joint_asset_cfg = get_amp_ref_observations.joint_asset_cfg
    # ref_term_dims = env.unwrapped.ref_observation_manager.compute_amp_dims(ref_observation_terms)
    # ref_observation_tensor, ref_observation_mask = ref_observations
    ref_terms = []
    # for i, term_name in enumerate(ref_observation_terms):
    #     term = ref_observation_tensor[..., ref_term_dims[i][0]:ref_term_dims[i][1]]
    #     if term_name == "target_link_poses":
    #         term = term.view(term.shape[0], -1, 7) # (num_envs, len(target_joints), 7)
    #         term = term[:, joint_asset_cfg.body_ids, :]
    #         term = term.view(term.shape[0], -1) # (num_envs, len(target_joints)*7)
    #     ref_terms.append(term)
    for i, term_name in enumerate(ref_observation_terms):
        term, mask = env.unwrapped.ref_observation_manager.get_term(term_name)
        if term_name == "target_link_poses":
            term = term.view(term.shape[0], -1, 7) # (num_envs, len(target_joints), 7)
            term = term[:, joint_asset_cfg.body_ids, :]
            term = term.view(term.shape[0], -1) # (num_envs, len(target_joints)*7)
        term = term.reshape(term.shape[0], -1) # (num_envs, dim_total)
        if i == 0:
            ref_observation_mask = mask
        else:
            ref_observation_mask = ref_observation_mask & mask
        ref_terms.append(term)
    
    ref_terms = torch.cat(ref_terms, dim=-1)
    return ref_terms, ref_observation_mask



@configclass
class UnitreeG1ObservationsCfg:

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
        imu_orientation = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.05, n_max=0.05)) # (4,)
        

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
        imu_orientation = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.05, n_max=0.05)) # (4,)
        robot_position = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.1, n_max=0.1))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = GLOBAL_HISTORY_LENGTH
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class UnitreeG1RefObservationCfg(RefObsCfg):
    @configclass
    class PolicyCfg(RefObsGroup):
        target_actions = RefObsTerm(func=reference_action_reshape, name="actions", noise=None, symmetry=actions_symmetry, symmetry_params={"flipper": flipper}, params={
            "urdf_path": DATA_PATHS.urdf_path,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "add_default_joint_pos": False,
        }) # it is acctually the target joint positions, not joint actions
        target_actions_t1 = RefObsTerm(func=reference_action_reshape, load_seq_delay=-0.4, name="actions", noise=None, symmetry=actions_symmetry, symmetry_params={"flipper": flipper}, params={
            "urdf_path": DATA_PATHS.urdf_path,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "add_default_joint_pos": False,
        })
        target_actions_t2 = RefObsTerm(func=reference_action_reshape, load_seq_delay=-0.8, name="actions", noise=None, symmetry=actions_symmetry, symmetry_params={"flipper": flipper}, params={
            "urdf_path": DATA_PATHS.urdf_path,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "add_default_joint_pos": False,
        })
        target_actions_t3 = RefObsTerm(func=reference_action_reshape, load_seq_delay=-1.6, name="actions", noise=None, symmetry=actions_symmetry, symmetry_params={"flipper": flipper}, params={
            "urdf_path": DATA_PATHS.urdf_path,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "add_default_joint_pos": False,
        })

        target_projected_gravity = RefObsTerm(
            name="project_gravity", noise=None, symmetry=pos_symmetry,
        )

        base_ang_vel = RefObsTerm(name="ang_vel", noise=Unoise(n_min=-0.04, n_max=0.04), symmetry=rot_symmetry)
        # base_ang_vel_t1 = RefObsTerm(name="ang_vel", load_seq_delay=-1.0, noise=Unoise(n_min=-0.04, n_max=0.04), symmetry=rot_symmetry)
        # base_ang_vel_t2 = RefObsTerm(name="ang_vel", load_seq_delay=-2.0, noise=Unoise(n_min=-0.04, n_max=0.04), symmetry=rot_symmetry)
        # base_ang_vel_t3 = RefObsTerm(name="ang_vel", load_seq_delay=-4.0, noise=Unoise(n_min=-0.04, n_max=0.04), symmetry=rot_symmetry)

        target_actions_diff = RefObsTerm(func=reference_action_reshape, load_seq_delay=0.02, name="actions", noise=None, env_func=reference_actions_diff, env_func_params={"history_length": GLOBAL_HISTORY_LENGTH}, symmetry=actions_symmetry, symmetry_params={"flipper": flipper}, params={
            "urdf_path": DATA_PATHS.urdf_path,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "add_default_joint_pos": False,
        }) # this is actually the target joint velocities
        
        target_quaternion = RefObsTerm(func=reference_root_quat_from_rot, name="root_orient", noise=None, symmetry=quat_symmetry, params={})
        target_quaternion_t1 = RefObsTerm(func=reference_root_quat_from_rot, load_seq_delay=-0.4, name="root_orient", noise=None, symmetry=quat_symmetry, params={})
        target_quaternion_t2 = RefObsTerm(func=reference_root_quat_from_rot, load_seq_delay=-0.8, name="root_orient", noise=None, symmetry=quat_symmetry, params={})
        target_quaternion_t3 = RefObsTerm(func=reference_root_quat_from_rot, load_seq_delay=-1.6, name="root_orient", noise=None, symmetry=quat_symmetry, params={})
        
        target_quaternion_diff = RefObsTerm(func=reference_root_quat_from_rot, load_seq_delay=0.02, name="root_orient", noise=None, env_func=reference_quaternion_diff, env_func_params={"history_length": GLOBAL_HISTORY_LENGTH}, symmetry=no_symmetry, params={})

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
        target_actions_t1 = RefObsTerm(func=reference_action_reshape, load_seq_delay=-0.4, name="actions", noise=None, symmetry=actions_symmetry, symmetry_params={"flipper": flipper}, params={
            "urdf_path": DATA_PATHS.urdf_path,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "add_default_joint_pos": False,
        })
        target_actions_t2 = RefObsTerm(func=reference_action_reshape, load_seq_delay=-0.8, name="actions", noise=None, symmetry=actions_symmetry, symmetry_params={"flipper": flipper}, params={
            "urdf_path": DATA_PATHS.urdf_path,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "add_default_joint_pos": False,
        })
        target_actions_t3 = RefObsTerm(func=reference_action_reshape, load_seq_delay=-1.6, name="actions", noise=None, symmetry=actions_symmetry, symmetry_params={"flipper": flipper}, params={
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
        # base_lin_vel_t1 = RefObsTerm(name="lin_vel", load_seq_delay=-1.0, noise=Unoise(n_min=-0.04, n_max=0.04), symmetry=pos_symmetry)
        # base_lin_vel_t2 = RefObsTerm(name="lin_vel", load_seq_delay=-2.0, noise=Unoise(n_min=-0.04, n_max=0.04), symmetry=pos_symmetry)
        # base_lin_vel_t3 = RefObsTerm(name="lin_vel", load_seq_delay=-4.0, noise=Unoise(n_min=-0.04, n_max=0.04), symmetry=pos_symmetry)
        base_ang_vel = RefObsTerm(name="ang_vel", noise=Unoise(n_min=-0.04, n_max=0.04), symmetry=rot_symmetry)
        # base_ang_vel_t1 = RefObsTerm(name="ang_vel", load_seq_delay=-1.0, noise=Unoise(n_min=-0.04, n_max=0.04), symmetry=rot_symmetry)
        # base_ang_vel_t2 = RefObsTerm(name="ang_vel", load_seq_delay=-2.0, noise=Unoise(n_min=-0.04, n_max=0.04), symmetry=rot_symmetry)
        # base_ang_vel_t3 = RefObsTerm(name="ang_vel", load_seq_delay=-4.0, noise=Unoise(n_min=-0.04, n_max=0.04), symmetry=rot_symmetry)
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
            in_obs_tensor=False,
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
        
        target_actions_diff = RefObsTerm(func=reference_action_reshape, load_seq_delay=0.02, name="actions",    noise=None, env_func=reference_actions_diff, env_func_params={"history_length": GLOBAL_HISTORY_LENGTH}, symmetry=actions_symmetry, symmetry_params={"flipper": flipper}, params={
                "urdf_path": DATA_PATHS.urdf_path,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "add_default_joint_pos": False,
        }) # this is actually the target joint velocities
        
        target_quaternion = RefObsTerm(func=reference_root_quat_from_rot, name="root_orient", noise=None, symmetry=quat_symmetry, params={})
        target_quaternion_t1 = RefObsTerm(func=reference_root_quat_from_rot, load_seq_delay=-0.4, name="root_orient", noise=None, symmetry=quat_symmetry, params={})
        target_quaternion_t2 = RefObsTerm(func=reference_root_quat_from_rot, load_seq_delay=-0.8, name="root_orient", noise=None, symmetry=quat_symmetry, params={})
        target_quaternion_t3 = RefObsTerm(func=reference_root_quat_from_rot, load_seq_delay=-1.6, name="root_orient", noise=None, symmetry=quat_symmetry, params={})
        
        target_quaternion_diff = RefObsTerm(func=reference_root_quat_from_rot, load_seq_delay=0.02, name="root_orient", noise=None, env_func=reference_quaternion_diff, env_func_params={"history_length": GLOBAL_HISTORY_LENGTH}, symmetry=no_symmetry, params={})
        
        

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
class UnitreeG1NoRefObservationCfg(RefObsCfg):
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
class UnitreeG1Rewards:
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1000.0)
    tracking_lin_vel = RewTerm(
        func=track_lin_vel_xy_yaw_frame_exp_custom,
        weight=3.0,
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
        weight=5.0,
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
    
    # ang_vel_xy_l2 = RewTerm(
    #     func=mdp.ang_vel_xy_l2,
    #     weight=-0.1,
    # )

    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)

    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
    pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
    

    feet_contact = RewTerm(
        func=phase_feet_contact,
        weight=2.0,
        params={
            "period": 0.8,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["l.*ankle.*roll.*", "r.*ankle.*roll.*"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["l.*ankle.*roll.*", "r.*ankle.*roll.*"]),
        }
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-40.0,
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
    
    relative_feet_height = RewTerm(
        func=relative_feet_height,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["l.*ankle.*roll.*", "r.*ankle.*roll.*"]),
            "command_name": "base_velocity"
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
    
    # feet_contact_forces_delta_rate = RewTerm(
    #     func=feet_contact_forces_delta_rate,
    #     weight=-0.2,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["l.*ankle.*roll.*", "r.*ankle.*roll.*"]),
    #     }
    # )

    # oriention_non_violation = RewTerm(
    #     func=reward_orientation_non_violation,
    #     weight=1.0,
    # )

    base_height = RewTerm(
        func=reward_base_height,
        weight=0.2,
    )

    # base_lin_acc = RewTerm(
    #     func=base_lin_acc_exp,
    #     weight=0.2,
    #     params={
    #         "std": 0.6,
    #     },
    # )

    # base_ang_acc_exp = RewTerm(
    #     func=base_ang_acc_exp,
    #     weight=0.1,
    #     params={
    #         "std": 0.4,
    #     },
    # )

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

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle.*roll.*_link.*"),
            "threshold": 0.4,
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*shoulder.*", ".*elbow.*"])},
    )
    joint_deviation_wrist = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*wrist.*")}
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist.*"])},
    )
    survival = RewTerm(func=survival, weight=5.0)
    
    illegal_contact = RewTerm(
        func=mdp.illegal_contact,
        weight=-100.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*pelvis.*", ".*elbow.*", ".*wrist.*", ".*hip.*", ".*shoulder.*", ".*torso.*", ".*knee.*"]), "threshold": 1.0},
    )
    
    # gravity_projection_violation = RewTerm(
    #      func=body_link_gravity_projection_exceed_threshold,
    #         params={
    #             "threshold": 0.3,
    #             "asset_cfg": SceneEntityCfg("robot", body_names=[".*torso.*"]),
    #         },
    #         weight=-20.0,
    # )


@configclass
class UnitreeG1RefRewards(UnitreeG1Rewards):

    ###################################################################
    #
    # Basic joint tracking rewards
    #
    ###################################################################

    
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

    tracking_target_actions_knee = RewTerm(
        func=tracking_target_actions_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[ ".*knee.*"]),
            "method": "mae",
            "std_updater_cfg": {
                "std_list": [0.35, 0.3, 0.28, 0.25, 0.2, 0.15, 0.12, 0.1],
                "reward_threshold": 0.35,
                "reward_key": "tracking_target_actions_knee",
            }
        }
    )
    
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

    
    tracking_base_xyz = RewTerm(
        func=tracking_base_xyz_exp,
        weight=4.0,
        params={
            "std_updater_cfg":{
                "std_list": [1, 0.85, 0.7, 0.55, 0.45, 0.38, 0.33, 0.3],
                "reward_threshold": 0.5,
                "reward_key": "tracking_base_xyz",
            },
        }
    )
    
    # tracking_base_quat = RewTerm(
    #     func=tracking_base_orient_l2,
    #     weight=-0.5,
    # )

    
    tracking_target_lower_body_xyz = RewTerm(
        func=tracking_body_pos_robot_frame,
        weight=1.0,
        params={
            "std": 0.07,
            "xyz_dim": (0, 1, 2),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*knee.*", ".*hip_roll.*", ".*ankle.*"]),
        }
    )

    tracking_target_upper_body_xyz = RewTerm(
        func=tracking_body_pos_robot_frame,
        weight=1.0,
        params={
            "std": 0.07,
            "xyz_dim": (0, 1, 2),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*shoulder.*", ".*wrist.*", ".*elbow.*"]),
        }
    )

    tracking_target_feet_height = RewTerm(
        func=tracking_body_pos_robot_frame,
        weight=1.0,
        params={
            "std": 0.07,
            "xyz_dim": (2),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
        }
    )

    tracking_target_upper_body_orn = RewTerm(
        func=tracking_body_quat_robot_frame,
        weight=0.3,
        params={
            "std": 0.3,
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*shoulder.*", ".*wrist.*", ".*elbow.*"]),
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
    
    tracking_target_quaternion = RewTerm(
        func=tracking_base_quaternion_exp,
        weight=2.0,
        params={
            "std_updater_cfg":{
                "std_list": [0.65, 0.6, 0.58, 0.55, 0.5, 0.45, 0.42, 0.4],
                "reward_threshold": 0.5,
                "reward_key": "tracking_target_quaternion",
            },
        }
    )
    
    tracking_target_quaternion_diff = RewTerm(
        func=tracking_base_quaternion_diff_exp,
        weight=2.0,
        params={
            "std_updater_cfg":{
                "std_list": [0.65, 0.6, 0.58, 0.55, 0.5, 0.45, 0.42, 0.4],
                "reward_threshold": 0.5,
                "reward_key": "tracking_target_quaternion_diff",
            },
        }
    )
    
    tracking_target_heading = RewTerm(
        func=tracking_base_heading_exp,
        weight=5.0,
    )

    ###################################################################
    #
    # contact phase rewards & other rewards
    #
    ###################################################################

    tracking_feet_contact_left = RewTerm(
        func=tracking_feet_contact_phase,
        weight=0.5, # 15.0
        params={
            "ref_id": [0],
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["l.*ankle.*roll.*"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["l.*ankle.*roll.*"]),
        }
    )

    tracking_feet_contact_right = RewTerm(
        func=tracking_feet_contact_phase,
        weight=0.5, # 15.0
        params={
            "ref_id": [1],
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["r.*ankle.*roll.*"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["r.*ankle.*roll.*"]),
        }
    )

    feet_always_contact_penalty = RewTerm(
        func=penalize_one_foot_always_contact,
        weight=-10.0,
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
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*pelvis.*", ".*torso.*"]), "threshold": 1.0},
    ) #".*pelvis.*", ".*elbow.*", ".*wrist.*", ".*hip.*", ".*shoulder.*", ".*torso.*", ".*knee.*"
    
    #"pelvis", ".*elbow.*", ".*wrist.*", ".*hip.*", ".*shoulder.*", ".*torso.*", ".*knee.*
    # base_height_too_low = DoneTerm(
    #     func=root_height_below_minimum_with_reference,
    #     params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("robot")}
    # )
    
    link_gravity_projection_exceed_threshold = DoneTerm(
        func=body_link_gravity_projection_exceed_threshold,
        params={
            "threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*torso.*"]),
        },
    )
    
    # base_gravity_projection_exceed_threshold = DoneTerm(
    #     func=body_link_gravity_projection_exceed_threshold,
    #     params={
    #         "threshold": 0.5,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*pelvis.*"]),
    #     },
    # )
                           



@configclass
class UnitreeG1EventCfg(EventCfg):

    # startup
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis.*"),
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
    
    reset_ref_status = EventTerm(
        func=do_the_fucking_all_three_resets_since_isaaclab_always_reset_joints_first_rather_than_given_order_and_i_dont_know_why,
        mode="reset",
        params={
            "sample_episode_ratio": 1.0,
        }
    )

    # reset_start_time = EventTerm(
    #     func=randomize_initial_start_time,
    #     mode="reset",
    #     params={
    #         "sample_episode_ratio": 1.0,
    #     }
    # )
    
    # reset_base_new =  EventTerm(
    #         func=reset_root_state_by_start_time,
    #         mode="reset",
    # ) #ensure that this is operated after reset_start_time
    
    # reset_robot_joints_new = EventTerm(
    #         func=reset_joints_by_start_time,
    #         mode="reset",
    # ) #ensure that this is operated after reset_start_time

    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(8.0, 12.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )

@configclass
class UnitreeG1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: UnitreeG1ObservationsCfg = UnitreeG1ObservationsCfg()
    ref_observation: UnitreeG1NoRefObservationCfg = UnitreeG1NoRefObservationCfg()

    rewards: UnitreeG1Rewards = UnitreeG1Rewards()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.sim.dt = 0.001
        self.decimation = 20 # set to 20Hz for longer horizon
        self.sim.render_interval = self.decimation
        self.actions.joint_pos.scale = ACTION_SCALE
        self.episode_length_s = 60.0
        # Scene
        self.scene.robot = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.contact_forces.history_length = 16 # keep contact history for better observation
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"

        # Randomization
        # self.events.push_robot = None
        # self.events.add_base_mass = None
        self.events.add_base_mass.params["asset_cfg"].body_names = [".*pelvis"]
        self.events.physics_material.params["static_friction_range"] = (0.8, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.8, 1.0)
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.1)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*pelvis"]
        # self.events.reset_base = None
        # self.events.reset_robot_joints = None
        # self.events.reset_robot_joints = EventTerm(
        #     func=reset_joints_by_start_time,
        #     mode="reset",
        # )
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
class UnitreeG1RoughEnvCfg_PLAY(UnitreeG1RoughEnvCfg):
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
class UnitreeG1RoughRefEnvCfg(UnitreeG1RoughEnvCfg):
    # observations: H1DAGObs = H1DAGObs()
    observations: UnitreeG1ObservationsCfg = UnitreeG1ObservationsCfg()
    ref_observation: UnitreeG1RefObservationCfg = UnitreeG1RefObservationCfg()
    rewards: UnitreeG1RefRewards = UnitreeG1RefRewards()
    events: UnitreeG1EventCfg = UnitreeG1EventCfg()
    # physics_modifiers: PhysicsModifiersCfg = PhysicsModifiersCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base = None
        self.events.reset_robot_joints = None

        # self.events.reset_base = EventTerm(
        #     func=reset_root_state_by_start_time,
        #     mode="reset",
        # )

        # self.events.reset_robot_joints = EventTerm(
        #     func=reset_joints_by_start_time,
        #     mode="reset",
        # )

        self.rewards.base_height = None  # base height is not used in the reference environment
        
        self.episode_length_s = 20.0
        # self.scene.robot.spawn.articulation_props.fix_root_link = True

@configclass
class UnitreeG1RoughRefEnvCfg_PLAY(UnitreeG1RoughEnvCfg_PLAY):
    observations: UnitreeG1ObservationsCfg = UnitreeG1ObservationsCfg()
    ref_observation: UnitreeG1RefObservationCfg = UnitreeG1RefObservationCfg()
    rewards: UnitreeG1RefRewards = UnitreeG1RefRewards()
    events: UnitreeG1EventCfg = UnitreeG1EventCfg()
    # events: H1EventCfg = H1EventCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base = None
        self.events.reset_robot_joints = None
