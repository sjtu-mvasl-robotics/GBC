from GBC.gyms.isaaclab_45.lab_assets.turing import TURING_CFG


from isaaclab.assets import Articulation # , RigidObject
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, quat_from_matrix, matrix_from_quat, combine_frame_transforms, subtract_frame_transforms
from GBC.utils.data_preparation.robot_flip_left_right import RobotFlipLeftRight

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    ObservationsCfg,
    ObsGroup,
    ObsTerm,
)

from GBC.gyms.isaaclab_45.managers.ref_obs_term_cfg import ReferenceObservationTermCfg as RefObsTerm
from GBC.gyms.isaaclab_45.managers.ref_obs_term_cfg import ReferenceObservationCfg as RefObsCfg
from GBC.gyms.isaaclab_45.managers.ref_obs_term_cfg import ReferenceObservationGroupCfg as RefObsGroup

from isaaclab.envs import ManagerBasedRLEnv
from GBC.gyms.isaaclab_45.envs import ManagerBasedRefRLEnv
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from GBC.utils.base.base_fk import RobotKinematics
from GBC.utils.base.math_utils import angle_axis_to_quaternion, rot_mat_to_vec, rot_vec_to_mat, contact_to_phase, is_foot_parallel_from_rot_matrix
from isaaclab.utils.math import quat_error_magnitude, quat_from_matrix, quat_inv, quat_mul, quat_apply, euler_xyz_from_quat
from GBC.gyms.isaaclab_45.managers.physics_modifier_function_wrapper import update
import torch
from collections import deque
import sys
import os
from .curriculum_utils import StdUpdater
import numpy as np

###################################################################################################

# symmetry functions

###################################################################################################

def dim_symmetry(env: ManagerBasedRLEnv, inputs: torch.Tensor, symmetry_dims: list[int], history_length: int = 1, **kwargs) -> torch.Tensor:
    """Symmetry the inputs by the dimensions."""
    sym_inputs = inputs.clone()
    sym_inputs = sym_inputs.reshape(-1, history_length, inputs.shape[-1] // history_length)
    sym_inputs[..., symmetry_dims] = sym_inputs[..., symmetry_dims] * -1
    sym_inputs = sym_inputs.reshape(inputs.shape)
    return sym_inputs

def actions_symmetry(env: ManagerBasedRLEnv, inputs: torch.Tensor, flipper: RobotFlipLeftRight, is_velocity: bool = True, history_length: int = 1, **kwargs) -> torch.Tensor:
    """Symmetry the actions and action related terms.
    
    This symmetry swaps the left and right actuator values.
    """
    # Handle case where flipper might be a class instead of instance (due to serialization issues)
    # Store the created instance to avoid repeated construction
    if isinstance(flipper, type):
        if not hasattr(actions_symmetry, '_flipper_instance'):
            actions_symmetry._flipper_instance = flipper()
        flipper = actions_symmetry._flipper_instance
    
    if not flipper.prepared:
        flipper.prepare_flip_joint_ids(dof_names=env.unwrapped.scene.articulations['robot'].joint_names)
    sym_inputs = inputs.clone()
    sym_inputs = sym_inputs.reshape(-1, history_length, inputs.shape[-1] // history_length)
    # sym_inputs_orig = sym_inputs.clone()
    sym_inputs = flipper.flip(q=sym_inputs, is_velocity=is_velocity)
    # sym_inputs[..., actions_symmetry.symmetry_pair[0]] = sym_inputs_orig[..., actions_symmetry.symmetry_pair[1]]
    # sym_inputs[..., actions_symmetry.symmetry_pair[1]] = sym_inputs_orig[..., actions_symmetry.symmetry_pair[0]]

    sym_inputs = sym_inputs.reshape(inputs.shape)
    return sym_inputs

def pos_symmetry(env: ManagerBasedRLEnv, inputs: torch.Tensor, history_length: int = 1, **kwargs) -> torch.Tensor:
    """Symmetry the position and position related terms.

    This symmetry keeps original x and z values, and multiplies y by -1.
    """
    # assert inputs.shape[-1] == 3, "Position-wise symmetry only supports data in form (..., 3), but got shape {}".format(inputs.shape)
    sym_inputs = inputs.clone()
    sym_inputs = sym_inputs.reshape(-1, history_length, inputs.shape[-1] // history_length)
    sym_inputs[..., 1] = -sym_inputs[..., 1]
    sym_inputs = sym_inputs.reshape(inputs.shape)
    return sym_inputs

def rot_symmetry(env: ManagerBasedRLEnv, inputs: torch.Tensor, history_length: int = 1, **kwargs) -> torch.Tensor:
    """Symmetry the rotation and rotation related terms.

    This symmetry keeps original y value, and multiplies x and z by -1.
    """
    # assert inputs.shape[-1] == 3, "Rotation-wise symmetry only supports data in form (..., 3), but got shape {}".format(inputs.shape)
    sym_inputs = inputs.clone()
    sym_inputs = sym_inputs.reshape(-1, history_length, inputs.shape[-1] // history_length)
    sym_inputs[..., 0] = -sym_inputs[..., 0]
    sym_inputs[..., 2] = -sym_inputs[..., 2]
    sym_inputs = sym_inputs.reshape(inputs.shape)
    return sym_inputs



def symmetry_by_ref_term_name(env: ManagerBasedRefRLEnv, inputs: torch.Tensor, term_name: str, **kwargs) -> torch.Tensor:
    """Symmetry the inputs by the term name.
    
    This function is used to symmetry the inputs by query the term name from reference observation manager.
    """
    ref_obs, mask = env.ref_observation_manager.get_term(term_name)
    if mask is None:
        mask = torch.zeros_like(ref_obs[..., 0], dtype=torch.bool)
    sym_inputs = torch.where(mask.unsqueeze(-1), inputs, ref_obs)
    return sym_inputs

def symmetry_by_term_name(env: ManagerBasedRLEnv, inputs: torch.Tensor, term_name: str, **kwargs) -> torch.Tensor:
    """Symmetry the inputs by the term name.
    
    NVIDIA, IF YOU ARE READING THIS, PLEASE ADD GET_TERM IMPLEMENTATION TO OBSERVATION MANAGER.
    """
    raise NotImplementedError("Symmetry by term name is not implemented for ManagerBasedRLEnv.")


def get_ref_observation_symmetry(env: ManagerBasedRefRLEnv, ref_observations: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the symmetry of the reference observation."""
    return env.unwrapped.ref_observation_manager.compute_policy_symmetry(ref_observations)
###################################################################################################

# tool functions

###################################################################################################

        
def get_phase(env: ManagerBasedRefRLEnv, period: float = 0.8, offset: float = 0.0, ref_name: str | None = None) -> torch.Tensor:
    """Get the phase of the robot. The phase is calculated as the time since the start of the episode divided by the period, and then wrapped to [0, 1]."""
        
    ref_phase = None
    if hasattr(env.unwrapped, "episode_length_buf"):
        cur_time = env.episode_length_buf.unsqueeze(1).to(torch.float32) * env.step_dt
        try:
            if hasattr(env.unwrapped, "ref_observation_manager") and ref_name is not None:
                ref_phase, mask = env.unwrapped.ref_observation_manager.get_term(ref_name)
        except:
            pass
    else:
        # episode_length_buf unavailable at initializing observations
        cur_time = torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)
    phase = cur_time % period / period
    phase = (phase + offset) % 1
    
    
    phase = torch.sin(phase * 2 * torch.pi)
    if ref_phase is not None:
        phase = torch.where(mask.unsqueeze(1), ref_phase, phase)
    return phase
    

###################################################################################################

# reward functions for unitree_rl_gym

###################################################################################################

def joint_pos_l2(env: ManagerBasedRefRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_pos), dim=1)
    
def phase_feet_contact(env, period: float, sensor_cfg: SceneEntityCfg, contact_phase_thresh: float = 0.55, contact_time_thresh = 0.01) -> torch.Tensor:
    """Reward the agent for tracking the target feet contact."""
    if hasattr(env.unwrapped, "episode_length_buf"):
        cur_time = env.episode_length_buf.to(torch.float32) * env.step_dt
    else:
        # episode_length_buf unavailable at initializing observations
        cur_time = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    leg_phase = torch.stack([
        cur_time % period / period,
        (cur_time + period / 2) % period / period,
    ])
    leg_phase = leg_phase.transpose(0, 1)
    assert leg_phase.shape == (env.num_envs, 2)
    target_feet_contact = leg_phase < contact_phase_thresh

    contact_sensor = env.scene.sensors[sensor_cfg.name]
    has_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > contact_time_thresh
    feet_link_quat = env.scene[asset_cfg.name].data.body_state_w[:, asset_cfg.body_ids, 3:7]
    lft_feet_quat = feet_link_quat[:, 0]
    rht_feet_quat = feet_link_quat[:, 1]
    lft_feet_rot_mat = matrix_from_quat(lft_feet_quat).unsqueeze(1)
    rht_feet_rot_mat = matrix_from_quat(rht_feet_quat).unsqueeze(1)
    feet_rot_mat = torch.cat([lft_feet_rot_mat, rht_feet_rot_mat], dim=1)
    is_parallel = is_foot_parallel_from_rot_matrix(feet_rot_mat, tolerance_deg=10)
    has_contact &= is_parallel
    # has_contact &= (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 5.0 * contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2].norm(dim=-1))
    reward = (target_feet_contact == has_contact)
    return torch.sum(reward.float(), dim=1)

def contact_no_vel(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, contact_time_thresh: float = 0.01) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    has_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > contact_time_thresh

    asset = env.scene[asset_cfg.name]
    feet_vel = asset.data.body_state_w[:, asset_cfg.body_ids, 7:10]
    contact_feet_vel = feet_vel * has_contact.unsqueeze(-1)
    penalize = torch.square(contact_feet_vel[:, :, :3])
    return torch.sum(penalize, dim=(1, 2))

def feet_swing_height(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, contact_time_thresh: float = 0.01) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    has_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > contact_time_thresh

    asset = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_state_w[:, asset_cfg.body_ids, :3]
    pos_error = torch.square(feet_pos[:, :, 2] - 0.08) * ~has_contact
    return torch.sum(pos_error, dim=1)


def root_body_yaw_diff_l1(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    root_orn = yaw_quat(asset.data.root_state_w[:, 3:7]).unsqueeze(1).repeat(1, len(asset_cfg.body_ids), 1)
    body_orn = yaw_quat(asset.data.body_state_w[:, asset_cfg.body_ids, 3:7])
    orn_diff = quat_error_magnitude(root_orn, body_orn)
    return torch.mean(orn_diff, dim=1)

###################################################################################################

# reference input reshaping functions

###################################################################################################

def reference_action_reshape(inputs, env: ManagerBasedRLEnv, urdf_path: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), pickle_cfg = {}, add_default_joint_pos = False):
    """Reshape the reference actions to the action space of the robot."""
    robot_kinematics = RobotKinematics(urdf_path=urdf_path, device=env.device)
    orig_order = robot_kinematics.get_dof_names()
    targ_order = env.scene[asset_cfg.name].joint_names
    inputs = inputs[:, [orig_order.index(j) for j in targ_order]]
    if add_default_joint_pos:
        default_joint_pos = env.scene[asset_cfg.name].data.default_joint_pos
        inputs = inputs + default_joint_pos[0] # take first dimension since it is the same for all dimensions
    return inputs

def reference_action_std(inputs, env: ManagerBasedRLEnv, urdf_path: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), pickle_cfg = {}, add_default_joint_pos = False):
    """Calculate the std of reshaped actions"""
    actions = reference_action_reshape(inputs, env, urdf_path, asset_cfg, pickle_cfg, add_default_joint_pos)
    return torch.std(actions, dim=0)

def reference_rotation_refine(input, env: ManagerBasedRLEnv, pikle_cfg = {}):
    """Refine the reference rotations."""
    input = rot_vec_to_mat(input)
    rot_0 = input[:, 0]
    input = torch.einsum("ij, bjk -> bik", rot_0.transpose(0, 1), input) # set the first rotation to original direction
    return input

def reference_action_velocity(input, env: ManagerBasedRLEnv, urdf_path: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), pickle_cfg = {}):
    reshaped_actions = reference_action_reshape(input, env, urdf_path, asset_cfg, pickle_cfg)
    action_vel = torch.cat([torch.zeros_like(reshaped_actions[0:1, :]), torch.diff(reshaped_actions, dim=0)], dim=0) * pickle_cfg["fps"]
    return action_vel


def reference_link_poses(actions, env: ManagerBasedRLEnv, urdf_path: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), **kwargs):
    robot_kinematics = RobotKinematics(urdf_path=urdf_path, device=env.device)

    # The actions are directly extracted from pickle, so no need to change order
    tfs = robot_kinematics.forward_kinematics(actions)
    # But change the order of links to USD
    link_names = env.scene[asset_cfg.name].body_names
    tfs = torch.cat([tfs[name].unsqueeze(1) for name in link_names], dim=1)
    assert tfs.shape == (actions.shape[0], len(link_names), 4, 4)

    return torch.cat([tfs[..., :3, 3], quat_from_matrix(tfs[..., :3, :3])], dim=-1)

def reference_feet_contact_phase(input, env: ManagerBasedRLEnv, index: int, offset: float = 0.0, threshold: float = 0.55, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), pickle_cfg = {}):
    """Get the feet contact phase from the reference data."""
    contact = input[:, index]
    phase = contact_to_phase(contact, threshold=threshold)
    phase = (phase + offset) % 1
    return torch.sin(phase * 2 * torch.pi).unsqueeze(1)

def reference_link_velocities(data, env: ManagerBasedRLEnv, urdf_path: str, pickle_cfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), **kwargs):
    link_poses = reference_link_poses(data["actions"], env, urdf_path, asset_cfg)
    link_pos, link_quat = link_poses[..., :3], link_poses[..., 3:7]
    root_pos = data["trans"]
    root_quat = quat_from_matrix(rot_vec_to_mat(data["root_orient"]))
    len_links = link_pos.shape[1]
    root_pos_rep = root_pos.unsqueeze(1).repeat(1, len_links, 1)
    root_quat_rep = root_quat.unsqueeze(1).repeat(1, len_links, 1)

    link_pos, link_quat = combine_frame_transforms(
        root_pos_rep, root_quat_rep,
        link_pos, link_quat,
    )

    last_link_pos = torch.cat([link_pos[:1, ...], link_pos[:-1, ...]], dim=0)
    last_link_quat = torch.cat([link_quat[:1, ...], link_quat[:-1, ...]], dim=0)

    diff_link_pos = link_pos - last_link_pos
    diff_link_quat = quat_mul(quat_inv(last_link_quat), link_quat)
    diff_link_euler = torch.stack(euler_xyz_from_quat(diff_link_quat.reshape(-1, 4))).T.reshape(-1, len_links, 3)
    # Put them in the range of [-pi, pi)
    times = (diff_link_euler / (2 * torch.pi) + 0.5).long()
    diff_link_euler -= times * 2 * torch.pi
    assert torch.all(torch.abs(diff_link_euler) <= torch.pi + 1e-5)

    link_lin_vel = quat_apply(quat_inv(root_quat_rep), diff_link_pos * pickle_cfg["fps"])
    link_ang_vel = quat_apply(root_quat_rep, diff_link_euler * pickle_cfg["fps"])
    return torch.cat([link_lin_vel, link_ang_vel], dim=-1)


###################################################################################################

# custom observation functions (corresponding to the custom observation terms)

###################################################################################################

def reference_base_lin_vel(env: ManagerBasedRefRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    base_lin_vel_b = asset.data.root_lin_vel_b
    reference_lin_vel, ref_lin_vel_mask = env.ref_observation_manager.get_term("base_lin_vel")
    reference_lin_vel = torch.where(ref_lin_vel_mask.unsqueeze(1), reference_lin_vel, base_lin_vel_b)
    return reference_lin_vel

def reference_base_ang_vel(env: ManagerBasedRefRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    base_ang_vel_b = asset.data.root_ang_vel_b
    reference_ang_vel, ref_ang_vel_mask = env.ref_observation_manager.get_term("base_ang_vel")
    reference_ang_vel = torch.where(ref_ang_vel_mask.unsqueeze(1), reference_ang_vel, base_ang_vel_b)
    return reference_ang_vel

def reference_projected_gravity(env: ManagerBasedRefRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    projected_gravity_b = asset.data.projected_gravity_b
    reference_projected_gravity, ref_projected_gravity_mask = env.ref_observation_manager.get_term("target_projected_gravity")
    reference_projected_gravity = torch.where(ref_projected_gravity_mask.unsqueeze(1), reference_projected_gravity, projected_gravity_b)
    return reference_projected_gravity


def feet_height_flat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Observation of the height of the feet above the ground (under flat terrain)."""
    asset = env.scene[asset_cfg.name]
    feet_height = asset.data.body_state_w[:, asset_cfg.body_ids, 2]
    return feet_height


def survival(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Survival reward."""
    return torch.ones(env.num_envs).to(env.device)


def last_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Observation of the last velocity of the robot."""
    asset = env.scene[asset_cfg.name]
    return asset.data.linear_velocity_w

def ref_action_diff(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    ref_actions, mask = env.ref_observation_manager.get_term("target_actions")
    if mask is None:
        mask = torch.zeros_like(ref_actions[:, 0], dtype=torch.bool)
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    ref_actions = ref_actions[:, asset_cfg.joint_ids]
    ref_actions_diff = torch.where(mask.unsqueeze(1), ref_actions - joint_pos, torch.zeros_like(ref_actions))
    return ref_actions_diff

# def do_nothing(env, command_name: str) -> torch.Tensor:
#     """Do nothing."""
#     return torch.zeros(env.num_envs, 1)

def base_lin_vel_hist(env, stack_size: int, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    
    asset = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b
    if not hasattr(base_lin_vel_hist, "lin_vel_hist"):
        base_lin_vel_hist.lin_vel_hist = deque(maxlen=stack_size)
        for _ in range(stack_size):
            base_lin_vel_hist.lin_vel_hist.append(torch.zeros_like(lin_vel).to(env.device))
    base_lin_vel_hist.lin_vel_hist.append(lin_vel)
    return torch.concatenate(list(base_lin_vel_hist.lin_vel_hist), dim=1)

def base_ang_vel_hist(env, stack_size: int, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_ang_vel_b
    if not hasattr(base_ang_vel_hist, "ang_vel_hist"):
        base_ang_vel_hist.ang_vel_hist = deque(maxlen=stack_size)
        for _ in range(stack_size):
            base_ang_vel_hist.ang_vel_hist.append(torch.zeros_like(ang_vel).to(env.device))
    base_ang_vel_hist.ang_vel_hist.append(ang_vel)
    return torch.concatenate(list(base_ang_vel_hist.ang_vel_hist), dim=1)

def projected_gravity_hist(env, stack_size: int, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b
    if not hasattr(projected_gravity_hist, "projected_gravity_hist"):
        projected_gravity_hist.projected_gravity_hist = deque(maxlen=stack_size)
        for _ in range(stack_size):
            projected_gravity_hist.projected_gravity_hist.append(torch.zeros_like(projected_gravity).to(env.device))
    projected_gravity_hist.projected_gravity_hist.append(projected_gravity)
    return torch.concatenate(list(projected_gravity_hist.projected_gravity_hist), dim=1)


def joint_pos_hist(env, stack_size: int, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    if not hasattr(joint_pos_hist, "joint_pos_hist"):
        joint_pos_hist.joint_pos_hist = deque(maxlen=stack_size)
        for _ in range(stack_size):
            joint_pos_hist.joint_pos_hist.append(torch.zeros_like(joint_pos).to(env.device))
    joint_pos_hist.joint_pos_hist.append(joint_pos)
    return torch.concatenate(list(joint_pos_hist.joint_pos_hist), dim=1)

def joint_vel_hist(env, stack_size: int, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if not hasattr(joint_vel_hist, "joint_vel_hist"):
        joint_vel_hist.joint_vel_hist = deque(maxlen=stack_size)
        for _ in range(stack_size):
            joint_vel_hist.joint_vel_hist.append(torch.zeros_like(joint_vel).to(env.device))
    joint_vel_hist.joint_vel_hist.append(joint_vel)
    return torch.concatenate(list(joint_vel_hist.joint_vel_hist), dim=1)

def control_hist(env, stack_size: int, action_name: str | None = None):
    if action_name is None:
        action =  env.action_manager.action
    else:
        action = env.action_manager.get_term(action_name).raw_actions

    if not hasattr(control_hist, "control_hist"):
        control_hist.control_hist = deque(maxlen=stack_size)
        for _ in range(stack_size):
            control_hist.control_hist.append(torch.zeros_like(action).to(env.device))
    control_hist.control_hist.append(action)
    return torch.concatenate(list(control_hist.control_hist), dim=1)



###################################################################################################

# Event functions

###################################################################################################

def randomize_initial_start_time(
        env: ManagerBasedRefRLEnv,
        env_ids: torch.Tensor,
        sample_episode_ratio: float,
):
    """Randomize the start sampling time for reference observation.

    This function can be used for `ManagerBasedRefRLEnv` only, since original `ManagerBasedRLEnv` does not support reference observation manager.

    This function accepts a ratio between 0 and 1. For sampled environments, the imitation reads the data sequence starting from episode_length * ratio.
    """
    if not isinstance(env, ManagerBasedRefRLEnv):
        raise RuntimeError("This function can only be used for `ManagerBasedRefRLEnv`.")
    epi_len = env.max_episode_length_s # in seconds
    # randomized_sample_time = torch.random.uniform(0, 1, size=(env.num_envs,), device=env.device) * epi_len * (1 - sample_episode_ratio) + epi_len * sample_episode_ratio
    randomized_sample_time = torch.rand(env.num_envs, device=env.device) * epi_len * sample_episode_ratio
    env.ref_observation_manager.start_time[env_ids] = randomized_sample_time[env_ids]

def reset_root_state_by_start_time(
        env: ManagerBasedRefRLEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        drop_z: bool = True, # drop z axis, since converted translation z from AMASS/HUMANML/MotionX is not accurate
):
    """Reset the root state of the robot by start time.
    
    This replaces the `reset_base` functions in velocity_env_cfg. Resetting robot initial position according to the start time and reference data.

    This term must be called **AFTER** `randomize_initial_start_time` function.

    While registering terms, make sure to register initial_start_first.
    """
    asset = env.scene[asset_cfg.name]
    # get original root state
    original_root_states = asset.data.default_root_state[env_ids].clone()
    # get reference root pos at start time
    base_pose_w, _ = env.ref_observation_manager.compute_term("target_base_pose", torch.zeros(env.num_envs, dtype=torch.float32).to(env.device))
    root_lin_velocities, _ = env.ref_observation_manager.compute_term("base_lin_vel", torch.zeros(env.num_envs, dtype=torch.float32).to(env.device))
    root_ang_velocities, mask = env.ref_observation_manager.compute_term("base_ang_vel", torch.zeros(env.num_envs, dtype=torch.float32).to(env.device))
    # root_orientations = angle_axis_to_quaternion(root_orientations)
    # root_positions += env.scene.env_origins

    root_positions = base_pose_w[:, :3] + env.scene.env_origins
    root_orientations = base_pose_w[:, 3:7]
    
    root_mask = mask[env_ids]
    root_positions = root_positions[env_ids]
    root_positions = torch.where(root_mask.unsqueeze(1), root_positions, original_root_states[:, :3])
    if drop_z:
        root_positions[:, 2] = original_root_states[:, 2]
    root_orientations = root_orientations[env_ids]
    root_orientations = torch.where(root_mask.unsqueeze(1), root_orientations, original_root_states[:, 3:7])
    root_lin_velocities = root_lin_velocities[env_ids]
    root_lin_velocities = torch.where(root_mask.unsqueeze(1), root_lin_velocities, original_root_states[:, 7:10])
    root_ang_velocities = root_ang_velocities[env_ids]
    root_ang_velocities = torch.where(root_mask.unsqueeze(1), root_ang_velocities, original_root_states[:, 10:13])
    # set the root state
    asset.write_root_pose_to_sim(torch.cat([root_positions, root_orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.cat([root_lin_velocities, root_ang_velocities], dim=-1), env_ids=env_ids)

    
def reset_joints_by_start_time(
        env: ManagerBasedRefRLEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        add_joint_vel: bool = True,
):
    """Reset the joints of the robot by start time.
    
    This replaces the `reset_joint` functions in velocity_env_cfg. Resetting robot initial position according to the start time and reference data.

    This term must be called **AFTER** `randomize_initial_start_time` function.

    While registering terms, make sure to register initial_start_first.
    """
    asset = env.scene[asset_cfg.name]
    # get original joint pos and vel
    original_joint_pos = asset.data.default_joint_pos[env_ids].clone()
    original_joint_vel = asset.data.default_joint_vel[env_ids].clone()
    # get reference joint pos and vel at start time
    joint_pos, mask = env.ref_observation_manager.compute_term("target_actions", torch.zeros(env.num_envs, dtype=torch.float32).to(env.device))
    # joint_vel, mask = env.ref_observation_manager.compute_term("target_joint_velocities", torch.zeros(env.num_envs, dtype=torch.float32).to(env.device))
    joint_pos = joint_pos[env_ids]
    joint_pos = torch.where(mask[env_ids].unsqueeze(1), joint_pos, original_joint_pos)
    # joint_vel = joint_vel[env_ids]
    # if add_joint_vel:
    #     joint_vel = torch.where(mask[env_ids].unsqueeze(1), joint_vel, original_joint_vel)
    # else:
    #     joint_vel = original_joint_vel
    joint_vel = original_joint_vel
    joint_pos_limits = asset.data.joint_limits[0]
    joint_pos = joint_pos.clamp_(joint_pos_limits[:, 0], joint_pos_limits[:, 1])

    joint_vel_limits = asset.data.joint_velocity_limits[0]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


###################################################################################################

# physics modifier functions

###################################################################################################

class ExternalForceUpdater:
    def __init__(self):
        self.history_base_contact = deque(maxlen=320)
        self.last_update_cnt = 0

    def add_base_contact_val(self, val):
        self.history_base_contact.append(val)

    def get_base_contact_avg(self):
        return sum(self.history_base_contact) / len(self.history_base_contact)

    def update_external_z_force_base(self, current_overrides: dict, new_kwargs: dict) -> dict:
        env = new_kwargs["env"]
        
        # fill your update strategy here
        apply_offset_range = new_kwargs["apply_offset_range"]
        if "apply_offset_range" in current_overrides:
            apply_offset_range = current_overrides["apply_offset_range"]

        base_contact = env.unwrapped.extras["log"]["Episode_Termination/base_contact"]
        self.add_base_contact_val(base_contact)
        base_contact = self.get_base_contact_avg()
        self.last_update_cnt += 1

        update_threshold = new_kwargs["update_threshold"]
        update_ratio = new_kwargs["update_ratio"]
        offset_bound = new_kwargs["apply_offset_range_bound"]
        update_interval = new_kwargs["update_interval"]

        if self.last_update_cnt >= update_interval:
            if base_contact < update_threshold:
                apply_offset_range *= update_ratio
            else:
                apply_offset_range /= update_ratio
            if apply_offset_range > offset_bound[1]:
                apply_offset_range = offset_bound[1]
            if apply_offset_range < offset_bound[0]:
                apply_offset_range = offset_bound[0]

            print("Current apply_offset_range:", apply_offset_range)
            self.last_update_cnt = 0

        current_overrides["apply_offset_range"] = apply_offset_range
        return current_overrides

@update(update_strategy=ExternalForceUpdater().update_external_z_force_base)
def external_z_force_base(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor,
    max_force: float,
    apply_offset_range: float,
    apply_force_duration_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    working_mode: str = "spring",
    ref_height_offset: float = 0.15,                          
    update_threshold: float = 1.1,
    update_ratio: float = 1.05,
    update_interval: int = 320,
    apply_offset_range_bound: tuple[float] = (0.01, 1.05),
):
    """Apply external z direction force to the base of the robot.
    
    This function is used for curriculum learning for robot imitation tasks. The external force is applied when base height is lower than reference height (ref_height - z > apply_offset_range). The force is applied on z axis to prevent the robot from falling down, like a toddler's baby walker. The force can work in two modes: "spring" and "constant", described as follows:
    * Spring mode:
        The force is applied as a spring force. The force is calculated as:
        F = -k * (z - ref_height) if 0.1 * apply_offset_range < (ref_height - z) < apply_offset_range, max_force if (ref_height - z) >= apply_offset_range, 0 if (ref_height - z) <= 0. k = max_force / apply_offset_range.
    * Constant mode:
        The force is applied as a constant force. The force is calculated as:
        F = max_force if ref_height - z > apply_offset_range, 0 otherwise.
        
    This force is applied to the robot only during episode_length_buf < episode_length * apply_force_duration_ratio.
    
    While using this function, we recomment to shorten the apply_force_duration_ratio and widen the apply_offset_range as the curriculum progresses, these two parameters can be updated using the `update_external_z_force_base` function.
    """
    assert working_mode in ["spring", "constant"], f"Working mode {working_mode} is not supported. Expected 'spring' or 'constant'."
    assert apply_force_duration_ratio <=1 and apply_force_duration_ratio >= 0, f"apply_force_duration_ratio should be in [0, 1]."
    assert max_force >= 0, f"max_force should be non-negative."
    assert apply_offset_range >= 0, f"apply_offset_range should be non-negative."
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
        
    root_states = asset.data.root_state_w[env_ids].clone()
    root_height = root_states[:, 2]
    ref_pos, _ = env.ref_observation_manager.compute_term("target_base_pose", torch.zeros(env.num_envs, dtype=torch.float32).to(env.device))
    ref_height = ref_pos[env_ids, 2]
    ref_height += ref_height_offset
    height_offset = root_height - ref_height
    height_offset = torch.where(height_offset < 0, height_offset, torch.zeros_like(height_offset)) # only apply force when height is lower than reference height
    if working_mode == "spring":
        k = max_force / apply_offset_range
        z_force = torch.where((0.1 * apply_offset_range < torch.abs(height_offset)), -k * height_offset, torch.zeros_like(height_offset))
        z_force = torch.clamp_max(z_force, max_force)
    else:
        z_force = torch.where(torch.abs(height_offset) > apply_offset_range, max_force, torch.zeros_like(height_offset))
        
    # z force shape: (env_ids, ) -> (env_ids, body_ids, 3)
    
    cur_time = (env.episode_length_buf.to(torch.float32)) * env.step_dt
    cur_time = cur_time[env_ids]
    z_force = torch.where(cur_time < env.max_episode_length_s * apply_force_duration_ratio, z_force, torch.zeros_like(z_force))
    
    forces = torch.zeros((len(env_ids), len(asset_cfg.body_ids), 3), device=env.device)
    forces[:, :, 2] = z_force.unsqueeze(1).repeat(1, len(asset_cfg.body_ids))

    tf_mat = matrix_from_quat(root_states[:, 3:7]).transpose(1, 2)
    forces = torch.einsum("ijk, ilk -> ilj", tf_mat, forces)

    torque = torch.zeros_like(forces)
    asset.set_external_force_and_torque(forces=forces, torques=torque, body_ids=asset_cfg.body_ids, env_ids=env_ids)
        


###################################################################################################

# reference observation reward functions

###################################################################################################

# lin_vel_std_updater = StdUpdater(
#     std_list=[0.35, 0.34, 0.33, 0.25, 0.2],
#     reward_threshold=0.7,
#     reward_key="track_lin_vel_xy_exp"
# )


def track_lin_vel_xy_yaw_frame_exp_custom(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std_updater_cfg = None
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    if not hasattr(track_lin_vel_xy_yaw_frame_exp_custom, "std_updater_dict"):
        track_lin_vel_xy_yaw_frame_exp_custom.std_updater_dict = {}

    if std_updater_cfg is not None:
        std_updater_dict = track_lin_vel_xy_yaw_frame_exp_custom.std_updater_dict
        key = std_updater_cfg["reward_key"]
        if key not in std_updater_dict:
            std_updater_dict[key] = StdUpdater(**std_updater_cfg)
        std = std_updater_dict[key].update(env)

    # std=lin_vel_std_updater.update(env)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


# ang_vel_std_updater = StdUpdater(
#     std_list=[0.5, 0.4, 0.33, 0.25],
#     reward_threshold=0.75,
#     reward_key="track_ang_vel_z_exp"
# )

def track_ang_vel_z_world_exp_custom(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std_updater_cfg = None
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    # std = ang_vel_std_updater.update(env)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])

    if not hasattr(track_ang_vel_z_world_exp_custom, "std_updater_dict"):
        track_ang_vel_z_world_exp_custom.std_updater_dict = {}
    
    if std_updater_cfg is not None:
        std_updater_dict = track_ang_vel_z_world_exp_custom.std_updater_dict
        key = std_updater_cfg["reward_key"]
        if key not in std_updater_dict:
            std_updater_dict[key] = StdUpdater(**std_updater_cfg)
        std = std_updater_dict[key].update(env)

    return torch.exp(-ang_vel_error / std**2)


def tracking_target_actions_exp(env, std: float = 0.5, type = None, method = "norm", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std_updater_cfg = None) -> torch.Tensor:
    """Reward the agent for tracking the target actions.
    
    **Note**: Target actions are now implemented in absolute joint angles instead of relative joint angles, therefore, current action is measured in absolute joint angles.
    """
    if not hasattr(env, "ref_observation_manager"):
        raise RuntimeError("Reference observation manager is not available, function `tracking_target_actions_exp` cannot be called.")
    if not hasattr(tracking_target_actions_exp, "std_updater_dict"):
        tracking_target_actions_exp.std_updater_dict = {}

    # cur_time = (env.episode_length_buf.to(torch.float32) - 1) * env.step_dt
    actions, mask = env.ref_observation_manager.get_term("target_actions")
    if mask is None:
        mask = torch.zeros(actions.shape[0], dtype=torch.bool).to(actions.device)
    actions = actions[:, asset_cfg.joint_ids]
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    actions_diff = torch.where(mask.unsqueeze(1), actions - joint_pos, torch.zeros_like(actions))
    if method == "norm":
        actions_error = torch.norm(actions_diff, dim=1)
    else:
        actions_error = torch.mean(torch.abs(actions_diff), dim=1)
    if std_updater_cfg is not None:
        std_updater_dict = tracking_target_actions_exp.std_updater_dict
        key = std_updater_cfg["reward_key"]
        if key not in std_updater_dict:
            std_updater_dict[key] = StdUpdater(**std_updater_cfg)
        std = std_updater_dict[key].update(env)
    return torch.exp(-actions_error / std**2)

def tracking_target_actions_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    actions, mask = env.ref_observation_manager.get_term("target_actions")
    actions = actions[:, asset_cfg.joint_ids]
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    actions_diff = torch.where(mask.unsqueeze(1), actions - joint_pos, torch.zeros_like(actions))
    return torch.norm(actions_diff, dim=1)

def unbalanced_tracking_left_right(env, left_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), right_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sigma: float = 0.4, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    actions, mask = env.ref_observation_manager.get_term("target_actions")
    asset = env.scene[asset_cfg.name]
    left_actions = actions[:, left_asset_cfg.joint_ids]
    right_actions = actions[:, right_asset_cfg.joint_ids]
    left_joint_pos = asset.data.joint_pos[:, left_asset_cfg.joint_ids]
    right_joint_pos = asset.data.joint_pos[:, right_asset_cfg.joint_ids]
    left_actions_diff = torch.where(mask.unsqueeze(1), left_actions - left_joint_pos, torch.zeros_like(left_actions))
    right_actions_diff = torch.where(mask.unsqueeze(1), right_actions - right_joint_pos, torch.zeros_like(right_actions))
    left_reward = torch.exp(-torch.norm(left_actions_diff, dim=1) / sigma**2)
    right_reward = torch.exp(-torch.norm(right_actions_diff, dim=1) / sigma**2)
    unbalance_ratio = (left_reward - right_reward) / (left_reward + right_reward) + 1e-3
    return torch.abs(unbalance_ratio)

def tracking_target_actions_normalized_exp(
    env,
    std: float = 0.5,
    type = None,
    method = "norm",
    actions_std_add_constant: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std_updater_cfg = None):
    """Reward the agent for tracking the target actions (normalized with the action std).
    
    **Note**: Target actions are now implemented in absolute joint angles instead of relative joint angles, therefore, current action is measured in absolute joint angles.
    """
    if not hasattr(env, "ref_observation_manager"):
        raise RuntimeError("Reference observation manager is not available, function `tracking_target_actions_normalized_exp` cannot be called.")
    if not hasattr(tracking_target_actions_normalized_exp, "std_updater_dict"):
        tracking_target_actions_normalized_exp.std_updater_dict = {}


    actions, mask = env.ref_observation_manager.get_term("target_actions")
    actions_std, _ = env.ref_observation_manager.get_term("target_actions_std")
    if mask is None:
        mask = torch.zeros(actions.shape[0], dtype=torch.bool).to(actions.device)

    actions = actions[:, asset_cfg.joint_ids]
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    actions_diff = torch.where(mask.unsqueeze(1), actions - joint_pos, torch.zeros_like(actions))
    actions_diff /= actions_std[:, asset_cfg.joint_ids] + actions_std_add_constant

    if method == "norm":
        actions_error = torch.norm(actions_diff, dim=1)
    else:
        actions_error = torch.mean(torch.abs(actions_diff), dim=1)
    if std_updater_cfg is not None:
        std_updater_dict = tracking_target_actions_exp.std_updater_dict
        key = std_updater_cfg["reward_key"]
        if key not in std_updater_dict:
            std_updater_dict[key] = StdUpdater(**std_updater_cfg)
        std = std_updater_dict[key].update(env)
    return torch.exp(-actions_error / std**2)


def tracking_target_actions_velocities_exp(
    env,
    std: float = 4,
    obs_joint_pos_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    if not hasattr(tracking_target_actions_velocities_exp, "joint_pos_his_buf"):
        tracking_target_actions_velocities_exp.joint_pos_his_buf = [] # (common_step_counter, episode_length_buffer, joint_pos)

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_his_buf = tracking_target_actions_velocities_exp.joint_pos_his_buf
    if not len(joint_pos_his_buf) or env.unwrapped.common_step_counter > joint_pos_his_buf[-1][0]:
        cur_joint_pos = asset.data.joint_pos - asset.data.default_joint_pos
        joint_pos_his_buf.append([
            env.unwrapped.common_step_counter,
            env.unwrapped.episode_length_buf.clone(),
            cur_joint_pos.clone(),
        ])
        if len(joint_pos_his_buf) > 2:
            del joint_pos_his_buf[0] # Save only the last two step

    if len(joint_pos_his_buf) == 1:
        # Only have the current joint pos yet
        return torch.zeros(env.num_envs).to(env.device)

    # Find the joint_pos in obs_buf
    prv_eps_len, prv_joint_pos = joint_pos_his_buf[0][1:]
    cur_eps_len, cur_joint_pos = joint_pos_his_buf[1][1:]

    joint_pos_diff = cur_joint_pos[:, asset_cfg.joint_ids] - prv_joint_pos[:, asset_cfg.joint_ids]
    joint_vel = joint_pos_diff / env.unwrapped.step_dt
    mask = cur_eps_len > prv_eps_len

    # Get the reference target joints pos
    target_joint_vel, ref_mask = env.unwrapped.ref_observation_manager.get_term("target_joint_velocities")
    target_joint_vel = target_joint_vel[:, asset_cfg.joint_ids]
    error = torch.square(joint_vel - target_joint_vel)
    error = torch.mean(error, dim = -1)

    error = torch.where(mask & ref_mask, error, torch.zeros_like(error))
    return torch.exp(-error / std)

    

def balanced_tracking_target_actions_exp(env, std: float, left_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), right_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    tracking_left = tracking_target_actions_exp(env, std, left_asset_cfg)
    tracking_right = tracking_target_actions_exp(env, std, right_asset_cfg)
    tracking_diff = torch.abs(tracking_left - tracking_right) / (tracking_left + tracking_right + 1e-6)
    return tracking_diff

def tracking_target_joint_velocities_exp(env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    if not hasattr(env, "ref_observation_manager"):
        raise RuntimeError("Reference observation manager is not available, function `tracking_target_joint_velocities_exp` cannot be called.")
    # cur_time = (env.episode_length_buf.to(torch.float32)) * env.step_dt
    # last_time = cur_time - env.step_dt
    # cur_action, cur_mask = env.ref_observation_manager.compute_term("target_actions", cur_time)
    # last_action, last_mask = env.ref_observation_manager.compute_term("target_actions", last_time)
    target_joint_velocities, cur_mask = env.ref_observation_manager.get_term("target_joint_velocities")
    if cur_mask is None:
        cur_mask = torch.zeros(target_joint_velocities.shape[0], dtype=torch.bool).to(target_joint_velocities.device)
    target_joint_velocities = target_joint_velocities[:, asset_cfg.joint_ids]
    asset = env.scene[asset_cfg.name]
    joint_velocities = asset.data.joint_vel[:, asset_cfg.joint_ids]
    joint_velocities_diff = torch.where(cur_mask.unsqueeze(1), target_joint_velocities - joint_velocities, torch.zeros_like(target_joint_velocities))
    joint_velocities_error = torch.norm(joint_velocities_diff, dim=1)
    return torch.exp(-joint_velocities_error / std**2)
    
def tracking_target_root_orient_exp(env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for tracking the target root orientation.
    
    """
    if not hasattr(env, "ref_observation_manager"):
        raise RuntimeError("Reference observation manager is not available, function `tracking_target_root_orient` cannot be called.")
    
    # cur_time = (env.episode_length_buf.to(torch.float32) - 1) * env.step_dt
    root_orient, mask = env.ref_observation_manager.get_term("root_orient")
    if mask is None:
        mask = torch.zeros(root_orient.shape[0], dtype=torch.bool).to(root_orient.device)
    asset: Articulation = env.scene[asset_cfg.name]
    asset_root_orient = asset.data.root_quat_w
    root_orient = angle_axis_to_quaternion(root_orient)
    root_orient = torch.where(mask.unsqueeze(1), root_orient, torch.zeros_like(root_orient))
    asset_root_orient = torch.where(mask.unsqueeze(1), asset_root_orient, torch.ones_like(asset_root_orient) * 0.5)
    root_orient_error = quat_error_magnitude(root_orient, asset_root_orient)
    return torch.exp(-root_orient_error / std**2)

def tracking_projected_gravity_exp(env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for tracking the target projected gravity."""
    if not hasattr(env, "ref_observation_manager"):
        raise RuntimeError("Reference observation manager is not available, function `tracking_projected_gravity_exp` cannot be called.")
    # cur_time = (env.episode_length_buf.to(torch.float32) - 1) * env.step_dt
    projected_gravity, mask = env.ref_observation_manager.get_term("target_projected_gravity")
    if mask is None:
        mask = torch.zeros(projected_gravity.shape[0], dtype=torch.bool).to(projected_gravity.device)
    asset = env.scene[asset_cfg.name]
    asset_projected_gravity = asset.data.projected_gravity_b
    projected_gravity = torch.where(mask.unsqueeze(1), projected_gravity, torch.zeros_like(projected_gravity))
    projected_gravity_error = torch.norm(projected_gravity - asset_projected_gravity, dim=1)
    return torch.exp(-projected_gravity_error / std**2)

def tracking_feet_contact(env, sensor_cfg: SceneEntityCfg = None, ref_id: list[int] = slice(None), contact_time_thresh: float = 0.01, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for tracking the target feet contact."""
    if ref_id is None:
        ref_id = [0, 1]
    if not hasattr(env, "ref_observation_manager"):
        raise RuntimeError("Reference observation manager is not available, function `tracking_feet_contact` cannot be called.")
    target_feet_contact, mask = env.ref_observation_manager.get_term("feet_contact")
    target_feet_contact = target_feet_contact.clone()
    mask = mask.clone()
    target_feet_contact = target_feet_contact[:, ref_id] > contact_time_thresh
    mask = mask.unsqueeze(1)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    has_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > contact_time_thresh
    has_contact = has_contact[:, ref_id]
    reward = (target_feet_contact == has_contact)
    reward &= mask
    return torch.all(reward, dim=1)

def calc_target_feet_contact(env, phase_thresh):
    if not hasattr(env, "ref_observation_manager"):
        raise RuntimeError("Reference observation manager is not available, function `tracking_feet_contact` cannot be called.")
    # target_feet_contact, mask = env.ref_observation_manager.get_term("feet_contact")
    # target_feet_contact = target_feet_contact.clone()
    try:
        lft_sin_phase, mask = env.ref_observation_manager.get_term("lft_sin_phase")
        lft_cos_phase, _ = env.ref_observation_manager.get_term("lft_cos_phase")
        rht_sin_phase, _ = env.ref_observation_manager.get_term("rht_sin_phase")
        rht_cos_phase, _ = env.ref_observation_manager.get_term("rht_cos_phase")
    except:
        lft_sin_phase = get_phase(env, offset=0.0)
        lft_cos_phase = get_phase(env, offset=0.25)
        rht_sin_phase = get_phase(env, offset=0.5)
        rht_cos_phase = get_phase(env, offset=0.75)
        mask = torch.zeros(lft_sin_phase.shape[0], dtype=torch.bool).to(lft_sin_phase.device)
    
    lft_phi = torch.atan2(lft_sin_phase, lft_cos_phase)
    lft_phi = (lft_phi + 2 * np.pi) % (2 * np.pi)
    lft_phi /= 2 * np.pi
    rht_phi = torch.atan2(rht_sin_phase, rht_cos_phase)
    rht_phi = (rht_phi + 2 * np.pi) % (2 * np.pi)
    rht_phi /= 2 * np.pi
    l_contact = lft_phi < phase_thresh
    r_contact = rht_phi < phase_thresh
    target_feet_contact = torch.concat([l_contact, r_contact], dim=1)
    target_feet_contact = target_feet_contact.clone()
    mask = mask.clone()
    return target_feet_contact, mask

def tracking_feet_contact_phase(env, sensor_cfg: SceneEntityCfg = None, ref_id: list[int] = slice(None), contact_time_thresh: float = 0.05, phase_thresh: float = 0.55, penalty_false: float = 0.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for tracking the target feet contact."""
    if ref_id is None:
        ref_id = [0, 1]

    target_feet_contact, mask = calc_target_feet_contact(env, phase_thresh)
    
    target_feet_contact = target_feet_contact[:, ref_id]
    # mask = mask.unsqueeze(1)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    has_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > contact_time_thresh
    # feet_link_quat = env.scene[asset_cfg.name].data.body_state_w[:, asset_cfg.body_ids, 3:7]
    # lft_feet_quat = feet_link_quat[:, 0]
    # rht_feet_quat = feet_link_quat[:, 1]
    # lft_feet_rot_mat = matrix_from_quat(lft_feet_quat).unsqueeze(1)
    # rht_feet_rot_mat = matrix_from_quat(rht_feet_quat).unsqueeze(1)
    # feet_rot_mat = torch.cat([lft_feet_rot_mat, rht_feet_rot_mat], dim=1)
    # parallel_contact = is_foot_parallel_from_rot_matrix(feet_rot_mat, tolerance_deg=10)
    # has_contact &= is_parallel
    # has_contact &= (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 5.0 * contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2].norm(dim=-1))
    # has_contact = has_contact[:, ref_id]
    # parallel_contact = parallel_contact[:, ref_id]
    
    reward = (target_feet_contact == has_contact)
    reward = torch.all(reward, dim=1) # left contact + right contact
    # reward &= mask
    # penalty = (target_feet_contact != has_contact)
    # penalty = torch.any(penalty, dim=1)
    # parallel_contact = (has_contact & target_feet_contact) & parallel_contact
    # penalty &= mask
    # reward = reward.float() - penalty.float() * penalty_false
    # parallel_reward = torch.sum(parallel_contact, dim=1)
    # reward = reward.float() + parallel_reward.float() * 2.5
    # reward *= mask
    return reward

def penalize_feet_contact_not_in_phase(env, sensor_cfg: SceneEntityCfg = None, ref_id: list[int] = slice(None), contact_time_thresh: float = 0.01, phase_thresh: float = 0.55, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    if ref_id is None:
        ref_id = [0, 1]
    target_feet_contact, mask = calc_target_feet_contact(env, phase_thresh)
    target_feet_contact = target_feet_contact[:, ref_id]
    mask = mask.unsqueeze(1)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    has_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > contact_time_thresh
    has_contact = has_contact[:, ref_id]
    not_in_contact = (target_feet_contact & ~has_contact)
    # not_in_contact &= mask
    return not_in_contact

def penalize_one_foot_always_contact(env, sensor_cfg: SceneEntityCfg = None, phase_thresh: float = 0.55, contact_time_thresh: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    still_has_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > contact_time_thresh
    target_feet_contact, mask = calc_target_feet_contact(env, phase_thresh)
    target_feet_not_contact = ~target_feet_contact
    not_in_contact_penalty = target_feet_not_contact & still_has_contact
    # not_in_contact_penalty *= mask
    return torch.sum(not_in_contact_penalty, dim=1) * mask

    

def reward_continuous_feet_contact(env, sensor_cfg: SceneEntityCfg = None, ref_id: list[int] = slice(None), upper_bound: float = 0.5, phase_thresh: float = 0.55, penalty_false: float = 2) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    contact_time = contact_time[:, ref_id].clamp(max=upper_bound)
    target_feet_contact = calc_target_feet_contact(env, phase_thresh)
    target_feet_contact, mask = calc_target_feet_contact(env, phase_thresh)
    target_feet_contact = target_feet_contact[:, ref_id]
    true_contact = torch.where((target_feet_contact | (~mask).unsqueeze(1)), contact_time, torch.zeros_like(contact_time))
    true_contact = true_contact.clamp(max=upper_bound)
    false_contact = torch.where((~target_feet_contact) & mask.unsqueeze(1), contact_time, torch.zeros_like(contact_time))
    res = true_contact - penalty_false * false_contact
    res = res[:, ref_id]
    return torch.sum(res, dim=1)

def compute_actual_body_pose_robot_frame(env, asset_cfg: SceneEntityCfg):
    # Compute the actual link poses in the robot base frame
    actual_link_w = env.scene[asset_cfg.name].data.body_state_w[:, asset_cfg.body_ids, :7]
    actual_root_w = env.scene[asset_cfg.name].data.root_state_w[:, :7]
    actual_root_w = actual_root_w.unsqueeze(1).repeat(1, len(asset_cfg.body_ids), 1)
    actual_link_w = actual_link_w.reshape(-1, 7)
    actual_root_w = actual_root_w.reshape(-1, 7)
    actual = torch.cat(subtract_frame_transforms(
        actual_root_w[:, :3], actual_root_w[:, 3:7],
        actual_link_w[:, :3], actual_link_w[:, 3:7],
    ), dim=-1)
    actual = actual.reshape(-1, len(asset_cfg.body_ids), 7)
    return actual

def tracking_body_pos_robot_frame(env, std: float, xyz_dim: tuple = (0, 1, 2), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    target, mask = env.ref_observation_manager.get_term("target_link_poses")
    target = target[:, asset_cfg.body_ids]
    actual = compute_actual_body_pose_robot_frame(env, asset_cfg)
    if isinstance(xyz_dim, tuple): # none singluar dimension
        actual = actual[..., xyz_dim]
        target = target[..., xyz_dim]
    else:
        actual = actual[..., xyz_dim].unsqueeze(-1)
        target = target[..., xyz_dim].unsqueeze(-1)
    error = torch.mean(torch.square(target - actual), dim=(1, 2))
    error = torch.where(mask, error, torch.zeros_like(error))
    return torch.exp(-error / std**2)

def tracking_body_quat_robot_frame(env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    target, mask = env.ref_observation_manager.get_term("target_link_poses")
    target = target[:, asset_cfg.body_ids]
    actual = compute_actual_body_pose_robot_frame(env, asset_cfg)
    actual = actual[..., 3:]
    target = target[..., 3:]
    error = quat_error_magnitude(yaw_quat(target.reshape(-1, 4)), yaw_quat(actual.reshape(-1, 4)))
    error = error.reshape(-1, len(asset_cfg.body_ids))
    error = torch.where(mask.unsqueeze(1), error, torch.zeros_like(error))
    error = torch.mean(error, dim=1)
    return torch.exp(-error / std**2)

def compute_target_body_pose_world_frame(env, asset_cfg: SceneEntityCfg):
    target_link_r, mask = env.ref_observation_manager.get_term("target_link_poses")
    target_link_r = target_link_r[:, asset_cfg.body_ids, :]
    target_base_w, _ = env.ref_observation_manager.get_term("target_base_pose")
    target_base_w = target_base_w.clone()
    target_base_w[:, :3] += env.scene.env_origins
    target_base_w = target_base_w.unsqueeze(1).repeat(1, len(asset_cfg.body_ids), 1)
    target_link_r = target_link_r.reshape(-1, 7)
    target_base_w = target_base_w.reshape(-1, 7)
    target_link_w = torch.cat(combine_frame_transforms(
        target_base_w[:, :3], target_base_w[:, 3:],
        target_link_r[:, :3], target_link_r[:, 3:],
    ), dim=-1)
    target_link_w = target_link_w.reshape(-1, len(asset_cfg.body_ids), 7)
    return target_link_w, mask

def tracking_body_pos_world_frame(env, std: float, xyz_dim: tuple = (0, 1, 2), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    actual = env.scene[asset_cfg.name].data.body_state_w[:, asset_cfg.body_ids, :7]
    target, mask = compute_target_body_pose_world_frame(env, asset_cfg)
    actual = actual[..., xyz_dim]
    target = target[..., xyz_dim]
    error = torch.mean(torch.square(target - actual), dim=(1, 2))
    error = torch.where(mask, error, torch.zeros_like(error))
    return torch.exp(-error / std**2)

def tracking_body_lin_vel_robot_frame(env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    target, mask = env.ref_observation_manager.get_term("target_link_velocities")
    target = target[:, asset_cfg.body_ids, :3]
    root_quat = env.scene[asset_cfg.name].data.root_state_w[:, 3:7]
    actual_w = env.scene[asset_cfg.name].data.body_state_w[:, asset_cfg.body_ids, 7:10]
    actual = quat_apply(
        quat_inv(root_quat).unsqueeze(1).repeat(1, len(asset_cfg.body_ids), 1),
        actual_w,
    )
    error = torch.mean(torch.square(target - actual), dim=2)
    error = torch.where(mask.unsqueeze(1), error, torch.zeros_like(error))
    error = torch.mean(error, dim=1)
    return torch.exp(-error / std**2)

def feet_distance_relative_to_target(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), coef: float = 0.9) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    feet_states = asset.data.body_state_w[:, asset_cfg.body_ids, :3] # x,y,z (world frame)
    # feet_states = feet_states - env.scene.env_origins
    # root_link_quat = env.scene[asset_cfg.name].data.root_state_w[:, 3:7]
    # feet_states = quat_rotate_inverse(root_link_quat, feet_states)

    actual_feet_distance = torch.norm(feet_states[:, 0] - feet_states[:, 1], dim=1)

    target, mask = env.ref_observation_manager.get_term("target_link_poses")
    target = target[:, asset_cfg.body_ids, :3]
    target_feet_distance = torch.norm(target[:, 0] - target[:, 1], dim=1)

    distance_diff = target_feet_distance * coef - actual_feet_distance
    return torch.where(mask, distance_diff.clamp(min=0), torch.zeros_like(distance_diff))

def tracking_base_height(env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    # std = feet_xy_std_updater.update(env)
    actual = env.scene[asset_cfg.name].data.root_state_w[:, 2]
    target, mask = env.ref_observation_manager.get_term("target_base_pose")
    target = target[:, 2]
    error = torch.square(target - actual)
    error = torch.where(mask, error, torch.zeros_like(error))
    return torch.exp(-error / std**2)

def base_height_square(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    actual = env.scene[asset_cfg.name].data.root_state_w[:, 2]
    target, mask = env.ref_observation_manager.get_term("target_base_pose")
    target = target[:, 2]
    height_diff = torch.square(target - actual)
    height_diff = torch.where(mask, height_diff, torch.zeros_like(height_diff))
    return -height_diff

def action_diff_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    action_diff = ref_action_diff(env, asset_cfg)
    action_diff = torch.norm(action_diff, dim=1)
    return -action_diff

def target_gravity_diff_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    target_gravity = env.ref_observation_manager.get_term("target_projected_gravity")[0]
    asset_gravity = asset.data.projected_gravity_b
    gravity_diff = target_gravity - asset_gravity
    gravity_diff = torch.norm(gravity_diff, dim=1)
    return -gravity_diff

def tracking_gravity_diff_exp(env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    target_gravity = env.ref_observation_manager.get_term("target_projected_gravity")[0]
    asset_gravity = asset.data.projected_gravity_b
    gravity_diff = target_gravity - asset_gravity
    gravity_diff = torch.norm(gravity_diff, dim=1)
    return torch.exp(-gravity_diff / std**2)

###################################################################################################

# reward functions

###################################################################################################

def soft_computed_torque_limit(env, ratio: float = 0.35, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    torque = asset.data.computed_torque
    actuators = env.scene.articulations[asset_cfg.name].actuators
    rtn = torch.zeros(env.num_envs).to(env.device)
    for name, actuator in actuators.items():
        soft_torque_limit = actuator.effort_limit * ratio
        actuator_torque = torch.clip(torch.abs(torque[:, actuator.joint_indices]) - soft_torque_limit, min=0.0)
        rtn += torch.sum(actuator_torque, dim=1)
    return rtn

def standing_still(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for standing still."""
    asset: Articulation =env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return -torch.sum(torch.abs(angle), dim=1) * (torch.norm(env.command_manager.get_command(command_name)[:,:2], dim=1) < 0.1).float()

def one_side_limit(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), limit: float = 0.0, above: bool = True) -> torch.Tensor:
    """penalize for exceeding one side limit"""
    asset: Articulation =env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    if above:
        angle = torch.where(angle > limit, angle - limit, torch.zeros_like(angle))
    else:
        angle = torch.where(angle < limit, limit - angle, torch.zeros_like(angle))

    return torch.sum(torch.abs(angle), dim=1)

def both_side_limit(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), limit: float = 0.0) -> torch.Tensor:
    """penalize for exceeding both side limit"""
    asset: Articulation =env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    angle_above = torch.where(angle > limit, angle - limit, torch.zeros_like(angle))
    angle_below = torch.where(angle < -limit, -limit - angle, torch.zeros_like(angle))
    angle = angle_above + angle_below
    return torch.sum(torch.abs(angle), dim=1)
    
def feet_distance(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_min: float = 0.1, distance_max: float = 1.0) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    feet_states = asset.data.body_state_w[:, asset_cfg.body_ids, :2]
    feet_distance = torch.norm(feet_states[:, 0] - feet_states[:, 1], dim=1)
    # penalize if the feet distance is too small or too large
    return torch.where(feet_distance < distance_min, distance_min - feet_distance, torch.zeros_like(feet_distance)) + torch.where(feet_distance > distance_max, feet_distance - distance_max, torch.zeros_like(feet_distance))

def knee_distance(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_min: float = 0.1, distance_max: float = 0.5) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    knee_states = asset.data.body_state_w[:, asset_cfg.body_ids, :2]
    knee_distance = torch.norm(knee_states[:, 0] - knee_states[:, 1], dim=1)
    # penalize if the feet distance is too small or too large
    return torch.where(knee_distance < distance_min, distance_min - knee_distance, torch.zeros_like(knee_distance)) + torch.where(knee_distance > distance_max, knee_distance - distance_max, torch.zeros_like(knee_distance))

def feet_distance_y(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_min: float = 0.1, distance_max: float = 0.5) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    feet_states = asset.data.body_state_w[:, asset_cfg.body_ids, :3] # x,y,z (world frame)
    # feet_states = feet_states - env.scene.env_origins
    root_link_quat = env.scene[asset_cfg.name].data.root_state_w[:, 3:7]
    feet_states_left = quat_rotate_inverse(root_link_quat, feet_states[:, 0])
    feet_states_right = quat_rotate_inverse(root_link_quat, feet_states[:, 1])
    feet_distance = torch.norm(feet_states_left[:, 1:2] - feet_states_right[:, 1:2], dim=1)
    # penalize if the feet distance is too small or too large
    return torch.where(feet_distance < distance_min, distance_min - feet_distance, torch.zeros_like(feet_distance)) + torch.where(feet_distance > distance_max, feet_distance - distance_max, torch.zeros_like(feet_distance))

def knee_distance_y(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_min: float = 0.1, distance_max: float = 0.3) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    knee_states = asset.data.body_state_w[:, asset_cfg.body_ids, 1:2]
    knee_distance = torch.norm(knee_states[:, 0] - knee_states[:, 1], dim=1)
    # penalize if the feet distance is too small or too large
    return torch.where(knee_distance < distance_min, distance_min - knee_distance, torch.zeros_like(knee_distance)) + torch.where(knee_distance > distance_max, knee_distance - distance_max, torch.zeros_like(knee_distance))
    

def feet_contact_parallel(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    feet_link_quat = env.scene[asset_cfg.name].data.body_state_w[:, asset_cfg.body_ids, 3:7]
    lft_feet_quat = feet_link_quat[:, 0]
    rht_feet_quat = feet_link_quat[:, 1]
    lft_feet_rot_mat = matrix_from_quat(lft_feet_quat).unsqueeze(1)
    rht_feet_rot_mat = matrix_from_quat(rht_feet_quat).unsqueeze(1)
    feet_rot_mat = torch.cat([lft_feet_rot_mat, rht_feet_rot_mat], dim=1)
    is_parallel = is_foot_parallel_from_rot_matrix(feet_rot_mat, tolerance_deg=10)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.current_contact_time[:, asset_cfg.body_ids] > 0.0
    contact_parallel = contact & is_parallel
    return torch.sum(contact_parallel, dim=1)
    # check for contact that is not parallel
    # non_parallel_contact = contact & ~is_parallel
    # return torch.sum(non_parallel_contact, dim=1)


def no_fly(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward no flying

    Reward the agent for having at least one foot in contact with the ground while moving.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    single_contact = contacts.sum(dim=1) == 1
    return single_contact.float()

def sole_contact(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-sole contact
    
    Ensure that when feet are in contact with the ground, the forces in the z direction are much larger than the force norm of the other two directions. (Ensure that the feet is not contact to ground with a strange angle)
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = contact_forces.norm(dim=-1) > 1.0
    contact_force_z = contact_forces[:, :, 2]
    contact_force_xy = contact_forces[:, :, :2].norm(dim=-1)
    penalty = torch.where(in_contact, contact_force_z < 5.0 * contact_force_xy, torch.zeros_like(contact_force_z))
    return torch.sum(penalty, dim=1)



def no_jump(env, command_name:str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward no jumping

    Reward the agent for not both feet in contact with the ground while zero command.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    double_contact = contacts.sum(dim=1) == 2
    return double_contact.float() * (torch.norm(env.command_manager.get_command(command_name)[:,:2], dim=1) < 0.1).float()

def jumping_penalty(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize jumping

    Penalize the agent for both feet not in contact with ground (whenever)
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    no_contact = contacts.sum(dim=1) == 0
    return no_contact.float()

def feet_stumble(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    reward = torch.any(torch.norm(contacts[:, :, :2], dim=-1) > 5 * torch.abs(contacts[:, :, 2]), dim=1).float()
    return reward

def feet_contact_forces(env, sensor_cfg: SceneEntityCfg, max_contact_force: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return torch.sum((torch.norm(contacts, dim=-1) - max_contact_force).clamp(min=0.0), dim=1)


def feet_air_time_balancing(env, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 0.4) -> torch.Tensor:
    '''Reward both feet having the same air time while moving.
    Threshold: the maximum time for which the feet can be in the air.
    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    Recommend to be used with feet_air_time    
    '''
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    air_time = torch.where(current_air_time > last_air_time, current_air_time, last_air_time)
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    contact_time = torch.where(current_contact_time > last_contact_time, current_contact_time, last_contact_time)

    air_time_min = torch.min(air_time, dim=1)[0]
    air_time_diff = torch.abs(air_time[:, 0] - air_time[:, 1]) * (air_time_min > 0.0).float()
    contact_time_min = torch.min(contact_time, dim=1)[0]
    contact_time_max = torch.max(contact_time, dim=1)[0]
    contact_time_diff = torch.abs(contact_time[:, 0] - contact_time[:, 1]) * (contact_time_min > 0.0).float()
    long_contact_penalty = torch.where(contact_time_max > threshold * 4, contact_time_max - threshold * 4, torch.zeros_like(contact_time_max))
    low_air_time_penalty = torch.where(air_time_min < threshold, threshold - air_time_min, torch.zeros_like(air_time_min))
    # penalize double stance
    double_stance_penalty = (current_contact_time[:, 0] > 0.0) & (current_contact_time[:, 1] > 0.0)
    # penalize jumping
    jump_penalty = (current_air_time[:, 0] > 0.0) & (current_air_time[:, 1] > 0.0)

    reward = air_time_diff + contact_time_diff + double_stance_penalty.float() + jump_penalty.float() + long_contact_penalty + low_air_time_penalty
    return reward * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1).float()

def balanced_feet_air_time_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    Threshold: the maximum time for which the feet can be in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    # Get maximum of current and last air/contact time
    last_air_time = torch.where(last_air_time > current_air_time, last_air_time, current_air_time)
    last_contact_time = torch.where(last_contact_time > current_contact_time, last_contact_time, current_contact_time)
    last_time_diff = torch.abs(last_air_time - last_contact_time).norm(dim=1)
    # last_time_min = torch.min(last_air_time, last_contact_time).norm(dim=1)
    # time_penalty = torch.exp(-last_time_min)
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    non_single_stance = ~single_stance
    l_in_contact = in_contact[:, 0] & single_stance
    r_in_contact = in_contact[:, 1] & single_stance
    # in_mode_time = torch.where(l_in_contact, contact_time[:, 0], air_time[:, 0])
    last_air_time_contact = torch.where(l_in_contact, last_air_time[:, 0], last_air_time[:, 1])
    last_contact_time_air = torch.where(l_in_contact, last_contact_time[:, 1], last_contact_time[:, 0])
    l_diff = torch.abs(last_air_time_contact - last_contact_time_air)
    l_min = torch.min(last_air_time_contact, last_contact_time_air)
    last_air_time_contact = torch.where(r_in_contact, last_air_time[:, 1], last_air_time[:, 0])
    last_contact_time_air = torch.where(r_in_contact, last_contact_time[:, 0], last_contact_time[:, 1])
    r_diff = torch.abs(last_air_time_contact - last_contact_time_air)
    r_min = torch.min(last_air_time_contact, last_contact_time_air)
    lr_min = torch.clamp(torch.min(l_min, r_min), max=threshold)
    # lr_min = torch.clamp(torch.min(last_air_time[:, 0], last_air_time[:, 1]), max=threshold)
    lr_min_zero = torch.where(lr_min > 0.05, 0.0, 1.0)
    # # penalize double stance
    double_stance_penalty = non_single_stance.float()
    reward = l_diff + r_diff + double_stance_penalty
    time_penalty = torch.exp(-lr_min)
    # reward *= torch.exp(-lr_min)
    # reward += lr_min_zero
    # reward *= (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1)
    # reward = 1.0 - reward
    reward *= 0.25
    reward += last_time_diff
    reward += lr_min_zero
    reward *= time_penalty
    reward *= (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1)
    reward = 0.8 - reward


    return reward

def feet_contact(env, command_name: str, sensor_cfg: SceneEntityCfg, period: float = 0.8, penalty_false: float = 0.05, lft_ref_name: str = "lft_sin_phase", rht_ref_name: str = "rht_sin_phase", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    lft_sin_phase = get_phase(env, period=period, offset=0.0, ref_name=lft_ref_name)
    rht_sin_phase = get_phase(env, period=period, offset=0.5, ref_name=rht_ref_name)
    lft_contact_ref = lft_sin_phase > 0.0
    rht_contact_ref = rht_sin_phase > 0.0
    should_contact = torch.cat([lft_contact_ref, rht_contact_ref], dim=1)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    has_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    feet_link_quat = env.scene[asset_cfg.name].data.body_state_w[:, asset_cfg.body_ids, 3:7]
    lft_feet_quat = feet_link_quat[:, 0]
    rht_feet_quat = feet_link_quat[:, 1]
    lft_feet_rot_mat = matrix_from_quat(lft_feet_quat).unsqueeze(1)
    rht_feet_rot_mat = matrix_from_quat(rht_feet_quat).unsqueeze(1)
    feet_rot_mat = torch.cat([lft_feet_rot_mat, rht_feet_rot_mat], dim=1)
    is_parallel = is_foot_parallel_from_rot_matrix(feet_rot_mat, tolerance_deg=10)
    has_contact &= is_parallel



    # has_contact &= (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 5.0 * contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2].norm(dim=-1))
    contact_match = (should_contact == has_contact).float()
    contact_match_reward = torch.all(contact_match, dim=1).float()
    contact_mismatch = (should_contact != has_contact).float()
    contact_mismatch_reward = torch.any(contact_mismatch, dim=1).float()
    reward = contact_match_reward - contact_mismatch_reward * penalty_false
    reward *= (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1)
    return reward


def feet_height_relative_positive(env: ManagerBasedRLEnv, height_limit: float, soft_limit_ratio: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for having the feet at a certain height range relative to the ground.

    Takes effect only when at least one foot is in contact with the ground.
    """
    asset = env.scene[asset_cfg.name]
    feet_height = asset.data.body_state_w[:, sensor_cfg.body_ids, 2]
    feet_height_diff = torch.abs(feet_height[:, 0] - feet_height[:, 1])
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    stances = torch.sum(in_contact.int(), dim=1) > 0
    soft_limit = height_limit * soft_limit_ratio
    reward = torch.where((feet_height_diff < soft_limit) & stances, feet_height_diff, torch.zeros_like(feet_height_diff))
    reward += torch.where(((feet_height_diff >= soft_limit) & (feet_height_diff < height_limit)) & stances, feet_height_diff, torch.zeros_like(feet_height_diff)) * 0.1
    return reward

def action_scale_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the magnitude of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


def action_rate_l2_dt(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)/env.step_dt

def action_rate_acc_l2_dt(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    if not hasattr(action_rate_acc_l2_dt, "prev_prev_action"):
        action_rate_acc_l2_dt.prev_prev_action = torch.zeros_like(env.action_manager.action)

    dt2 = env.step_dt ** 2
    acc = (env.action_manager.action - 2 * env.action_manager.prev_action + action_rate_acc_l2_dt.prev_prev_action) / dt2
    action_rate_acc_l2_dt.prev_prev_action = env.action_manager.prev_action
    return torch.sum(torch.square(acc), dim=1)

def base_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of the base using L2 squared kernel."""
    asset = env.scene[asset_cfg.name]
    if not hasattr(base_lin_acc_l2, "prev_lin_vel"):
        base_lin_acc_l2.prev_lin_vel = torch.zeros_like(asset.data.root_lin_vel_b)
    lin_acc = (asset.data.root_lin_vel_b - base_lin_acc_l2.prev_lin_vel) / env.step_dt
    base_lin_acc_l2.prev_lin_vel = asset.data.root_lin_vel_b
    return torch.sum(torch.square(lin_acc), dim=1)

def base_lin_acc_exp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.1) -> torch.Tensor:
    """Penalize the linear acceleration of the base using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    if not hasattr(base_lin_acc_exp, "prev_lin_vel"):
        base_lin_acc_exp.prev_lin_vel = torch.zeros_like(asset.data.root_lin_vel_b)
    lin_acc = (asset.data.root_lin_vel_b - base_lin_acc_exp.prev_lin_vel) / env.step_dt
    base_lin_acc_exp.prev_lin_vel = asset.data.root_lin_vel_b
    return torch.exp(-torch.sum(torch.square(lin_acc), dim=1) / std**2)

def base_ang_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the angular acceleration of the base using L2 squared kernel."""
    asset = env.scene[asset_cfg.name]
    if not hasattr(base_ang_acc_l2, "prev_ang_vel"):
        base_ang_acc_l2.prev_ang_vel = torch.zeros_like(asset.data.root_ang_vel_b)
    ang_acc = (asset.data.root_ang_vel_b- base_ang_acc_l2.prev_ang_vel) / env.step_dt
    base_ang_acc_l2.prev_ang_vel = asset.data.root_ang_vel_b
    return torch.sum(torch.square(ang_acc), dim=1)

def base_ang_acc_exp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.1) -> torch.Tensor:
    """Penalize the angular acceleration of the base using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    if not hasattr(base_ang_acc_exp, "prev_ang_vel"):
        base_ang_acc_exp.prev_ang_vel = torch.zeros_like(asset.data.root_ang_vel_b)
    ang_acc = (asset.data.root_ang_vel_b - base_ang_acc_exp.prev_ang_vel) / env.step_dt
    base_ang_acc_exp.prev_ang_vel = asset.data.root_ang_vel_b
    return torch.exp(-torch.sum(torch.square(ang_acc), dim=1) / std**2)

def reward_ori_d(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the deviation of the orientation of the robot using L2 squared kernel."""
    asset = env.scene[asset_cfg.name]
    cur_ori = asset.data.projected_gravity_b[:, :2]
    if not hasattr(reward_ori_d, "prev_ori"):
        reward_ori_d.prev_ori = torch.zeros_like(cur_ori)
    ori_d = cur_ori - reward_ori_d.prev_ori / env.step_dt
    reward_ori_d.prev_ori = cur_ori
    return torch.sum(torch.square(ori_d), dim=1)



###################################################################################################

# xbot additional reward functions

###################################################################################################

def reward_speed_matching(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for matching speed, and penalize for non-matching speed."""

    # speed:
    asset = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b[:, :2]
    ang_vel = asset.data.root_ang_vel_b[:, 2]
    command = env.command_manager.get_command(command_name)
    target_lin_vel = command[:, :2]
    target_ang_vel = command[:, 2]

    lin_x_matching = torch.zeros(lin_vel.shape[0]).to(env.device)
    lin_y_matching = torch.zeros(lin_vel.shape[0]).to(env.device)
    ang_matching = torch.zeros(ang_vel.shape[0]).to(env.device)

    # linear related
    # lin_x too low: speed < 0.5*abs(cmd_x)
    lin_x_too_low = torch.abs(lin_vel[:, 0]) < 0.5 * torch.abs(target_lin_vel[:, 0])
    lin_x_too_high = torch.abs(lin_vel[:, 0]) > 1.2 * torch.abs(target_lin_vel[:, 0])
    lin_x_matching = (~lin_x_too_low & ~lin_x_too_high).float() * 1.2
    sgn_mismatch = torch.sign(lin_vel[:, 0]) != torch.sign(target_lin_vel[:, 0])
    lin_x_matching[lin_x_too_low] = -1.0
    lin_x_matching[lin_x_too_high] = 0.0
    lin_x_matching[sgn_mismatch] = -2.0
    lin_x_matching

    # lin_y too low: speed < 0.5*abs(cmd_y)
    lin_y_too_low = torch.abs(lin_vel[:, 1]) < 0.5 * torch.abs(target_lin_vel[:, 1])
    lin_y_too_high = torch.abs(lin_vel[:, 1]) > 1.2 * torch.abs(target_lin_vel[:, 1])
    lin_y_matching = (~lin_y_too_low & ~lin_y_too_high).float() * 1.2
    sgn_mismatch = torch.sign(lin_vel[:, 1]) != torch.sign(target_lin_vel[:, 1])
    lin_y_matching[lin_y_too_low] = -1.0
    lin_y_matching[lin_y_too_high] = 0.0
    lin_y_matching[sgn_mismatch] = -2.0

    # angular related
    ang_too_low = torch.abs(ang_vel) < 0.8 * torch.abs(target_ang_vel)
    ang_too_high = torch.abs(ang_vel) > 1.2 * torch.abs(target_ang_vel)
    ang_matching = (~ang_too_low & ~ang_too_high).float() * 1.2
    sgn_mismatch = torch.sign(ang_vel) != torch.sign(target_ang_vel)
    ang_matching[ang_too_low] = -1.0
    ang_matching[ang_too_high] = -0.5
    ang_matching[sgn_mismatch] = -2.0

    speed_matching = lin_x_matching * 2. + lin_y_matching * 1. + ang_matching * 1.5
    return speed_matching


def reward_speed_non_violation(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the agent for exceeding the speed limit."""
    asset = env.scene[asset_cfg.name]
    lin_vel_z = asset.data.root_lin_vel_b[:, 2]
    ang_vel_xy = asset.data.root_ang_vel_b[:, :2].norm(dim=1)

    lin_violation = torch.exp(-torch.square(lin_vel_z) / 0.3**2)
    ang_violation = torch.exp(-torch.square(ang_vel_xy) / 0.2**2)

    return lin_violation + ang_violation

def reward_orientation_non_violation(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the agent for exceeding the orientation limit."""
    asset = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b[:, :2]
    orientation_violation = torch.exp(-torch.sum(torch.square(projected_gravity)) / 0.1**2)
    return orientation_violation


def reward_base_height(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for keeping the base height within a certain range."""
    asset = env.scene[asset_cfg.name]
    base_height = asset.cfg.init_state.pos[2] # base height
    current_base_height = asset.data.root_state_w[:, 2]
    height_diff = base_height - current_base_height
    height_error = torch.exp(-torch.square(height_diff) / 0.1**2)
    return height_error