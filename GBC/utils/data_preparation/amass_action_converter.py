from GBC.utils.data_preparation.pose_transformer import PoseTransformer
from GBC.utils.data_preparation.amass_loader import AMASSDatasetInterpolate
import os
import torch
import torch.nn.functional as F
from GBC.utils.data_preparation.data_preparation_cfg import *
from torchaudio_filters import LowPass
from tqdm import tqdm
from GBC.utils.base.math_utils import rot_mat_to_vec, rot_vec_to_mat, quaternion_to_angle_axis, find_longest_cyclic_subsequence, angle_axis_to_quaternion, batch_angle_axis_to_ypr, interpolate_trans, pad_to_len, filt_feet_contact, hampel_filter, quat_inv, quat_conjugate, quat_rotate, q_mul, quat_rotate_inverse, quat_fix, unwrap_and_smooth_rot_vecs, smooth_quat_savgol
from GBC.utils.base.rotation_repair import repair_rotation_sequence
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from GBC.utils.base.base_fk import RobotKinematics
from typing import Dict
from glob import glob
from copy import deepcopy
from GBC.utils.base import DATA_PATHS
from GBC.utils.data_preparation.robot_flip_left_right import RobotFlipLeftRight, flip_rot_mat_left_right
from human_body_prior.body_model.body_model import BodyModel

def adjust_shape_0(sequence: torch.Tensor, shape_0: int):
    if sequence.shape[0] > shape_0:
        sequence = sequence[:shape_0]
    elif sequence.shape[0] < shape_0:
        sequence = torch.cat([sequence, sequence[-1:].repeat(shape_0 - sequence.shape[0], 1)], dim=0)
    return sequence


class TrackingDataPostProcess:
    def __init__(self, filter_cfg: FilterCfg):
        self.cfg = filter_cfg
        self.device = filter_cfg.device
        self.lowpass = LowPass(
            cutoff=filter_cfg.filter_cutoff,
            sample_rate=filter_cfg.filter_sample_rate,
            order=filter_cfg.filter_order,
        ).to(self.device)
        
    def to(self, device: str):
        '''
            Move the filter to the specified device
        '''
        self.device = device
        self.lowpass = self.lowpass.to(device)
        return self

    def filt(self, sequence: torch.Tensor, use_padding: bool = True, window_size = 3):
        '''
            Apply lowpass filter to the sequence

            Args:
                sequence (torch.Tensor): The input sequence to be filtered (T, D)
                use_padding (bool): Whether to use padding to reduce edge artifacts
                window_size (int): The size of the window for the Hampel filter

            Returns:
                torch.Tensor: The filtered sequence (T, D)
        '''

        # Apply Hampel filter
        sequence, has_outlier = hampel_filter(sequence, window_size=window_size)
        # if has_outlier:
        #     print("Warning: Hampel filter detected outlier in the sequence"

        epsilon = 1e-6
        sequence += epsilon # torchaudio_filters has bugs dealing with 0 values, therefore we add a small epsilon

        sequence_shape_0 = sequence.shape[0]
        if sequence.shape[0] <= 2:
            return sequence
        
        if use_padding:
            # Add padding to reduce edge artifacts - use longer padding for better results
            pad_length = min(sequence.shape[0] // 2, 50)  # Use half of sequence length or 50 frames, whichever is smaller
            
            # Create padded sequence
            first_frame = sequence[0:1].repeat(pad_length, 1)  # Repeat first frame
            last_frame = sequence[-1:].repeat(pad_length, 1)   # Repeat last frame
            padded_sequence = torch.cat([first_frame, sequence, last_frame], dim=0)
            
            # Apply filter to padded sequence
            padded_sequence = padded_sequence.permute(1, 0).unsqueeze(0)
            padded_sequence = self.lowpass(padded_sequence)
            padded_sequence = padded_sequence.squeeze(0).permute(1, 0)
            
            # Extract original sequence length from the filtered padded sequence
            filtered_sequence = padded_sequence[pad_length:pad_length + sequence_shape_0]
            
            # Use adjust_shape_0 as safety measure to ensure correct output length
            filtered_sequence = adjust_shape_0(filtered_sequence, sequence_shape_0)

            filtered_sequence -= epsilon # remove the epsilon added earlier
            
            return filtered_sequence
        
        else:
            # Original implementation without padding
            sequence = sequence.permute(1, 0).unsqueeze(0)
            sequence = self.lowpass(sequence)
            sequence = sequence.squeeze(0).permute(1, 0)
            sequence = adjust_shape_0(sequence, sequence_shape_0)
            sequence -= epsilon # remove the epsilon added earlier
            return sequence
    
    
    def adjust_root_orient(self, root_orient: torch.Tensor):
        '''
            Adjust the root orientation to the correct format

            Args:
                root_orient (torch.Tensor): The input root orientation to be adjusted (T, 3)
o be adjusted (T, 3)

            Returns:
                torch.Tensor: The adjusted root orientation (T, 3)
        '''

        rot_quat = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32, device=self.device).repeat(root_orient.shape[0], 1)
        rot_vec = quaternion_to_angle_axis(rot_quat)

        root_rot_mat = rot_vec_to_mat(root_orient) # (T, 3, 3)
        rot_mat = rot_vec_to_mat(rot_vec) # (T, 3, 3)

        # transpose the rotation matrix R -> R^T
        rot_mat = rot_mat.permute(0, 2, 1)
    
        rot = torch.bmm(root_rot_mat, rot_mat) # (T, 3, 3)
        root_orient = rot_mat_to_vec(rot)
        root_quat = angle_axis_to_quaternion(root_orient)
        root_quat = quat_fix(root_quat)  # Ensure quaternion is continuous
        root_quat = smooth_quat_savgol(root_quat, window_size=11, polyorder=3)
        root_quat = F.normalize(root_quat, p=2, dim=1)
        root_orient = quaternion_to_angle_axis(root_quat)
        
        return root_orient
    
    def get_projected_gravity(self, root_orient: torch.Tensor):
        '''
            Get the projected gravity vector in the global coordinate

            Args:
                root_orient (torch.Tensor): The input root orientation (T, 3)

            Returns:
                torch.Tensor: The projected gravity vector (T, 3)
        '''
        gravity = torch.tensor([0, 0, -1], dtype=torch.float32, device=self.device).repeat(root_orient.shape[0], 1) # (T, 3) gravity vector with normalized length
        root_rot_mat = rot_vec_to_mat(root_orient) # (T, 3, 3)
        gravity = torch.einsum("ijk, ik -> ij", root_rot_mat, gravity) # (T, 3)
        return gravity

    def get_yaw_only_rot_mat(self, rot_mat):
        '''
            Calculate the rotation matrix where only the yaw component is preserved
            
            Args:
                rot_mat (torch.Tensor): rotation matrix (..., 3, 3)
                
            Note:
                This implementation handles gimbal lock (when pitch ≈ ±90°) robustly.
                In gimbal lock situations, we extract the "effective yaw" that produces
                the same projected motion in the XY plane.
        '''
        # Check if we're near gimbal lock (|cos(pitch)| ≈ 0)
        cos_pitch = torch.sqrt(rot_mat[..., 0, 0]**2 + rot_mat[..., 1, 0]**2)
        gimbal_threshold = 1e-6
        
        # Default method: yaw = atan2(R_21, R_11) = atan2(sin_yaw*cos_pitch, cos_yaw*cos_pitch)
        yaw = torch.atan2(rot_mat[..., 1, 0], rot_mat[..., 0, 0])
        
        # Handle gimbal lock cases where cos(pitch) ≈ 0
        gimbal_mask = cos_pitch < gimbal_threshold
        
        if torch.any(gimbal_mask):
            # In gimbal lock, we extract the effective yaw from other matrix elements
            # For pitch = +90°: effective_yaw = atan2(-R[0,1], R[1,1])
            # For pitch = -90°: effective_yaw = atan2(R[0,1], R[1,1])
            
            # Determine sign of pitch from R[2,0] = -sin(pitch)
            pitch_sign = -torch.sign(rot_mat[..., 2, 0])  # +1 for +90°, -1 for -90°
            yaw_gimbal = torch.atan2(-pitch_sign * rot_mat[..., 0, 1], rot_mat[..., 1, 1])
            
            # Use gimbal-corrected yaw where needed
            yaw = torch.where(gimbal_mask, yaw_gimbal, yaw)

        res = torch.zeros_like(rot_mat)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        res[..., 0, 0] = cos_yaw
        res[..., 0, 1] = -sin_yaw
        res[..., 1, 0] = sin_yaw
        res[..., 1, 1] = cos_yaw
        res[..., 2, 2] = 1.0
        return res

    def get_batch_tf_matrix(self, tsl, rot_vec):
        tf_mat_3x4 = torch.concatenate((rot_vec_to_mat(rot_vec), tsl.unsqueeze(2)), dim=2)
        tf_mat_1x3 = torch.zeros_like(tsl).unsqueeze(1)
        tf_mat_1x1 = torch.ones_like(tsl[:, 0]).reshape(-1, 1, 1)
        tf_mat = torch.concatenate((
            tf_mat_3x4,
            torch.concatenate((tf_mat_1x3, tf_mat_1x1), dim=2),
        ), dim=1)
        assert tf_mat.shape == (tsl.shape[0], 4, 4)
        return tf_mat

    def transform_for_intial_pose(self, tsl, rot_vec):
        '''
            Apply a global transform to all frames s.t. the first frame has x = 0, y = 0, yaw = 0

            Args:
                tsl (torch.Tensor): translation (T, 3)
                rot_vec (torch.Tensor): rotation vector (T, 3)

            Return:
                tuple of transformed translation tensor and rotation vector tensor
        '''
        tf_mat = self.get_batch_tf_matrix(tsl, rot_vec)
        inv_tf = torch.eye(4, dtype=tf_mat.dtype, device=tf_mat.device)
        # Set roll = 0, pitch = 0, only rotation around the z axis
        inv_tf[:3, :3] = self.get_yaw_only_rot_mat(tf_mat[0, :3, :3].T)
        # Get translation
        inv_tf[:, 3:] = -inv_tf @ tf_mat[0, :, 3:]
        # Set translation z = 0
        inv_tf[2, 3] = 0

        # Apply transformation to each frame
        new_tf_mat = torch.einsum("ij, bjk -> bik", inv_tf, tf_mat)

        return new_tf_mat[:, :3, 3], rot_mat_to_vec(new_tf_mat[:, :3, :3])

    def translate_z(self, trans, root_orient, contact_tf_mats):
        root_tf_mat = self.get_batch_tf_matrix(trans, root_orient)
        link_world_mat = torch.einsum("tij, tljk -> tlik", root_tf_mat, contact_tf_mats)
        z_coor = torch.min(link_world_mat[..., 2, 3])
        # print("z_coor:", z_coor)
        trans[:, 2] -= z_coor
        return trans

    def transform_poses_data(self, save_dict, filter_pose):
        if filter_pose:
            # trans, root_orient = map(self.filt, (save_dict["trans"], save_dict["root_orient"]))
            trans = self.filt(save_dict["trans"])
            root_orient = save_dict["root_orient"]
        else:
            trans, root_orient = save_dict["trans"], save_dict["root_orient"]
        new_trans, new_root_orient = self.transform_for_intial_pose(trans, root_orient)
        save_dict["trans"], save_dict["root_orient"] = new_trans, new_root_orient
        return save_dict

    def calc_lin_vel_yaw_frame(self, trans, rot_vec, fps):
        lin_vel = torch.diff(trans, dim=0) # (T-1, 3)
        lin_vel *= fps
        lin_vel = torch.cat([lin_vel, lin_vel[-1:]], dim=0) # (T, 3)

        # To robot yaw frame
        rot_mat = rot_vec_to_mat(rot_vec)
        tf_mat = self.get_yaw_only_rot_mat(rot_mat).transpose(2, 1)
        lin_vel = torch.einsum("bij, bj -> bi", tf_mat, lin_vel)
        return self.filt(lin_vel), lin_vel

    def calc_ang_vel_world(self, rot_vec, fps):
        rot_quat = angle_axis_to_quaternion(rot_vec)
        # rot_quat = quat_fix(rot_quat)  # Ensure quaternion is continuous
        # rot_quat = self.filt(rot_quat)
        delta_q = q_mul(rot_quat[1:], quat_inv(rot_quat[:-1])) # quaternion diff
        neg_w_mask = delta_q[..., 0] < 0
        delta_q[neg_w_mask] *= -1.0
        angle = 2 * torch.acos(torch.clamp(delta_q[...,0], -1.0, 1.0))
        axis = delta_q[...,1:]
        axis_norm = torch.norm(axis, p=2, dim=-1, keepdim=True)
        axis = axis / (axis_norm + 1e-8)
        rot_diff_vec = axis * angle.unsqueeze(-1)
        ang_vel = rot_diff_vec * fps
        ang_vel = torch.cat([ang_vel, ang_vel[-1:]], dim=0)

        return ang_vel

    def calc_ang_vel(self, rot_vec, fps):
        rot_quat = angle_axis_to_quaternion(rot_vec)
        # rot_quat = quat_fix(rot_quat)  # Ensure quaternion is continuous
        # rot_quat = self.filt(rot_quat)
        delta_q = q_mul(rot_quat[1:], quat_inv(rot_quat[:-1])) # quaternion diff
        neg_w_mask = delta_q[..., 0] < 0
        delta_q[neg_w_mask] *= -1.0
        angle = 2 * torch.acos(torch.clamp(delta_q[...,0], -1.0, 1.0))
        axis = delta_q[...,1:]
        axis_norm = torch.norm(axis, p=2, dim=-1, keepdim=True)
        axis = axis / (axis_norm + 1e-8)
        rot_diff_vec = axis * angle.unsqueeze(-1)
        ang_vel = rot_diff_vec * fps
        ang_vel = torch.cat([ang_vel, ang_vel[-1:]], dim=0)
        # ang_vel = quat_rotate(quat_inv(rot_quat), ang_vel)
        ang_vel = quat_rotate_inverse(rot_quat, ang_vel) # (T, 3)

        return ang_vel
    
    def calc_lin_vel_world(self, trans, orient, fps):
        lin_vel = torch.diff(trans, dim=0) # (T-1, 3)
        lin_vel *= fps
        lin_vel = torch.cat([lin_vel, lin_vel[-1:]], dim=0) # (T, 3)
        return lin_vel
    
    def calc_lin_vel(self, trans, orient, fps):
        lin_vel = torch.diff(trans, dim=0) # (T-1, 3)
        lin_vel *= fps
        lin_vel = torch.cat([lin_vel, lin_vel[-1:]], dim=0) # (T, 3)
        rot_quat = angle_axis_to_quaternion(orient)
        rot_quat = quat_fix(rot_quat)  # Ensure quaternion is continuous
        rot_quat = smooth_quat_savgol(rot_quat, window_size=11, polyorder=3)
        rot_quat = F.normalize(rot_quat, p=2, dim=1)
        
        lin_vel = quat_rotate_inverse(rot_quat, lin_vel) # (T,
        return lin_vel

    def add_extra_data_from_poses(self, data_dict):
        # data_dict["lin_vel"], data_dict["lin_vel_orig"] = self.calc_lin_vel_yaw_frame(data_dict["trans"], data_dict["root_orient"], data_dict["fps"])
        data_dict["lin_vel"] = self.calc_lin_vel(data_dict["trans"], data_dict["root_orient"], data_dict["fps"])
        data_dict["lin_vel_orig"] = self.calc_lin_vel_world(data_dict["trans"], data_dict["root_orient"], data_dict["fps"])
        data_dict["ang_vel"] = self.calc_ang_vel(data_dict["root_orient"], data_dict["fps"])
        data_dict["ang_vel_world"] = self.calc_ang_vel_world(data_dict["root_orient"], data_dict["fps"])

        project_gravity = self.get_projected_gravity(data_dict["root_orient"]) # (T, 3)
        # print("project_gravity is: ", project_gravity)
        data_dict["project_gravity"] = project_gravity
        return data_dict

    def calc_actions_vel(self, data_dict):
        actions = data_dict["actions"]
        actions_vel = torch.diff(actions, dim = 0) / data_dict["fps"]
        actions_vel = torch.cat([actions_vel, actions_vel[-1:]])
        # print("actions:", actions[:2])
        # print("fps:", data_dict["fps"])
        # print("actions_vel:", actions_vel[0])
        return actions_vel

    def post_process(self, data_dict, contact_tf_mats=None, filter_pose=False):
        data_dict = self.transform_poses_data(data_dict, filter_pose=filter_pose)
        if contact_tf_mats is not None:
            data_dict["trans"] = self.translate_z(data_dict["trans"], data_dict["root_orient"], contact_tf_mats)
        data_dict = self.add_extra_data_from_poses(data_dict)
        data_dict["actions_vel"] = self.calc_actions_vel(data_dict)
        return data_dict

    def __call__(self, data_dict, contact_tf_mats=None, filter_pose=False):
        return self.post_process(data_dict, contact_tf_mats=contact_tf_mats, filter_pose=filter_pose)


class AMASSActionConverter:
    def __init__(self,
                 cfg: BaseCfg,
                 **kwargs
                    ):
        self.cfg = cfg
        self.dataset = AMASSDatasetInterpolate.from_cfg(cfg)
        self.load_hands = cfg.load_hands
        self.fk = RobotKinematics(cfg.urdf_path, device=cfg.device)
        self.model = PoseTransformer(num_actions=self.fk.num_dofs, load_hands=cfg.load_hands).to(cfg.device)
        self.model.load_state_dict(torch.load(cfg.pose_transformer_path, map_location=torch.device(cfg.device),weights_only=False))
        jit_compile = kwargs.get("jit_compile", False)
        if jit_compile:
            self.model = torch.jit.trace(self.model, torch.zeros((1, 63 if not cfg.load_hands else 153), device=cfg.device))
        self.model.eval() # set model to evaluation mode to improve performance
        self.post_process = TrackingDataPostProcess(filter_cfg=cfg.filter).to(cfg.device)
        self.batch_size = cfg.batch_size
        self.device = cfg.device
        self.export_path = cfg.export_path
        self.visualize = cfg.visualize
        self.default_body_map = None
        self.smplh_body_names = ['Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist']
        self.smplh_extend_names = ['L_Index1', 'L_Index2', 'L_Index3', 'L_Middle1', 'L_Middle2', 'L_Middle3', 'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 'L_Ring1', 'L_Ring2', 'L_Ring3', 'L_Thumb1', 'L_Thumb2', 'L_Thumb3', 'R_Index1', 'R_Index2', 'R_Index3', 'R_Middle1', 'R_Middle2', 'R_Middle3', 'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 'R_Ring1', 'R_Ring2', 'R_Ring3', 'R_Thumb1', 'R_Thumb2', 'R_Thumb3']
        self.action_flip_left_right = None
        self.set_body_model(cfg.smplh_model_path, cfg.dmpls_model_path, cfg.smpl_fits_dir)
        
    def set_flipper(self, flipper: RobotFlipLeftRight):
        '''
            Set the flipper for the action converter

            Args:
                flipper (RobotFlipLeftRight): The flipper to be set
        '''
        self.action_flip_left_right = flipper

    @classmethod
    def from_cfg(cls, cfg: BaseCfg):
        return cls(cfg)
    @torch.no_grad()
    def pose_to_action(self, pose: torch.Tensor):
        '''
            Convert pose to action using the PoseTransformer model

            Args:
                pose (torch.Tensor): The input pose to be converted (T, 63 if not load_hands else 63+90)

            Returns:
                torch.Tensor: The converted action (T, 63 if not load_hands else 63+90)
        '''
        num_splits = pose.shape[0] // self.batch_size
        if pose.shape[0] % self.batch_size != 0:
            num_splits += 1

        actions = []

        for i in range(num_splits):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, pose.shape[0])
            pose_batch = pose[start:end].to(self.device)
            action_batch = self.model(pose_batch)
            if self.default_body_map is not None:
                pose_body_indices, action_indices = self.default_body_map
                pose_body_indices = torch.tensor(pose_body_indices, dtype=torch.long, device=self.device)
                action_indices = torch.tensor(action_indices, dtype=torch.long, device=self.device)
                pose_body = pose_batch[:, :63].reshape(-1, 21, 3)
                ypr = batch_angle_axis_to_ypr(pose_body)
                # ypr[:,:,2] *= 0.01

    
                action_batch[:, action_indices] = ypr[:, pose_body_indices[:, 0], pose_body_indices[:, 1]]

            actions.append(action_batch)

        actions = torch.cat(actions, dim=0)
        actions = adjust_shape_0(actions, pose.shape[0])
        return actions
    
    def set_body_model(self, smplh_model_path: str, dmpls_model_path: str, smpl_fits_dir: str):
        self.body_model = BodyModel(bm_fname=smplh_model_path, dmpl_fname=dmpls_model_path, num_betas=16, num_dmpls=8).to(self.device)
        try:
            self.fit_data = torch.load(smpl_fits_dir, map_location=self.device)
        except FileNotFoundError:
            print(f"Warning: {smpl_fits_dir} not found")
            self.fit_data = None

    def adjust_trans_height(self, trans: torch.Tensor, poses: torch.Tensor):
        body_parms = {
            "betas": self.fit_data["beta"],
            "dmpls": self.fit_data["dmpls"],
            "pose_body": poses[:, 3:66],
            "trans": trans,
            "root_orient": poses[:, :3],
        }
        positions = self.body_model(**body_parms).Jtr.detach()
        height = positions[:, :, 1]
        trans[:, 1] -= height.min()
        return trans
    
    def set_direct_joint_map_body(self, direct_joint_map: Dict[str, Dict[str, str]]):
        '''
        Set direct joint map_body

        Args:
            direct_joint_map (Dict[str, Dict[str, str]]): Direct joint map


        Format:
        Dict{
            SMPLH joint name: Dict{
                'yaw'/ 'pitch'/ 'roll': Humanoid joint name
                }
            }
        '''
        # order: yaw, pitch, roll
        angles = ['yaw', 'pitch', 'roll']
        pose_body_indices = []
        action_indices = []
        for smplh_joint, humanoid_joint in direct_joint_map.items():
            for angle_name, humanoid_joint_name in humanoid_joint.items():
                pose_body_indices.append([self.smplh_body_names.index(smplh_joint) - 1, angles.index(angle_name)])
                action_indices.append(self.fk.dof_names.index(humanoid_joint_name))

        # if "body" not in self.direct_joint_map.keys():
        #     self.direct_joint_map["body"] = (pose_body_indices, action_indices)
        # else:
        #     self.direct_joint_map["body"] = (self.direct_joint_map["body"][0] + pose_body_indices, self.direct_joint_map["body"][1] + action_indices)
        self.default_body_map = (pose_body_indices, action_indices)
    
    def visualize_filter(self, origin: torch.Tensor, filtered: torch.Tensor, max_duration: float = 25.0, title: str = "Filter Visualization", save_dir: str = None, cyclic_subseq: tuple = None):
        '''
            Visualize the filter effect on the sequence

            Args:
                origin (torch.Tensor): The original sequence (T, 29)
                filtered (torch.Tensor): The filtered sequence (T, 29)
                max_duration (float): The maximum duration to visualize
                title (str): The title of the plot
                save_dir (str): The directory to save the plot
        '''
        # select 3 random dimensions to visualize
        dims = torch.randint(0, origin.shape[1], (3,))
        dims = torch.unique(dims)
        origin = origin[:, dims].cpu().numpy()
        filtered = filtered[:, dims].cpu().numpy()
        dims = dims.numpy()
        frames = min(int(max_duration * self.dataset.interpolate_fps), origin.shape[0])
        origin = origin[:frames]
        filtered = filtered[:frames]

        fig = plt.figure(figsize=(12, 6))
        for i in range(dims.shape[0]):
            plt.subplot(3, 1, i + 1)
            plt.plot(origin[:, i], label="origin")
            plt.plot(filtered[:, i], label="filtered")
            if cyclic_subseq is not None:
                start, end = cyclic_subseq
                plt.axvline(start, color='r', linestyle='--', label="start")
                plt.axvline(end, color='g', linestyle='--', label="end")
            plt.title(f"Dimension {dims[i]}")
            plt.legend()
        plt.suptitle(title)
        if save_dir is not None:
            plt.savefig(save_dir)

        plt.close(fig)

    def visualize_trajectory(self, trans: torch.Tensor, root_orient: torch.Tensor, axis_length: float, title: str, save_dir: str = None):
        '''
            Visualize the trajectory of the sequence

            Args:
                trans (torch.Tensor): The translation sequence (T, 3)
                root_orient (torch.Tensor): The root orientation sequence (T, 3)
                axis_length (float): The length of the axis
                title (str): The title of the plot video
                save_dir (str): The directory to save the plot video
        '''
        T = trans.shape[0]
        root_mat = rot_vec_to_mat(root_orient)
        trans_np = trans.cpu().numpy()
        trans_np_zeros = np.zeros((T, 3))
        root_mat_np = root_mat.cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        min_range = np.min(trans_np, axis=0)
        min_range[2] = 0
        max_range = np.max(trans_np, axis=0)
        center = (max_range + min_range) * 0.5
        scale = np.max(max_range - min_range) * 0.5 * 1.1
        center[2] = scale
        for i, c in enumerate("xyz"):
            getattr(ax, f"set_{c}lim")(center[i] - scale, center[i] + scale)
            getattr(ax, f"set_{c}label")(c.upper())
        ax.set_title(title)
        traj_line, = ax.plot([], [], [], 'b-', lw=2, label='Trajectory')
        axis_x, = ax.plot([], [], [], 'r-', lw=1, label='X-axis')
        axis_y, = ax.plot([], [], [], 'g-', lw=1, label='Y-axis')
        axis_z, = ax.plot([], [], [], 'b-', lw=1, label='Z-axis')
        ax.legend()

        def init():
            traj_line.set_data([], [])
            traj_line.set_3d_properties([])
            axis_x.set_data([], [])
            axis_x.set_3d_properties([])
            axis_y.set_data([], [])
            axis_y.set_3d_properties([])
            axis_z.set_data([], [])
            axis_z.set_3d_properties([])
            return traj_line, axis_x, axis_y, axis_z
        
        def update(frame):
            x_data = trans_np[:frame + 1, 0]
            y_data = trans_np[:frame + 1, 1]
            z_data = trans_np_zeros[:frame + 1, 2]
            traj_line.set_data(x_data, y_data)
            traj_line.set_3d_properties(z_data)
            R = root_mat_np[frame]
            origin = trans_np[frame]
            ex = np.array([axis_length, 0, 0])
            ey = np.array([0, axis_length, 0])
            ez = np.array([0, 0, axis_length])
            ex = np.dot(R, ex) + origin
            ey = np.dot(R, ey) + origin
            ez = np.dot(R, ez) + origin
            axis_x.set_data([origin[0], ex[0]], [origin[1], ex[1]])
            axis_x.set_3d_properties([origin[2], ex[2]])
            axis_y.set_data([origin[0], ey[0]], [origin[1], ey[1]])
            axis_y.set_3d_properties([origin[2], ey[2]])
            axis_z.set_data([origin[0], ez[0]], [origin[1], ez[1]])
            axis_z.set_3d_properties([origin[2], ez[2]])
            return traj_line, axis_x, axis_y, axis_z
        
        ani = FuncAnimation(fig, update, frames=T, init_func=init, blit=False, interval=1000 / self.dataset.interpolate_fps)
        if save_dir is not None:
            ani.save(save_dir, writer='ffmpeg')
        
        
        # print(f"Visualized trajectory for {title}")
        # plt.show()

    def visualize_lin_vel(self, lin_vel: torch.Tensor, save_path: str = None):
        lin_vel = lin_vel.cpu().detach().numpy().T
        t = np.arange(lin_vel.shape[1], dtype=np.float64) / self.dataset.interpolate_fps
        plt.plot(t, lin_vel[0], label="x")
        plt.plot(t, lin_vel[1], label="y")
        plt.plot(t, lin_vel[2], label="z")
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def flip_pose_actions_left_right(self, data_dict):
        assert self.action_flip_left_right is not None, "Action flip left right is not set. Please set it using set_flipper method."
        # Create a deepcopy
        data_dict = dict(
            (k, v.clone() if isinstance(v, torch.Tensor) else deepcopy(v))
            for k, v in data_dict.items()
        )
        data_dict["trans"][:, 1] *= -1
        rot_mat = rot_vec_to_mat(data_dict["root_orient"])
        rot_mat = flip_rot_mat_left_right(rot_mat)
        data_dict["root_orient"] = rot_mat_to_vec(rot_mat)
        data_dict["actions"] = self.action_flip_left_right.flip(data_dict["actions"])
        data_dict["feet_contact"] = data_dict["feet_contact"][:, [1, 0]].clone()
        return data_dict

    def convert(self, foot_names: list[str], min_subseq_ratio: float = 0.2, data_correction_path: str = None, data_correction_keys: dict[str, str] = None, add_flipped_data = False, floor_contact_offset_map = None, foot_ground_offset: float = 0.0, nan_check: bool = False):
        '''
            Convert the AMASS dataset to the action dataset

            Args:
                foot_names (list[str]): The names of the foot links
                min_subseq_ratio (float): The minimum ratio of the cyclic subsequences to the total sequence length. If len(subseq) < min_subseq_ratio * len(sequence), the sequence is considered as non-cyclic
                data_correction_path (str): The path to the data correction file, currently implemented for AMASS/ACCAD dataset
                data_correction_keys (map): map of key_name: corrected_data_name to be used in the data correction file
                foot_ground_offset (float): The offset to be added to the foot contact points. This is used to correct the foot contact points to the ground level. The offset is in meters.
                nan_check (bool): Whether to check for NaN values in the data
            The data correction file (in .npz format) should contain at least the following:
            - 'root_lin_vel': The root linear velocity (T, 3)
            - 'root_ang_vel': The root angular velocity (T, 3)
            - 'frame_rate': The frame rate of the original data

        '''
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False)
        for idx, data in enumerate(tqdm(dataloader)):
            try:
                fname = data['title'][0]
                # if fname[:3] != "B3 ": continue
                trans = data['trans'].to(self.device).squeeze(0) # (T, 3)

                poses = data['poses'].to(self.device).squeeze(0)
                if len(poses.shape) == 1:
                    poses = poses.unsqueeze(0)

                if len(trans.shape) == 1:
                    trans = trans.unsqueeze(0)
                root_orient = poses[:, :3].to(self.device) # Rotation vector (T, 3)
                root_orient = self.post_process.adjust_root_orient(root_orient)
                
                # self.test_pose_to_action_fps() # Test the pose to action conversion FPS

                # ensure trans & root_orient has same shape[0] as poses
                trans = adjust_shape_0(trans, poses.shape[0])
                root_orient = adjust_shape_0(root_orient, poses.shape[0])
                trans = self.adjust_trans_height(trans, poses)

                pose = poses[:, 3:66].to(self.device) if not self.load_hands else poses[:, 3:].to(self.device)
                actions = self.pose_to_action(pose)
                actions_filtered = self.post_process.filt(actions)

                # import time
                # t_begin = time.time()
                start, end, seq_len, dist = find_longest_cyclic_subsequence(actions_filtered, max_distance=0.15)
                # print(f'find_longest_cyclic_subsequence time: {time.time() - t_begin} for sequence {fname} with length {actions_filtered.shape[0]}')
                # print("sequence dist is: ", dist)
                if seq_len < min_subseq_ratio * actions_filtered.shape[0] or dist > 0.15:
                    # print(f"Cyclic subsequence not found for {fname}")
                    cyclic_subseq = None
                else:
                    cyclic_subseq = (start, end)
                    # print(f"Found cyclic subsequence for {fname} with length {seq_len}")
                    
                export_path = os.path.join(self.export_path, data['fpath'][0])
                os.makedirs(export_path, exist_ok=True)

                if self.visualize:
                    export_dir = os.path.join(export_path, "visualize")
                    if not os.path.exists(export_dir):
                        os.makedirs(export_dir)
                    save_dir = os.path.join(export_dir, f"{fname}.png")
                    self.visualize_filter(actions, actions_filtered, title=fname, save_dir=save_dir, cyclic_subseq=cyclic_subseq)

                save_dict = {
                    "name": fname,
                    "trans": trans,
                    "root_orient": root_orient,
                    "actions_orig": actions,
                    "actions": actions_filtered,
                    "cyclic_subseq": cyclic_subseq,
                    "fps": self.dataset.interpolate_fps,
                }
                
                feet_contact_indices, filtered_root_poses = filt_feet_contact(
                    actions=actions_filtered.unsqueeze(0), # shape (1, num_frames, num_dofs)
                    root_pos=trans.unsqueeze(0), # shape (1, num_frames, 3)
                    root_rot=root_orient.unsqueeze(0), # shape (1, num_frames, 3)
                    fk=RobotKinematics(self.cfg.urdf_path, device=self.device),
                    foot_names=foot_names,
                    threshold=0.065,
                    debug_visualize=False,
                    disable_height=True,
                    debug_save_dir=export_path,
                )
                
                filtered_root_poses = self.post_process.filt(filtered_root_poses.squeeze(0))

                save_dict["trans"] = filtered_root_poses.squeeze(0)
                
                feet_contact_indices = feet_contact_indices.squeeze(0)
                save_dict["feet_contact"] = feet_contact_indices
                

                if data_correction_path is not None:
                    correction_dir = os.path.join(data_correction_path, data['fpath'][0])
                    file_prefix = data['title'][0].split('poses')[0] # ACCAD naming format. Change this if loading other datasets like CMU
                    file_str = correction_dir + '/' + file_prefix + '*.npz'
                    correction_files = glob(file_str)
                    if len(correction_files) == 0:
                        print(f"No correction file found for {fname}, skipping")
                        continue # we prefer to skip the file if no correction file is found and correction mode is enabled
                    correction_file = correction_files[0]
                    correction_data = np.load(correction_file)
                    correction_frame_rate = int(correction_data['frame_rate'])

                    if data_correction_keys is not None:
                        for save_key, key in data_correction_keys.items():
                            if key in correction_data.files:
                                # print(f"Saving {key} to {save_key}")
                                save_data = torch.tensor(correction_data[key], dtype=torch.float32, device=self.device)
                                if "feet_contact" not in key:
                                    save_data = interpolate_trans(save_data, target_fps=self.dataset.interpolate_fps, source_fps=correction_frame_rate)
                                    save_data = self.post_process.filt(save_data)
                                else:
                                    save_data = interpolate_trans(save_data.T, target_fps=self.dataset.interpolate_fps, source_fps=correction_frame_rate)
                                save_data = pad_to_len(save_data, actions_filtered.shape[0])
                                save_dict[save_key] = save_data

                if floor_contact_offset_map is not None:
                    link_tf_mat = self.fk.forward_kinematics(save_dict["actions"])
                    contact_tf_mats = []
                    for name, offset in floor_contact_offset_map.items():
                        tf_mat = link_tf_mat[name]
                        # print(tf_mat.shape)
                        offset = torch.tensor(offset, device=self.device)
                        tf_mat[:, :3, 3] += torch.einsum("bij, j -> bi", tf_mat[:, :3, :3], offset)
                        contact_tf_mats.append(tf_mat.unsqueeze(0))
                    contact_tf_mats = torch.cat(contact_tf_mats).transpose(1, 0)
                else:
                    contact_tf_mats = None

                save_dict = self.post_process(save_dict, contact_tf_mats=contact_tf_mats, filter_pose=True)
                
                
                self.fk.set_target_links(self.cfg.mapping_table.values())
                link_positions = self.fk.forward(actions_filtered, root_trans_offset=save_dict["trans"], root_rot=save_dict["root_orient"]) # (T, N_joints, 3)
                link_velocities = torch.diff(link_positions[:, :, :], dim=0) * self.dataset.interpolate_fps
                link_velocities = torch.cat([torch.zeros_like(link_positions[0:1, :, :]), link_velocities], dim=0)
                rot_mat = rot_vec_to_mat(save_dict["root_orient"])
                link_velocities_local = torch.einsum("ijk, ilj -> ilk", rot_mat, link_velocities)
                save_dict["link_positions"] = link_positions # global positions
                link_positions_local = link_positions
                # link_positions_local[:, :, 0] -= link_positions[0:1, :, 0]
                # link_positions_local[:, :, 1] -= link_positions[0:1, :, 1]
                # Lines above assume the root is the first link which is now deprecated. We calculate by using root translation instead.
                link_positions_local[:, :, 0] -= save_dict["trans"][:, 0:1]
                link_positions_local[:, :, 1] -= save_dict["trans"][:, 1:2]
                
                link_positions_local = torch.einsum("ijk, ilj -> ilk", rot_mat, link_positions_local)
                save_dict["link_positions_local"] = link_positions_local # local positions
                link_velocities_local = torch.einsum("ijk, ilj -> ilk", rot_mat, link_velocities_local)
                save_dict["link_velocities"] = link_velocities_local # local velocities
                
                if self.visualize:
                    export_dir = os.path.join(export_path, "visualize")
                    self.visualize_lin_vel(save_dict["lin_vel"], save_path=os.path.join(export_dir, f"{fname}_lin_vel.png"))

                # Save the converted action using .pkl format
                save_path = os.path.join(export_path, f"{fname}.pkl")

                if nan_check:
                    for key, value in save_dict.items():
                        if not isinstance(value, torch.Tensor):
                            continue
                        if torch.isnan(value).any():
                            print(f"NaN found in {key} for {fname}")
                            continue
                # print(list(save_dict.keys()))
                # print("save_path:", save_path)
                torch.save(save_dict, save_path)

                if add_flipped_data:
                    # Add flipped data and save
                    flipped_save_dict = self.flip_pose_actions_left_right(save_dict)
                    flipped_save_dict = self.post_process(flipped_save_dict, filter_pose=True)
                    flipped_fname = fname + "_flipped"
                    flipped_save_path = os.path.join(export_path, f"{flipped_fname}.pkl")
                    torch.save(flipped_save_dict, flipped_save_path)
            except Exception as e:
                print(f"Error processing {fname}: {e}")
                continue
        print(f"Converted {len(dataloader)} samples to {self.export_path}")