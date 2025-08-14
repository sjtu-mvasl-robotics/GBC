# Convert unitree retargeting dataset to supported pickle
import pandas as pd
import numpy as np
import torch
from GBC.utils.base.base_fk import RobotKinematics as RK
from GBC.utils.base.math_utils import quaternion_to_angle_axis, filt_feet_contact, find_longest_cyclic_subsequence
from GBC.utils.data_preparation.amass_action_converter import TrackingDataPostProcess
from GBC.utils.data_preparation.data_preparation_cfg import *
from GBC.utils.data_preparation.robot_visualizer import RobotVisualizer

from glob import glob
import os
from copy import deepcopy
from types import SimpleNamespace
from GBC.utils.base.math_utils import rot_mat_to_vec, rot_vec_to_mat, quaternion_to_angle_axis
from GBC.utils.data_preparation.robot_flip_left_right import UnitreeH12FlipLeftRight, flip_rot_mat_left_right


H1_2_ORDER = [
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

supported_robots = {
    "H1_2": H1_2_ORDER,
}

def flip_pose_actions_left_right(data_dict, action_flip_left_right):
    # Create a deepcopy
    data_dict = dict(
        (k, v.clone() if isinstance(v, torch.Tensor) else deepcopy(v))
        for k, v in data_dict.items()
    )
    data_dict["trans"][:, 1] *= -1
    rot_mat = rot_vec_to_mat(data_dict["root_orient"])
    rot_mat = flip_rot_mat_left_right(rot_mat)
    data_dict["root_orient"] = rot_mat_to_vec(rot_mat)
    data_dict["actions"] = action_flip_left_right.flip(data_dict["actions"])
    data_dict["feet_contact"] = data_dict["feet_contact"][:, [1, 0]].clone()
    return data_dict

# export data:
# name, trans, root_orient, actions, cyclic_subseq, fps
def export_unitree_rt_to_pkl(
    dataset_path: str,
    export_path: str,
    urdf_path: str,
    fps: int = 60,
    reduce_step: int = 3,
    robot_type: str = "H1_2",
    device: str = "cuda",
    add_flipped_data = False
    ):
    assert robot_type in supported_robots.keys(), f"robot_type {robot_type} not supported"
    fk = RK(urdf_path, device=device)
    joint_order = supported_robots[robot_type]
    assert len(joint_order) == fk.num_dofs, f"number of dof mismatch, expected {len(joint_order)}, got {fk.num_dofs} from urdf: {urdf_path}"

    filter_cfg = FilterCfg()
    cutoff_relative_freq = filter_cfg.filter_cutoff / filter_cfg.filter_sample_rate
    filter_cfg.filter_sample_rate = fps / reduce_step
    filter_cfg.filter_cutoff = cutoff_relative_freq * fps
    post_process = TrackingDataPostProcess(filter_cfg)

    data_files = glob(f"{dataset_path}/*.csv")
    action_mapping = [joint_order.index(joint) for joint in fk.get_dof_names()]
    if add_flipped_data:
        cfg = SimpleNamespace(urdf_path=urdf_path, device=device)
        action_flip_left_right = UnitreeH12FlipLeftRight(cfg)

    for data_file in data_files:
        if "walk1_subject1" not in data_file:
            continue
        data = pd.read_csv(data_file) # shape (num_frames, 3+4+num_dofs)
        name = data_file.split("/")[-1].split(".")[0]
        data = torch.tensor(data.values, dtype=torch.float32).to(device)
        data = data[::reduce_step, ...]
        root_pos = data[:, :3]
        root_quat = data[:, 3:7] # xyzw
        root_quat = root_quat[:, [3, 0, 1, 2]] # wxyz
        root_orient = quaternion_to_angle_axis(root_quat)
        # root_orient = action_converter.adjust_root_orient(root_orient)
        actions = data[:, 7:]
        actions = actions[:, action_mapping]
        
        start, end, seq_len, dist = find_longest_cyclic_subsequence(actions, max_distance=0.25)
        print("sequence dist is: ", dist)
        if seq_len < 30 or dist > 0.25:
            print(f"Cyclic subsequence not found")
            cyclic_subseq = None
        else:
            cyclic_subseq = (start, end)
            print(f"Found cyclic subsequence with length {seq_len}")

        save_dict = {
            "name": name,
            "trans": root_pos,
            "root_orient": root_orient,
            "actions": actions,
            "cyclic_subseq": cyclic_subseq,
            "fps": fps / reduce_step,
        }
        save_dict = post_process(save_dict)
        from scipy.spatial.transform import Rotation as R
        root_orient = R.from_rotvec(save_dict["root_orient"].cpu().detach().numpy()).as_euler("xyz")
        # root_orient = angle_axis_to_ypr(save_dict["root_orient"])
        # root_orient = root_orient[..., [2, 1, 0]].unsqueeze(0)
        root_orient = torch.tensor(root_orient, device=device).unsqueeze(0)

        feet_contact_indices, filtered_root_poses = filt_feet_contact(
            actions=actions.unsqueeze(0), # shape (1, num_frames, num_dofs)
            root_pos=save_dict["trans"].unsqueeze(0), # shape (1, num_frames, 3)
            root_rot=root_orient, # shape (1, num_frames, 3)
            fk=RK(urdf_path, device=device),
            foot_names=["left_ankle_pitch_link", "right_ankle_pitch_link"],
            threshold=0.05,
            debug_visualize=True,
            debug_rv=RobotVisualizer(urdf_path=urdf_path, device=device),
        )
        feet_contact_indices = feet_contact_indices.squeeze(0)
        save_dict["feet_contact"] = feet_contact_indices
        save_path = os.path.join(export_path, f"{name}.pkl")
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        torch.save(save_dict, save_path)
        print(f"converted {name} to {save_path}")

        if add_flipped_data:
            flipped_dict = flip_pose_actions_left_right(save_dict, action_flip_left_right)
            flipped_save_path = os.path.join(export_path, f"{name}_flipped.pkl")
            torch.save(flipped_dict, flipped_save_path)
            print(f"save flipped {name} to {flipped_save_path}")
    