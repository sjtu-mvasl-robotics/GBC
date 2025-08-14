import os
import re
import glob
import numpy as np
from bvh import Bvh # https://github.com/20tab/bvh-python
from tqdm import tqdm
from scipy.spatial.transform import Rotation


scale = 1e-2
transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
root_rot_joint_list = [
    ("ToSpine", np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)),
    ("Spine", np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32))]
euler_convention = "XYZ"

def process(file_path, feet_percentile=50, feet_coeff=1):
    with open(file_path) as f:
        mocap = Bvh(f.read())

    root_info = map(str, mocap.root)

    root_joint_name = None
    frame_rate = None
    for item in root_info:
        if item[:5] == "ROOT ":
            root_joint_name = item[5:]
        elif item[:12] == "Frame Time: ":
            frame_rate = 1 / float(item[12:])

    assert root_joint_name is not None, f"root joint name not found in {root_info}"
    assert frame_rate is not None, f"frame time not found in {root_info}"

    joints = mocap.get_joints_names()
    root_rot_joint = None
    for joint, tf in root_rot_joint_list:
        if joint in joints:
            root_rot_joint = joint
            root_rot_transform = tf
            break
    assert root_rot_joint is not None, f"joint for calculating root rotation {root_rot_joint_list} not found in {joints}"

    joints_par_id = [mocap.joint_parent_index(joint) for joint in joints]
    calc_joints_tf_order = np.argsort(joints_par_id)

    joints_tf = []
    for joint in joints:
        tf = np.eye(4, dtype=np.float32)
        if joint != root_joint_name:
            tf[:3, 3] = mocap.joint_offset(joint)
            tf[:3, 3] *= scale
        joints_tf.append(tf)

    nb_frames = mocap.nframes
    frame_vals = np.float32(mocap.frames)

    # Get the list of position channel indices for each joint and channel
    root_pos_channel_ids = []
    root_joint_channel_id = mocap.get_joint_channels_index(root_joint_name)
    for pos_label in "XYZ":
        # Refer to line 173 in bvh.py
        channel_id = mocap.get_joint_channel_index(root_joint_name, pos_label + "position")
        root_pos_channel_ids.append(root_joint_channel_id + channel_id)

    root_pos = frame_vals[:, root_pos_channel_ids]
    root_pos = root_pos @ transform.T * scale

    root_lin_vel = np.diff(root_pos, axis=0) * frame_rate
    root_lin_vel = np.concatenate((root_lin_vel, root_lin_vel[-1:]))

    # Get the list of rotation channel indices for each joint and channel
    rot_joint = []
    for joint in joints:
        joint_channel_id = mocap.get_joint_channels_index(joint)
        channel_ids, order = [], ""
        for channel in mocap.joint_channels(joint):
            if channel[-len("rotation"):] == "rotation":
                channel_ids.append(joint_channel_id + mocap.get_joint_channel_index(joint, channel))
                order += channel[0]
        rot_joint.append((channel_ids, order))

    # Calculate the transformation matrices for each joint in each frame
    joints_frame_tf = [None for _ in range(len(joints))]
    for joint_id in calc_joints_tf_order:
        joint = joints[joint_id]
        joint_frame_tf = joints_tf[joint_id][None, ...].repeat(nb_frames, axis=0)

        rot_euler = frame_vals[:, rot_joint[joint_id][0]]
        rot = Rotation.from_euler(rot_joint[joint_id][1], rot_euler, degrees=True)
        rot_mat = rot.as_matrix()

        if joint == root_joint_name:
            joint_frame_tf[:, :3, 3] = root_pos
            joint_frame_tf[:, :3, :3] = np.matmul(transform, rot_mat)
        else:
            joint_par_id = joints_par_id[joint_id]
            joint_frame_tf[:, :3, :3] = rot_mat
            joint_frame_tf = np.matmul(joints_frame_tf[joint_par_id], joint_frame_tf)

        joints_frame_tf[joint_id] = joint_frame_tf
    joints_frame_tf = np.array(joints_frame_tf)

    joints_frame_pos = joints_frame_tf[..., :3, 3]
    joints_frame_rot = joints_frame_tf[..., :3, :3]

    feet_joints_list = ["LeftToeBase", "RightToeBase"]
    feet_pos = joints_frame_pos[[joints.index(joint) for joint in feet_joints_list], :, :]
    diff = np.linalg.norm(np.diff(feet_pos, axis=1), axis=2)
    thresh = np.percentile(diff.flatten(), feet_percentile) * feet_coeff
    feet_contact = diff < thresh

    feet_contact_time = np.zeros(feet_contact.shape, dtype=float)
    cur_contact_frames = np.zeros(feet_contact.shape[0], dtype=int)
    for i in range(feet_contact.shape[1]):
        cur_contact_frames = (cur_contact_frames + feet_contact[:, i]) * feet_contact[:, i]
        feet_contact_time[:, i] = np.asarray(cur_contact_frames, dtype=float) / frame_rate

    def rot2euler(rot):
        res_shape = rot.shape[:-1]
        rot = rot.reshape(-1, 3, 3)
        res = Rotation.from_matrix(rot).as_euler(euler_convention)
        return res.reshape(res_shape)

    root_rot = joints_frame_tf[joints.index(root_rot_joint), :, :3, :3]
    root_rot = np.matmul(root_rot, root_rot_transform)
    root_euler = rot2euler(root_rot)
    joints_frame_euler = rot2euler(joints_frame_rot)

    root_rot_diff = np.matmul(root_rot[:-1].transpose(0, 2, 1), root_rot[1:])
    root_rot_vec = Rotation.from_matrix(root_rot_diff).as_rotvec() * frame_rate
    root_rot_vec = np.matmul(root_rot_vec[:, None, :], root_rot[:-1].transpose(0, 2, 1)).squeeze(1)
    root_rot_vec = np.concatenate((root_rot_vec, root_rot_vec[-1:]))

    pose_data = {
        "nb_frames": nb_frames, # number of frames
        "frame_rate": frame_rate, # fps
        "euler_convention": euler_convention, # euler angle convention for rotation, the same as scipy.spatial.transform.Rotation for lower and upper case
        "root_pos": root_pos, # root position in each frame
        "root_rot": root_euler, # root rotation in each frame
        "root_lin_vel": root_lin_vel, # root linear velocity
        "root_ang_vel": root_rot_vec, # root angular velocity (in xyz)
        "feet_contact": feet_contact, # shape (2, nb_frames)
        "feet_contact_time": feet_contact_time,
        "joints": joints, # joint names
        "joints_par_id": joints_par_id, # the indices of the parent joints
        "joints_pos": joints_frame_pos, # joint positions in each frame
        "joints_rot": joints_frame_euler, # joint rotations in each frame
    }
    return pose_data

    
if __name__ == '__main__':
    amass_accad_path = "/home/yifei/dataset/amass/ACCAD"
    accad_bvh_path = "/home/yifei/dataset/accad/bvh"
    save_path = "/home/yifei/dataset/amass_accad_data/ACCAD"

    for file in tqdm(glob.glob(os.path.join(amass_accad_path, "**", "B3 - walk1_poses.npz"), recursive=True)):
        file_prefix = file[:-len("_poses.npz")]

        folder = os.path.basename(os.path.dirname(file))
        # print("File:", file)

        # Check the name of the folder, Male or Female + a digit
        match = re.match(r'^(Male|Female)\d', folder)
        if not match:
            continue
        subject = match.group(0)
        # print("Subject:", subject)

        filename = os.path.basename(file)
        match = re.search(r'[A-Z]\d+', filename)
        if not match:
            continue
        file_id = match.group(0)

        # Allow leading zeros
        file_id = file_id[0] + "*" + file_id[1:]

        # Check if there is extra ID behind the file ID
        extra_name = filename[match.end():].strip()
        extra_match = re.match(r'^[A-Za-z]\d+', extra_name)
        if extra_match:
            file_id += '_' + extra_match.group(0).upper()
        # print("File ID:", file_id)

        bvh_files = glob.glob(os.path.join(accad_bvh_path, "**", f"{subject}_{file_id}_*.bvh"), recursive=True)
        if not bvh_files:
            print("No BVH files found for", file)
            continue

        # Get the shortest matching file (in case of extra id)
        bvh_file = min(bvh_files, key=lambda x: len(x))

        data = process(bvh_file)
        category_path = os.path.relpath(os.path.dirname(file), amass_accad_path)
        os.makedirs(os.path.join(save_path, category_path), exist_ok=True)
        file_prefix = os.path.basename(file_prefix)
        np.savez(os.path.join(save_path, category_path, f"{file_prefix}_accad_data.npz"), **data)