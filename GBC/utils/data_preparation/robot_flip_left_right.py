import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from GBC.utils.data_preparation.data_preparation_cfg import AMASSActionConverterCfg
from GBC.utils.base.base_fk import RobotKinematics
from GBC.utils.base.math_utils import *
import roma


def flip_rot_mat_left_right(rot_mat):
    shape = rot_mat.shape
    rot_mat = rot_mat.view(-1, 3, 3)
    rot_mat[:, 1, 0] *= -1
    rot_mat[:, 1, 2] *= -1
    rot_mat[:, 0, 1] *= -1
    rot_mat[:, 2, 1] *= -1
    return rot_mat.view(shape)


class RobotFlipLeftRight:
    ANGLE_ID = "rpy"
    AXIS_ID = "xyz"
    FILE_NAME = "_flipped_poses.pkl"

    def __init__(self, cfg: AMASSActionConverterCfg | None = None):
        if cfg is not None:
            self.cfg = cfg
            self.urdf_path = cfg.urdf_path
            self.device = cfg.device
        else:
            self.urdf_path = None
            self.device = None
            self.cfg = None

        self.flip_sign_ids = [] # Only flip the sign of joint positions
        self.swap_dict = {} # Only swap between left and right
        self.flip_rpy_dict = {} # Swap and flip
        self.is_prepared = False

        if cfg is not None:
            self.prepare_flip_joint_ids()

    def set_device(self, device: str):
        self.device = device

    def prepare_flip_joint_ids(self, dof_names: list[str] | None = None):
        raise NotImplementedError

    def __call__(self, q):
        return self.flip(q)

    def flip(self, q: torch.Tensor, is_velocity: bool = True):
        assert self.is_prepared, "Please prepare the flip joint ids first"
        # set device according to q
        self.set_device(q.device)
        original_shape = q.shape
        assert original_shape[-1] == self.num_dofs, f"Last dimension must be num_dofs ({self.num_dofs}), but got {original_shape[-1]}"
        
        # Flatten all dimensions except the last one
        q_flat = q.view(-1, self.num_dofs)
        new_q = q_flat.clone()
        self.flip_sign_ids = self.flip_sign_ids.to(self.device)

        # Flip sign
        if self.flip_sign_ids.numel() > 0:
            new_q[:, self.flip_sign_ids] *= -1

        # Swap left and right joints
        for li, ri in self.swap_dict.values():
            new_q[:, li] = q_flat[:, ri]
            new_q[:, ri] = q_flat[:, li]

        # Flip left and right rpy joints
        for flip_rpy in self.flip_rpy_dict.values():
            if not is_velocity:
                rot_mats = []
                for i in range(2):
                    axis, ids = flip_rpy[i]
                    # angles = q_flat[:, ids].cpu().detach().numpy()
                    # rot_mat = Rotation.from_euler(axis, angles).as_matrix()
                    # rot_mat_val = euler_to_rot_mat(axis,q_flat[:, ids]).cpu().detach().numpy()
                    # assert np.allclose(rot_mat, rot_mat_val)
                    rot_mat = euler_to_rot_mat(axis, q_flat[:, ids])
                    rot_mats.append(rot_mat)
                # rot_mats = torch.tensor(np.array(rot_mats), device=self.device)
                rot_mats = torch.stack(rot_mats, dim=0)
                rot_mats = flip_rot_mat_left_right(rot_mats)
                # Convert the numpy operations to PyTorch
                rot_mats = torch.flip(rot_mats, dims=[0])  # Equivalent to rot_mats[::-1, ...]
                # rot_mats = rot_mats.cpu().detach().numpy()
                for i in range(2):
                    joint_axis, ids = flip_rpy[i]
                    full_joint_axis = joint_axis
                    for c in self.AXIS_ID:
                        if c not in full_joint_axis:
                            full_joint_axis += c

                    # angles = Rotation.from_matrix(rot_mats[i].detach().cpu().numpy()).as_euler(full_joint_axis)
                    # angles_val = roma.rotmat_to_euler(convention=full_joint_axis, rotmat=rot_mats[i])
                    # assert np.allclose(angles, angles_val.detach().cpu().numpy())
                    # print("full joint axis:", full_joint_axis)
                    # angles = torch.tensor(angles, dtype=new_q.dtype, device=self.device)
                    angles = roma.rotmat_to_euler(convention=full_joint_axis, rotmat=rot_mats[i])
                    for i, axis in enumerate(full_joint_axis):
                        if axis in joint_axis:
                            joint_id = ids[joint_axis.index(axis)]
                            new_q[:, joint_id] = angles[:, i]
                        # else:
                        #     assert torch.allclose(angles[:, i], torch.zeros_like(angles[:, i]))

            else:
                # Based on the rotation matrix flip, the reflection is across the XZ plane.
                # An angular velocity vector ω = [ωx, ωy, ωz] transforms to ω' = [-ωx, ωy, -ωz].
                left_joint_axis, left_ids = flip_rpy[0]
                right_joint_axis, right_ids = flip_rpy[1]

                # Assume left and right joints use the same axis convention (e.g., 'xyz')
                assert left_joint_axis == right_joint_axis
                joint_axis = left_joint_axis.lower()

                # Define the sign change for each axis ('xyz')
                sign_flips_map = {'x': -1.0, 'y': 1.0, 'z': -1.0}
                
                sign_flips = torch.tensor(
                    [sign_flips_map[axis] for axis in joint_axis],
                    device=q_flat.device, dtype=q_flat.dtype
                )
                
                # Get original left and right velocities
                left_vel = q_flat[:, left_ids]
                right_vel = q_flat[:, right_ids]

                # Apply the flip and swap the joints
                # New left joint gets the flipped velocity of the old right joint
                new_q[:, left_ids] = right_vel * sign_flips
                # New right joint gets the flipped velocity of the old left joint
                new_q[:, right_ids] = left_vel * sign_flips

        return new_q.view(original_shape)

    @property
    def __name__(self):
        return self.__class__.__name__
    
    @property
    def prepared(self):
        return getattr(self, 'is_prepared', False)

    def __str__(self):
        return "RobotFlipLeftRight"

    def __repr__(self):
        return "RobotFlipLeftRight"


class UnitreeH12FlipLeftRight(RobotFlipLeftRight):
    def __init__(self, cfg=None):
        super().__init__(cfg)

    def prepare_flip_joint_ids(self, dof_names: list[str] | None = None):
        if dof_names is None:
            self.fk = RobotKinematics(self.urdf_path)
            self.dof_names = self.fk.dof_names
        else:
            self.dof_names = dof_names

        self.num_dofs = len(self.dof_names)

        self.flip_sign_ids = [
            self.dof_names.index(name) for name in ("torso_joint",)
        ]
        self.flip_sign_ids = torch.tensor(self.flip_sign_ids, device=self.device)

        self.swap_dict = {}
        self.flip_rpy_dict = {}
        for i in reversed(range(len(self.dof_names))):
            joint_name = self.dof_names[i]
            if "left" in joint_name or "right" in joint_name:
                names = joint_name.split("_")
                name = names[1]
                if len(names) == 3:
                    # No roll, pitch, yaw, simply swap left and right
                    if name not in self.swap_dict:
                        self.swap_dict[name] = []
                    self.swap_dict[name].append(i)
                else:
                    if name not in self.flip_rpy_dict:
                        self.flip_rpy_dict[name] = [["", []] for _ in range(2)]
                    side_id = ["left", "right"].index(names[0])
                    angle_id = names[2][0]
                    self.flip_rpy_dict[name][side_id][0] += self.AXIS_ID[self.ANGLE_ID.index(angle_id)]
                    self.flip_rpy_dict[name][side_id][1].append(i)

        self.is_prepared = True
        
class TurinV3FlipLeftRight(RobotFlipLeftRight):
    def __init__(self, cfg = None):
        super().__init__(cfg)

    def prepare_flip_joint_ids(self, dof_names: list[str] | None = None):
        if dof_names is None:
            self.fk = RobotKinematics(self.urdf_path)
            self.dof_names = self.fk.dof_names
        else:
            self.dof_names = dof_names
        self.num_dofs = len(self.dof_names)
        
        # design principle: switch mirroring joints and flip the sign for non-mirroring joints
        self.flip_sign_ids = [
            self.dof_names.index(name) for name in ("head_yaw", "waist_yaw", "waist_roll") # no non-mirroring joints
        ]
        self.flip_sign_ids = torch.tensor(self.flip_sign_ids, device=self.device)

        self.swap_dict = {} # for joints that contains single dof (no roll, pitch, yaw)
        self.flip_rpy_dict = {} # for joints that contains roll, pitch, yaw
        for i in reversed(range(len(self.dof_names))):
            if i in self.flip_sign_ids:
                continue
            joint_name = self.dof_names[i]
            names = joint_name.split("_")
            name = "_".join(names[1:]) # remove the first left-right identifier
            if len(names) == 2: # single dof
                if name not in self.swap_dict:
                    self.swap_dict[name] = []
                self.swap_dict[name].append(i)
            else: # roll, pitch, yaw
                if name not in self.flip_rpy_dict:
                    self.flip_rpy_dict[name] = [["", []] for _ in range(2)]
                side_id = ["l", "r"].index(names[0])
                angle_id = names[2][0]
                self.flip_rpy_dict[name][side_id][0] += self.AXIS_ID[self.ANGLE_ID.index(angle_id)]
                self.flip_rpy_dict[name][side_id][1].append(i)

        self.is_prepared = True
