import os
import numpy as np
import sys, contextlib
from urdf_parser_py.urdf import URDF
from scipy.spatial.transform import Rotation as R
import networkx as nx
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum, matmul
from typing import Optional, List
from GBC.utils.base.math_utils import euler_xyz_to_rot_mat


    
class RobotKinematics(nn.Module):
    def __init__(self, urdf_path: str, device: str = 'cpu'):
        """
        Initialize the RobotKinematics with a URDF file and specify the computation device.

        Parameters:
        - urdf_path (str): Path to the URDF file.
        - device (str): Computation device ('cpu' or 'cuda'). Default is 'cpu'.
        """
        super(RobotKinematics, self).__init__()

        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        self.device = torch.device(device if torch.cuda.is_available() and str(device)[:4] == 'cuda' else 'cpu')
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            self.robot = URDF.from_xml_file(urdf_path)
        self.link_map = {link.name: link for link in self.robot.links}
        self.joint_map = {joint.name: joint for joint in self.robot.joints}
        # remove the fixed joints
        self.joint_map = {k: v for k, v in self.joint_map.items() if v.type != 'fixed'}
        self.base_link = self.find_base_link()
        self.kinematic_tree = self.build_kinematic_tree()
        self.global_transforms = {}  # To store global transformation matrices

        self.dof_names = self.get_joint_names()
        self.num_dofs = len(self.dof_names)
        self.target_links = None  # To be set via set_target_links
        # self.apply_joint_limits = apply_joint_limits

    def to(self, device):
        self.device = torch.device(device)
        if self.global_transforms:
            for link_name, T in self.global_transforms.items():
                self.global_transforms[link_name] = T.to(device)
        return self


    def find_base_link(self):
        """
        Find the base link of the robot (link with no parent joint).

        Returns:
        - base_link_name (str)
        """
        child_links = set(joint.child for joint in self.robot.joints)
        all_links = set(link.name for link in self.robot.links)
        base_links = all_links - child_links
        if len(base_links) != 1:
            raise ValueError("URDF should have exactly one base link.")
        return base_links.pop()       

    def get_joint_names(self):
        """
        Get the names of all joints in the robot.

        Returns:
        - joint_names (list)
        """
        return list(self.joint_map.keys())
    
    def get_link_names(self):
        """
        Get the names of all links in the robot.

        Returns:
        - link_names (list)
        """
        return [link.name for link in self.robot.links]
    
    def build_kinematic_tree(self):
        """
        Build a kinematic tree using NetworkX.

        Returns:
        - G (networkx.DiGraph): Directed graph representing the kinematic chain.
        """
        G = nx.DiGraph()
        for joint in self.robot.joints:
            parent = joint.parent
            child = joint.child
            G.add_edge(parent, child, joint=joint)
        return G
    
    def rotation_matrix_batch(self, rot_vecs):
        """
        Compute the rotation matrix from a rotation vector using PyTorch.

        Parameters:
        - rot_vecs (torch.Tensor): Rotation vector (angle-axis), shape (B, 3).

        Returns:
        - rot_mat (torch.Tensor): Rotation matrix, shape (B, 3, 3).
        """
        angles = torch.norm(rot_vecs, dim=1, keepdim=True)  # (B, 1)
        zero_mask = angles.squeeze(1) == 0  # (B,)
        angles = angles.squeeze(1)  # (B,)

        # Avoid division by zero
        angles_safe = torch.where(zero_mask, torch.ones_like(angles), angles)  # (B,)
        axis = rot_vecs / angles_safe.unsqueeze(1)  # (B, 3)

        # Compute sine and cosine
        sin_theta = torch.sin(angles_safe)
        cos_theta = torch.cos(angles_safe)
        one_minus_cos = 1 - cos_theta

        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]  # Each (B,)
        sin_theta = sin_theta.unsqueeze(1)  # (B, 1)
        cos_theta = cos_theta.unsqueeze(1)  # (B, 1)
        one_minus_cos = one_minus_cos.unsqueeze(1)  # (B, 1)

        # Compute rotation matrices
        rot_mat = torch.zeros((rot_vecs.shape[0], 3, 3), device=self.device)
        rot_mat[:, 0, 0] = cos_theta.squeeze(1) + x * x * one_minus_cos.squeeze(1)
        rot_mat[:, 0, 1] = x * y * one_minus_cos.squeeze(1) - z * sin_theta.squeeze(1)
        rot_mat[:, 0, 2] = x * z * one_minus_cos.squeeze(1) + y * sin_theta.squeeze(1)

        rot_mat[:, 1, 0] = y * x * one_minus_cos.squeeze(1) + z * sin_theta.squeeze(1)
        rot_mat[:, 1, 1] = cos_theta.squeeze(1) + y * y * one_minus_cos.squeeze(1)
        rot_mat[:, 1, 2] = y * z * one_minus_cos.squeeze(1) - x * sin_theta.squeeze(1)

        rot_mat[:, 2, 0] = z * x * one_minus_cos.squeeze(1) - y * sin_theta.squeeze(1)
        rot_mat[:, 2, 1] = z * y * one_minus_cos.squeeze(1) + x * sin_theta.squeeze(1)
        rot_mat[:, 2, 2] = cos_theta.squeeze(1) + z * z * one_minus_cos.squeeze(1)

        # Handle zero rotation vectors
        rot_mat[zero_mask] = torch.eye(3, device=self.device)
        
        return rot_mat  # (B, 3, 3)
    

    def get_joint_transform_batch(self, joint, joint_values):
        """
        Compute the transformation matrix for a given joint and joint value using PyTorch.

        Parameters:
        - joint (urdf_parser_py.urdf.Joint): The joint object.
        - joint_values (torch.Tensor): The joint position value(s) of shape (B,).

        Returns:
        - T (torch.Tensor): Bx4x4 transformation matrix.
        """

        # Joint origin
        origin_xyz = torch.tensor(joint.origin.xyz if joint.origin.xyz else [0.0, 0.0, 0.0],
                                    dtype=torch.float32, device=self.device, requires_grad=False)
        origin_xyz = origin_xyz.unsqueeze(0).expand(joint_values.shape[0], -1)  # (B, 3)
        origin_rpy = torch.tensor(joint.origin.rpy if joint.origin.rpy else [0.0, 0.0, 0.0],
                                    dtype=torch.float32, device=self.device, requires_grad=False)
        origin_rpy = origin_rpy.unsqueeze(0).expand(joint_values.shape[0], -1)  # (B, 3)

        # Convert RPY to rotation matrix using PyTorch
        rot_mat_origin = euler_xyz_to_rot_mat(origin_rpy)  # (B, 3, 3)

        # Initialize transformation with origin
        T = torch.eye(4, device=self.device).unsqueeze(0).repeat(joint_values.shape[0], 1, 1)  # (B, 4, 4)
        T[:, :3, :3] = rot_mat_origin
        T[:, :3, 3] = origin_xyz  # (B, 4, 4)

        # Joint movement
        joint_type = joint.type
        axis = torch.tensor(joint.axis if joint.axis else [0.0, 0.0, 1.0],
                            dtype=torch.float32, device=self.device)
        axis = axis / torch.norm(axis)  # Ensure it's a unit vector
        axis = axis.unsqueeze(0).expand(joint_values.shape[0], -1)  # (B, 3)

        if joint_type in ['revolute', 'continuous']:
            # Rotation about the joint axis
            rot_vecs = joint_values.unsqueeze(1) * axis  # (B, 3)
            rot_mats = self.rotation_matrix_batch(rot_vecs)  # (B, 3, 3)

            T_joint = torch.eye(4, device=self.device).unsqueeze(0).repeat(joint_values.shape[0], 1, 1)  # (B, 4, 4)
            T_joint[:, :3, :3] = rot_mats
            # T_joint[:, :3, 3] remains [0,0,0]

            T = torch.bmm(T, T_joint)  # (B, 4, 4)

        elif joint_type == 'prismatic':
            # Translation along the joint axis
            displacements = joint_values.unsqueeze(1) * axis  # (B, 3)
            T_joint = torch.eye(4, device=self.device).unsqueeze(0).repeat(joint_values.shape[0], 1, 1)  # (B, 4, 4)
            T_joint[:, :3, 3] = displacements
            # T_joint[:, :3, :3] remains identity

            T = torch.bmm(T, T_joint)  # (B, 4, 4)

        elif joint_type == 'fixed':
            # No additional transformation
            pass

        else:
            raise NotImplementedError(f"Joint type '{joint_type}' is not supported.")

        return T  # (B, 4, 4)

    def get_dof_names(self):
        """
        Return the names of degrees of freedom (DoF) for the robot in the order of actions.

        Returns:
        - dof_names (list): List of joint names in the order of DoF.
        """
        return self.dof_names
    
    def get_dof_limits(self):
        """
        Return the joint limits for the robot.

        Returns:
        - min_vals (torch.Tensor): Minimum values for each joint.
        - max_vals (torch.Tensor): Maximum values for each joint.
        """
        if not hasattr(self, 'joint_map'):
            raise ValueError("Joint map not found. Load a URDF file first.")
        
        if not hasattr(self, 'dof_min_vals') or not hasattr(self, 'dof_max_vals'):
            dof_limits = {}
            for joint_name, joint in self.joint_map.items():
                if joint.limit:
                    dof_limits[joint_name] = (joint.limit.lower, joint.limit.upper)
                else:
                    dof_limits[joint_name] = (-np.inf, np.inf)

            dof_min_vals = torch.tensor([dof_limits[joint_name][0] for joint_name in self.dof_names], device=self.device)
            dof_max_vals = torch.tensor([dof_limits[joint_name][1] for joint_name in self.dof_names], device=self.device)
            self.dof_min_vals = dof_min_vals
            self.dof_max_vals = dof_max_vals

        return self.dof_min_vals, self.dof_max_vals
    
    def get_dof_related_link_names(self):
        """
        Return the names of the links that are connected to the free joints.

        Returns:
        - dof_related_link_names (list): List of link names connected to free joints.
        """
        if not hasattr(self, 'dof_related_link_names'):
            queue = [self.base_link]
            links = []
            joints = []
            while queue:
                current_link = queue.pop(0)
                
                for child in self.kinematic_tree.successors(current_link):
                    joint = self.kinematic_tree[current_link][child]['joint']
                    if joint.type != 'fixed':
                        links.append(child)
                        joints.append(joint.name)
                    queue.append(child)

            assert len(links) == self.num_dofs, "Number of links connected to free joints is not equal to the number of DoFs. Please check the URDF file."
            # sort links and joints by dof_names
            sorted_links = []
            for joint in list(self.dof_names):
                idx = joints.index(joint)
                sorted_links.append(links[idx])

            self.dof_related_link_names = sorted_links
        return self.dof_related_link_names
        
    def get_end_effectors(self):
        """
        Return the names of the end effectors for the robot.

        Returns:
        - end_effectors (dict): Dictionary of end effector names and their corresponding joint names.
        """
        if not hasattr(self, 'end_effectors'):
            end_effector_map = {}
            queue = [(self.base_link, None, "")]
            while queue:
                current_link, current_joint, cur_joint_type = queue.pop(0)

                # End effector: current link has no child or all children are fixed
                if not self.kinematic_tree.successors(current_link) or (all(self.kinematic_tree[current_link][child]['joint'].type == 'fixed' for child in self.kinematic_tree.successors(current_link)) and 'waist' not in current_link): # special fix for fourier attributes
                    if cur_joint_type != 'fixed':
                        end_effector_map[current_link] = current_joint
                else:
                    for child in self.kinematic_tree.successors(current_link):
                        joint = self.kinematic_tree[current_link][child]['joint']
                        # if joint.type == 'fixed':
                        #     continue
                        queue.append((child, joint.name, joint.type))
            self.end_effectors = end_effector_map
        return self.end_effectors
    
    def get_certain_part_body_link_and_joint_names(self, root_joints: List[str]):
        """
        Return the names of the links and joints that are connected to the root joints.

        Parameters:
        - root_joints (list): List of root joint names.
        """
        lj_map = {}
        base_successors_joints = [self.kinematic_tree[self.base_link][child]['joint'].name for child in self.kinematic_tree.successors(self.base_link)]
        assert len(base_successors_joints) > 0, "Base link has no successors. Check the URDF file."
        # check if root_joints are in base_successors_joints
        for root_joint in root_joints:
            assert root_joint in base_successors_joints, f"Root joint '{root_joint}' not found in base successors. Available base successors: {base_successors_joints}"

        queue = []
        for child in self.kinematic_tree.successors(self.base_link):
            link_name = child
            joint_name = self.kinematic_tree[self.base_link][child]['joint'].name
            joint_type = self.kinematic_tree[self.base_link][child]['joint'].type
            if joint_name not in root_joints:
                continue
            queue.append((link_name, joint_name, joint_type))

        while queue:
            current_link, current_joint, cur_joint_type = queue.pop(0)
            if cur_joint_type != 'fixed':
                lj_map[current_link] = current_joint
            for child in self.kinematic_tree.successors(current_link):
                joint = self.kinematic_tree[current_link][child]['joint']
                # if joint.type == 'fixed':
                #     continue
                queue.append((child, joint.name, joint.type))
        return lj_map



    def forward_kinematics(self, dof_pos: torch.Tensor, root_trans_offset: Optional[torch.Tensor] = None, root_rot: Optional[torch.Tensor] = None, link_trans_offset: Optional[dict[str, torch.Tensor]] = None):
        """
        Compute forward kinematics for the robot given joint values using PyTorch.

        Parameters:
        - dof_pos (torch.Tensor): BxN tensor of joint values, where N is the number of DoFs.
        - root_trans_offset (torch.Tensor): Bx3 tensor of root translation offsets. Default is None.
        - root_rot (torch.Tensor): Bx3 tensor of root rotation angles in XYZ order. Default is None.
        - link_trans_offset (dict): Dictionary of link names to (3,) translation offsets. Default is None. This translation is applied after the joint transformation.

        Returns:
        - global_transforms (dict): Mapping from link names to their Bx4x4 global transformation matrices (torch.Tensor).
        """

        if dof_pos.dim() != 2:
            raise ValueError("dof_pos must be a 2D tensor.")
        if dof_pos.shape[1] != self.num_dofs:
            raise ValueError(f"dof_pos must have {self.num_dofs} columns.")
        
        if root_trans_offset is not None:
            root_T = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(dof_pos.shape[0], 1, 1)  # (B, 4, 4)
            root_T[:, :3, 3] = root_trans_offset
            self.global_transforms = {self.base_link: root_T}
        else:
            self.global_transforms = {self.base_link: torch.eye(4, device=self.device).unsqueeze(0).repeat(dof_pos.shape[0], 1, 1)}

        if root_rot is not None:
            root_rot_mat = euler_xyz_to_rot_mat(root_rot)
            self.global_transforms[self.base_link][:, :3, :3] = root_rot_mat

        # Traverse the tree in breadth-first order
        queue = [self.base_link]
        while queue:
            current_link = queue.pop(0)
            children = list(self.kinematic_tree.successors(current_link))
            
            for child in children:
                # Find the joint connecting current_link to child
                joint = self.kinematic_tree[current_link][child]['joint']
                joint_name = joint.name
                if joint_name in self.dof_names:
                    idx = list(self.dof_names).index(joint_name)
                    joint_val = dof_pos[:, idx]  # (B,)
                    # Compute transformation for this joint
                elif joint.type == "fixed":
                    # SPECIAL NOTICE FOR UNITREE G1 USERS:
                    # UNITREE TENDS TO ADD NO GEOMETRY XYZ IN URDF, WHICH CAUSES FAILURE TO THE FOLLOWING CALCULATION SINCE WE STILL REQUIRES TO INCLUDE SOME OF THE FIXED JOINTS TO BUILD THE KINEMATIC TREE.
                    # IF ANY UNITREE DEVELOPER IS READING THIS, PLEASE ADD THE XYZ IN URDF FOR FIXED JOINTS.
                    
                    # FURTHER NOTICE FOR FOURIER GR1 USERS:
                    # WE DO NOT RECOMMEND TO ADD FIXED JOINTS TO CONNECT BOTH PARTS OF THE ROBOT
                    
                    # special fix for G1 series
                    if not hasattr(joint, "origin") or not hasattr(joint.origin, "xyz") or joint.origin.xyz is None:
                        continue
                    
                    joint_val = torch.zeros_like(dof_pos[:, 0])
                else:
                    raise ValueError(f"Joint '{joint_name}' not found in DoF order.")
                T_joint = self.get_joint_transform_batch(joint, joint_val)  # (B, 4, 4)
                
                # Compute global transformation
                T_parent = self.global_transforms[current_link]  # (B, 4, 4)
                T_child = torch.bmm(T_parent, T_joint)  # (B, 4, 4)

                # Store the transformation
                if link_trans_offset is not None and child in link_trans_offset.keys():
                    child_translation_mat = torch.eye(4, dtype=T_joint.dtype, device=T_joint.device).unsqueeze(0).repeat(T_child.shape[0], 1, 1)
                    child_translation_mat[:, :3, 3] = link_trans_offset[child].unsqueeze(0)
                    T_child = torch.bmm(T_child, child_translation_mat)

                self.global_transforms[child] = T_child  # (B, 4, 4)
                
                # Add child to the queue
                queue.append(child)

        return self.global_transforms
    
    def inverse_kinematics(self, target_link_pose: torch.Tensor, root_trans_offset: Optional[torch.Tensor] = None, root_rot: Optional[torch.Tensor] = None, max_iter: int = 1000, tol: float = 1e-6, verbose: bool = False, lr: float = 0.01, log_interval: int = 10):
        """
        Inverse Kinematics for given link global translation. All link positions must be provided.
        
        Parameters:
        - target_link_pose (torch.Tensor): Target link positions, shape (B, num_target_links, 3).
        - root_rot (torch.Tensor): Root rotation angles in XYZ order, shape (B, 3).
        - max_iter (int): Maximum number of iterations. Default is 100.
        - tol (float): Tolerance for convergence. Default is 1e-6.        
        - verbose (bool): Whether to print loss during optimization. Default is False.
        - lr (float): Learning rate for optimization. Default is 0.01.
        - log_interval (int): Log interval for printing loss. Default is 10.
        """
        # check if target link pose matches number of target links
        if target_link_pose.shape[1] != len(self.target_links):
            raise ValueError("Number of target links does not match target link pose.")
        
        actions = torch.randn(target_link_pose.shape[0], self.num_dofs, device=self.device)
        actions = torch.autograd.Variable(actions, requires_grad=True)
        optimizer = torch.optim.Adam([actions], lr=0.01)
        for i in range(max_iter):
            link_pose = self.forward(actions, root_trans_offset=root_trans_offset, root_rot=root_rot)
            loss = F.mse_loss(link_pose, target_link_pose) * 5000.0
            action_violation = F.relu(torch.abs(actions) - 1.57).sum(dim=1).mean() * 400.0
            loss += action_violation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose and i % log_interval == 0:
                print(f"Iteration {i}, Loss: {loss.item()}, Violation Loss: {action_violation.item()}")
            if loss.item() < tol:
                break
        return actions.detach()
    
    def set_target_links(self, target_links: List[str]):
        """
        Set target links for forward kinematics.

        Parameters:
        - target_links (List[str]): List of link names to compute forward kinematics for.
        """
        self.target_links = target_links

    def forward(self, dof_pos: torch.Tensor, root_trans_offset: Optional[torch.Tensor] = None, root_rot: Optional[torch.Tensor] = None, link_trans_offset: Optional[dict[str, torch.Tensor]] = None):
        """
        Perform forward kinematics and retrieve target link positions.

        Parameters:
        - dof_pos (torch.Tensor): Joint values, shape (B, num_dof).
        - root_trans_offset (Optional[torch.Tensor]): Root translation offsets, shape (B, 3).
        - root_rot (Optional[torch.Tensor]): Root rotation angles in XYZ order, shape (B, 3).
        - link_trans_offset (Optional[dict[str, torch.Tensor]]): Dictionary of link names to translation offsets, shape (3,)

        Returns:
        - target_link_positions (torch.Tensor): Positions of target links, shape (B, num_target_links, 3).
        """
        if self.target_links is None:
            raise ValueError("Target links not set. Call set_target_links first.")
        
        self.forward_kinematics(dof_pos, root_trans_offset, root_rot, link_trans_offset)
        target_link_positions = []
        for link_name in self.target_links:
            if link_name not in self.global_transforms:
                raise ValueError(f"Link '{link_name}' has not been transformed yet. Run forward_kinematics first.")
            T = self.global_transforms[link_name]  # (B, 4, 4)
            position = T[:, :3, 3]  # (B, 3)
            target_link_positions.append(position)

        return torch.stack(target_link_positions, dim=1)  # (B, num_target_links, 3)
    
    def visualize(self):
        """
        Visualize the kinematic tree using NetworkX and Matplotlib.
        """
        pos = nx.spring_layout(self.kinematic_tree)
        plt.figure(figsize=(8, 6))
        nx.draw(self.kinematic_tree, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
        plt.title("Robot Kinematic Tree")
        plt.show()

    def visualize_joints(self, show: bool =True, axis_length: float =0.2, default_batch: int = 0, bm: Optional[torch.Tensor] = None, title_prefix: str = "", save_path: str = None, title_override: Optional[str] = None, fixed_axis: bool = False):
        """
        Visualize the robot's joint positions and orientations in 3D.

        Parameters:
        - show (bool): Whether to display the plot immediately. Default is True.
        - figsize (tuple): Size of the matplotlib figure. Default is (10, 8).
        - axis_length (float): Length of the orientation axes arrows. Default is 0.2.
        - default_batch (int): Index of the batch to visualize. Default is 0.
        - bm (torch.Tensor): BxNx3 tensor of body model vertices (to compare with fk results). Default is None.
        - title_prefix (str): Prefix for the plot title. Default is "".
        - save_path (str): Path to save the plot. Default is None.
        - title_override (str): Override the title with this string. Default is None.
        - fixed_axis (bool): Whether to fix the plot axis ranges. Default is False.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title_prefix + "Joint Positions and Orientations after FK")
        ax.view_init(elev=30, azim=45)  # Adjust viewing angle as needed

        if bm is not None:
            bm = bm[default_batch].cpu().numpy()
            bm = bm - bm[0]

            print("bm shape:", bm.shape)

        # Define colors for the axes
        colors = ['r', 'g', 'b']  # X: red, Y: green, Z: blue

        # Iterate through each link to plot its joint position and orientation
        for link_name, T in self.global_transforms.items():
            pos = T[default_batch, :3, 3].detach().cpu().numpy()
            ori = T[default_batch, :3, :3].detach().cpu().numpy()

            # Plot the joint position
            if link_name in self.target_links:
                ax.scatter(pos[0], pos[1], pos[2], color='b', s=40)
                ax.text(pos[0], pos[1], pos[2], link_name, color='b', fontsize=8)
            else:
                if bm is not None:
                    continue
                ax.scatter(pos[0], pos[1], pos[2], color='k', s=30)

            # Plot the orientation axes
            for i in range(3):
                axis = ori[:, i] * axis_length
                ax.quiver(pos[0], pos[1], pos[2],
                          axis[0], axis[1], axis[2],
                          color=colors[i], length=axis_length, normalize=True)
                
            if bm is not None:
                ax.scatter(bm[:, 0], bm[:, 1], bm[:, 2], color='red', s=50)
                
            

        # Optionally, connect the joints with lines to show the kinematic chain
        for parent, child in self.kinematic_tree.edges():
            if parent in self.global_transforms and child in self.global_transforms:
                parent_pos = self.global_transforms[parent][default_batch, :3, 3].detach().cpu().numpy()
                child_pos = self.global_transforms[child][default_batch, :3, 3].detach().cpu().numpy()
                ax.plot([parent_pos[0], child_pos[0]],
                        [parent_pos[1], child_pos[1]],
                        [parent_pos[2], child_pos[2]],
                        color='gray', linewidth=1)

        # Adjust the plot limits based on joint positions
        all_positions = np.array([T[default_batch, :3, 3].detach().cpu().numpy() for T in self.global_transforms.values()])
        max_range = (all_positions.max(axis=0) - all_positions.min(axis=0)).max() / 2.0
        mid_x, mid_y, mid_z = all_positions.mean(axis=0)
        if not fixed_axis:
            if bm is not None:
                bm_range = (bm.max(axis=0) - bm.min(axis=0)).max() / 2.0
                max_range = max(max_range, bm_range)
                if bm_range > max_range:
                    mid_x, mid_y, mid_z = bm.mean(axis=0)
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        else:
            ax.set_xlim(-0.7, 0.7)
            ax.set_ylim(-0.7, 0.7)
            ax.set_zlim(-0.7, 0.7)


        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])

        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not title_override:
                plt.savefig(save_path + '/' + title_prefix + "joint_positions_orientations.png")
            else:
                plt.savefig(save_path + '/' + title_override + ".png")
            
            
            # print("Saved plot to:", save_path + '/' + title_prefix + "joint_positions_orientations.png")

        if show:
            plt.show()

        plt.close(fig)

