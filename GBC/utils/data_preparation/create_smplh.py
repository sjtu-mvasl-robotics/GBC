import numpy as np
import torch
from GBC.utils.base.base_fk import RobotKinematics
from scipy.spatial.transform import Rotation as R
from human_body_prior.body_model.body_model import BodyModel
from typing import List, Tuple, Dict, Any, Optional
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer

import os
# body_parms = {
#     'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
#     'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
#     'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
#     'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
#     'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
#     'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
# }

class SMPLHFitter:
    def __init__(self,
                 smplh_model_path: str,
                 dmpls_model_path: str,
                 urdf_path: str,
                 device: str = "cuda"):
        self.robot = RobotKinematics(urdf_path, device=device)
        num_betas = 16
        num_dmpls = 8
        self.device = device
        self.body_model = BodyModel(bm_fname=smplh_model_path, dmpl_fname=dmpls_model_path, num_betas=num_betas, num_dmpls=num_dmpls).to(device)
        self.smplh_body_names = ['Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist']
        self.smplh_extend_names = ['L_Index1', 'L_Index2', 'L_Index3', 'L_Middle1', 'L_Middle2', 'L_Middle3', 'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 'L_Ring1', 'L_Ring2', 'L_Ring3', 'L_Thumb1', 'L_Thumb2', 'L_Thumb3', 'R_Index1', 'R_Index2', 'R_Index3', 'R_Middle1', 'R_Middle2', 'R_Middle3', 'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 'R_Ring1', 'R_Ring2', 'R_Ring3', 'R_Thumb1', 'R_Thumb2', 'R_Thumb3']
        self.mapping_table = None
        self.fit_res = None
        self.body_parms = None

    def set_mapping_table(self, mapping_table: Dict[str, str]):
        '''
        Mapping table: Dict[str, str]

        Mapping names from the SMPLH model to the URDF model
        '''
        self.mapping_table = mapping_table

    def create_model_input(self, root_orient: torch.Tensor, pose_body: torch.Tensor, pose_hand: torch.Tensor, trans: torch.Tensor, betas: torch.Tensor, dmpls: torch.Tensor):
        
        body_parms = {
            'root_orient': root_orient,
            'pose_body': pose_body,
            'pose_hand': pose_hand,
            'trans': trans,
            'betas': betas,
            'dmpls': dmpls
        }
        return body_parms
    
    # def reset_urdf_pose(self, root_trans_offset: torch.Tensor = None):
    #     # set forward kinematics to zero for all joints
    #     self.robot.reset_urdf_pose(root_trans_offset)


    def fit(self, 
            max_iter: int = 1000, 
            lr : float = 0.01, 
            tol: float = 1e-6, 
            verbose: bool = False, 
            offset_map: Optional[Dict[str, torch.Tensor]] = None, 
            rotation_map: Optional[Dict[str, list[float]]] = None, 
            t_pose_action: Optional[torch.Tensor] = None, 
            additional_fitting_poses: Optional[List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]] = None,
            joint_weights: Optional[torch.Tensor] = None
            ):
        '''
        Fit the SMPLH model to the URDF model

        Params:
        - max_iter: int, maximum iteration
        - lr: float, learning rate
        - tol: float, tolerance for the loss
        - verbose: bool, print the loss during the fitting process
        - offset_map: Dict[str, torch.Tensor], offset of URDF model to better fit SMPLH. e.g. Creating offset for ankle_pitch can help match with the position of toe
        - rotation_map: Dict[str, list[float]], rotation of the SMPLH model to fit the zero pose of URDF model. e.g. For Unitree H1 / H1-2 series, elbows should be rotated to certain angles
        - t_pose_action: torch.Tensor, action to set the URDF model to T-pose. None if no T-pose is needed
        - additional_fitting_poses: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]], additional fitting poses to fit the SMPLH model to URDF model. Each tuple consists of [rotation_map, action]. The rotation required for the SMPLH model and the action required for the URDF model.
        '''
        if self.mapping_table is None:
            raise ValueError("Mapping table is not set, call `set_mapping_table` first")
        
        # Prepare SMPL default pose
        root_orient = R.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec()
        pose_body = np.zeros(21 * 3)
        pose_body = pose_body.reshape(-1, 3)
        pose_body[self.smplh_body_names.index('L_Shoulder') - 1] = R.from_euler("xyz", [0, 0, -np.pi/2], degrees=False).as_rotvec()
        pose_body[self.smplh_body_names.index('R_Shoulder') - 1] = R.from_euler("xyz", [0, 0, np.pi/2], degrees=False).as_rotvec()
        if rotation_map is not None:
            for k, v in rotation_map.items():
                pose_body[self.smplh_body_names.index(k) - 1] = R.from_euler("xyz", v, degrees=False).as_rotvec()
        
        pose_body = pose_body.reshape(-1)
        pose_hand = np.zeros(30 * 3)
        trans = np.zeros(3)
        beta = np.zeros(16)
        dmpls = np.zeros(8)

        # shape poses from (n) to (1, n)
        root_orient = root_orient.reshape(1, -1)
        pose_body = pose_body.reshape(1, -1)
        pose_hand = pose_hand.reshape(1, -1)
        trans = trans.reshape(1, -1)
        beta = beta.reshape(1, -1)
        dmpls = dmpls.reshape(1, -1)


        root_orient = torch.from_numpy(root_orient).float().to(self.device)
        pose_body = torch.from_numpy(pose_body).float().to(self.device)
        pose_hand = torch.from_numpy(pose_hand).float().to(self.device)
        trans = torch.from_numpy(trans).float().to(self.device)
        beta = torch.from_numpy(beta).float().to(self.device)
        dmpls = torch.from_numpy(dmpls).float().to(self.device)

        body_parms = self.create_model_input(root_orient, pose_body, pose_hand, trans, beta, dmpls)
        body_parm_tpose = body_parms.copy()
        body_parm_tpose['pose_body'] = torch.zeros_like(body_parm_tpose['pose_body'])
        self.body_parms = body_parms

        if joint_weights is None:
            joint_weights = torch.ones(len(self.mapping_table.keys()))

        joint_weights = joint_weights.to(self.device)

        joints = self.body_model(**{k:v for k,v in body_parms.items() if k in ['root_orient', 'pose_body', 'betas', 'trans']}).Jtr
        joints_tpose = self.body_model(**{k:v for k,v in body_parm_tpose.items() if k in ['root_orient', 'pose_body', 'betas', 'trans']}).Jtr
        offset = joints[:,0] - trans
        root_trans_offset = trans + offset
        root_trans_offset = root_trans_offset.flatten(0)


        # Prepare URDF default pose
        self.robot.set_target_links(self.mapping_table.values())
        dof = self.robot.num_dofs
        actions = torch.zeros(dof).to(self.device).unsqueeze(0)
        
        robot_link_pos = self.robot(actions, link_trans_offset=offset_map)
        if t_pose_action is not None:
            robot_link_pos_tpose = self.robot(t_pose_action.unsqueeze(0), link_trans_offset=offset_map)
        else:
            robot_link_pos_tpose = None

        # self.robot.visualize_joints(show=True)

        self.robot_link_pos = robot_link_pos

        robot_link_pos = robot_link_pos.reshape(1, -1, 3)
        robot_link_pos = robot_link_pos - robot_link_pos[:, 0, :]

        if robot_link_pos_tpose is not None:
            robot_link_pos_tpose = robot_link_pos_tpose.reshape(1, -1, 3)
            robot_link_pos_tpose = robot_link_pos_tpose - robot_link_pos_tpose[:, 0, :]
            self.robot_link_pos_tpose = robot_link_pos_tpose

        else:
            self.robot_link_pos_tpose = None

        additional_poses = []
        if additional_fitting_poses is not None:
            for rotation_map, action in additional_fitting_poses:
                pose_body = np.zeros(21 * 3)
                pose_body = pose_body.reshape(-1, 3)
                for k, v in rotation_map.items():
                    pose_body[self.smplh_body_names.index(k) - 1] = R.from_euler("xyz", v, degrees=False).as_rotvec()
                pose_body = pose_body.reshape(-1)
                pose_body = torch.from_numpy(pose_body).float().to(self.device)
                cur_body_parms = self.create_model_input(root_orient, pose_body, pose_hand, trans, beta, dmpls)
                cur_robot_link_pos = self.robot(action.unsqueeze(0), link_trans_offset=offset_map)
                cur_robot_link_pos = cur_robot_link_pos.reshape(1, -1, 3)
                cur_robot_link_pos = cur_robot_link_pos - cur_robot_link_pos[:, 0, :]
                additional_poses.append((cur_body_parms, cur_robot_link_pos))
        

        beta_opt = Variable(torch.zeros([1, 16]).to(self.device), requires_grad=True)
        scale = Variable(torch.ones([1]).to(self.device), requires_grad=True)
        dmpls_opt = Variable(torch.zeros([1, 8]).to(self.device), requires_grad=True)
        optimizer_shape = torch.optim.Adam([beta_opt, dmpls_opt, scale], lr=lr)

        for iteration in range(max_iter):
            # when iter > 50000, decrease lr to 0.001
            if iteration > 50000:
                optimizer_shape.param_groups[0]['lr'] = 0.001
            body_parms['betas'] = beta_opt
            body_parms['dmpls'] = dmpls_opt
            body_parm_tpose['betas'] = beta_opt
            body_parm_tpose['dmpls'] = dmpls_opt
            joints = self.body_model(**{k:v for k,v in body_parms.items() if k in ['root_orient', 'pose_body', 'betas', 'dmpls']}).Jtr
            joints_tpose = self.body_model(**{k:v for k,v in body_parm_tpose.items() if k in ['root_orient', 'pose_body', 'betas', 'dmpls']}).Jtr
            root_pos = joints[:, 0]
            joints = (joints - root_pos) * scale # + root_pos
            root_pos = joints_tpose[:, 0]
            joints_tpose = (joints_tpose - root_pos) * scale # + root_pos
            loss = F.mse_loss(torch.einsum('ijk, j->ijk', joints[:,[self.smplh_body_names.index(k) for k in self.mapping_table.keys()],:], joint_weights), torch.einsum('ijk, j->ijk', robot_link_pos, joint_weights))
            
            if robot_link_pos_tpose is not None:
                t_pose_loss = F.mse_loss(torch.einsum('ijk, j->ijk', joints_tpose[:,[self.smplh_body_names.index(k) for k in self.mapping_table.keys()],:], joint_weights), torch.einsum('ijk, j->ijk', robot_link_pos_tpose, joint_weights))
                loss += t_pose_loss

            if additional_poses:
                for cur_body_parms, cur_robot_link_pos in additional_poses:
                    cur_body_parms['betas'] = beta_opt
                    cur_body_parms['dmpls'] = dmpls_opt
                    cur_joints = self.body_model(**{k:v for k,v in cur_body_parms.items() if k in ['root_orient', 'pose_body', 'betas', 'dmpls']}).Jtr
                    cur_root_pos = cur_joints[:, 0]
                    cur_joints = (cur_joints - cur_root_pos) * scale # + cur_root_pos
                    loss += F.mse_loss(torch.einsum('ijk, j->ijk', cur_joints[:,[self.smplh_body_names.index(k) for k in self.mapping_table.keys()],:], joint_weights), torch.einsum('ijk, j->ijk', cur_robot_link_pos, joint_weights))


            if verbose and iteration % 100 == 0:
                print("-" * 100)
                print(f"Iteration {iteration}, Loss: {loss.item() * 1000}")
                joint_error = (joints[:,[self.smplh_body_names.index(k) for k in self.mapping_table.keys()],:] - robot_link_pos).norm(dim=-1)
                end_effectors = ['L_Wrist', 'R_Wrist', 'L_Ankle', 'R_Ankle']
                end_effector_bm_idx = [self.smplh_body_names.index(k) for k in end_effectors]
                end_effector_robot_idx = [list(self.mapping_table.keys()).index(k) for k in end_effectors]
                end_effectors_error = (joints[:,[end_effector_bm_idx],:] - robot_link_pos[:,[end_effector_robot_idx],:]).norm(dim=-1)
                print(f"Max joint error: {joint_error.max().item() * 100:.2f}cm")
                print(f"Min joint error: {joint_error.min().item() * 100:.2f}cm")
                print(f"Mean joint error: {joint_error.mean().item() * 100:.2f}cm")
                print(f"Max end effectors error: {end_effectors_error.max().item() * 100:.2f}cm")
                print(f"Min end effectors error: {end_effectors_error.min().item() * 100:.2f}cm")
                print(f"Mean end effectors error: {end_effectors_error.mean().item() * 100:.2f}cm")
                if robot_link_pos_tpose is not None:
                    print(f"t_pose_loss: {t_pose_loss.item() * 1000}")
                    t_pose_joint_error = (joints_tpose[:,[end_effector_bm_idx],:] - robot_link_pos_tpose[:,[end_effector_robot_idx],:]).norm(dim=-1)
                    end_effectors_tpose_error = (joints_tpose[:,[end_effector_bm_idx],:] - robot_link_pos_tpose[:,[end_effector_robot_idx],:]).norm(dim=-1)
                    print(f"Max t_pose joint error: {t_pose_joint_error.max().item() * 100:.2f}cm")
                    print(f"Min t_pose joint error: {t_pose_joint_error.min().item() * 100:.2f}cm")
                    print(f"Mean t_pose joint error: {t_pose_joint_error.mean().item() * 100:.2f}cm")
                    print(f"Max t_pose end effectors error: {end_effectors_tpose_error.max().item() * 100:.2f}cm")
                    print(f"Min t_pose end effectors error: {end_effectors_tpose_error.min().item() * 100:.2f}cm")
                    print(f"Mean t_pose end effectors error: {end_effectors_tpose_error.mean().item() * 100:.2f}cm")
            optimizer_shape.zero_grad()
            loss.backward()
            optimizer_shape.step()

            if loss < tol:
                break

        self.fit_res = {
            'beta': beta_opt,
            'dmpls': dmpls_opt,
            'scale': scale
        }

        return self.fit_res
    
    def get_fit_result(self):
        if self.fit_res is None:
            raise ValueError("Call `fit` method first")
        
        return self.fit_res
    
    def visualize_fit_model(self, imw: int = 1600, imh: int = 1600):
        if self.fit_res is None or self.body_parms is None:
            raise ValueError("Call `fit` method first")
        

        mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

        body_parms = self.body_parms
        body_parms['betas'] = self.fit_res['beta']
        body_pose = self.body_model(**{k:v for k,v in body_parms.items() if k in [ 'pose_body', 'betas', 'pose_hand']})
        faces = self.body_model.f
        body_mesh = trimesh.Trimesh(vertices=body_pose.v[0].detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), vertex_colors=np.tile(colors['grey'], (6890, 1)))
        mv.set_static_meshes([body_mesh])
        body_image = mv.render(render_wireframe=False)
        import cv2
        cv2.imshow("Body Image", body_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def visualize_all_23_joints(self, imw: int = 1600, imh: int = 1600):
        mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
        body_parms = self.body_parms
        
        for i in range(len(self.smplh_body_names) - 1):

            body_parms['pose_body'] = torch.zeros_like(body_parms['pose_body'])
            body_parms['pose_body'] = body_parms['pose_body'].reshape(-1, 3)
            body_parms['pose_body'][i] = torch.Tensor(R.from_euler('xyz', [0, 0, np.pi/2], degrees=False).as_rotvec())
            body_parms['pose_body'] = body_parms['pose_body'].reshape(1, -1)
            body_pose = self.body_model(**{k:v for k,v in body_parms.items() if k in ['root_orient', 'pose_body', 'betas', 'pose_hand']})
            faces = self.body_model.f
            body_mesh = trimesh.Trimesh(vertices=body_pose.v[0].detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), vertex_colors=np.tile(colors['grey'], (6890, 1)))
            mv.set_static_meshes([body_mesh])
            body_image = mv.render(render_wireframe=False)
            # write image to file
            from PIL import Image
            im = Image.fromarray(body_image)
            im.save(f"joint_{self.smplh_body_names[i + 1]}.png")
            # show_image(body_image)

    def visualize_open3d(self):
        robot_link_pos = self.robot_link_pos.reshape(-1, 3)
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        body_parms = self.body_parms
        body_parms['betas'] = self.fit_res['beta']
        body_parms['dmpls'] = self.fit_res['dmpls']
        body_pose = self.body_model(**{k:v for k,v in body_parms.items() if k in ['root_orient', 'pose_body', 'betas', 'pose_hand', 'dmpls']})
        faces = self.body_model.f
        mesh.vertices = o3d.utility.Vector3dVector(body_pose.v[0].detach().cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(faces.detach().cpu().numpy())
        list_geometries = [mesh, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)]
        for pos in robot_link_pos:
            pos[1] += 1.0
            frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            frame_mesh = frame_mesh.translate(pos.cpu().detach().numpy())
            list_geometries.append(frame_mesh)
        o3d.visualization.draw_geometries(list_geometries)

    def visualize_open3d_tpose(self):
        if self.robot_link_pos_tpose is None:
            raise ValueError("No tpose data")
        
        robot_link_pos = self.robot_link_pos_tpose.reshape(-1, 3)
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        body_parms = self.body_parms
        body_parms['betas'] = self.fit_res['beta']
        body_parms['dmpls'] = self.fit_res['dmpls']
        body_parms['pose_body'] = torch.zeros_like(body_parms['pose_body'])
        body_pose = self.body_model(**{k:v for k,v in body_parms.items() if k in ['root_orient', 'pose_body', 'betas', 'pose_hand', 'dmpls']})
        faces = self.body_model.f
        mesh.vertices = o3d.utility.Vector3dVector(body_pose.v[0].detach().cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(faces.detach().cpu().numpy())
        list_geometries = [mesh, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)]
        for pos in robot_link_pos:
            pos[1] += 1.0
            frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            frame_mesh = frame_mesh.translate(pos.cpu().detach().numpy())
            list_geometries.append(frame_mesh)
        o3d.visualization.draw_geometries(list_geometries)

    def save_fit_result(self, path: str = "best_fit.pt"):
        if self.fit_res is None:
            raise ValueError("Call `fit` method first")
        
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        print("Fitting result:")
        for k, v in self.fit_res.items():
            print(f"{k}: {v}")
        
        torch.save(self.fit_res, path)

    def load_fit_result(self, path: str = "new_best_fit_org.pt"):
        if not os.path.exists(path):
            raise ValueError(f"File {path} does not exist")
        
        self.fit_res = torch.load(path)