import os
import torch
import numpy as np
import trimesh
import pyrender
import pytorch_kinematics as pk
from typing import Union, Optional, List
from GBC.utils.base.base_fk import RobotKinematics


class RobotVisualizer:
    def __init__(
        self,
        urdf_path,
        use_offscreen=True,
        width=800,
        height=1080,
        robot_transmission=0.1,
        vis_world_frame_length=0.1,
        vis_world_frame_color=(0, 0, 0),
        vis_link_frame_length=0.05,
        vis_link_frame_color=(0.5, 0.5, 0.5),
        vis_points_size=0.01,
        vis_points_color=(0.9, 0.1, 0.1),
        device="cpu",
    ):
        self.urdf_path = urdf_path
        self.urdf_dir_path = os.path.dirname(urdf_path)

        self.use_offscreen = use_offscreen
        self.offscreen_width = width
        self.offscreen_height = height

        self.robot_transmission = robot_transmission
        self.vis_world_frame_length = vis_world_frame_length
        self.vis_world_frame_color = vis_world_frame_color
        self.vis_link_frame_length = vis_link_frame_length
        self.vis_link_frame_color = vis_link_frame_color
        self.vis_points_size = vis_points_size
        self.vis_points_color = vis_points_color

        self.device = device

        self.scene = pyrender.Scene()

        self.load_robot()

        self.robot_kinematics = RobotKinematics(self.urdf_path, self.device)

    def __call__(self, q, root_tf=None, **kwargs):
        return self.vis_robot(q, root_tf=root_tf, **kwargs)

    def load_robot(self):
        # Load URDF with pytorch_kinematics
        self.chain = pk.build_chain_from_urdf(open(self.urdf_path, mode="rb").read())
        self.chain = self.chain.to(device=self.device)

        self.link_names = self.chain.get_link_names()

        # Get DOF informations
        assert isinstance(self.chain.joint_indices, torch.Tensor)
        self.joint_names = self.chain.get_joint_parameter_names()
        self.num_joints = len(self.joint_names)
        self.joint_frames = [(-1, -1) for _ in range(self.num_joints)]
        for child_idx, joint_idx in enumerate(self.chain.joint_indices):
            joint_idx = int(joint_idx.item())
            if joint_idx < 0:
                continue
            parent_idx = self.chain.parents_indices[child_idx][:-1]
            if parent_idx.shape == (0,):
                continue
            self.joint_frames[joint_idx] = (int(parent_idx[-1].item()), child_idx)

        # Load all meshes
        self.link_meshes = {}
        for link_name in self.link_names:
            self.link_meshes[link_name] = []
            link = self.chain.find_link(link_name)
            assert isinstance(link, pk.Link)

            for i, visual in enumerate(link.visuals):
                # Check visual geometry type
                if visual.geom_type is None:
                    continue
                if visual.geom_type != "mesh":
                    raise ValueError(f"Visual geometry with type {visual.geom_type} is not supported")
                mesh_path = os.path.join(self.urdf_dir_path, visual.geom_param[0])
                mesh = trimesh.load_mesh(mesh_path)
                mesh.visual.vertex_colors = np.ones([mesh.vertices.shape[0], 4]) * [0.5, 0.5, 0.5, 1-self.robot_transmission]

                mesh_name = f"{link_name}_visual_{i}"
                self.scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), name=mesh_name)
                self.link_meshes[link_name].append((visual.offset.get_matrix().to(self.device), mesh))

    @property
    def robot_structure(self):
        return str(self.chain)

    def get_frame_name(self, frame_idx: int):
        return self.chain.idx_to_frame[frame_idx]

    def get_joint_frame_names(self, joint_idx: int):
        return tuple(map(self.get_frame_name, self.joint_frames[joint_idx]))

    def vis_robot(
        self,
        q,
        root_tf=None,
        vis_world_frame=False,
        vis_link_frame: Union[bool, list]=False,
        extra_pts: Optional[Union[np.ndarray, torch.Tensor]] = None,
        extra_pyrender_objs: List[Union[pyrender.Mesh, pyrender.Light]] = [],
        cam_pose=np.array([[1, 0, 0, 0], [0, 1, 0, -0.2], [0, 0, 1, 1.5], [0, 0, 0, 1]]),
        cam_yfov=np.pi * .4,
    ):
        self.extra_nodes = []
        self.deleted_nodes = []
        if root_tf is None:
            root_tf = torch.zeros((4, 4), dtype=torch.float32)
            root_tf[1, 2] = root_tf[2, 0] = root_tf[0, 1] = root_tf[3, 3] = 1
        self.set_robot_mesh_pose(q, root_tf=root_tf)

        if vis_world_frame:
            self.add_axes(np.eye(4), self.vis_world_frame_length, self.vis_world_frame_color)
        if vis_link_frame:
            list_link_tfs = tuple(self.link_transform.values())
            if isinstance(vis_link_frame, list):
                list_link_tfs = [self.link_transform[name] for name in vis_link_frame]
            link_tfs = torch.stack(list_link_tfs)
            link_tfs = link_tfs.cpu().detach().numpy()
            self.add_axes(link_tfs, self.vis_link_frame_length, self.vis_link_frame_color)

        if extra_pts is not None:
            if isinstance(extra_pts, torch.Tensor):
                extra_pts = extra_pts.cpu().detach().numpy()
            if extra_pts.shape[0]:
                self.add_points(extra_pts, self.vis_points_size, self.vis_points_color)

        for pyrender_obj in extra_pyrender_objs:
            self.extra_nodes.append(self.scene.add(pyrender_obj))

        camera = pyrender.PerspectiveCamera(yfov=cam_yfov)
        self.extra_nodes.append(self.scene.add(camera, pose=cam_pose))

        if self.use_offscreen:
            self.add_raymond_lights()
            viewer = pyrender.OffscreenRenderer(self.offscreen_width, self.offscreen_height)
            RenderFlags = pyrender.constants.RenderFlags
            flags =  RenderFlags.RGBA
            color_img, _ = viewer.render(self.scene, flags=flags)
            res = color_img
        else:
            pyrender.Viewer(self.scene, use_raymond_lighting=True)
            res = None

        for node in self.extra_nodes:
            assert self.scene.has_node(node)
            self.scene.remove_node(node)
        self.extra_nodes = []
        for node in self.deleted_nodes:
            assert not self.scene.has_node(node)
            self.scene.add_node(node)

        return res

    def set_robot_mesh_pose(self, q, root_tf=None, debug_ankle=False):
        if len(q.shape) == 1:
            q = q.unsqueeze(0)

        if root_tf is None:
            root_tf = torch.eye(4)
        if isinstance(root_tf, np.ndarray):
            root_tf = torch.tensor(root_tf).to(torch.float32)
        root_tf = root_tf.to(self.device)

        fk = self.robot_kinematics.forward_kinematics(q)
        self.link_transform = {}

        for link_name, link_tf in fk.items():
            link_tf = torch.matmul(root_tf, link_tf[0])
            self.link_transform[link_name] = link_tf

        for link_name, link_meshes in self.link_meshes.items():
            for i, (offset, mesh) in enumerate(link_meshes):
                mesh_name = f"{link_name}_visual_{i}"
                nodes = list(self.scene.get_nodes(name=mesh_name))
                assert len(nodes) == 1

                if link_name in self.link_transform:
                    link_tf = self.link_transform[link_name]
                    mesh_tf = torch.matmul(self.link_transform[link_name], offset)
                    mesh_tf = mesh_tf.cpu().detach().numpy()[0]
                    if "ankle" in link_name and debug_ankle:
                        mesh_pts = np.asarray(mesh.vertices)
                        mesh_pts = np.einsum("ij, kj -> ki", mesh_tf[:3, :3], mesh_pts) + mesh_tf[np.newaxis, :3, 3]
                    self.scene.set_pose(nodes[0], mesh_tf)
                else:
                    self.deleted_nodes.append(nodes[0])
                    self.scene.remove_node(nodes[0])


    def add_axes(self, transforms: np.ndarray, length, color):
        mesh = trimesh.creation.axis(axis_length=length, origin_size=0.1*length, axis_radius=0.025*length, origin_color=color)
        self.extra_nodes.append(self.scene.add(pyrender.Mesh.from_trimesh(mesh, poses=transforms, smooth=False)))

    def add_points(self, pts: np.ndarray, size, color):
        mesh = trimesh.creation.uv_sphere(radius=size)
        mesh.visual.vertex_colors = list(color) + [1.]

        transforms = np.tile(np.eye(4), (pts.shape[0], 1, 1))
        transforms[:, :3, 3] = pts
        self.extra_nodes.append(self.scene.add(pyrender.Mesh.from_trimesh(mesh, poses=transforms)))

    def add_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3,:3] = np.c_[x,y,z]
            self.extra_nodes.append(self.scene.add(
                pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                pose=matrix
            ))

