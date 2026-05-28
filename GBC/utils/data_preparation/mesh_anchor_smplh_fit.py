from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def barycentric_sample(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_indices: torch.Tensor,
    barycentric_weights: torch.Tensor,
) -> torch.Tensor:
    """Sample mesh points while keeping gradients to `vertices`.

    Args:
        vertices: `(B, V, 3)` or `(V, 3)` vertex tensor.
        faces: `(F, 3)` long tensor.
        face_indices: `(N,)` long tensor selecting faces.
        barycentric_weights: `(N, 3)` float tensor.

    Returns:
        `(B, N, 3)` sampled points.
    """
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)
    if vertices.dim() != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"vertices must have shape (B, V, 3) or (V, 3), got {tuple(vertices.shape)}")

    face_indices = face_indices.to(device=vertices.device, dtype=torch.long)
    faces = faces.to(device=vertices.device, dtype=torch.long)
    barycentric_weights = barycentric_weights.to(device=vertices.device, dtype=vertices.dtype)
    tri_vertex_indices = faces[face_indices]  # (N, 3)
    tri_vertices = vertices[:, tri_vertex_indices, :]  # (B, N, 3, 3)
    return (tri_vertices * barycentric_weights.view(1, -1, 3, 1)).sum(dim=2)


def transform_points(transform: torch.Tensor, local_points: torch.Tensor) -> torch.Tensor:
    """Transform local points with homogeneous transforms.

    Args:
        transform: `(B, 4, 4)` or `(4, 4)`.
        local_points: `(N, 3)` or `(B, N, 3)`.

    Returns:
        `(B, N, 3)`.
    """
    if transform.dim() == 2:
        transform = transform.unsqueeze(0)
    if local_points.dim() == 2:
        local_points = local_points.unsqueeze(0).expand(transform.shape[0], -1, -1)
    elif local_points.shape[0] == 1 and transform.shape[0] != 1:
        local_points = local_points.expand(transform.shape[0], -1, -1)
    if transform.dim() != 3 or transform.shape[-2:] != (4, 4):
        raise ValueError(f"transform must have shape (B, 4, 4) or (4, 4), got {tuple(transform.shape)}")
    if local_points.dim() != 3 or local_points.shape[-1] != 3:
        raise ValueError(f"local_points must have shape (B, N, 3) or (N, 3), got {tuple(local_points.shape)}")
    rot = transform[:, :3, :3]
    trans = transform[:, :3, 3]
    return torch.einsum("bij,bnj->bni", rot, local_points) + trans[:, None, :]


def look_at_camera_pose(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> np.ndarray:
    """Build a pyrender camera pose looking from `eye` to `target`."""
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    z_axis = eye - target
    z_axis = z_axis / np.linalg.norm(z_axis).clip(min=1e-6)
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis).clip(min=1e-6)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye
    return pose


def robot_viewer_root_transform(device: torch.device | str = "cpu") -> torch.Tensor:
    """Coordinate transform used by the existing RobotVisualizer/PoseRenderer."""
    root_tf = torch.zeros((4, 4), dtype=torch.float32, device=device)
    root_tf[range(4), [1, 2, 0, 3]] = 1.0
    return root_tf


def extreme_patch_grid(
    vertices: torch.Tensor,
    normal_axis: int,
    side: str,
    u_axis: int,
    v_axis: int,
    grid_size: tuple[int, int] = (4, 4),
    band_fraction: float = 0.08,
) -> torch.Tensor:
    """Select a stable local patch grid from an extreme side of a mesh.

    This is intended for offline anchor construction. It returns existing vertex
    positions closest to a normalized grid on the selected patch.
    """
    if side not in {"min", "max"}:
        raise ValueError("side must be 'min' or 'max'")
    if vertices.dim() == 3:
        if vertices.shape[0] != 1:
            raise ValueError("extreme_patch_grid expects unbatched vertices or a single batch")
        vertices = vertices[0]
    vertices = vertices.detach()
    coord = vertices[:, normal_axis]
    min_val = coord.min()
    max_val = coord.max()
    band = (max_val - min_val).clamp(min=1e-6) * band_fraction
    if side == "min":
        mask = coord <= min_val + band
    else:
        mask = coord >= max_val - band
    patch = vertices[mask]
    if patch.shape[0] == 0:
        patch = vertices

    u_vals = torch.linspace(patch[:, u_axis].min(), patch[:, u_axis].max(), grid_size[0], device=vertices.device)
    v_vals = torch.linspace(patch[:, v_axis].min(), patch[:, v_axis].max(), grid_size[1], device=vertices.device)
    grid = torch.stack(torch.meshgrid(u_vals, v_vals, indexing="ij"), dim=-1).reshape(-1, 2)
    patch_uv = patch[:, [u_axis, v_axis]]
    distances = torch.cdist(grid.to(patch_uv.dtype), patch_uv)
    return patch[distances.argmin(dim=1)]


def extreme_patch_vertex_indices(
    vertices: torch.Tensor,
    normal_axis: int,
    side: str,
    u_axis: int,
    v_axis: int,
    grid_size: tuple[int, int] = (4, 4),
    band_fraction: float = 0.08,
    side_axis: int | None = None,
    side_sign: float | None = None,
) -> torch.Tensor:
    """Return vertex indices for an extreme semantic patch.

    Optional `side_axis` and `side_sign` split left/right or front/back groups.
    """
    if vertices.dim() == 3:
        vertices = vertices[0]
    vertices = vertices.detach()
    candidate = torch.arange(vertices.shape[0], device=vertices.device)
    if side_axis is not None and side_sign is not None:
        side_mask = vertices[:, side_axis] * side_sign >= 0
        if side_mask.any():
            candidate = candidate[side_mask]
            vertices_for_patch = vertices[candidate]
        else:
            vertices_for_patch = vertices
    else:
        vertices_for_patch = vertices

    patch_points = extreme_patch_grid(
        vertices_for_patch,
        normal_axis=normal_axis,
        side=side,
        u_axis=u_axis,
        v_axis=v_axis,
        grid_size=grid_size,
        band_fraction=band_fraction,
    )
    distances = torch.cdist(patch_points, vertices[candidate])
    return candidate[distances.argmin(dim=1)].long()


@dataclass(frozen=True)
class RobotLinkMesh:
    link_name: str
    source: str
    vertices: np.ndarray
    faces: np.ndarray
    mesh_path: str | None = None


@dataclass(frozen=True)
class SurfaceAnchorSpec:
    """Robot-independent description of one semantic surface correspondence."""

    name: str
    robot_link_candidates: tuple[str, ...]
    human_mode: str
    weight: float = 1.0
    human_joint_indices: tuple[int, ...] = ()
    human_side_axis: int | None = None
    human_side_sign: float | None = None
    human_normal_axis: int = 2
    human_u_axis: int = 0
    human_v_axis: int = 1
    human_grid_size_2d: tuple[int, int] = (4, 4)
    human_grid_size_3d: tuple[int, int, int] = (3, 3, 3)
    human_band_fraction: float = 0.05
    human_candidate_count: int = 600
    robot_mode: str = "bbox"
    robot_normal_axis: int = 2
    robot_u_axis: int = 0
    robot_v_axis: int = 1
    robot_grid_size_2d: tuple[int, int] = (4, 4)
    robot_grid_size_3d: tuple[int, int, int] = (3, 3, 3)
    robot_band_fraction: float = 0.20


@dataclass(frozen=True)
class SurfaceAnchorPair:
    name: str
    human_vertex_indices: torch.Tensor
    robot_link_name: str
    robot_local_points: torch.Tensor
    weight: float = 1.0


def resolve_link_name(link_meshes: dict[str, list[RobotLinkMesh]], candidates: Iterable[str]) -> str | None:
    """Resolve ordered link candidates against a concrete URDF mesh set."""
    for name in candidates:
        if name in link_meshes:
            return name
    return None


def g1_surface_anchor_specs(
    foot_links: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    foot_grid_size: tuple[int, int] = (4, 4),
) -> list[SurfaceAnchorSpec]:
    """Unitree G1 semantic surface anchors as an explicit preset."""
    return [
        SurfaceAnchorSpec(
            name="pelvis_surface",
            robot_link_candidates=("pelvis", "base_link"),
            human_mode="joint_patch",
            human_joint_indices=(0, 1, 2),
            human_candidate_count=700,
            weight=2.0,
        ),
        SurfaceAnchorSpec(
            name="waist_surface",
            robot_link_candidates=("waist_yaw_link", "waist_roll_link", "torso_link"),
            human_mode="joint_patch",
            human_joint_indices=(3, 6),
            human_grid_size_3d=(2, 2, 2),
            robot_grid_size_3d=(2, 2, 2),
            human_candidate_count=500,
            weight=2.5,
        ),
        SurfaceAnchorSpec(
            name="torso_surface",
            robot_link_candidates=("torso_link", "Chest", "Link_torso"),
            human_mode="joint_patch",
            human_joint_indices=(6, 9, 12),
            human_candidate_count=900,
            weight=2.5,
        ),
        SurfaceAnchorSpec(
            name="left_foot_sole",
            robot_link_candidates=(foot_links[0],),
            human_mode="foot_sole",
            weight=2.0,
            human_side_axis=1,
            human_side_sign=1.0,
            human_grid_size_2d=foot_grid_size,
            robot_mode="extreme",
            robot_grid_size_2d=foot_grid_size,
        ),
        SurfaceAnchorSpec(
            name="right_foot_sole",
            robot_link_candidates=(foot_links[1],),
            human_mode="foot_sole",
            weight=2.0,
            human_side_axis=1,
            human_side_sign=-1.0,
            human_grid_size_2d=foot_grid_size,
            robot_mode="extreme",
            robot_grid_size_2d=foot_grid_size,
        ),
        SurfaceAnchorSpec(
            name="head_surface",
            robot_link_candidates=("head_link", "Link_head_yaw"),
            human_mode="joint_patch",
            human_joint_indices=(15,),
            human_candidate_count=900,
            weight=3.0,
        ),
        SurfaceAnchorSpec(
            name="left_hand_surface",
            robot_link_candidates=("left_rubber_hand", "left_wrist_yaw_link", "Link_arm_l_06"),
            human_mode="joint_patch",
            human_joint_indices=tuple(range(22, 37)),
            human_candidate_count=700,
            weight=2.0,
        ),
        SurfaceAnchorSpec(
            name="right_hand_surface",
            robot_link_candidates=("right_rubber_hand", "right_wrist_yaw_link", "Link_arm_r_06"),
            human_mode="joint_patch",
            human_joint_indices=tuple(range(37, 52)),
            human_candidate_count=700,
            weight=2.0,
        ),
        SurfaceAnchorSpec(
            name="left_shoulder_surface",
            robot_link_candidates=("left_shoulder_roll_link", "left_shoulder_pitch_link", "Link_arm_l_01"),
            human_mode="joint_patch",
            human_joint_indices=(16,),
            human_grid_size_3d=(2, 2, 2),
            robot_grid_size_3d=(2, 2, 2),
            human_candidate_count=350,
            weight=1.5,
        ),
        SurfaceAnchorSpec(
            name="right_shoulder_surface",
            robot_link_candidates=("right_shoulder_roll_link", "right_shoulder_pitch_link", "Link_arm_r_01"),
            human_mode="joint_patch",
            human_joint_indices=(17,),
            human_grid_size_3d=(2, 2, 2),
            robot_grid_size_3d=(2, 2, 2),
            human_candidate_count=350,
            weight=1.5,
        ),
    ]


def g1_init_rotation_map() -> dict[str, tuple[float, float, float]]:
    """SMPLH init-pose rotations used by the existing G1 SMPLH fit."""
    return {
        "L_Elbow": (0.0, -np.pi / 2, 0.0),
        "R_Elbow": (0.0, np.pi / 2, 0.0),
    }


def default_g1_t_pose_action(
    dof_names: Iterable[str],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the G1 T-pose action used by the existing SMPLH fitter."""
    dof_names = list(dof_names)
    action = torch.zeros(len(dof_names), dtype=dtype, device=device)
    joint_values = {
        "right_shoulder_roll_joint": -np.pi / 2,
        "left_shoulder_roll_joint": np.pi / 2,
        "right_elbow_joint": np.pi / 2,
        "left_elbow_joint": np.pi / 2,
    }
    for joint_name, value in joint_values.items():
        if joint_name in dof_names:
            action[dof_names.index(joint_name)] = float(value)
    return action


def bbox_surface_vertex_indices(
    vertices: torch.Tensor,
    grid_size: tuple[int, int, int] = (3, 3, 3),
    candidate_indices: torch.Tensor | None = None,
    axes: tuple[int, int, int] = (0, 1, 2),
) -> torch.Tensor:
    """Return fixed-order surface vertex indices closest to a bbox grid."""
    if vertices.dim() == 3:
        if vertices.shape[0] != 1:
            raise ValueError("bbox_surface_vertex_indices expects unbatched vertices or a single batch")
        vertices = vertices[0]
    vertices = vertices.detach()
    if candidate_indices is None:
        candidate_indices = torch.arange(vertices.shape[0], device=vertices.device)
    else:
        candidate_indices = candidate_indices.to(device=vertices.device, dtype=torch.long)
    if candidate_indices.numel() == 0:
        candidate_indices = torch.arange(vertices.shape[0], device=vertices.device)
    candidates = vertices[candidate_indices]
    axis_values = [
        torch.linspace(
            candidates[:, axis].min(),
            candidates[:, axis].max(),
            grid_size[i],
            device=vertices.device,
            dtype=vertices.dtype,
        )
        for i, axis in enumerate(axes)
    ]
    grid = torch.stack(torch.meshgrid(*axis_values, indexing="ij"), dim=-1).reshape(-1, 3)
    candidate_xyz = candidates[:, list(axes)]
    distances = torch.cdist(grid, candidate_xyz)
    return candidate_indices[distances.argmin(dim=1)].long()


def nearest_surface_patch_vertex_indices(
    vertices: torch.Tensor,
    centers: torch.Tensor,
    grid_size: tuple[int, int, int] = (3, 3, 3),
    candidate_count: int = 600,
) -> torch.Tensor:
    """Build a semantic surface patch around one or more body-space centers."""
    if vertices.dim() == 3:
        vertices = vertices[0]
    if centers.dim() == 1:
        centers = centers.unsqueeze(0)
    vertices = vertices.detach()
    centers = centers.detach().to(vertices.device, vertices.dtype)
    distances = torch.cdist(vertices, centers).min(dim=1).values
    count = min(max(int(candidate_count), int(np.prod(grid_size))), vertices.shape[0])
    candidate_indices = distances.topk(count, largest=False).indices
    return bbox_surface_vertex_indices(vertices, grid_size=grid_size, candidate_indices=candidate_indices)


def _origin_matrix_np(origin) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    mat = np.eye(4, dtype=np.float64)
    if origin is None:
        return mat
    xyz = origin.xyz if origin.xyz is not None else [0.0, 0.0, 0.0]
    rpy = origin.rpy if origin.rpy is not None else [0.0, 0.0, 0.0]
    mat[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    mat[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return mat


def _as_trimesh(mesh_obj):
    import trimesh

    if isinstance(mesh_obj, trimesh.Scene):
        return trimesh.util.concatenate(tuple(mesh_obj.geometry.values()))
    return mesh_obj


def _mesh_from_geometry(geometry, urdf_dir: Path):
    import trimesh

    geom_name = geometry.__class__.__name__
    mesh_path = None
    if geom_name == "Mesh":
        mesh_path = Path(geometry.filename)
        if not mesh_path.is_absolute():
            mesh_path = urdf_dir / mesh_path
        mesh = _as_trimesh(trimesh.load_mesh(str(mesh_path), process=False))
        if getattr(geometry, "scale", None) is not None:
            scale = np.asarray(geometry.scale, dtype=np.float64)
            mesh = mesh.copy()
            mesh.vertices *= scale.reshape(1, 3)
    elif geom_name == "Box":
        mesh = trimesh.creation.box(extents=np.asarray(geometry.size, dtype=np.float64))
    elif geom_name == "Cylinder":
        mesh = trimesh.creation.cylinder(radius=float(geometry.radius), height=float(geometry.length), sections=32)
    elif geom_name == "Sphere":
        mesh = trimesh.creation.uv_sphere(radius=float(geometry.radius), count=[16, 16])
    else:
        return None, None
    return mesh, str(mesh_path) if mesh_path is not None else None


def load_robot_link_meshes(
    urdf_path: str | os.PathLike,
    link_names: Iterable[str] | None = None,
    prefer_collision: bool = True,
    visual_fallback: bool = True,
) -> dict[str, list[RobotLinkMesh]]:
    """Load robot link meshes in each link's local frame."""
    from urdf_parser_py.urdf import URDF

    urdf_path = Path(urdf_path)
    urdf_dir = urdf_path.parent
    robot = URDF.from_xml_file(str(urdf_path))
    selected = set(link_names) if link_names is not None else None
    output: dict[str, list[RobotLinkMesh]] = {}
    for link in robot.links:
        if selected is not None and link.name not in selected:
            continue
        groups = []
        if prefer_collision:
            groups.append(("collision", getattr(link, "collisions", []) or []))
            if visual_fallback:
                groups.append(("visual", getattr(link, "visuals", []) or []))
        else:
            groups.append(("visual", getattr(link, "visuals", []) or []))
            groups.append(("collision", getattr(link, "collisions", []) or []))

        link_meshes: list[RobotLinkMesh] = []
        for source, geoms in groups:
            if link_meshes and source == "visual" and prefer_collision:
                continue
            for geom_with_origin in geoms:
                mesh, mesh_path = _mesh_from_geometry(geom_with_origin.geometry, urdf_dir)
                if mesh is None:
                    continue
                origin_mat = _origin_matrix_np(getattr(geom_with_origin, "origin", None))
                vertices_h = np.concatenate([np.asarray(mesh.vertices), np.ones((len(mesh.vertices), 1))], axis=1)
                vertices = (origin_mat @ vertices_h.T).T[:, :3]
                faces = np.asarray(mesh.faces, dtype=np.int64)
                link_meshes.append(
                    RobotLinkMesh(
                        link_name=link.name,
                        source=source,
                        vertices=vertices.astype(np.float32),
                        faces=faces,
                        mesh_path=mesh_path,
                    )
                )
        if link_meshes:
            output[link.name] = link_meshes
    return output


def concatenate_link_vertices(meshes: list[RobotLinkMesh], device: torch.device | str = "cpu") -> torch.Tensor:
    if not meshes:
        raise ValueError("Expected at least one mesh")
    return torch.as_tensor(np.concatenate([m.vertices for m in meshes], axis=0), dtype=torch.float32, device=device)


def sample_robot_anchor_grid(
    link_meshes: dict[str, list[RobotLinkMesh]],
    link_name: str,
    normal_axis: int,
    side: str,
    u_axis: int,
    v_axis: int,
    grid_size: tuple[int, int] = (4, 4),
    band_fraction: float = 0.08,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    vertices = concatenate_link_vertices(link_meshes[link_name], device=device)
    return extreme_patch_grid(
        vertices,
        normal_axis=normal_axis,
        side=side,
        u_axis=u_axis,
        v_axis=v_axis,
        grid_size=grid_size,
        band_fraction=band_fraction,
    )


def sample_robot_bbox_grid(
    link_meshes: dict[str, list[RobotLinkMesh]],
    link_name: str,
    grid_size: tuple[int, int, int] = (3, 3, 3),
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    vertices = concatenate_link_vertices(link_meshes[link_name], device=device)
    indices = bbox_surface_vertex_indices(vertices, grid_size=grid_size)
    return vertices[indices]


class MeshAnchorSMPLHFitter:
    """Prototype mesh-anchor fitter for SMPLH-to-robot surface anchors."""

    def __init__(
        self,
        smplh_model_path: str,
        dmpls_model_path: str,
        urdf_path: str,
        device: str = "cuda",
    ):
        from human_body_prior.body_model.body_model import BodyModel
        from GBC.utils.base.base_fk import RobotKinematics

        self.device = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
        self.urdf_path = str(urdf_path)
        self.body_model = BodyModel(
            bm_fname=smplh_model_path,
            dmpl_fname=dmpls_model_path,
            num_betas=16,
            num_dmpls=8,
        ).to(self.device)
        self.robot = RobotKinematics(self.urdf_path, device=str(self.device))
        self.robot_meshes = load_robot_link_meshes(urdf_path)
        self.faces = self.body_model.f.long().to(self.device)
        self.smplh_body_names = [
            "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine",
            "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe", "Neck",
            "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder",
            "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
        ]
        self.init_rotation_map: dict[str, tuple[float, float, float]] = {}

    def set_init_rotation_map(self, rotation_map: dict[str, Iterable[float]] | None) -> None:
        if rotation_map is None:
            self.init_rotation_map = {}
            return
        self.init_rotation_map = {name: tuple(float(v) for v in values) for name, values in rotation_map.items()}

    def apply_pose_preset(self, pose_preset: str) -> None:
        if pose_preset == "g1":
            self.set_init_rotation_map(g1_init_rotation_map())
        elif pose_preset == "none":
            self.set_init_rotation_map(None)
        else:
            raise ValueError(f"Unknown pose preset '{pose_preset}'")

    def default_body_params(
        self,
        betas: torch.Tensor,
        dmpls: torch.Tensor | None = None,
        pose_name: str = "init",
    ) -> dict[str, torch.Tensor]:
        from scipy.spatial.transform import Rotation as R

        root = torch.tensor(R.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec(), dtype=torch.float32, device=self.device).view(1, 3)
        pose_body = torch.zeros(1, 21, 3, dtype=torch.float32, device=self.device)
        if pose_name == "init":
            pose_body[:, 15] = torch.tensor(R.from_euler("xyz", [0.0, 0.0, -np.pi / 2]).as_rotvec(), dtype=torch.float32, device=self.device)
            pose_body[:, 16] = torch.tensor(R.from_euler("xyz", [0.0, 0.0, np.pi / 2]).as_rotvec(), dtype=torch.float32, device=self.device)
            for joint_name, euler_xyz in self.init_rotation_map.items():
                if joint_name not in self.smplh_body_names:
                    raise ValueError(f"Unknown SMPLH joint in init rotation map: {joint_name}")
                pose_body[:, self.smplh_body_names.index(joint_name) - 1] = torch.tensor(
                    R.from_euler("xyz", euler_xyz, degrees=False).as_rotvec(),
                    dtype=torch.float32,
                    device=self.device,
                )
        elif pose_name != "tpose":
            raise ValueError(f"Unknown body pose '{pose_name}', expected 'init' or 'tpose'")
        return {
            "root_orient": root,
            "pose_body": pose_body.reshape(1, -1),
            "pose_hand": torch.zeros(1, 90, dtype=torch.float32, device=self.device),
            "trans": torch.zeros(1, 3, dtype=torch.float32, device=self.device),
            "betas": betas,
            "dmpls": torch.zeros(1, 8, dtype=torch.float32, device=self.device) if dmpls is None else dmpls,
        }

    def pose_actions(self, pose_names: Iterable[str], pose_preset: str = "g1") -> dict[str, torch.Tensor]:
        actions: dict[str, torch.Tensor] = {}
        dof_names = self.robot.get_dof_names()
        for pose_name in pose_names:
            pose_name = pose_name.strip()
            if pose_name == "init":
                actions[pose_name] = torch.zeros(1, self.robot.num_dofs, device=self.device)
            elif pose_name == "tpose":
                if pose_preset != "g1":
                    raise ValueError("T-pose action requires a robot pose preset or a caller-provided action dictionary")
                actions[pose_name] = default_g1_t_pose_action(dof_names, device=self.device).unsqueeze(0)
            elif pose_name:
                raise ValueError(f"Unknown fit pose '{pose_name}', expected 'init' or 'tpose'")
        if not actions:
            raise ValueError("At least one fit pose is required")
        return actions

    def build_default_anchor_pairs(
        self,
        foot_links: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
        grid_size: tuple[int, int] = (4, 4),
        anchor_specs: Iterable[SurfaceAnchorSpec] | None = None,
    ) -> list[SurfaceAnchorPair]:
        if anchor_specs is None:
            anchor_specs = g1_surface_anchor_specs(foot_links=foot_links, foot_grid_size=grid_size)
        return self.build_anchor_pairs(anchor_specs)

    def build_anchor_pairs(self, anchor_specs: Iterable[SurfaceAnchorSpec]) -> list[SurfaceAnchorPair]:
        with torch.no_grad():
            params = self.default_body_params(torch.zeros(1, 16, device=self.device))
            body = self.body_model(**params)
            vertices = body.v[0] - body.Jtr[0, 0:1, :]
            joints = body.Jtr[0] - body.Jtr[0, 0:1, :]

        pairs: list[SurfaceAnchorPair] = []
        for spec in anchor_specs:
            link_name = resolve_link_name(self.robot_meshes, spec.robot_link_candidates)
            if link_name not in self.robot_meshes:
                continue
            if spec.human_mode == "foot_sole":
                human_indices = extreme_patch_vertex_indices(
                    vertices,
                    normal_axis=spec.human_normal_axis,
                    side="min",
                    u_axis=spec.human_u_axis,
                    v_axis=spec.human_v_axis,
                    grid_size=spec.human_grid_size_2d,
                    band_fraction=spec.human_band_fraction,
                    side_axis=spec.human_side_axis,
                    side_sign=spec.human_side_sign,
                )
            elif spec.human_mode == "joint_patch":
                valid_joint_indices = [idx for idx in spec.human_joint_indices if idx < joints.shape[0]]
                if not valid_joint_indices:
                    continue
                human_indices = nearest_surface_patch_vertex_indices(
                    vertices,
                    joints[valid_joint_indices],
                    grid_size=spec.human_grid_size_3d,
                    candidate_count=spec.human_candidate_count,
                )
            else:
                raise ValueError(f"Unknown human anchor mode '{spec.human_mode}'")

            if spec.robot_mode == "extreme":
                robot_points = sample_robot_anchor_grid(
                    self.robot_meshes,
                    link_name=link_name,
                    normal_axis=spec.robot_normal_axis,
                    side="min",
                    u_axis=spec.robot_u_axis,
                    v_axis=spec.robot_v_axis,
                    grid_size=spec.robot_grid_size_2d,
                    band_fraction=spec.robot_band_fraction,
                    device=self.device,
                )
            elif spec.robot_mode == "bbox":
                robot_points = sample_robot_bbox_grid(
                    self.robot_meshes,
                    link_name=link_name,
                    grid_size=spec.robot_grid_size_3d,
                    device=self.device,
                )
            else:
                raise ValueError(f"Unknown robot anchor mode '{spec.robot_mode}'")

            pairs.append(
                SurfaceAnchorPair(
                    name=spec.name,
                    human_vertex_indices=human_indices.to(self.device),
                    robot_link_name=link_name,
                    robot_local_points=robot_points.to(self.device),
                    weight=spec.weight,
                )
            )
        return pairs

    def robot_anchor_world_points(self, pairs: list[SurfaceAnchorPair], actions: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if actions is None:
            actions = torch.zeros(1, self.robot.num_dofs, device=self.device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        transforms = self.robot.forward_kinematics(actions)
        output = {}
        for pair in pairs:
            if pair.robot_link_name not in transforms:
                continue
            output[pair.name] = transform_points(transforms[pair.robot_link_name], pair.robot_local_points)
        return output

    def human_anchor_points(
        self,
        pairs: list[SurfaceAnchorPair],
        betas: torch.Tensor,
        dmpls: torch.Tensor,
        scale: torch.Tensor,
        pose_name: str = "init",
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        body = self.body_model(**self.default_body_params(betas, dmpls, pose_name=pose_name))
        root = body.Jtr[:, 0:1, :]
        vertices = (body.v - root) * scale.view(1, 1, 1)
        output = {}
        for pair in pairs:
            output[pair.name] = vertices[:, pair.human_vertex_indices, :]
        return output, body.v

    def fit(
        self,
        pairs: list[SurfaceAnchorPair],
        max_iter: int = 300,
        lr: float = 0.01,
        verbose: bool = True,
        pose_actions: dict[str, torch.Tensor] | None = None,
        log_interval: int = 500,
    ) -> dict[str, torch.Tensor | float | dict[str, float]]:
        betas = torch.zeros(1, 16, device=self.device, requires_grad=True)
        dmpls = torch.zeros(1, 8, device=self.device, requires_grad=True)
        scale = torch.ones(1, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([betas, dmpls, scale], lr=lr)
        if pose_actions is None:
            pose_actions = self.pose_actions(["init"])
        robot_points_by_pose = {
            pose_name: self.robot_anchor_world_points(pairs, action.to(self.device))
            for pose_name, action in pose_actions.items()
        }

        last_loss = None
        for step in range(max_iter):
            losses = []
            for pose_name, robot_points in robot_points_by_pose.items():
                human_points, _ = self.human_anchor_points(pairs, betas, dmpls, scale, pose_name=pose_name)
                for pair in pairs:
                    if pair.name not in robot_points:
                        continue
                    losses.append(pair.weight * F.smooth_l1_loss(human_points[pair.name], robot_points[pair.name]))
            if not losses:
                raise ValueError("No valid anchor pairs to fit")
            prior = 0.001 * (betas.square().mean() + dmpls.square().mean()) + 0.01 * (scale - 1.0).square().mean()
            loss = torch.stack(losses).sum() + prior
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            if verbose and (step == 0 or (step + 1) % log_interval == 0 or step == max_iter - 1):
                print(f"[mesh-anchor-fit] step={step + 1:04d} loss={last_loss:.6f} scale={float(scale.detach().cpu()):.4f}")

        metrics = self.evaluate_pairs(
            pairs,
            betas.detach(),
            dmpls.detach(),
            scale.detach(),
            pose_actions=pose_actions,
            robot_points_by_pose=robot_points_by_pose,
        )
        return {
            "beta": betas.detach().cpu(),
            "dmpls": dmpls.detach().cpu(),
            "scale": scale.detach().cpu(),
            "loss": last_loss,
            "metrics": metrics,
            "fit_poses": tuple(pose_actions.keys()),
        }

    def evaluate_pairs(
        self,
        pairs: list[SurfaceAnchorPair],
        betas: torch.Tensor,
        dmpls: torch.Tensor,
        scale: torch.Tensor,
        pose_actions: dict[str, torch.Tensor] | None = None,
        robot_points_by_pose: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, float]:
        if pose_actions is None:
            pose_actions = self.pose_actions(["init"])
        if robot_points_by_pose is None:
            robot_points_by_pose = {
                pose_name: self.robot_anchor_world_points(pairs, action.to(self.device))
                for pose_name, action in pose_actions.items()
            }
        metrics = {}
        use_pose_prefix = len(pose_actions) > 1
        for pose_name, robot_points in robot_points_by_pose.items():
            human_points, _ = self.human_anchor_points(
                pairs,
                betas.to(self.device),
                dmpls.to(self.device),
                scale.to(self.device),
                pose_name=pose_name,
            )
            prefix = f"{pose_name}_" if use_pose_prefix else ""
            pose_errors = []
            for pair in pairs:
                if pair.name not in robot_points:
                    continue
                error = torch.linalg.norm(human_points[pair.name] - robot_points[pair.name], dim=-1)
                pose_errors.append(error.reshape(-1))
                metrics[f"{prefix}{pair.name}_mean_error_m"] = float(error.mean().detach().cpu())
                metrics[f"{prefix}{pair.name}_max_error_m"] = float(error.max().detach().cpu())
            if pose_errors:
                all_error = torch.cat(pose_errors)
                metrics[f"{prefix}all_surface_mean_error_m"] = float(all_error.mean().detach().cpu())
                metrics[f"{prefix}all_surface_max_error_m"] = float(all_error.max().detach().cpu())
        return metrics

    def export_debug_scene(
        self,
        pairs: list[SurfaceAnchorPair],
        path: str | os.PathLike,
        betas: torch.Tensor | None = None,
        dmpls: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        pose_name: str = "init",
        actions: torch.Tensor | None = None,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if betas is None:
            betas = torch.zeros(1, 16, device=self.device)
        if dmpls is None:
            dmpls = torch.zeros(1, 8, device=self.device)
        if scale is None:
            scale = torch.ones(1, device=self.device)
        human_points, _ = self.human_anchor_points(
            pairs,
            betas.to(self.device),
            dmpls.to(self.device),
            scale.to(self.device),
            pose_name=pose_name,
        )
        robot_points = self.robot_anchor_world_points(pairs, actions)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        for pair in pairs:
            if pair.name not in robot_points:
                continue
            hp = human_points[pair.name][0].detach().cpu().numpy()
            rp = robot_points[pair.name][0].detach().cpu().numpy()
            ax.scatter(hp[:, 0], hp[:, 1], hp[:, 2], s=12, label=f"human {pair.name}")
            ax.scatter(rp[:, 0], rp[:, 1], rp[:, 2], s=12, marker="^", label=f"robot {pair.name}")
            for h, r in zip(hp, rp):
                ax.plot([h[0], r[0]], [h[1], r[1]], [h[2], r[2]], linewidth=0.5, color="gray")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(loc="best")
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def export_debug_meshes(
        self,
        pairs: list[SurfaceAnchorPair],
        out_dir: str | os.PathLike,
        betas: torch.Tensor | None = None,
        dmpls: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        pose_name: str = "init",
        actions: torch.Tensor | None = None,
    ) -> tuple[Path, Path]:
        """Export fitted human mesh and anchored robot link meshes for inspection."""
        import trimesh

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if betas is None:
            betas = torch.zeros(1, 16, device=self.device)
        if dmpls is None:
            dmpls = torch.zeros(1, 8, device=self.device)
        if scale is None:
            scale = torch.ones(1, device=self.device)

        body = self.body_model(**self.default_body_params(betas.to(self.device), dmpls.to(self.device), pose_name=pose_name))
        root = body.Jtr[:, 0:1, :]
        human_vertices = ((body.v - root) * scale.to(self.device).view(1, 1, 1))[0].detach().cpu().numpy()
        human_faces = self.faces.detach().cpu().numpy()
        human_mesh_path = out_dir / f"{pose_name}_human_fit_mesh.ply"
        trimesh.Trimesh(vertices=human_vertices, faces=human_faces, process=False).export(human_mesh_path)

        if actions is None:
            actions = torch.zeros(1, self.robot.num_dofs, device=self.device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        transforms = self.robot.forward_kinematics(actions)
        vertices_all = []
        faces_all = []
        face_offset = 0
        for link_name in sorted({pair.robot_link_name for pair in pairs}):
            if link_name not in self.robot_meshes or link_name not in transforms:
                continue
            transform = transforms[link_name]
            for mesh in self.robot_meshes[link_name]:
                local_vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=self.device)
                world_vertices = transform_points(transform, local_vertices)[0].detach().cpu().numpy()
                vertices_all.append(world_vertices)
                faces_all.append(mesh.faces.astype(np.int64) + face_offset)
                face_offset += world_vertices.shape[0]
        robot_mesh_path = out_dir / f"{pose_name}_robot_anchor_links_mesh.ply"
        if vertices_all:
            trimesh.Trimesh(
                vertices=np.concatenate(vertices_all, axis=0),
                faces=np.concatenate(faces_all, axis=0),
                process=False,
            ).export(robot_mesh_path)
        return human_mesh_path, robot_mesh_path

    def render_human_robot_overlay(
        self,
        out_dir: str | os.PathLike,
        betas: torch.Tensor,
        dmpls: torch.Tensor,
        scale: torch.Tensor,
        actions: torch.Tensor | None = None,
        pose_name: str = "init",
        views: tuple[str, ...] = ("front", "side", "iso"),
        width: int = 1400,
        height: int = 1000,
    ) -> dict[str, Path]:
        """Render fitted SMPLH mesh inside the existing robot pyrender scene."""
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import pyrender
        import trimesh
        from PIL import Image

        from GBC.utils.data_preparation.robot_visualizer import RobotVisualizer

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if actions is None:
            actions = torch.zeros(1, self.robot.num_dofs, device=self.device)
        if actions.dim() == 2:
            q = actions[0]
        else:
            q = actions

        with torch.no_grad():
            body = self.body_model(**self.default_body_params(betas.to(self.device), dmpls.to(self.device), pose_name=pose_name))
            vertices = ((body.v - body.Jtr[:, 0:1, :]) * scale.to(self.device).view(1, 1, 1))[0]
        root_tf = robot_viewer_root_transform(self.device)
        vertices_h = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1)
        vertices_vis = (root_tf @ vertices_h.T).T[:, :3].detach().cpu().numpy()
        faces = self.faces.detach().cpu().numpy()

        human_mesh = trimesh.Trimesh(vertices=vertices_vis, faces=faces, process=False)
        human_mesh.visual.vertex_colors = np.tile(np.array([58, 127, 255, 115], dtype=np.uint8), (vertices_vis.shape[0], 1))
        human_obj = pyrender.Mesh.from_trimesh(human_mesh, smooth=False)

        vis = RobotVisualizer(
            self.urdf_path,
            use_offscreen=True,
            width=width,
            height=height,
            robot_transmission=0.30,
            device=str(self.device),
        )
        q = q.to(vis.device)
        fk = vis.robot_kinematics.forward_kinematics(q.unsqueeze(0) if q.dim() == 1 else q)
        robot_points = []
        for link_name, link_meshes in vis.link_meshes.items():
            if link_name not in fk:
                continue
            link_tf = root_tf @ fk[link_name][0].to(self.device)
            for offset, mesh in link_meshes:
                offset_tf = offset.to(self.device)
                if offset_tf.dim() == 3:
                    offset_tf = offset_tf[0]
                mesh_tf = link_tf @ offset_tf
                mesh_vertices = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=self.device)
                mesh_vertices_h = torch.cat([mesh_vertices, torch.ones_like(mesh_vertices[:, :1])], dim=1)
                robot_points.append((mesh_tf @ mesh_vertices_h.T).T[:, :3].detach().cpu().numpy())
        if robot_points:
            all_points = np.concatenate([vertices_vis] + robot_points, axis=0)
        else:
            all_points = vertices_vis
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        target = (mins + maxs) * 0.5
        extent = float(np.max(maxs - mins))
        dist = max(extent * 2.2, 1.0)
        eye_by_view = {
            "front": target + np.array([0.0, -0.20 * dist, 1.00 * dist], dtype=np.float32),
            "side": target + np.array([1.00 * dist, -0.12 * dist, 0.25 * dist], dtype=np.float32),
            "iso": target + np.array([0.70 * dist, 0.35 * dist, 0.80 * dist], dtype=np.float32),
        }

        paths: dict[str, Path] = {}
        for view in views:
            if view not in eye_by_view:
                raise ValueError(f"Unknown view '{view}', expected one of {sorted(eye_by_view)}")
            image = vis(
                q,
                root_tf=root_tf,
                extra_pyrender_objs=[human_obj],
                cam_pose=look_at_camera_pose(eye_by_view[view], target),
                cam_yfov=np.pi * 0.23,
            )
            path = out_dir / f"{pose_name}_human_robot_mesh_overlay_{view}.png"
            Image.fromarray(image).save(path)
            paths[view] = path
        return paths


def main() -> None:
    from GBC.utils.base.assets import DATA_PATHS

    parser = argparse.ArgumentParser(description="Prototype mesh-anchor SMPLH fit.")
    parser.add_argument("--urdf-path", default=DATA_PATHS.urdf_path)
    parser.add_argument("--smplh-model-path", default=DATA_PATHS.smplh_model_path)
    parser.add_argument("--dmpls-model-path", default=DATA_PATHS.dmpls_model_path)
    parser.add_argument("--out-dir", default="outputs/mesh_anchor_smplh_fit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--fit-poses", default="init,tpose")
    parser.add_argument("--pose-preset", choices=("g1", "none"), default="g1")
    parser.add_argument("--anchor-preset", choices=("g1",), default="g1")
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--no-fit", action="store_true")
    parser.add_argument("--render-overlay", action="store_true")
    parser.add_argument("--overlay-width", type=int, default=1400)
    parser.add_argument("--overlay-height", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fitter = MeshAnchorSMPLHFitter(
        smplh_model_path=args.smplh_model_path,
        dmpls_model_path=args.dmpls_model_path,
        urdf_path=args.urdf_path,
        device=args.device,
    )
    fitter.apply_pose_preset(args.pose_preset)
    if args.anchor_preset == "g1":
        anchor_specs = g1_surface_anchor_specs()
    else:
        raise ValueError(f"Unsupported anchor preset '{args.anchor_preset}'")
    pairs = fitter.build_anchor_pairs(anchor_specs)
    if not pairs:
        raise RuntimeError("No default mesh-anchor pairs could be built for this robot")
    print(f"Built {len(pairs)} mesh-anchor pairs: {[p.name for p in pairs]}")
    pose_names = tuple(name.strip() for name in args.fit_poses.split(",") if name.strip())
    pose_actions = fitter.pose_actions(pose_names, pose_preset=args.pose_preset)
    print(f"Fit poses: {list(pose_actions.keys())}")
    zero_beta = torch.zeros(1, 16)
    zero_dmpls = torch.zeros(1, 8)
    one_scale = torch.ones(1)
    before = fitter.evaluate_pairs(pairs, zero_beta, zero_dmpls, one_scale, pose_actions=pose_actions)
    print("Before fit:", before)
    if args.no_fit:
        result = {"beta": zero_beta, "dmpls": zero_dmpls, "scale": one_scale, "metrics": before, "fit_poses": pose_names}
    else:
        result = fitter.fit(
            pairs,
            max_iter=args.iters,
            lr=args.lr,
            pose_actions=pose_actions,
            log_interval=args.log_interval,
        )
        print("After fit:", result["metrics"])
    torch.save(result, out_dir / "mesh_anchor_fit.pt")
    mesh_paths = []
    for pose_name, action in pose_actions.items():
        fitter.export_debug_scene(
            pairs,
            out_dir / f"{pose_name}_mesh_anchor_debug.png",
            betas=result["beta"],
            dmpls=result["dmpls"],
            scale=result["scale"],
            pose_name=pose_name,
            actions=action,
        )
        mesh_paths.extend(
            fitter.export_debug_meshes(
                pairs,
                out_dir,
                betas=result["beta"],
                dmpls=result["dmpls"],
                scale=result["scale"],
                pose_name=pose_name,
                actions=action,
            )
        )
    print(f"Saved {out_dir / 'mesh_anchor_fit.pt'}")
    for pose_name in pose_actions:
        print(f"Saved {out_dir / f'{pose_name}_mesh_anchor_debug.png'}")
    for mesh_path in mesh_paths:
        print(f"Saved {mesh_path}")
    if args.render_overlay:
        for pose_name, action in pose_actions.items():
            overlay_paths = fitter.render_human_robot_overlay(
                out_dir,
                betas=result["beta"],
                dmpls=result["dmpls"],
                scale=result["scale"],
                actions=action,
                pose_name=pose_name,
                width=args.overlay_width,
                height=args.overlay_height,
            )
            for view, path in overlay_paths.items():
                print(f"Saved {pose_name} {view} overlay {path}")


if __name__ == "__main__":
    main()
