from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn

from GBC.utils.base.base_fk import RobotKinematics
from GBC.utils.base.math_utils import euler_xyz_to_rot_mat


@dataclass(frozen=True)
class _CachedJoint:
    parent: str
    child: str
    name: str
    joint_type: str
    dof_index: int


def _axis_angle_rot_mat(theta: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Build rotation matrices for a fixed unit axis and batched angles."""
    axis = axis.to(device=theta.device, dtype=theta.dtype)
    x, y, z = axis.unbind(dim=-1)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    one_minus = 1.0 - cos_t
    zeros = torch.zeros_like(theta)

    row0 = torch.stack(
        [
            cos_t + x * x * one_minus,
            x * y * one_minus - z * sin_t,
            x * z * one_minus + y * sin_t,
        ],
        dim=-1,
    )
    row1 = torch.stack(
        [
            y * x * one_minus + z * sin_t,
            cos_t + y * y * one_minus,
            y * z * one_minus - x * sin_t,
        ],
        dim=-1,
    )
    row2 = torch.stack(
        [
            z * x * one_minus - y * sin_t,
            z * y * one_minus + x * sin_t,
            cos_t + z * z * one_minus,
        ],
        dim=-1,
    )
    rot = torch.stack([row0, row1, row2], dim=-2)
    return rot + zeros.view(-1, 1, 1) * 0.0


class CachedTargetLinkKinematics(nn.Module):
    """Cached FK for a fixed target-link set.

    The existing :class:`RobotKinematics` remains the correctness baseline. This
    module caches URDF constants, DoF indices, and the minimal kinematic subtree
    needed by `target_links` so repeated SFT loss calls avoid per-joint URDF and
    networkx work.
    """

    def __init__(
        self,
        urdf_path: str,
        target_links: Iterable[str],
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.reference = RobotKinematics(urdf_path, device=str(device))
        self.base_link = self.reference.base_link
        self.dof_names = tuple(self.reference.get_dof_names())
        self.num_dofs = self.reference.num_dofs
        self.dof_edges = tuple(self.reference.get_dof_edges())
        self.dof_feature_names = tuple(self.reference.get_dof_feature_names())
        self.register_buffer("dof_features", self.reference.get_dof_features().detach().clone())
        self.target_links = tuple(target_links)
        if not self.target_links:
            raise ValueError("target_links must not be empty")

        self._joints = self._build_cached_joints()
        origin_xyz = []
        origin_rot = []
        axes = []
        for cached in self._joints:
            joint = self.reference.joint_map.get(cached.name)
            if joint is None:
                joint = self.reference.kinematic_tree[cached.parent][cached.child]["joint"]
            xyz = joint.origin.xyz if getattr(joint, "origin", None) is not None and joint.origin.xyz else [0.0, 0.0, 0.0]
            rpy = joint.origin.rpy if getattr(joint, "origin", None) is not None and joint.origin.rpy else [0.0, 0.0, 0.0]
            axis = joint.axis if joint.axis else [0.0, 0.0, 1.0]
            axis_tensor = torch.as_tensor(axis, dtype=torch.float32)
            axis_tensor = axis_tensor / axis_tensor.norm().clamp_min(1e-12)
            origin_xyz.append(torch.as_tensor(xyz, dtype=torch.float32))
            origin_rot.append(
                euler_xyz_to_rot_mat(torch.as_tensor(rpy, dtype=torch.float32).view(1, 3))[0]
            )
            axes.append(axis_tensor)

        self.register_buffer("origin_xyz", torch.stack(origin_xyz, dim=0))
        self.register_buffer("origin_rot", torch.stack(origin_rot, dim=0))
        self.register_buffer("axes", torch.stack(axes, dim=0))
        self.to(device)

    def _build_cached_joints(self) -> tuple[_CachedJoint, ...]:
        tree = self.reference.kinematic_tree
        required_links = {self.base_link}
        required_edges = set()
        for target_link in self.target_links:
            if target_link not in self.reference.get_link_names():
                raise ValueError(f"Unknown target link: {target_link}")
            path = self._path_to_target(target_link)
            required_links.update(path)
            required_edges.update(zip(path[:-1], path[1:]))

        dof_index = {name: i for i, name in enumerate(self.dof_names)}
        cached = []
        queue = [self.base_link]
        while queue:
            parent = queue.pop(0)
            for child in tree.successors(parent):
                if (parent, child) not in required_edges:
                    continue
                joint = tree[parent][child]["joint"]
                if joint.type == "fixed" and (
                    not hasattr(joint, "origin")
                    or not hasattr(joint.origin, "xyz")
                    or joint.origin.xyz is None
                ):
                    continue
                cached.append(
                    _CachedJoint(
                        parent=parent,
                        child=child,
                        name=joint.name,
                        joint_type=joint.type,
                        dof_index=dof_index.get(joint.name, -1),
                    )
                )
                queue.append(child)
        return tuple(cached)

    def _path_to_target(self, target_link: str) -> list[str]:
        tree = self.reference.kinematic_tree
        queue = [(self.base_link, [self.base_link])]
        while queue:
            link, path = queue.pop(0)
            if link == target_link:
                return path
            for child in tree.successors(link):
                queue.append((child, path + [child]))
        raise ValueError(f"No path from {self.base_link} to {target_link}")

    def get_dof_edges(self, include_root_edges: bool = True) -> tuple[tuple[int, int], ...]:
        # Cached `self.dof_edges` is the include_root_edges=True graph (root
        # clique). When root edges are off, recompute the chain-only graph from
        # the live reference kinematics (cheap, called once at model build).
        if include_root_edges:
            return self.dof_edges
        return tuple(self.reference.get_dof_edges(include_root_edges=False))

    def get_dof_edge_names(self, include_root_edges: bool = True) -> tuple[tuple[str, str], ...]:
        return tuple(
            (self.dof_names[i], self.dof_names[j])
            for i, j in self.get_dof_edges(include_root_edges=include_root_edges)
        )

    def get_dof_feature_names(self) -> tuple[str, ...]:
        return self.dof_feature_names

    def get_dof_features(self) -> torch.Tensor:
        return self.dof_features.to(device=self.device)

    @property
    def device(self) -> torch.device:
        return self.origin_xyz.device

    def forward(
        self,
        dof_pos: torch.Tensor,
        root_trans_offset: Optional[torch.Tensor] = None,
        root_rot: Optional[torch.Tensor] = None,
        link_trans_offset: Optional[dict[str, torch.Tensor]] = None,
        return_full: bool = False,
    ) -> torch.Tensor:
        if dof_pos.dim() != 2:
            raise ValueError("dof_pos must be a 2D tensor.")
        if dof_pos.shape[1] != self.num_dofs:
            raise ValueError(f"dof_pos must have {self.num_dofs} columns. Current shape: {dof_pos.shape}")

        batch_size = dof_pos.shape[0]
        dtype = dof_pos.dtype
        device = dof_pos.device
        eye3 = torch.eye(3, dtype=dtype, device=device).expand(batch_size, 3, 3)
        if root_rot is None:
            root_R = eye3
        else:
            root_R = euler_xyz_to_rot_mat(root_rot.to(device=device, dtype=dtype))
        if root_trans_offset is None:
            root_t = dof_pos.new_zeros(batch_size, 3)
        else:
            root_t = root_trans_offset.to(device=device, dtype=dtype)

        rotations = {self.base_link: root_R}
        translations = {self.base_link: root_t}
        for edge_idx, cached in enumerate(self._joints):
            parent_R = rotations[cached.parent]
            parent_t = translations[cached.parent]
            origin_R = self.origin_rot[edge_idx].to(device=device, dtype=dtype).expand(batch_size, 3, 3)
            origin_t = self.origin_xyz[edge_idx].to(device=device, dtype=dtype).expand(batch_size, 3)
            axis = self.axes[edge_idx].to(device=device, dtype=dtype)

            if cached.joint_type in {"revolute", "continuous"}:
                joint_R = _axis_angle_rot_mat(dof_pos[:, cached.dof_index], axis)
                local_R = torch.bmm(origin_R, joint_R)
                local_t = origin_t
            elif cached.joint_type == "prismatic":
                disp = dof_pos[:, cached.dof_index].unsqueeze(-1) * axis.view(1, 3)
                local_R = origin_R
                local_t = origin_t + torch.bmm(origin_R, disp.unsqueeze(-1)).squeeze(-1)
            elif cached.joint_type == "fixed":
                local_R = origin_R
                local_t = origin_t
            else:
                raise NotImplementedError(f"Joint type '{cached.joint_type}' is not supported.")

            child_R = torch.bmm(parent_R, local_R)
            child_t = parent_t + torch.bmm(parent_R, local_t.unsqueeze(-1)).squeeze(-1)
            if link_trans_offset is not None and cached.child in link_trans_offset:
                offset = link_trans_offset[cached.child].to(device=device, dtype=dtype)
                child_t = child_t + torch.bmm(child_R, offset.view(1, 3, 1).expand(batch_size, 3, 1)).squeeze(-1)

            rotations[cached.child] = child_R
            translations[cached.child] = child_t

        pos_out = torch.stack([translations[name] for name in self.target_links], dim=1)
        if not return_full:
            return pos_out
        rot_out = torch.stack([rotations[name] for name in self.target_links], dim=1)
        return pos_out, rot_out


def make_cached_target_fk_from_mujica_cfg(
    cfg_path: str,
    target_links_key: str = "foot_names",
    device: str | torch.device = "cpu",
) -> CachedTargetLinkKinematics:
    import yaml

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    base = cfg["base"]
    return CachedTargetLinkKinematics(
        urdf_path=base["urdf_path"],
        target_links=base[target_links_key],
        device=device,
    )
