from __future__ import annotations

import inspect
import torch
import weakref
from abc import abstractmethod
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

# import omni.kit.app

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers.command_manager import CommandManager


class ReferenceCommandManager(CommandManager):
    """Manages the reference commands for a reference observation manager.

    Overrides command manager's commands when reference are available.
    """
    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __str__(self):
        msg = super().__str__()
        return msg.replace("CommandManager", "ReferenceCommandManager")

    def get_command(self, name):
        """Only available if command is of (lin_x, lin_y, ang_z) format. Otherwise, modify this function by yourself."""
        orig_command = super().get_command(name)
        if hasattr(self._env, "ref_observation_manager") and name == "base_velocity":
            try:
                # cur_time = self._env.episode_length_buf.to(torch.float32) * self._env.step_dt
                lin_vel, mask = self._env.ref_observation_manager.get_term("base_lin_vel")
                # (Warning!) Using lin_vel calculated from AMASS translation seems to be extremely inaccurate, the reward weight for tracking_lin_vel should be relatively small.
                ang_vel, mask = self._env.ref_observation_manager.get_term("base_ang_vel")
                if mask is None:
                    mask = torch.zeros_like(lin_vel[:, 0], dtype=torch.bool)
                override_command = torch.cat([lin_vel[:, :2], ang_vel[:, 2:]], dim=1)
                # mask: shape (batch_size,). Only replace the command where mask is True.
                orig_command[mask] = override_command[mask]
            except Exception as e:
                return orig_command
        return orig_command

