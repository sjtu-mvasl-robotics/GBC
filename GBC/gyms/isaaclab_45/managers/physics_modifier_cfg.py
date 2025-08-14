from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any, Tuple, Union, Literal

from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierCfg
from isaaclab.utils.noise import NoiseCfg
from isaaclab.managers.manager_base import ManagerTermBaseCfg

@configclass
class PhysicsModifierTermCfg(ManagerTermBaseCfg):
    func: Callable = MISSING
    """The function or class to modify the term."""
    description: str = ""
    """Description of the term."""
    