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
class ReferenceObservationTermCfg(ManagerTermBaseCfg):
    """Configuration for a reference observation term."""
    name: str | dict | None = None
    """The name of the reference observation term."""
    func: Callable | None = None
    """The function or class to modify the term."""
    symmetry: Callable | None = None
    """The function or class to symmetry the term."""
    symmetry_params: dict[str, Any] = {}
    """The parameters for the symmetry function."""
    params: dict[str, Any] | None = None
    modifiers: list[ModifierCfg] | None = None
    noise: NoiseCfg | None = None
    clip: tuple[float, float] | None = None
    
    scale: float | None = None
    
    make_empty: bool = False
    
    in_obs_tensor = True
    is_constant = False
    is_base_pose: bool = False
    
    load_seq_delay: float = 0.0
    """Delay for loading the sequence data. Unit is in seconds. Defaults to 0.0."""
    
    history_length: int = 0
    """Number of past observations to store in the observation buffers. Defaults to 0, meaning no history.

    Observation history initializes to empty, but is filled with the first append after reset or initialization. Subsequent history
    only adds a single entry to the history buffer. If flatten_history_dim is set to True, the source data of shape
    (N, H, D, ...) where N is the batch dimension and H is the history length will be reshaped to a 2D tensor of shape
    (N, H*D*...). Otherwise, the data will be returned as is.
    """
    flatten_history_dim: bool = True
    """Whether or not the observation manager should flatten history-based observation terms to a 2D (N, D) tensor.
    Defaults to True."""


@configclass
class ReferenceObservationGroupCfg:
    """Configuration for a group of reference observation terms.
    
    Important:
        The reference action must be the first term in the group if `teacher_coef` for PPO is not None (for critic only).
        We did not implement forced check for this, so please make sure to follow this rule.
        If you want to adjust the order of the terms, parse the index of the terms in rsl_rl_ref_ppo_cfg.py.
    """
    concatenate_terms: bool = True
    enable_corruption: bool = False
    
    load_seq_delay: float = 0.0
    """Delay for loading the sequence data. Unit is in seconds. Defaults to 0.0.
    
    This parameter will override all :attr:`ReferenceObservationTermCfg.load_seq_delay` in the group if
    :attr:`ReferenceObservationTermCfg.load_seq_delay` is set.
    """
    
    history_length: int | None = None
    """Number of past observation to store in the observation buffers for all observation terms in group.

    This parameter will override :attr:`ObservationTermCfg.history_length` if set. Defaults to None. If None, each
    terms history will be controlled on a per term basis. See :class:`ObservationTermCfg` for details on history_length
    implementation.
    """

    flatten_history_dim: bool = True
    """Flag to flatten history-based observation terms to a 2D (num_env, D) tensor for all observation terms in group.
    Defaults to True.

    This parameter will override all :attr:`ObservationTermCfg.flatten_history_dim` in the group if
    ObservationGroupCfg.history_length is set.
    """

@configclass
class ReferenceObservationCfg:
    # terms: ReferenceObservationGroupCfg = MISSING
    # """The configuration for the reference observation terms."""
    data_dir: list[str] = MISSING
    """The directories containing the reference observation data."""
    working_mode: Literal["recurrent", "recurrent_strict", "singular"] = "recurrent"
    """The working mode of the reference observation manager."""
    static_delay: float = 0.0
    """Static delay for retrieving reference observation"""
    