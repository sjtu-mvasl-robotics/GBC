'''
    Reference Observation Manager
    This manager is responsible for generating reference observations for the policy to learn from.
    Under our current implementation, for any humanoid task, the reference observation is fixed to the following:
    - ref_action: torch.Tensor of shape (num_envs, num_actions)
    - ref_translation: torch.Tensor of shape (num_envs, 3). Global translation of the robot calibrated to the robot's initial position.
    - ref_rotation: torch.Tensor of shape (num_envs, 4). Global rotation of the robot calibrated to the robot's initial orientation.
    - ref_velocity: torch.Tensor of shape (num_envs, 3). Global velocity of the robot calibrated to the robot's initial velocity. Order: [linear_x, linear_y, angular_z]
    - (optional) ref_root_height: torch.Tensor of shape (num_envs, 1). Global height of the robot's root link calibrated to the robot's initial height.
    - (optional) ref_joint_positions: torch.Tensor of shape (num_envs, num_required_joints). Global joint positions of the robot calibrated to the robot's initial joint positions. You need to specify the required joints in the configuration file.

    Additionally, you have to specify the observation manager's working mode:
    - "singular": The reference observation only plays the whole episode once. After that, the reference observation will be zeroed out, and corresponding masks will be set to False.
    - "recurrent": Each MOCAP sequence with `recurrent` tuple will be played repeatedly in the episode until timeout. The sequence without `recurrent` tuple will be played once. After that, the reference observation will be zeroed out, and corresponding masks will be set to False.
    - "recurrent_strict": Working in recurrent mode, but the MOCAP sequence without `recurrent` will be discarded.

'''
import os
import time
import torch
from collections.abc import Sequence
from prettytable import PrettyTable
import inspect
import copy
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerBase, ManagerTermBase
from isaaclab.utils import modifiers
from isaaclab.utils.buffers import CircularBuffer
from .ref_obs_term_cfg import ReferenceObservationCfg, ReferenceObservationTermCfg, ReferenceObservationGroupCfg
from glob import glob
from GBC.utils.buffer.ref_buffer import BufferManager
import numpy as np

ref_obs_type = tuple[torch.Tensor, torch.Tensor] | None


class RefObservationManager(ManagerBase):
    def __init__(self, cfg: ReferenceObservationCfg, env: ManagerBasedEnv):
        assert cfg is not None, "The Configuration for RefObservationManager shouldn't be empty (in order to call _prepare_terms())"
        self._env = env
        self._env_start_time = torch.zeros(self.num_envs, device=self.device)
        self._load_pickles(cfg)
        self.static_delay = cfg.static_delay
        self.debug_show_time_between_steps = 100
        self.debug_count_steps = 0
        self.debug_not_compute_obs_time = 0
        self.debug_compute_obs_time = 0
        self.debug_last_done_time = None
        super().__init__(cfg, env)
        self._group_ref_obs_dim: dict[str, tuple[int, ...] | list[tuple[int, ...]]] = dict()
        for group_name, group_term_dims in self._group_ref_obs_term_dim.items():
            if self._group_ref_obs_concatenate[group_name]:
                try:
                    term_dims = [torch.tensor(term_dim) for term_dim in group_term_dims]
                    self._group_ref_obs_dim[group_name] = tuple(torch.sum(torch.stack(term_dims, dim=0), dim=0).tolist())
                except RuntimeError:
                    raise RuntimeError(
                        f"Unable to concatenate observation terms in group '{group_name}'."
                        f" The shapes of the terms are: {group_term_dims}."
                        " Please ensure that the shapes are compatible for concatenation."
                        " Otherwise, set 'concatenate_terms' to False in the group configuration."
                    )
            else:
                self._group_ref_obs_dim[group_name] = group_term_dims


    @property
    def start_time(self):
        return self._env_start_time
    
    @start_time.setter
    def start_time(self, time: torch.Tensor):
        assert time.shape == (self.num_envs,), "start_time should be of shape (num_envs,)"
        self._env_start_time = time


    @property
    def active_terms(self):
        return self._group_ref_obs_term_names
    
    @property
    def detailed_active_terms(self):
        names = []
        for group_name in self._group_ref_obs_term_names:
            names += [f"{group_name}/{term_name}" for term_name in self._group_ref_obs_term_names[group_name]]
        return names
    
    @property
    def group_ref_obs_dim(self):
        return self._group_ref_obs_dim
    
    @property
    def group_ref_obs_term_dim(self):
        return self._group_ref_obs_term_dim
    
    @property
    def group_ref_obs_concatenate(self):
        return self._group_ref_obs_concatenate
    
    @property
    def group_ref_obs_term_buffer(self):
        return self._group_ref_obs_term_buffer_manager
    
    def reset(self, env, env_ids:Sequence[int] | None = None):
        for group_name, group_cfg in self._group_ref_obs_class_term_cfgs.items():
            for term_cfg in group_cfg:
                term_cfg.func.reset(env_ids = env_ids)

            for term_name in self._group_ref_obs_term_names[group_name]:
                if term_name in self._group_ref_obs_term_history_buffer[group_name]:
                    self._group_ref_obs_term_history_buffer[group_name][term_name].reset(batch_ids = env_ids)
        for mod in self._group_ref_obs_class_modifiers:
            mod.reset(env_ids = env_ids)
        for buf_manager in self._group_ref_obs_term_buffer_manager.values():
            buf_manager.reset(env, env_ids)
        return {}
    
    def compute(self, cur_time: torch.Tensor, add_start_time: bool = True, symmetry: bool = False) -> dict[str, tuple[torch.Tensor, torch.Tensor] | None | dict[str, tuple[torch.Tensor, torch.Tensor] | None]]:
        ref_obs = dict()
        if add_start_time:
            cur_time = cur_time + self._env_start_time
        for group_name in self._group_ref_obs_term_names:
            ref_obs[group_name] = self.compute_group(group_name, cur_time, symmetry)
        return ref_obs
    
    def compute_policy_symmetry(self, ref_observations_tuple: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        assert "policy" in self._group_ref_obs_term_names, "policy group is not defined, cannot compute policy symmetry"
        group_term_names = self._group_ref_obs_term_names["policy"]
        group_term_cfgs = self._group_ref_obs_term_cfgs["policy"]
        group_term_concatenate_last_dim = copy.deepcopy(self._group_ref_obs_term_concatenate_last_dim["policy"])
        group_term_dim = self._group_ref_obs_term_dim["policy"]
        assert len(group_term_concatenate_last_dim) == len(group_term_names), "policy group term concatenate last dim is not properly initialized"
        ref_observations, ref_masks = ref_observations_tuple
        group_term_concatenate_last_dim += [ref_observations.shape[-1]]
        symmetry_ref_observations = torch.zeros_like(ref_observations)
        for i in range(len(group_term_names)):
            term_name = group_term_names[i]
            term_cfg = group_term_cfgs[i]
            if not term_cfg.in_obs_tensor:
                continue
            assert term_cfg.symmetry is not None, f"Symmetry is not implemented for term {term_name} in group policy"
            if term_cfg.symmetry_params and term_cfg.symmetry_params.get("term_name", None) is not None:
                target_term_name = term_cfg.symmetry_params["term_name"]
                assert target_term_name in group_term_names, f"Target term {target_term_name} is not defined in group policy"
                target_term_idx = group_term_names.index(target_term_name)
                symmetry_ref_observations[..., group_term_concatenate_last_dim[i]:group_term_concatenate_last_dim[i+1]] = ref_observations[..., group_term_concatenate_last_dim[target_term_idx]:group_term_concatenate_last_dim[target_term_idx+1]]
            else:
                if term_cfg.history_length > 0 and term_cfg.flatten_history_dim:
                    target_obs = ref_observations[..., group_term_concatenate_last_dim[i]:group_term_concatenate_last_dim[i+1]].reshape(-1, term_cfg.history_length, group_term_dim[i][-1] // term_cfg.history_length)
                    target_obs = term_cfg.symmetry(self._env, target_obs, **term_cfg.symmetry_params)
                    target_obs = target_obs.reshape(ref_observations.shape[0], -1)
                    symmetry_ref_observations[..., group_term_concatenate_last_dim[i]:group_term_concatenate_last_dim[i+1]] = target_obs
                else:
                    symmetry_ref_observations[..., group_term_concatenate_last_dim[i]:group_term_concatenate_last_dim[i+1]] = term_cfg.symmetry(self._env, ref_observations[..., group_term_concatenate_last_dim[i]:group_term_concatenate_last_dim[i+1]], **term_cfg.symmetry_params)
        return (symmetry_ref_observations, ref_masks)

    def _compute_policy_term_idx(self, term_name: str) -> tuple[int, int]:
        group_term_names = self._group_ref_obs_term_names["critic"]
        group_term_concatenate_last_dim = copy.deepcopy(self._group_ref_obs_term_concatenate_last_dim["critic"])
        group_term_concatenate_last_dim += [-1]
        assert term_name in group_term_names, f"Term {term_name} is not defined in group critic"
        term_idx = group_term_names.index(term_name)
        offset = 0
        # AMP requires only current state, so we need to remove history dimension (if any)
        if self._group_ref_obs_term_cfgs["critic"][term_idx].history_length > 0 and self._group_ref_obs_term_cfgs["critic"][term_idx].flatten_history_dim:
            history_length = self._group_ref_obs_term_cfgs["critic"][term_idx].history_length
            term_dim = self._group_ref_obs_term_dim["critic"][term_idx][-1]
            offset = (term_dim // history_length) * (history_length - 1)
        return group_term_concatenate_last_dim[term_idx] + offset, group_term_concatenate_last_dim[term_idx+1]   

    def compute_amp_dims(self, term_names: list[str]) -> list[tuple[int, int]]:
        if hasattr(self, "last_amp_term_names"):
            if term_names == self.last_amp_term_names and hasattr(self, "last_amp_dims"):
                return self.last_amp_dims
        amp_dims = []
        for term_name in term_names:
            amp_dims.append(self._compute_policy_term_idx(term_name))
        self.last_amp_term_names = term_names
        self.last_amp_dims = amp_dims
        return amp_dims            

    def compute_term(self, term_name: str, cur_time: torch.Tensor, add_start_time: bool = True) -> tuple[torch.Tensor, torch.Tensor] | None:
        if add_start_time:
            cur_time = cur_time + self._env_start_time
        for group_name in self._group_ref_obs_term_names:
            if term_name in self._group_ref_obs_term_names[group_name]:
                idx = self._group_ref_obs_term_names[group_name].index(term_name)
                term_cfg = self._group_ref_obs_term_cfgs[group_name][idx]
                term_delay = self._group_ref_obs_init_delay[group_name][idx]
                if term_delay != 0:
                    cp_cur_time = torch.clamp(cur_time - term_delay, min=0.0)
                else:
                    cp_cur_time = cur_time
                # if term_cfg.is_base_pose:
                #     raise ValueError("don't call compute_term on link pose")
                buf_manager = self._group_ref_obs_term_buffer_manager[group_name]
                if term_cfg.is_base_pose:
                    return buf_manager.calc_base_pose_cumulative(cp_cur_time, term_cfg.params["lin_vel_name"], term_cfg.params["ang_vel_name"]), buf_manager.calc_mask(cp_cur_time)
                obs, mask = buf_manager.calc_obs(term_name, cp_cur_time), buf_manager.calc_mask(cp_cur_time)
                if term_cfg.modifiers is not None:
                    for mod in term_cfg.modifiers:
                        obs = mod.func(obs, **mod.params)
                        
                 # apply noise
                if term_cfg.noise is not None:
                    obs = term_cfg.noise.func(obs, term_cfg.noise)
                # apply clipping
                if term_cfg.clip is not None:
                    obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
                # apply scaling
                if term_cfg.scale is not None:
                    obs = obs.mul_(term_cfg.scale)
                        
                if term_cfg.env_func is not None:
                    func, func_params = term_cfg.env_func, term_cfg.env_func_params
                    obs = func(self._env, obs, **func_params)
                return obs, mask
        
        raise ValueError(f"Invalid term name '{term_name}'. Expected one of: {self._group_ref_obs_term_names.values()}.")
    
    def get_term(self, term_name: str, symmetry: bool = False) -> tuple[torch.Tensor, torch.Tensor] | None:
        for group_name in self._group_ref_obs_term_names:
            if term_name in self._group_ref_obs_term_tmp_storage[group_name].keys():
                if not symmetry:
                    return self._group_ref_obs_term_tmp_storage[group_name][term_name]
                else:
                    assert term_cfg.symmetry is not None, f"Symmetry is not implemented for term {term_name} in group {group_name}."
                    term_cfg = self._group_ref_obs_term_cfgs[group_name][self._group_ref_obs_term_names[group_name].index(term_name)]
                    inputs, mask = self._group_ref_obs_term_tmp_storage[group_name][term_name]
                    inputs = term_cfg.symmetry(self._env, inputs, **term_cfg.symmetry_params)
                    return inputs, mask
        
        raise ValueError(f"Invalid term name '{term_name}'. Expected one of: {self._group_ref_obs_term_names.values()}.")
        

    def _set_term_tmp_storage(self):
        cur_time = torch.zeros(self.num_envs, device=self.device)
        for group_name in self._group_ref_obs_term_names:
            self.compute_group(group_name, cur_time + self.static_delay + 1e-6)


    def compute_group(self, group_name: str, cur_time: torch.Tensor, symmetry: bool = False) -> tuple[torch.Tensor, torch.Tensor] | None | dict[str, tuple[torch.Tensor, torch.Tensor] | None]:
        """Computes the observations for a given group.

        The observations for a given group are computed by calling the registered functions for each
        term in the group. The functions are called in the order of the terms in the group. The functions
        are expected to return a tensor with shape (num_envs, ...).

        The following steps are performed for each observation term:

        1. Compute observation term by querying the corresponding buffer using current_time
        2. Apply custom modifiers in the order specified in :attr:`ReferenceObservationTermCfg.modifiers`
        3. Apply corruption/noise model based on :attr:`ReferenceObservationTermCfg.noise`
        4. Apply clipping based on :attr:`ReferenceObservationTermCfg.clip`
        5. Apply scaling based on :attr:`ReferenceObservationTermCfg.scale`

        We apply noise to the computed term first to maintain the integrity of how noise affects the data
        as it truly exists in the real world (from data collection). If the noise is applied after clipping or scaling, the noise
        could be artificially constrained or amplified, which might misrepresent how noise naturally occurs
        in the data.

        Args:
            group_name: The name of the group for which to compute the reference observations. Defaults to None,
                in which case reference observations for all the groups are computed and returned.
            cur_time: The current time of the environment.
            symmetry: Whether to apply symmetry to the computed observations. Defaults to False. If True, tmp_storage and history_buffer will not be updated.

        Returns:
            Depending on the group's configuration, the tensors for individual observation terms are
            concatenated along the last dimension into a single tensor. Otherwise, they are returned as
            a dictionary with keys corresponding to the term's name.
            If all reference_masks (second element of the tuple) are False, the observation is set to None.

        Raises:
            ValueError: If input ``group_name`` is not a valid group handled by the manager.
        """

        # DEBUG for time
        if self.debug_last_done_time is not None:
            self.debug_not_compute_obs_time += time.time() - self.debug_last_done_time
            self.debug_count_steps += 1
            if self.debug_count_steps == self.debug_show_time_between_steps:
                # print(f"Time compute obs: {self.debug_compute_obs_time}, not computing obs: {self.debug_not_compute_obs_time}")
                self.debug_not_compute_obs_time = 0
                self.debug_compute_obs_time = 0
                self.debug_count_steps = 0
        self.debug_last_done_time = time.time()

        cur_time = cur_time - self.static_delay

        if group_name not in self._group_ref_obs_term_names:
            raise ValueError(f"Invalid group name '{group_name}'. Expected one of: {self._group_ref_obs_term_names.keys()}.")
        # iterate over all the terms in each group
        group_term_names = self._group_ref_obs_term_names[group_name]
        group_buffer_manager = self._group_ref_obs_term_buffer_manager[group_name]
        group_ref_obs = dict.fromkeys(group_term_names, None)
        obs_terms = zip(group_term_names, self._group_ref_obs_term_cfgs[group_name])
        obs_mask: torch.Tensor | None = None
        for term_name, term_cfg in obs_terms:
            term_delay = self._group_ref_obs_init_delay[group_name][self._group_ref_obs_term_names[group_name].index(term_name)]
            cp_cur_time = torch.clamp(cur_time - term_delay, min=0.0)
            if term_cfg.is_base_pose:
                obs = group_buffer_manager.calc_base_pose(
                    cp_cur_time,
                    term_cfg.params["lin_vel_name"],
                    term_cfg.params["ang_vel_name"],
                )
            else:
                obs = group_buffer_manager.calc_obs(term_name, cp_cur_time)
            if obs_mask is None and not term_cfg.make_empty:
                obs_mask = group_buffer_manager.calc_mask(cp_cur_time)
            # apply modifiers
            if term_cfg.modifiers is not None:
                for mod in term_cfg.modifiers:
                    obs = mod.func(obs, **modifiers.params)
            # apply noise
            if term_cfg.noise is not None:
                obs = term_cfg.noise.func(obs, term_cfg.noise)
            # apply clipping
            if term_cfg.clip is not None:
                obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
            # apply scaling
            if term_cfg.scale is not None:
                obs = obs.mul_(term_cfg.scale)
                
            if term_cfg.env_func is not None:
                func, func_params = term_cfg.env_func, term_cfg.env_func_params
                obs = func(self._env, obs, **func_params)

            if not symmetry:
                self._group_ref_obs_term_tmp_storage[group_name][term_name] = (obs, obs_mask)

            if term_cfg.in_obs_tensor:
                if term_cfg.history_length > 0 and not symmetry:
                    self._group_ref_obs_term_history_buffer[group_name][term_name].append(obs.unsqueeze(1))
                    if term_cfg.flatten_history_dim:
                        obs = self._group_ref_obs_term_history_buffer[group_name][term_name].buffer.reshape(
                                self._env.num_envs, -1
                        )
                        group_ref_obs[term_name] = (obs, obs_mask)

                    else:
                        obs = self._group_ref_obs_term_history_buffer[group_name][term_name].buffer
                        group_ref_obs[term_name] = (obs, obs_mask)
                elif term_cfg.history_length > 0 and symmetry:
                    obs = self._group_ref_obs_term_history_buffer[group_name][term_name].buffer # no need to update
                    obs = term_cfg.symmetry(self._env, obs, **term_cfg.symmetry_params)
                    if term_cfg.flatten_history_dim:
                        obs = obs.reshape(self._env.num_envs, -1)
                    group_ref_obs[term_name] = (obs, obs_mask)
                else:
                    if symmetry:
                        obs = term_cfg.symmetry(self._env, obs, **term_cfg.symmetry_params)
                    group_ref_obs[term_name] = (obs, obs_mask)
            

            # Update history buffer if observation term has history enabled (since this modification, this program no longer supports isaaclab v1)
        
        self.debug_compute_obs_time += time.time() - self.debug_last_done_time
        self.debug_last_done_time = time.time()

        if self._group_ref_obs_concatenate[group_name]:
            term_obs = [group_ref_obs[term_name] for term_name in group_term_names]
            if len(self._group_ref_obs_term_concatenate_last_dim[group_name]) == 0: # update last dim concatenate index
                idx = 0
                for term in term_obs:
                    self._group_ref_obs_term_concatenate_last_dim[group_name].append(idx)
                    if term is not None:
                        idx += term[0].shape[-1]
            term_obs = filter(lambda x: x is not None, term_obs)
            
            obs = torch.cat([tup[0] for tup in term_obs], dim=-1)

            if obs_mask is None or not obs_mask.any():
                return None # all masks are False
            return obs, obs_mask
        else:
            return group_ref_obs
    
    def _load_pickles(self, cfg: object):
        self.data_dir = cfg.data_dir
        files = []
        for data_dir in self.data_dir:
            if os.path.isdir(data_dir):
                files += glob(data_dir + "/**/*.pkl", recursive=True)
            else:
                files += glob(data_dir)
        assert len(files) > 0, "No pickle files found in the data directory"
        self.file_indices = torch.randint(0, len(files), (self.num_envs,), device=self.device)
        self.files = [files[i] for i in self.file_indices]
        self.tmp_storage = [torch.load(file) for file in files]

    def _resolve_reference_term_cfg(self, term_name: str, term_cfg: ReferenceObservationTermCfg) -> tuple:
        if '/' in term_name:
            group_name, term_name = term_name.split('/')
        else:
            raise ValueError(f"Invalid term name {term_name}. Expected format: group_name/term_name")
        self._group_ref_obs_term_names[group_name].append(term_name)
        self._group_ref_obs_term_cfgs[group_name].append(term_cfg)
        if self._group_ref_obs_term_buffer_manager[group_name] is None:
            self._group_ref_obs_term_buffer_manager[group_name] = BufferManager(self.num_envs, len(self.tmp_storage), self.cfg.working_mode, self.device)
            self._group_ref_obs_term_buffer_manager[group_name].set_all_env_ref_id(self.file_indices)
            for idx in range(len(self.tmp_storage)):
                pkl = self.tmp_storage[idx]
                seq_len = pkl["trans"].shape[0]
                cyclic_subseq = pkl.get("cyclic_subseq", None)
                if cyclic_subseq is not None:
                    if cyclic_subseq[1] - cyclic_subseq[0] <= 35:
                        cyclic_subseq = (0, seq_len - 1)
                else:
                    cyclic_subseq = (0, seq_len - 1)
                z = pkl["trans"][0, 2]
                self._group_ref_obs_term_buffer_manager[group_name].set_env_origin_z(idx, z)
                self._group_ref_obs_term_buffer_manager[group_name].add_reference('trans', idx, pkl["trans"].to(self.device), False, pkl["fps"], cyclic_subseq=cyclic_subseq)
                self._group_ref_obs_term_buffer_manager[group_name].add_reference('root_orient', idx, pkl["root_orient"].to(self.device), False, pkl["fps"], cyclic_subseq=pkl["cyclic_subseq"])
            self._group_ref_obs_term_buffer_manager[group_name].prepare_buffers('trans')
            self._group_ref_obs_term_buffer_manager[group_name].prepare_buffers('root_orient')

        self._group_ref_obs_init_delay[group_name].append(term_cfg.load_seq_delay)

        if term_cfg.is_base_pose:
            ref_obs_dims = (0,) if not term_cfg.in_obs_tensor else (7,)
        else:
            buffer_manager = self._group_ref_obs_term_buffer_manager[group_name]
            for ref_id, pkl in enumerate(self.tmp_storage):
                if isinstance(term_cfg.name, dict):
                    data = dict((key, pkl[value].to(self.device)) for key, value in term_cfg.name.items())
                else:
                    term_real_name = term_cfg.name
                    if term_real_name not in pkl.keys():
                        raise ValueError(f"Invalid term name {term_real_name} in term {term_name}. Expected term name in the pickle file.")
                    data = pkl[term_real_name]
                    data = data.to(self.device)
                if term_cfg.func is not None:
                    data = term_cfg.func(data, env=self._env, pickle_cfg = {"fps": pkl["fps"]}, **term_cfg.params)

                if term_cfg.make_empty:
                    data = torch.zeros_like(data)
                buffer_manager.add_reference(term_name, ref_id, data, term_cfg.is_constant, pkl["fps"], cyclic_subseq=pkl["cyclic_subseq"])
            buffer_manager.prepare_buffers(term_name)
            ref_obs_dims = (0,) if not term_cfg.in_obs_tensor or term_cfg.make_empty else buffer_manager.get_dim(term_name)
            if term_cfg.env_func is not None:
                num_envs = self._env.num_envs
                dummy_obs = torch.zeros((num_envs, *ref_obs_dims), device=self._env.device)
                dummy_obs = term_cfg.env_func(self._env, dummy_obs, **term_cfg.env_func_params)
                ref_obs_dims = dummy_obs.shape[1:]
        
        if term_cfg.history_length > 0:
            old_dims = list(ref_obs_dims)
            old_dims.insert(1, term_cfg.history_length)
            ref_obs_dims = tuple(old_dims)
            if term_cfg.flatten_history_dim:
                ref_obs_dims = (np.prod(ref_obs_dims),)
        self._group_ref_obs_term_dim[group_name].append(ref_obs_dims)
        return ref_obs_dims


    def _prepare_terms(self):
        """Prepares a list of reference observation terms."""
        self._group_ref_obs_term_names:  dict[str, list[str]] = dict()
        self._group_ref_obs_term_dim: dict[str, list[tuple[int, ...]]] = dict()
        self._group_ref_obs_term_concatenate_last_dim: dict[str, list[int]] = dict()
        self._group_ref_obs_term_buffer_manager: dict[str, BufferManager | None] = dict()
        self._group_ref_obs_term_tmp_storage: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = dict()
        self._group_ref_obs_term_cfgs: dict[str, list[ReferenceObservationTermCfg]] = dict()
        self._group_ref_obs_class_term_cfgs: dict[str, list[ReferenceObservationTermCfg]] = dict()
        self._group_ref_obs_concatenate: dict[str, bool] = dict()
        self._group_ref_obs_term_history_buffer: dict[str, dict] = dict()
        self._group_ref_obs_init_delay: dict[str, list[float]] = dict()

        self._group_ref_obs_class_modifiers: dict[str, list[modifiers.ModifierBase]] = dict()
        
        if isinstance(self.cfg, dict):
            group_cfg_items = self.cfg.items()
        else:
            group_cfg_items = self.cfg.__dict__.items()

        for group_name, group_cfg in group_cfg_items:
            if group_cfg is None:
                continue
            if group_name in ["data_dir", "working_mode", "static_delay"]:
                continue
            assert isinstance(group_cfg, ReferenceObservationGroupCfg), f"Invalid configuration for group {group_name}. Expected ReferenceObservationGroupCfg, but got {type(group_cfg)}"

            self._group_ref_obs_term_names[group_name] = list()
            self._group_ref_obs_term_dim[group_name] = list()
            self._group_ref_obs_term_cfgs[group_name] = list()
            self._group_ref_obs_term_buffer_manager[group_name] = None
            self._group_ref_obs_term_tmp_storage[group_name] = dict()
            self._group_ref_obs_class_term_cfgs[group_name] = list()
            
            self._group_ref_obs_concatenate[group_name] = group_cfg.concatenate_terms
            self._group_ref_obs_init_delay[group_name] = list()
            self._group_ref_obs_term_concatenate_last_dim[group_name] = list()

            group_ref_entry_history_buffer: dict[str, CircularBuffer] = dict()

            if isinstance(group_cfg, dict):
                term_cfg_items = group_cfg.items()
            else:
                term_cfg_items = group_cfg.__dict__.items()
            
            for term_name, term_cfg in term_cfg_items:
                if term_name in ["concatenate_terms", "enable_corruption", "history_length", "flatten_history_dim", "load_seq_delay"]:
                    continue
                if term_cfg is None:
                    continue
                assert isinstance(term_cfg, ReferenceObservationTermCfg), f"Invalid configuration for term {term_name}. Expected ReferenceObservationTermCfg, but got {type(term_cfg)}"

                # check group history params and override terms
                if group_cfg.history_length is not None:
                    term_cfg.history_length = group_cfg.history_length
                    term_cfg.flatten_history_dim = group_cfg.flatten_history_dim
                
                if group_cfg.load_seq_delay != 0:
                    term_cfg.load_seq_delay = group_cfg.load_seq_delay
                
                ref_obs_dims = self._resolve_reference_term_cfg(f"{group_name}/{term_name}", term_cfg)
                # ref_obs_dims = self._group_ref_obs_term_dim[group_name][-1]
                # check noise settings
                if not group_cfg.enable_corruption:
                    term_cfg.noise = None

                if term_cfg.history_length > 0:
                    group_ref_entry_history_buffer[term_name] = CircularBuffer(
                        max_len=term_cfg.history_length,
                        batch_size=self._env.num_envs,
                        device=self._env.device
                    )

                if term_cfg.scale is not None:
                    if not isinstance(term_cfg.scale, (float, int, tuple)):
                        raise TypeError(
                            f"Scale for reference observation term '{term_name}' in group '{group_name}'"
                            f" is not of type float, int or tuple. Received: '{type(term_cfg.scale)}'."
                        )
                    if isinstance(term_cfg.scale, tuple) and len(term_cfg.scale) != ref_obs_dims[0]:
                        raise ValueError(
                            f"Scale for reference observation term '{term_name}' in group '{group_name}'"
                            f" does not match the dimensions of the observation. Expected: {ref_obs_dims[0]}"
                            f" but received: {len(term_cfg.scale)}."
                        )
                    
                    term_cfg.scale = torch.tensor(term_cfg.scale, dtype=torch.float, device=self._env.device)

                if term_cfg.modifiers is not None:
                    # initialize list of modifiers for term
                    for mod_cfg in term_cfg.modifiers:
                        # check if class modifier and initialize with observation size when adding
                        if isinstance(mod_cfg, modifiers.ModifierCfg):
                            # to list of modifiers
                            if inspect.isclass(mod_cfg.func):
                                if not issubclass(mod_cfg.func, modifiers.ModifierBase):
                                    raise TypeError(
                                        f"Modifier function '{mod_cfg.func}' for observation term '{term_name}'"
                                        f" is not a subclass of 'ModifierBase'. Received: '{type(mod_cfg.func)}'."
                                    )
                                mod_cfg.func = mod_cfg.func(cfg=mod_cfg, data_dim=ref_obs_dims, device=self._env.device)

                                # add to list of class modifiers
                                self._group_ref_obs_class_modifiers.append(mod_cfg.func)
                        else:
                            raise TypeError(
                                f"Modifier configuration '{mod_cfg}' of observation term '{term_name}' is not of"
                                f" required type ModifierCfg, Received: '{type(mod_cfg)}'"
                            )

                        # check if function is callable
                        if not callable(mod_cfg.func):
                            raise AttributeError(
                                f"Modifier '{mod_cfg}' of observation term '{term_name}' is not callable."
                                f" Received: {mod_cfg.func}"
                            )

                        # check if term's arguments are matched by params
                        term_params = list(mod_cfg.params.keys())
                        args = inspect.signature(mod_cfg.func).parameters
                        args_with_defaults = [arg for arg in args if args[arg].default is not inspect.Parameter.empty]
                        args_without_defaults = [arg for arg in args if args[arg].default is inspect.Parameter.empty]
                        args = args_without_defaults + args_with_defaults
                        # ignore first two arguments for env and env_ids
                        # Think: Check for cases when kwargs are set inside the function?
                        if len(args) > 1:
                            if set(args[1:]) != set(term_params + args_with_defaults):
                                raise ValueError(
                                    f"Modifier '{mod_cfg}' of observation term '{term_name}' expects"
                                    f" mandatory parameters: {args_without_defaults[1:]}"
                                    f" and optional parameters: {args_with_defaults}, but received: {term_params}."
                                )
                if isinstance(term_cfg.func, ManagerTermBase):
                    self._group_ref_obs_class_term_cfgs[group_name].append(term_cfg)
                    term_cfg.func.reset()

            self._group_ref_obs_term_history_buffer[group_name] = group_ref_entry_history_buffer




        # remove tmp storage
        del self.tmp_storage
        
        # set reference term tmp storage
        self._set_term_tmp_storage()


    def __str__(self) -> str:
        msg = f"<RefObservationManager> contains {len(self.active_terms)} active terms.\n"

        # create table for term information
        for group_name, group_dim in self._group_ref_obs_dim.items():
            table = PrettyTable()
            table.title = f"Active Reference Observation Terms in Group '{group_name}'"
            table.field_names = ["Index", "Name", "Dimension"]
            # set alignment of table columns
            table.align["Name"] = "l"
            ref_obs_terms = zip(
                self._group_ref_obs_term_names[group_name],
                self._group_ref_obs_term_dim[group_name]
            )

            for index, (term_name, term_dim) in enumerate(ref_obs_terms):
                tab_dim = tuple(term_dim)
                table.add_row([index, term_name, tab_dim])

            # convert table to string
            msg += table.get_string()
            msg += "\n"
        table = PrettyTable()
        table.title = f"Reference Observation Files"
        table.field_names = ["Index", "File"]
        # set alignment of table columns
        table.align["Index"] = "l"
        table.align["File"] = "r"
        files = [file.split("/")[-1] for file in self.files]
        # remove redundant files
        files = list(set(files))
        for index, file in enumerate(files):
            table.add_row([index, file])
        # convert table to string
        msg += table.get_string()
        msg += "\n"


        return msg