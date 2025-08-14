import torch
import warp as wp
from collections.abc import Sequence
from isaaclab.utils.math import yaw_quat, quat_apply, quat_inv, quat_mul, combine_frame_transforms, quat_from_euler_xyz
from GBC.utils.base.math_utils import angle_axis_to_quaternion
import numpy as np

class BufferType:
    singular = 0
    recurrent = 1
    recurrent_strict = 2


class BufferManager:
    def __init__(self, num_envs: int, num_ref: int, working_mode: str, device: str):
        self.num_envs = num_envs
        self.num_ref = num_ref
        self.device = device

        buffer_type_id = getattr(BufferType, working_mode)
        self.buffer_type = torch.ones(num_ref, dtype=torch.int8, device=self.device) * buffer_type_id
        self.frame_rate = torch.zeros(num_ref, dtype=torch.float32, device=self.device)
        self.max_len = torch.zeros(num_ref, dtype=torch.int32, device=self.device)
        self.recurrent_subseq = torch.ones((num_ref, 2), dtype=torch.int32, device=self.device) * -1
        self.env_ref_id = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self.start_index = None
        self.ref_buffer_list = dict()
        self.ref_buffer = dict()
        self.is_constant = dict()

        self.last_time = None
        self.last_idx = None
        self.last_pose_tme = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.base_pos = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        self.base_quat = torch.zeros((num_envs, 4), dtype=torch.float32, device=self.device)

    def add_reference(self, key: str, ref_id: int, buffer_raw: torch.Tensor, is_constant: bool, frame_rate: int, cyclic_subseq: tuple = None):
        buffer_type_id = BufferType.recurrent
        self.buffer_type[ref_id] = buffer_type_id
        self.frame_rate[ref_id] = frame_rate

        self.is_constant[key] = is_constant
        if not is_constant:
            if cyclic_subseq is not None:
                st, ed = cyclic_subseq
                buffer_raw = buffer_raw[:ed, ...]
                self.recurrent_subseq[ref_id, 0] = st
                self.recurrent_subseq[ref_id, 1] = ed
                assert st < ed, "Cyclic subsequence is not valid, expected (start < end), got {}".format(cyclic_subseq)
            self.max_len[ref_id] = buffer_raw.shape[0]

        if key not in self.ref_buffer_list:
            self.ref_buffer_list[key] = [None for _ in range(self.num_ref)]
        self.ref_buffer_list[key][ref_id] = buffer_raw
        
    def prepare_buffers(self, key):
        # Push a (1, dim) tensor to the front of the reference buffer
        if self.is_constant[key]:
            self.ref_buffer[key] = torch.stack(self.ref_buffer_list[key])
        else:
            self.ref_buffer_list[key] = [torch.zeros_like(self.ref_buffer_list[key][0])[:1,...]] + self.ref_buffer_list[key]
            self.ref_buffer[key] = torch.concatenate(self.ref_buffer_list[key], dim=0).contiguous()
            if self.start_index is None:
                len_sum = torch.cumsum(self.max_len, dim=0)
                self.start_index = torch.zeros(self.num_ref, dtype=torch.int32, device=self.device)
                self.start_index[1:] = len_sum[:-1]
                self.start_index += 1

    def get_dim(self, key):
        return tuple(self.ref_buffer[key].shape[1:])

    def set_single_env_ref_id(self, env_id: int, ref_id: int):
        self.env_ref_id[env_id] = ref_id

    def set_all_env_ref_id(self, env_ref_id: torch.Tensor):
        self.env_ref_id = env_ref_id.to(torch.int32)

    def reset(self, env, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.env_ref_id[env_ids] = torch.randint(
            0, self.num_ref,
            size=self.env_ref_id[env_ids].shape,
            dtype=self.env_ref_id.dtype,
            device=self.env_ref_id.device,
        )
        self.last_pose_tme[env_ids] = -1
        root_pose_w = env.scene["robot"].data.root_state_w[env_ids, :7]
        root_pose_w = root_pose_w.clone()
        assert root_pose_w.dtype == self.base_pos.dtype == self.base_quat.dtype
        self.base_pos[env_ids] = root_pose_w[:, :3] - env.scene.env_origins[env_ids]
        self.base_quat[env_ids] = root_pose_w[:, 3:]

    @wp.kernel
    def compute_idx(
        current_idx: wp.array(dtype=int),
        current_time: wp.array(dtype=float),
        env_ref_id: wp.array(dtype=int),
        frame_rate: wp.array(dtype=float),
        start_index: wp.array(dtype=int),
        max_len: wp.array(dtype=int),
        recurrent_subseq: wp.array(dtype=wp.vec2i),
    ) -> int:
        tid = wp.tid()
        rid = env_ref_id[tid]
        idx = wp.int32(current_time[tid] * frame_rate[rid])
        if current_time[tid] < 0:
            idx = 0
        elif recurrent_subseq[rid][0] == -1 or max_len[rid] == -1:
            # Singular buffer
            if idx >= max_len[rid]:
                idx = -1
            else:
                idx += start_index[rid]
        else:
            rec_st = recurrent_subseq[rid][0]
            rec_ed = recurrent_subseq[rid][1]
            rec_len = rec_ed - rec_st
            if idx >= rec_st:
                idx = rec_st + (idx - rec_st) % rec_len
            idx += start_index[rid]
        current_idx[tid] = idx

    @wp.kernel
    def compute_num_cyclic_subseq(
        num_cyclic_subseq: wp.array(dtype=int),
        current_idx: wp.array(dtype=int),
        begin_idx: wp.array(dtype=int),
        end_idx: wp.array(dtype=int),
        current_time: wp.array(dtype=float),
        env_ref_id: wp.array(dtype=int),
        frame_rate: wp.array(dtype=float),
        start_index: wp.array(dtype=int),
        max_len: wp.array(dtype=int),
        recurrent_subseq: wp.array(dtype=wp.vec2i),
    ) -> int:
        tid = wp.tid()
        rid = env_ref_id[tid]
        idx = wp.int32(current_time[tid] * frame_rate[rid])
        if current_time[tid] < 0:
            num_cyclic_subseq[tid] = 0
            begin_idx[tid] = 0
            end_idx[tid] = 0
            idx = 0
        elif recurrent_subseq[rid][0] == -1 or max_len[rid] == -1:
            # Singular buffer
            if idx >= max_len[rid]:
                idx = -1
            else:
                idx += start_index[rid]
            num_cyclic_subseq[tid] = 0
            begin_idx[tid] = 0
            end_idx[tid] = 0
        else:
            rec_st = recurrent_subseq[rid][0]
            rec_ed = recurrent_subseq[rid][1]
            rec_len = rec_ed - rec_st
            if idx >= rec_st:
                num_subseq = (idx - rec_st) // rec_len
                idx = rec_st + (idx - rec_st) % rec_len
                begin_idx[tid] = rec_st
                end_idx[tid] = rec_ed
            else:
                num_subseq = 0
                begin_idx[tid] = 0
                end_idx[tid] = 0

            # idx += start_index[rid]
            # begin_idx[tid] += start_index[rid]
            # end_idx[tid] += start_index[rid]
        current_idx[tid] = idx
        num_cyclic_subseq[tid] = num_subseq

    def get_warp(self, name):
        if name == "recurrent_subseq":
            return wp.from_torch(getattr(self, name), wp.vec2i)
        else:
            return wp.from_torch(getattr(self, name))
    
    def calc_idx(self, current_time: torch.Tensor):
        if self.last_time is not None and torch.allclose(current_time, self.last_time):
            return self.last_idx

        self.last_time = current_time.clone()
        current_idx = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        current_time = wp.from_torch(current_time)
        list_attrs = ["env_ref_id", "frame_rate", "start_index", "max_len", "recurrent_subseq"]
        wp.launch(
            self.compute_idx,
            self.num_envs,
            inputs=[current_idx, current_time, *map(self.get_warp, list_attrs)],
            device=self.device
        )
        self.last_idx = wp.to_torch(current_idx)
        return self.last_idx
    
    def calc_num_cyclic_subseq(self, current_time: torch.Tensor):
        current_num_cyclic_subseq = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        current_idx = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        current_begin_idx = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        current_end_idx = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        current_time = wp.from_torch(current_time)
        list_attrs = ["env_ref_id", "frame_rate", "start_index", "max_len", "recurrent_subseq"]
        wp.launch(
            self.compute_num_cyclic_subseq,
            self.num_envs,
            inputs=[current_num_cyclic_subseq, current_idx, current_begin_idx, current_end_idx, current_time, *map(self.get_warp, list_attrs)],
            device=self.device
        )
        return wp.to_torch(current_num_cyclic_subseq), wp.to_torch(current_idx), wp.to_torch(current_begin_idx), wp.to_torch(current_end_idx)

    def calc_obs(self, key, current_time: torch.Tensor):
        if self.is_constant[key]:
            return self.ref_buffer[key][self.env_ref_id, ...]
        current_idx = self.calc_idx(current_time)
        return self.ref_buffer[key][current_idx, ...]
    
    def calc_cumulative_obs(self, key, current_time: torch.Tensor):
        if self.is_constant[key]:
            return self.ref_buffer[key][self.env_ref_id, ...]
        current_num_cyclic_subseq, current_idx, current_begin_idx, current_end_idx = self.calc_num_cyclic_subseq(current_time)

        cumulative_obs = torch.zeros((self.num_envs, *self.ref_buffer[key].shape[1:]), dtype=self.ref_buffer[key].dtype, device=self.device)
        start_idx = self.start_index[self.env_ref_id]
        for i in range(self.num_envs):
            cumulative_obs[i] += torch.sum(
                self.ref_buffer[key][start_idx[i]:current_begin_idx[i]+start_idx[i], ...],
                dim=0
            )
            cumulative_obs[i] += torch.sum(
                self.ref_buffer[key][current_begin_idx[i]+start_idx[i]:current_end_idx[i]+start_idx[i], ...],
                dim=0
            ) * current_num_cyclic_subseq[i].to(torch.float32)
            cumulative_obs[i] += torch.sum(
                self.ref_buffer[key][current_begin_idx[i]+start_idx[i]:current_idx[i]+start_idx[i], ...],
                dim=0
            )
        
        return cumulative_obs

    def calc_cumulative_obs_v2(self, key, current_time: torch.Tensor):
        if self.is_constant[key]:
            return self.ref_buffer[key][self.env_ref_id, ...]
        
        current_num_cyclic_subseq, current_idx, current_begin_idx, current_end_idx = self.calc_num_cyclic_subseq(current_time)
        start_idx = self.start_index[self.env_ref_id]  # shape: (num_envs,)
        
        # Initialize output tensor
        cumulative_obs = torch.zeros(
            (self.num_envs, *self.ref_buffer[key].shape[1:]),
            dtype=self.ref_buffer[key].dtype,
            device=self.device
        )
        
        # Calculate indices for each environment
        begin_indices = current_begin_idx + start_idx  # shape: (num_envs,)
        end_indices = current_end_idx + start_idx      # shape: (num_envs,)
        current_indices = current_idx + start_idx      # shape: (num_envs,)
        
        # Get the maximum lengths needed for each part
        max_first_len = (begin_indices - start_idx).max()
        max_cyclic_len = (end_indices - begin_indices).max()
        max_last_len = (current_indices - begin_indices).max()
        
        # Create index tensors for all environments
        # Shape: (num_envs, max_len)
        first_indices = torch.arange(max_first_len, device=self.device).unsqueeze(0) + start_idx.unsqueeze(1)
        cyclic_indices = torch.arange(max_cyclic_len, device=self.device).unsqueeze(0) + begin_indices.unsqueeze(1)
        last_indices = torch.arange(max_last_len, device=self.device).unsqueeze(0) + begin_indices.unsqueeze(1)
        
        # Create masks for valid indices
        first_mask = first_indices < begin_indices.unsqueeze(1) # shape: (num_envs, max_first_len)
        cyclic_mask = cyclic_indices < end_indices.unsqueeze(1) # shape: (num_envs, max_cyclic_len)
        last_mask = last_indices < current_indices.unsqueeze(1) # shape: (num_envs, max_last_len)

        buffer_max_len = self.ref_buffer[key].shape[0]
        first_indices = torch.where(first_indices < buffer_max_len, first_indices, buffer_max_len - 1)
        cyclic_indices = torch.where(cyclic_indices < buffer_max_len, cyclic_indices, buffer_max_len - 1)
        last_indices = torch.where(last_indices < buffer_max_len, last_indices, buffer_max_len - 1)
        
        # Get the sequences using advanced indexing
        first_sequences = self.ref_buffer[key][first_indices] * first_mask.unsqueeze(-1)
        cyclic_sequences = self.ref_buffer[key][cyclic_indices] * cyclic_mask.unsqueeze(-1)
        last_sequences = self.ref_buffer[key][last_indices] * last_mask.unsqueeze(-1)
        
        # Calculate sums
        cumulative_obs += torch.sum(first_sequences, dim=1)
        if torch.any(current_end_idx > current_begin_idx):
            cumulative_obs += torch.sum(cyclic_sequences, dim=1) * current_num_cyclic_subseq.to(torch.float32).unsqueeze(1)
        cumulative_obs += torch.sum(last_sequences, dim=1)
        
        return cumulative_obs

    def calc_mask(self, current_time: torch.Tensor):
        current_idx = self.calc_idx(current_time)
        return current_idx >= 0

    def step_robot_base_pose(self, current_time: torch.Tensor, lin_vel_yaw_frame: torch.Tensor, ang_vel: torch.Tensor):
        dt = torch.where(self.last_pose_tme < 0, torch.zeros_like(self.last_pose_tme), current_time - self.last_pose_tme)
        assert torch.all(dt > -1e-5)
        dt = dt.clamp(min=0)

        lin_vel_yaw_frame = lin_vel_yaw_frame.to(self.base_pos.dtype)
        ang_vel = ang_vel.to(self.base_pos.dtype)
        quat_yaw = yaw_quat(self.base_quat)
        lin_vel = quat_apply(quat_yaw, lin_vel_yaw_frame)
        self.base_pos += lin_vel * dt.unsqueeze(1)
        rot_vec = quat_apply(quat_inv(self.base_quat), ang_vel) * dt.unsqueeze(1)
        self.base_quat = quat_mul(self.base_quat, quat_inv(angle_axis_to_quaternion(rot_vec)))
        self.last_pose_tme = torch.where(
            torch.logical_and(self.last_pose_tme < 0, current_time < 1e-5),
            self.last_pose_tme,
            current_time.clone()
        )

    def calc_base_pose(self, current_time: torch.Tensor, lin_vel_name: str, ang_vel_name: str):
        lin_vel_yaw_frame = self.calc_obs(lin_vel_name, current_time)
        ang_vel = self.calc_obs(ang_vel_name, current_time)
        self.step_robot_base_pose(current_time, lin_vel_yaw_frame, ang_vel)
        return torch.cat([self.base_pos, self.base_quat], dim=1)
    
    def calc_base_pose_cumulative(self, current_time: torch.Tensor, lin_vel_name: str, ang_vel_name: str):
        dt = 1.0 / self.frame_rate[self.env_ref_id]
        # import time
        # start_time = time.time()
        # for i in range(10):
        #     lin_pos = self.calc_cumulative_obs(lin_vel_name, current_time) * dt
        #     ang_pos = self.calc_cumulative_obs(ang_vel_name, current_time) * dt
        # end_time = time.time()
        # print("calc_cumulative_obs time: ", end_time - start_time)
        # start_time = time.time()
        # for i in range(10):
        #     lin_pos_val = self.calc_cumulative_obs_v2(lin_vel_name, current_time) * dt
        #     ang_pos_val = self.calc_cumulative_obs_v2(ang_vel_name, current_time) * dt
        # end_time = time.time()
        # print("calc_cumulative_obs_v2 time: ", end_time - start_time)
        # lin_pos_match = torch.allclose(lin_pos, lin_pos_val)
        # ang_pos_match = torch.allclose(ang_pos, ang_pos_val)
        # print("lin_pos_match: ", lin_pos_match)
        # print("ang_pos_match: ", ang_pos_match)
        lin_pos = self.calc_cumulative_obs_v2(lin_vel_name, current_time) * dt.unsqueeze(1)
        ang_pos = self.calc_cumulative_obs_v2(ang_vel_name, current_time) * dt.unsqueeze(1)

        # normalize ang_pos to 0~2pi
        ang_pos = ang_pos % (2 * np.pi)
        # base_quat = angle_axis_to_quaternion(ang_pos)
        base_quat = quat_from_euler_xyz(*ang_pos.T)
        return torch.cat([lin_pos, base_quat], dim=1)