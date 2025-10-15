import torch
import warp as wp
from collections.abc import Sequence
from isaaclab.utils.math import yaw_quat, quat_apply, quat_inv, quat_mul, combine_frame_transforms, quat_from_euler_xyz
from GBC.utils.base.math_utils import angle_axis_to_quaternion, quat_rotate_inverse
import numpy as np

@wp.func
def quat_mul_wp(q1: wp.quat, q2: wp.quat) -> wp.quat:
    # Warp quaternion format: (x, y, z, w)
    x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
    x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
    
    w_out = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x_out = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_out = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_out = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return wp.quat(x_out, y_out, z_out, w_out)

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
        self.env_origin_z = torch.zeros((num_ref,), dtype=torch.float32, device=self.device)
        self.base_quat = torch.zeros((num_envs, 4), dtype=torch.float32, device=self.device)
        
    def set_env_origin_z(self, ref_id: int, z: float):
        self.env_origin_z[ref_id] = z

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
                idx = 0
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

            idx += start_index[rid]
            begin_idx[tid] += start_index[rid]
            end_idx[tid] += start_index[rid]
            buffer_size = start_index[rid] + max_len[rid]
            if end_idx[tid] >= buffer_size:
                end_idx[tid] = buffer_size - 1
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
    
    def calc_idx_direct(self, current_time: torch.Tensor):
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
        return wp.to_torch(current_idx)
    
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
        if 'target_quaternion' in key:
            return self.calculate_quat_obs(key, current_time)
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

    @wp.kernel
    def compute_cumulative_obs_optimized(
        cumulative_obs: wp.array(dtype=float),
        ref_buffer: wp.array(dtype=float),
        current_num_cyclic_subseq: wp.array(dtype=int),
        current_idx: wp.array(dtype=int),
        current_begin_idx: wp.array(dtype=int),
        current_end_idx: wp.array(dtype=int),
        env_ref_id: wp.array(dtype=int),
        start_index: wp.array(dtype=int),
        obs_dim: int,
        buffer_max_len: int,
    ):
        """
        Optimized CUDA kernel for cumulative observation calculation.
        Each thread handles one environment-feature pair.
        """
        tid = wp.tid()
        env_id = tid // obs_dim
        feat_id = tid % obs_dim
        
        if env_id >= env_ref_id.shape[0]:
            return
            
        rid = env_ref_id[env_id]
        start_idx = start_index[rid]
        
        # Calculate actual buffer indices
        begin_idx = current_begin_idx[env_id] + start_idx
        end_idx = current_end_idx[env_id] + start_idx  
        curr_idx = current_idx[env_id] + start_idx
        num_cycles = current_num_cyclic_subseq[env_id]
        
        # Use float() to declare dynamic variable as required by warp
        cumsum = float(0.0)
        
        # Part 1: Sum from start_idx to begin_idx
        i = start_idx
        while i < begin_idx and i < buffer_max_len:
            cumsum += ref_buffer[i * obs_dim + feat_id]
            i += 1
        
        # Part 2: Cyclic part - sum from begin_idx to end_idx, multiply by num_cycles
        if end_idx > begin_idx and num_cycles > 0:
            cycle_sum = float(0.0)  # Dynamic variable
            i = begin_idx
            while i < end_idx and i < buffer_max_len:
                cycle_sum += ref_buffer[i * obs_dim + feat_id]
                i += 1
            cumsum += cycle_sum * float(num_cycles)
        
        # Part 3: Sum from begin_idx to curr_idx (current partial cycle)
        i = begin_idx
        while i < curr_idx and i < buffer_max_len:
            cumsum += ref_buffer[i * obs_dim + feat_id]
            i += 1
        
        # Store result
        cumulative_obs[env_id * obs_dim + feat_id] = cumsum

    def calc_cumulative_obs_cuda(self, key, current_time: torch.Tensor):
        """
        CUDA-accelerated version of calc_cumulative_obs_v2.
        
        This version uses a warp CUDA kernel to parallelize the cumulative sum computation
        across all environments and observation dimensions simultaneously.
        
        Performance benefits:
        - Eliminates large tensor creation and indexing operations
        - Parallelizes computation across GPU threads  
        - Reduces memory bandwidth requirements
        - Especially beneficial for large num_envs and observation dimensions
        """
        if self.is_constant[key]:
            return self.ref_buffer[key][self.env_ref_id, ...]
        
        # Get cyclic subsequence information
        current_num_cyclic_subseq, current_idx, current_begin_idx, current_end_idx = self.calc_num_cyclic_subseq(current_time)
        
        # Get buffer dimensions
        buffer_shape = self.ref_buffer[key].shape
        obs_dim = int(torch.prod(torch.tensor(buffer_shape[1:])))  # Flatten observation dimensions
        
        # Prepare flattened buffer and output
        ref_buffer_flat = self.ref_buffer[key].view(-1).contiguous()
        cumulative_obs_flat = wp.zeros(self.num_envs * obs_dim, dtype=wp.float32, device=self.device)
        
        # Convert to warp arrays
        ref_buffer_wp = wp.from_torch(ref_buffer_flat)
        current_num_cyclic_subseq_wp = wp.from_torch(current_num_cyclic_subseq.to(torch.int32))
        current_idx_wp = wp.from_torch(current_idx.to(torch.int32))
        current_begin_idx_wp = wp.from_torch(current_begin_idx.to(torch.int32)) 
        current_end_idx_wp = wp.from_torch(current_end_idx.to(torch.int32))
        env_ref_id_wp = wp.from_torch(self.env_ref_id)
        start_index_wp = wp.from_torch(self.start_index)
        
        # Launch kernel with one thread per (env, feature) pair
        total_threads = self.num_envs * obs_dim
        wp.launch(
            self.compute_cumulative_obs_optimized,
            total_threads,
            inputs=[
                cumulative_obs_flat,
                ref_buffer_wp,
                current_num_cyclic_subseq_wp,
                current_idx_wp,
                current_begin_idx_wp,
                current_end_idx_wp,
                env_ref_id_wp,
                start_index_wp,
                obs_dim,
                buffer_shape[0]
            ],
            device=self.device
        )
        
        # Convert back to torch and reshape to original dimensions
        cumulative_obs = wp.to_torch(cumulative_obs_flat).to(self.ref_buffer[key].dtype)
        cumulative_obs = cumulative_obs.view(self.num_envs, *buffer_shape[1:])
        
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
        begin_indices = current_begin_idx  # shape: (num_envs,)
        end_indices = current_end_idx  # shape: (num_envs,)
        current_indices = current_idx  # shape: (num_envs,)

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
    
    
    def calc_cumulative_obs_v2_lin_pos(self, lin_vel, ang_vel, current_time: torch.Tensor):
        
        current_num_cyclic_subseq, current_idx, current_begin_idx, current_end_idx = self.calc_num_cyclic_subseq(current_time)
        start_idx = self.start_index[self.env_ref_id]  # shape: (num_envs,)
        dt = 1.0 / self.frame_rate[self.env_ref_id]
        dt = dt.unsqueeze(1).unsqueeze(1) # shape (num_envs, 1, 1)
        
        # Initialize output tensor
        cumulative_obs = torch.zeros(
            (self.num_envs, *self.ref_buffer[lin_vel].shape[1:]),
            dtype=self.ref_buffer[lin_vel].dtype,
            device=self.device
        )
        
        ang_pos_cumulative_obs = torch.zeros(
            (self.num_envs, *self.ref_buffer[ang_vel].shape[1:]),
            dtype=self.ref_buffer[ang_vel].dtype,
            device=self.device
        )

        # Calculate indices for each environment
        begin_indices = current_begin_idx  # shape: (num_envs,)
        end_indices = current_end_idx    # shape: (num_envs,)
        current_indices = current_idx   # shape: (num_envs,)

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

        buffer_max_len = self.ref_buffer[lin_vel].shape[0]
        first_indices = torch.where(first_indices < buffer_max_len, first_indices, buffer_max_len - 1)
        cyclic_indices = torch.where(cyclic_indices < buffer_max_len, cyclic_indices, buffer_max_len - 1)
        last_indices = torch.where(last_indices < buffer_max_len, last_indices, buffer_max_len - 1)
        
        # Get the sequences using advanced indexing
        first_sequences = self.ref_buffer[lin_vel][first_indices] * first_mask.unsqueeze(-1)
        first_angvel_sequences = torch.cumsum(self.ref_buffer[ang_vel][first_indices] * first_mask.unsqueeze(-1), dim=1) * dt # shape (num_envs, max_first_len, 3) 
        cyclic_sequences = self.ref_buffer[lin_vel][cyclic_indices] * cyclic_mask.unsqueeze(-1)
        cyclic_sequences[...,2] *= 0. # disable velocity in z axis
        cyclic_angvel_sequences = torch.cumsum(self.ref_buffer[ang_vel][cyclic_indices] * cyclic_mask.unsqueeze(-1), dim=1) * dt
        cyclic_angvel_sequences[..., :2] *= 0. # disable rotation in x & y axis
        last_sequences = self.ref_buffer[lin_vel][last_indices] * last_mask.unsqueeze(-1)
        last_angvel_sequences = torch.cumsum(self.ref_buffer[ang_vel][last_indices] * last_mask.unsqueeze(-1), dim=1) * dt
        last_first_angvel = first_angvel_sequences[:, -1, :] if first_angvel_sequences.shape[1] > 0 else torch.zeros(first_angvel_sequences.shape[0], 3, device=first_angvel_sequences.device)
        last_cyclic_angvel = cyclic_angvel_sequences[:, -1, :] if cyclic_angvel_sequences.shape[1] > 0 else torch.zeros(cyclic_angvel_sequences.shape[0], 3, device=cyclic_angvel_sequences.device)
        last_last_angvel = last_angvel_sequences[:, -1, :] if last_angvel_sequences.shape[1] > 0 else torch.zeros(last_angvel_sequences.shape[0], 3, device=last_angvel_sequences.device)

        ang_pos_cumulative_obs += last_first_angvel + current_num_cyclic_subseq.to(torch.float32).unsqueeze(1) * last_cyclic_angvel + last_last_angvel

        # Calculate sums
        cumulative_obs += torch.sum(
            # first_sequences,
            quat_rotate_inverse(
                angle_axis_to_quaternion(first_angvel_sequences),
                first_sequences
            ), 
            dim=1)
        if torch.any(current_end_idx > current_begin_idx):
            # cumulative_obs += torch.sum(cyclic_sequences, dim=1) * current_num_cyclic_subseq.to(torch.float32).unsqueeze(1)
            max_num_cyclic_subseq = current_num_cyclic_subseq.max()
            for i in range(max_num_cyclic_subseq):
                cumulative_obs += torch.sum(
                    quat_rotate_inverse(
                        angle_axis_to_quaternion(cyclic_angvel_sequences + i * last_cyclic_angvel.unsqueeze(1) + last_first_angvel.unsqueeze(1)),
                        cyclic_sequences
                    ),
                    dim=1
                ) * (current_num_cyclic_subseq <= i+1).to(torch.float32).unsqueeze(1)
        cumulative_obs += torch.sum(
            quat_rotate_inverse(
                angle_axis_to_quaternion(ang_pos_cumulative_obs - last_last_angvel).unsqueeze(1),
                last_sequences
            ), dim=1)

        return cumulative_obs, ang_pos_cumulative_obs

    def calc_mask(self, current_time: torch.Tensor):
        current_idx = self.calc_idx(current_time)
        return current_idx >= 0

    def step_robot_base_pose(self, current_time: torch.Tensor, lin_vel_yaw_frame: torch.Tensor, ang_vel: torch.Tensor):
        dt = torch.where(self.last_pose_tme < 0, torch.zeros_like(self.last_pose_tme), current_time - self.last_pose_tme)
        assert torch.all(dt > -1e-5)
        assert torch.all(dt < 0.2)
        
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
        try:
            self.base_pos, self.base_quat = self.calc_base_pose_from_trans_quat(current_time)
            return torch.cat([self.base_pos, self.base_quat], dim=1)
        except:
            try:
                self.step_robot_base_pose(current_time, lin_vel_yaw_frame, ang_vel)
                return torch.cat([self.base_pos, self.base_quat], dim=1)
            except:
                self.last_pose_tme = torch.where(
                    torch.logical_and(self.last_pose_tme < 0, current_time < 1e-5),
                    self.last_pose_tme,
                    current_time.clone()
                )
                return self.calc_base_pose_cumulative(current_time, lin_vel_name, ang_vel_name)
    
    
    def calc_base_pose_from_trans_orient(self, current_time: torch.Tensor):
        assert 'trans' in self.ref_buffer.keys() and 'root_orient' in self.ref_buffer.keys()
        idx = self.calc_idx_direct(current_time)
        lin_pos = self.ref_buffer['trans'][idx]
        ang_pos = self.ref_buffer['root_orient'][idx]
        return lin_pos, ang_pos
    
    
    
    @wp.kernel
    def compute_cumulative_pose(
        final_pos_out: wp.array(dtype=wp.vec3),
        final_quat_out: wp.array(dtype=wp.quat),

        trans_buffer: wp.array(dtype=wp.vec3),
        quat_buffer: wp.array(dtype=wp.quat), 
        
        current_indices: wp.array(dtype=wp.int32),
        begin_indices: wp.array(dtype=wp.int32),
        current_num_cyclic_subseq: wp.array(dtype=wp.int32),
        
        cyclic_quat_diff: wp.array(dtype=wp.quat),
        cyclic_lin_xyz_diff_local: wp.array(dtype=wp.vec3)
    ):
        tid = wp.tid()

        c_idx = current_indices[tid]
        b_idx = begin_indices[tid]
        num_cycles = current_num_cyclic_subseq[tid]
        if c_idx < b_idx:
            final_pos_out[tid] = trans_buffer[c_idx]
            final_quat_out[tid] = quat_buffer[c_idx]
        else:
            
            accumulated_pos = trans_buffer[b_idx]
            accumulated_quat = quat_buffer[b_idx]

            i = wp.int32(0)
            while i < num_cycles:
                accumulated_pos += wp.quat_rotate(accumulated_quat, cyclic_lin_xyz_diff_local[tid])
                accumulated_quat = quat_mul_wp(accumulated_quat, cyclic_quat_diff[tid])
                i += 1

            q_current_in_cycle = quat_buffer[c_idx]
            p_current_in_cycle = trans_buffer[c_idx]
            q_begin_in_cycle = quat_buffer[b_idx]
            p_begin_in_cycle = trans_buffer[b_idx]
            q_diff_in_cycle = quat_mul_wp(q_current_in_cycle, wp.quat_inverse(q_begin_in_cycle))
            p_diff_in_cycle = p_current_in_cycle - p_begin_in_cycle
            p_diff_in_cycle_local = wp.quat_rotate(wp.quat_inverse(q_begin_in_cycle), p_diff_in_cycle)
            accumulated_pos += wp.quat_rotate(accumulated_quat, p_diff_in_cycle_local)
            accumulated_quat = quat_mul_wp(accumulated_quat, q_diff_in_cycle)
            final_pos_out[tid] = accumulated_pos
            final_quat_out[tid] = accumulated_quat
    
    def calc_base_pose_from_trans_quat(self, current_time: torch.Tensor):
        # this function is only available when "target_quaternion" is stored in the buffer (MUST BE THIS EXACT NAME)
        # however, this is pure table lookup. Time complexity is O(1). Only a little bit slower, since it has some adds and multiplies
        assert 'trans' in self.ref_buffer.keys() and 'target_quaternion' in self.ref_buffer.keys()
        def to_xyzw(q_wxyz):
            return q_wxyz.roll(shifts=-1, dims=-1).contiguous()
        def to_wxyz(q_xyzw):
            return q_xyzw.roll(shifts=1, dims=-1).contiguous()
        current_num_cyclic_subseq, current_idx, current_begin_idx, current_end_idx = self.calc_num_cyclic_subseq(current_time)
        begin_indices = current_begin_idx  # shape: (num_envs,)
        end_indices = current_end_idx      # shape: (num_envs,)
        end_indices = torch.clamp(end_indices, max=self.ref_buffer["trans"].shape[0]-1)
        current_indices = current_idx      # shape: (num_envs,)
        
        q_begin = self.ref_buffer['target_quaternion'][begin_indices]
        q_end = self.ref_buffer['target_quaternion'][end_indices]
        cyclic_quat_diff = quat_mul(q_end, quat_inv(q_begin))
        
        #    p_delta = p_end - p_begin
        p_begin = self.ref_buffer['trans'][begin_indices]
        p_end = self.ref_buffer['trans'][end_indices]
        cyclic_lin_xyz_diff = p_end - p_begin
        cyclic_lin_xyz_diff_local = quat_apply(quat_inv(q_begin), cyclic_lin_xyz_diff)
        final_pos_out = wp.zeros(self.num_envs, dtype=wp.vec3, device=self.device)
        final_quat_out = wp.zeros(self.num_envs, dtype=wp.quat, device=self.device)
        
        trans_buffer_wp = wp.from_torch(self.ref_buffer['trans'], dtype=wp.vec3)
        quat_buffer_wxyz = self.ref_buffer['target_quaternion']
        quat_buffer_xyzw = quat_buffer_wxyz.roll(shifts=-1, dims=-1).contiguous()
        quat_buffer_wp = wp.from_torch(quat_buffer_xyzw, dtype=wp.quat)
        
        wp.launch(
            kernel=self.compute_cumulative_pose,
            dim=self.num_envs,
            inputs=[
                final_pos_out, final_quat_out,
                trans_buffer_wp, quat_buffer_wp,
                wp.from_torch((current_indices).to(torch.int32)),
                wp.from_torch((begin_indices).to(torch.int32)),
                wp.from_torch((current_num_cyclic_subseq).to(torch.int32)),
                wp.from_torch(to_xyzw(cyclic_quat_diff), dtype=wp.quat),
                wp.from_torch(cyclic_lin_xyz_diff_local, dtype=wp.vec3)
                
            ],
        device=self.device
        )
        final_pos = wp.to_torch(final_pos_out).to(self.device)
        final_quat_xyzw = wp.to_torch(final_quat_out).to(self.device)
        final_quat = to_wxyz(final_quat_xyzw)
            
        return final_pos, final_quat
        
    @wp.kernel
    def compute_cumulative_quat(
        final_quat_out: wp.array(dtype=wp.quat),
        quat_buffer: wp.array(dtype=wp.quat), 
        current_indices: wp.array(dtype=wp.int32),
        begin_indices: wp.array(dtype=wp.int32),
        current_num_cyclic_subseq: wp.array(dtype=wp.int32),
        cyclic_quat_diff: wp.array(dtype=wp.quat)
    ):
        """
        CUDA kernel to compute cumulative quaternion observations with cyclic subsequence support.
        
        For each environment:
        - If current_idx < begin_idx: directly return quaternion at current_idx
        - Otherwise: accumulate quaternion rotations through cycles and current partial cycle
        
        Args:
            final_quat_out: Output quaternions for each environment
            quat_buffer: Buffer containing quaternion data
            current_indices: Current time indices for each environment  
            begin_indices: Start indices of cyclic subsequence for each environment
            current_num_cyclic_subseq: Number of completed cycles for each environment
            cyclic_quat_diff: Quaternion difference over one complete cycle
        """
        tid = wp.tid()

        c_idx = current_indices[tid]
        b_idx = begin_indices[tid]
        num_cycles = current_num_cyclic_subseq[tid]
        
        if c_idx < b_idx:
            # Before cyclic region: direct lookup
            final_quat_out[tid] = quat_buffer[c_idx]
        else:
            # Within cyclic region: accumulate rotations
            
            # Start with quaternion at beginning of cycle
            accumulated_quat = quat_buffer[b_idx]

            # Apply complete cycle rotations
            i = wp.int32(0)
            while i < num_cycles:
                accumulated_quat = quat_mul_wp(accumulated_quat, cyclic_quat_diff[tid])
                i += 1

            # Apply partial cycle rotation (from begin to current position)
            q_current_in_cycle = quat_buffer[c_idx]
            q_begin_in_cycle = quat_buffer[b_idx]
            q_diff_in_cycle = quat_mul_wp(q_current_in_cycle, wp.quat_inverse(q_begin_in_cycle))
            
            accumulated_quat = quat_mul_wp(accumulated_quat, q_diff_in_cycle)
            final_quat_out[tid] = accumulated_quat
    
    def calculate_quat_obs(self, key, current_time: torch.Tensor):
        """
        Calculate quaternion observations with cyclic subsequence support.
        
        This function computes quaternion values at given time points, handling:
        - Direct lookup for times before cyclic region
        - Cumulative quaternion multiplication for times within cyclic region
        - Proper quaternion accumulation across multiple cycles
        
        Args:
            key: Key name for quaternion buffer (e.g., 'target_quaternion')
            current_time: Time tensor of shape (num_envs,)
            
        Returns:
            torch.Tensor: Quaternion observations of shape (num_envs, 4) in w,x,y,z format
        """
        assert key in self.ref_buffer.keys(), f"Key '{key}' not found in reference buffer"
        
        # Helper functions for quaternion format conversion between PyTorch (w,x,y,z) and Warp (x,y,z,w)
        def to_xyzw(q_wxyz):
            """Convert quaternion from w,x,y,z to x,y,z,w format"""
            return q_wxyz.roll(shifts=-1, dims=-1).contiguous()
        
        def to_wxyz(q_xyzw):
            """Convert quaternion from x,y,z,w to w,x,y,z format"""
            return q_xyzw.roll(shifts=1, dims=-1).contiguous()
        
        # Get cyclic subsequence information
        current_num_cyclic_subseq, current_idx, current_begin_idx, current_end_idx = self.calc_num_cyclic_subseq(current_time)
        
        begin_indices = current_begin_idx  # shape: (num_envs,)
        end_indices = current_end_idx      # shape: (num_envs,)
        current_indices = current_idx      # shape: (num_envs,)
        
        # Calculate quaternion difference over one complete cycle
        q_begin = self.ref_buffer[key][begin_indices]  # shape: (num_envs, 4)
        q_end = self.ref_buffer[key][end_indices]      # shape: (num_envs, 4)
        cyclic_quat_diff = quat_mul(q_end, quat_inv(q_begin))  # Quaternion representing one cycle rotation
        
        # Prepare output tensor and convert quaternion buffer to warp format
        final_quat_out = wp.zeros(self.num_envs, dtype=wp.quat, device=self.device)
        
        quat_buffer_wxyz = self.ref_buffer[key]  # PyTorch format: w,x,y,z
        quat_buffer_xyzw = to_xyzw(quat_buffer_wxyz)  # Convert to Warp format: x,y,z,w
        quat_buffer_wp = wp.from_torch(quat_buffer_xyzw, dtype=wp.quat)
        
        # Launch CUDA kernel for parallel computation
        wp.launch(
            kernel=self.compute_cumulative_quat,
            dim=self.num_envs,
            inputs=[
                final_quat_out,
                quat_buffer_wp,
                wp.from_torch(current_indices.to(torch.int32)),
                wp.from_torch(begin_indices.to(torch.int32)),
                wp.from_torch(current_num_cyclic_subseq.to(torch.int32)),
                wp.from_torch(to_xyzw(cyclic_quat_diff), dtype=wp.quat)
            ],
            device=self.device
        )
        
        # Convert result back to PyTorch format
        final_quat_xyzw = wp.to_torch(final_quat_out).to(self.device)
        final_quat = to_wxyz(final_quat_xyzw)  # Convert back to w,x,y,z format
            
        return final_quat
    
    def calc_base_pose_cumulative(self, current_time: torch.Tensor, lin_vel_name: str, ang_vel_name: str):
        dt = 1.0 / self.frame_rate[self.env_ref_id]
        
        # ang_pos = self.calc_cumulative_obs_v2(ang_vel_name, current_time) * dt.unsqueeze(1)
        try:
            lin_pos, base_quat = self.calc_base_pose_from_trans_quat(current_time)
        except:
            try:
                lin_pos, ang_pos = self.calc_base_pose_from_trans_orient(current_time)
            
            except:
                lin_pos, ang_pos = self.calc_cumulative_obs_v2_lin_pos(lin_vel_name, ang_vel_name, current_time)# * dt.unsqueeze(1)
                lin_pos *= dt.unsqueeze(1)
                lin_pos[..., 2] += self.env_origin_z[self.env_ref_id]  # Adjust for environment origin z

            # normalize ang_pos to 0~2pi
            ang_pos = ang_pos % (2 * np.pi)
            # base_quat = angle_axis_to_quaternion(ang_pos)
            base_quat = quat_from_euler_xyz(*ang_pos.T)                
        return torch.cat([lin_pos, base_quat], dim=1)