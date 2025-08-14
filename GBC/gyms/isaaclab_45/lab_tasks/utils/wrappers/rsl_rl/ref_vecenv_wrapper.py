import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from GBC.gyms.isaaclab_45.envs import ManagerBasedRefRLEnv
from GBC.gyms.isaaclab_45.managers import ref_obs_type
# import time


class RslRlReferenceVecEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env: ManagerBasedRefRLEnv):
        super().__init__(env)

    def get_reference_observations(self) -> tuple[ref_obs_type, dict]:
        cur_time = self.unwrapped.episode_length_buf.to(torch.float32) * self.unwrapped.step_dt
        obs_dict = self.unwrapped.ref_observation_manager.compute(cur_time)
        return obs_dict["policy"], {"ref_observations": obs_dict}

    def reset(self) -> tuple[torch.Tensor, ref_obs_type, dict]:
        # obs, obs_dict = super().reset()
        # ref_obs, ref_obs_dict = self.get_reference_observations()
        # return obs, ref_obs, {**obs_dict, **ref_obs_dict}
        obs_dict, ref_obs_dict, _ = self.env.reset()
        # return observations
        return obs_dict["policy"], ref_obs_dict["policy"], {"observations": obs_dict, "ref_observations": ref_obs_dict}

    


    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, ref_obs_type, torch.Tensor, torch.Tensor, dict]:
        '''
        ** NOTE: ** `self.get_reference_observations()` MUST be called before `self.reward_manager.compute(self.step_dt)` in order to make sure the `<ReferenceObservationManager._ref_observation_group_term_tmp_storage>`  is updated before the `reward_manager` trying to call `get_term`       
        
        '''
        # record step information
        obs_dict, ref_obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs = obs_dict["policy"]
        ref_obs = ref_obs_dict["policy"]
        extras["observations"] = obs_dict
        extras["ref_observations"] = ref_obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return obs, ref_obs, rew, dones, extras

    # def step(self, action: torch.Tensor) -> tuple[torch.Tensor, ref_obs_type, torch.Tensor, torch.Tensor, dict]:
        
    #     # obs, rew, dones, extras = super().step(actions)
    #     obs, rew, dones, extras = super().step(action)
    #     ref_obs, ref_obs_dict = self.get_reference_observations()
    #     extras = {**extras, **ref_obs_dict}
    #     return obs, ref_obs, rew, dones, extras