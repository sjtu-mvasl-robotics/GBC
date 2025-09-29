import torch
from isaaclab.envs import ManagerBasedRLEnv

class StdUpdater:
    def __init__(self, std_list, reward_key, reward_threshold=0.8, reward_hist=128, step_threshold_down=32*1200, step_threshold_up=32*200):
        self.std_list = std_list
        self.reward_key = reward_key
        self.level = 0
        self.step_num_last_change = 0
        self.reward_threshold = reward_threshold
        self.step_threshold_down = step_threshold_down
        self.step_threshold_up = step_threshold_up
        self.reward_hist = reward_hist
        self.reward_buf = torch.ones(self.reward_hist, dtype=torch.float64) * -1e9
        self.reward_pos_id = 0

    def update(self, env: ManagerBasedRLEnv, bidirect=False):
        reward = env.extras["log"][f"Episode_Reward/{self.reward_key}"]
        try:
            reward_weight = env.reward_manager._term_cfgs[env.reward_manager.active_terms.index(self.reward_key)].weight
        except:
            print(f"Warning: {self.reward_key} not found in reward manager. Using default weight.")
            reward_weight = 1.0
        self.reward_buf[self.reward_pos_id] = reward
        self.reward_pos_id = (self.reward_pos_id + 1) % self.reward_hist
        reward = torch.mean(self.reward_buf).item()
        reward = reward / reward_weight

        changed = False
        if env.common_step_counter - self.step_num_last_change > self.step_threshold_up and reward > self.reward_threshold:
            if self.level < len(self.std_list) - 1:
                self.level += 1
                changed = True
        elif (env.common_step_counter - self.step_num_last_change > self.step_threshold_down and 
              reward < self.reward_threshold * 0.5 and bidirect):
            if self.level > 0:
                self.level -= 1
                changed = True
        if changed:
            self.step_num_last_change = env.common_step_counter
            print(f"Updated {self.reward_key} level to {self.level}")
        env.extras["log"][f"Curriculum/{self.reward_key}_std"] = self.std_list[self.level]
        return self.std_list[self.level]
