# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

parser.add_argument("--enable-feet-logger", type=bool, default=True)
parser.add_argument("--enable-ref-action-logger", type=bool, default=True)
parser.add_argument("--save-robot-data", type=bool, default=True)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR,"../.."))

# import lab_tasks
import GBC.gyms.isaaclab_45.lab_tasks

from rsl_rl.runners import OnPolicyRunnerMM

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.markers import VisualizationMarkers, FRAME_MARKER_CFG
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    # RslRlOnPolicyRunnerCfg,
    # RslRlVecEnvWrapper,
    export_policy_as_jit,
    # export_policy_as_onnx,
)
from GBC.gyms.isaaclab_45.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlRefOnPolicyRunnerCfg,
    RslRlReferenceVecEnvWrapper,
    export_policy_as_onnx,
)


class FeetContactLogger:
    def __init__(self, env, sensor_cfg):
        self.env = env
        self.sensor_cfg = sensor_cfg
        self.sensor_cfg.resolve(self.env.unwrapped.scene)

        self.list_contacts = []
        # self.list_target_contacts = []

    def step(self):
        contact_sensor = self.env.unwrapped.scene.sensors[self.sensor_cfg.name]
        current_contact_time = contact_sensor.data.current_contact_time[:, self.sensor_cfg.body_ids]
        self.list_contacts.append((current_contact_time > 0.0).unsqueeze(0))

        # target_feet_contact, _ = self.env.unwrapped.ref_observation_manager.get_term("feet_contact")
        # self.list_target_contacts.append((target_feet_contact > 0.0).unsqueeze(0))

    def print_info(self):
        contact = torch.cat(self.list_contacts)
        # target_contact = torch.cat(self.list_target_contacts)

        def calc_contact_and_air(tensor):
            list_times = [[], []]
            cur_times = [0, 0]
            flag = None
            for cur in tensor:
                cur = cur.item()
                if cur != flag:
                    if flag is not None:
                        list_times[flag].append(cur_times[flag])
                        cur_times[flag] = 0
                    flag = cur
                cur_times[cur] += 1
            list_times[flag].append(cur_times[flag])
            print("air: %f, contact: %f" % tuple(torch.mean(torch.tensor(times, dtype=float)).item() for times in list_times))
        for i in range(contact.shape[1]):
            print("Env:", i)
            print("Left:")
            calc_contact_and_air(contact[:, i, 0])
            print("Right:")
            calc_contact_and_air(contact[:, i, 1])


class RefActionLogger: 
    def __init__(self, env, asset_cfg):
        self.env = env
        self.asset_cfg = asset_cfg
        self.asset_cfg.resolve(self.env.unwrapped.scene)

        self.list_actual_jt_pos = []
        self.list_target_jt_pos = []

    def step(self):
        asset = self.env.unwrapped.scene[self.asset_cfg.name]
        actual_jt_pos = asset.data.joint_pos[:, self.asset_cfg.joint_ids]
        self.list_actual_jt_pos.append(actual_jt_pos)

        target_jt_pos, mask = self.env.unwrapped.ref_observation_manager.get_term("target_actions")
        assert mask.all()
        target_jt_pos = target_jt_pos[:, self.asset_cfg.joint_ids]
        self.list_target_jt_pos.append(target_jt_pos)

    def print_info(self):
        actual_jt_pos = torch.cat(self.list_actual_jt_pos)
        target_jt_pos = torch.cat(self.list_target_jt_pos)
        actions_diff = actual_jt_pos - target_jt_pos
        std_list = [0.5]

        def print_errors(errors, std_list=[]):
            print("Min error:", torch.min(errors).item(), torch.argmin(errors))
            print("Max error:", torch.max(errors).item(), torch.argmax(errors))
            print("Avg error:", torch.mean(errors).item())
            for std in std_list:
                print("Std:", std, "reward:", torch.mean(torch.exp(-errors / std**2)).item())

        print("Mean Absolute Error:")
        print_errors(torch.mean(torch.abs(actions_diff), dim=-1), std_list)
        print()

        print("Norm Error:")
        print_errors(torch.norm(actions_diff, dim=-1), std_list)
        print()


class SaveRobotDataLogger:
    save_names = ["root_state_w"] + ["joint_" + key for key in ("pos", "vel", "acc")]
    def __init__(self, env, asset_cfg):
        self.env = env
        self.asset_cfg = asset_cfg
        self.asset_cfg.resolve(self.env.unwrapped.scene)
        self.step_dt = env.unwrapped.step_dt

        self.joint_names = self.env.unwrapped.scene[self.asset_cfg.name].data.joint_names
        for save_name in self.save_names:
            setattr(self, "list_" + save_name, [])
        self.path = None

    def step(self):
        if hasattr(self.env.env, "video_folder") and hasattr(self.env.env, "_video_name") and self.path is None:
            path = os.path.join(self.env.env.video_folder, self.env.env._video_name)
            self.path = os.path.splitext(path)[0] + ".pkl"
            print(f"Robot data will be saved to {self.path}")

        asset = self.env.unwrapped.scene[self.asset_cfg.name]
        for save_name in self.save_names:
            l = getattr(self, "list_" + save_name)
            l.append(getattr(asset.data, save_name).clone())
            setattr(self, "list_" + save_name, l)

    def print_info(self):
        assert self.path is not None
        obj = {
            "step_dt": self.step_dt,
            "joint_names": self.joint_names,
        }
        for save_name in self.save_names:
            obj[save_name] = torch.stack(getattr(self, "list_" + save_name))
        torch.save(obj, self.path)
        joint_pos = obj["joint_pos"]
        joint_pos_d1 = torch.diff(joint_pos, dim=0)
        joint_pos_d2 = torch.diff(joint_pos_d1, dim=0)
        print("Joint pos diff2:", torch.mean(torch.sum(torch.square(joint_pos_d2), dim=(0, 2))).item())
        joint_acc = obj["joint_acc"]
        print("Joint acc:", torch.mean(torch.sum(torch.square(joint_acc), dim=(0, 2))).item())
        return obj


class ActionRateAccLogger:
    def __init__(self, env, asset_cfg):
        self.env = env
        self.asset_cfg = asset_cfg
        self.asset_cfg.resolve(self.env.unwrapped.scene)
        self.step_dt = env.unwrapped.step_dt
        self.prev_prev_action = None
        self.sum_acc = None

    def step(self):
        action_mgr = self.env.unwrapped.action_manager
        if self.prev_prev_action is None:
            self.prev_prev_action = torch.zeros_like(action_mgr.action)
        dt2 = self.step_dt ** 2
        acc = (action_mgr.action - 2 * action_mgr.prev_action + self.prev_prev_action) / dt2
        if self.sum_acc is None:
            self.sum_acc = torch.zeros_like(acc)
        self.sum_acc += acc

    def print_info(self):
        print("Sum of acc (mean across different environments):", torch.mean(self.sum_acc))


class RefBasePoseLogger:
    def __init__(self, env):
        self.env = env
        self.marker_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/World/Marker",
        )
        self.marker = VisualizationMarkers(self.marker_cfg)

    def step(self):
        pose, mask = self.env.unwrapped.ref_observation_manager.get_term("target_base_pose")
        pose = pose[mask]
        pos, quat = pose[:, :3], pose[:, 3:7]
        pos = pos + self.env.unwrapped.scene.env_origins
        self.marker.visualize(translations=pos, orientations=quat)

    def print_info(self):
        print("No information")


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlRefOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # agent_cfg.load_run = "2025-03-26_13-28*"
    # agent_cfg.load_checkpoint = ".*5000.pt"
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    iter_id = int(resume_path.split("model_")[1].split(".")[0])
    print("Iteration id:", iter_id)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "episode_trigger": lambda episode_id: episode_id == iter_id,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        env.episode_id = iter_id - 1 # To change name of the generated video

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlReferenceVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunnerMM(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(
    #     ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    # )
    # export_policy_as_onnx(
    #     ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    loggers = []
    if args_cli.enable_feet_logger:
        loggers.append(FeetContactLogger(
            env,
            SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            # SceneEntityCfg("contact_forces", body_names=["left_ankle_link", "right_ankle_link"]),
        )) 
    if args_cli.enable_ref_action_logger:
        loggers.append(RefActionLogger(
            env,
            SceneEntityCfg("robot", joint_names=[ ".*" ]),
        ))
    if args_cli.save_robot_data:
        loggers.append(SaveRobotDataLogger(
            env,
            SceneEntityCfg("robot"),
        ))
    if True:
        loggers.append(ActionRateAccLogger(
            env,
            SceneEntityCfg("robot"),
        ))
    if True:
        loggers.append(RefBasePoseLogger(
            env,
        ))

    # reset environment
    obs, _ = env.get_observations()
    ref_obs, _ = env.get_reference_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs, ref_obs)
            # env stepping
            obs, ref_obs, _, _, extras = env.step(actions)

            for logger in loggers:
                logger.step()

        if args_cli.video:
            timestep += 1
            if timestep % 50 == 0:
                print("timestep:", timestep)
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()

    for logger in loggers:
        print(type(logger).__name__)
        logger.print_info()
        print()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
