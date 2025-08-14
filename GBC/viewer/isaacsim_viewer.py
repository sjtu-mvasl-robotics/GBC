import time
import argparse

from isaaclab.app import AppLauncher
# default_pkl = "/home/yifei/codespace/GBC_45/output/h1_2/converted_actions/ACCAD/Female1Walking_c3d/B3 - walk1_poses.pkl"
# default_pkl = "/home/yifei/codespace/GBC_45/output/h1_2/converted_actions/ACCAD/Female1Walking_c3d/B3 - walk1_poses_flipped.pkl"
# default_pkl = "/home/yifei/codespace/GBC_45/output/h1_2/unitree_rt/walk1_subject1.pkl"
default_pkl = "output/converted_actions/turin_v3/ACCAD/Male2MartialArtsKicks_c3d/G8 -  roundhouse left_poses.pkl"
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--pkl", type=str, default=default_pkl)
parser.add_argument("--urdf", type=str, default="/home/turin/sjtu/dataset/turin_urdf/turin_humanoid_robot/turin_v3_02/urdf/turin_humanoid_29dof_v3_2.urdf")

parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import yaw_quat, quat_apply, quat_inv, quat_mul, quat_error_magnitude, matrix_from_quat, quat_from_euler_xyz
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, FRAME_MARKER_CFG
from GBC.utils.base.math_utils import angle_axis_to_quaternion, is_foot_parallel_from_rot_matrix
from GBC.gyms.isaaclab_45.lab_assets.unitree import H1_2_CFG, G1_29DOF_CFG
from GBC.gyms.isaaclab_45.lab_assets.fourier import GR1_CFG
from GBC.gyms.isaaclab_45.lab_assets.turin_v3 import TURIN_V3_CFG
from GBC.utils.base.base_fk import RobotKinematics as RK


class VelocityIntegrator:
    def __init__(self, num_envs=1, device="cuda:0"):
        self.num_envs = num_envs
        self.device = device
        self.pos = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        self.quat = torch.zeros((num_envs, 4), dtype=torch.float32, device=self.device)
        self.quat[:, 0] = 1

    def reset(self, pos=None, quat=None):
        if pos is not None:
            self.pos = pos.clone()
        else:
            self.pos.zero_()
        if quat is not None:
            self.quat = quat.clone()
        else:
            self.quat.zero_()
            self.quat[:, 0] = 1

    def step(self, lin_vel_yaw_frame, ang_vel, dt):
        quat_yaw = yaw_quat(self.quat)
        lin_vel = quat_apply(quat_yaw, lin_vel_yaw_frame)
        self.pos += lin_vel * dt

        rot_vec = quat_apply(quat_inv(self.quat), ang_vel) * dt
        self.quat = quat_mul(self.quat, quat_inv(angle_axis_to_quaternion(rot_vec)))


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot: ArticulationCfg = TURIN_V3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene,
                  fps: float, tsl: torch.Tensor, orn: torch.Tensor,
                  # lin_vel: torch.Tensor, ang_vel: torch.Tensor,
                  actions: torch.Tensor,
                  feet_contact: torch.Tensor | None = None,
                  feet_contact_links: tuple[str, str] | None = None,
                  cyclic_subseq: tuple[int] = None):
    if feet_contact is not None and feet_contact_links is not None:
        marker_lft = VisualizationMarkers(FRAME_MARKER_CFG.replace(prim_path="/Visuals/left_foot"))
        marker_rht = VisualizationMarkers(FRAME_MARKER_CFG.replace(prim_path="/Visuals/right_foot"))

    tsl = tsl.reshape(-1, 3)
    orn = orn.reshape(-1, 4)
    actions = actions.reshape(-1, actions.shape[-1])
    actions_vel = torch.cat([torch.zeros_like(actions[0:1, :]), torch.diff(actions, dim=0)], dim=0) * fps

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    cur_time = 0
    wait_time = 0
    not_init = True

    root_vel = robot.data.default_root_state[:, 7:].clone()

    # vel_int = VelocityIntegrator(
    #     num_envs=scene.num_envs,
    #     device=scene.device,
    # )

    while simulation_app.is_running():
        wait_refresh_time = time.time() + sim_dt

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()

        if cur_time >= wait_time:
            frame_id = int(cur_time * fps + .5)
            if cyclic_subseq is not None:
                print('cyclic subseq:', cyclic_subseq)
                st, ed = cyclic_subseq
                if frame_id >= st:
                    frame_id = (frame_id - st) % (ed - st) + st
            elif frame_id >= tsl.shape[0]:
                cur_time = 0
                not_init = True
                continue

            if feet_contact is not None and feet_contact_links is not None:
                body_cfg = SceneEntityCfg("robot", body_names=feet_contact_links)
                body_cfg.resolve(scene)
                feet_state = robot.data.body_state_w[:, body_cfg.body_ids, :7]
                # feet_quat = feet_state[:, :, 3:7]
                # l_feet_rot_mat = matrix_from_quat(feet_quat[:, 0, :]).unsqueeze(1)
                # r_feet_rot_mat = matrix_from_quat(feet_quat[:, 1, :]).unsqueeze(1)
                # feet_rot_mat = torch.cat([l_feet_rot_mat, r_feet_rot_mat], dim=1)
                # is_parallel = is_foot_parallel_from_rot_matrix(feet_rot_mat, 15.0)
                # is_parallel_acc = torch.sum((feet_contact[frame_id] == is_parallel)) / 2
                # print('is_parallel_acc:', is_parallel_acc.item())
                # print('is_parallel:', is_parallel.mean().item())
                marker_lft.visualize(feet_state[:, 0, :3], feet_state[:, 0, 3:])
                marker_rht.visualize(feet_state[:, 1, :3], feet_state[:, 1, 3:])
                marker_lft.set_visibility(feet_contact[frame_id, 0])
                marker_rht.set_visibility(feet_contact[frame_id, 1])
                # marker_lft.set_visibility(False)
                # marker_rht.set_visibility(False)

            root_pose = robot.data.default_root_state[:, :7].clone()
            root_pose[:, :3] = tsl[None, frame_id] + scene.env_origins
            root_pose[:, 3:7] = orn[None, frame_id]
            # roll = torch.zeros(scene.num_envs, dtype=orn.dtype, device=orn.device)
            # pitch = torch.zeros(scene.num_envs, dtype=orn.dtype, device=orn.device)
            # yaw = torch.ones(scene.num_envs, dtype=orn.dtype, device=orn.device) * -0.5
            # yaw_orn = quat_from_euler_xyz(roll, pitch, yaw)
            # root_pose[:, 3:7] = quat_mul(yaw_orn, orn[None, frame_id].repeat(scene.num_envs, 1))

            # if not_init:
            #     vel_int.reset(root_pose[:, :3], root_pose[:, 3:])
            #     not_init = False

            # cur_root_pos = vel_int.pos.clone()
            # cur_root_quat = vel_int.quat.clone()

            # tsl_error = torch.mean(torch.norm(cur_root_pos[:, :2] - root_pose[:, :2], dim=1))
            # rot_error = torch.mean(quat_error_magnitude(cur_root_quat, root_pose[:, 3:7]))
            # print('tsl, rot error:', tsl_error.item(), rot_error.item())

            # root_vel[:, :3] = lin_vel[None, frame_id]
            # root_vel[:, 3:] = ang_vel[None, frame_id]
            # vel_int.step(root_vel[:, :3], root_vel[:, 3:], sim_dt)

            joint_pos = actions[None, frame_id]
            target_joint_vel = actions_vel[None, frame_id]
            cur_joint_vel = robot.data.joint_vel[:, :].clone()
            target_joint_vel_diff = torch.norm(target_joint_vel - cur_joint_vel, dim=1)
            print('joint vel diff:', target_joint_vel_diff.mean().item())
            print('joint vel (first 5):', cur_joint_vel[0, :5])
            print('target joint vel (first 5):', target_joint_vel[0, :5])

            robot.write_root_pose_to_sim(root_pose)         
        
        robot.write_root_velocity_to_sim(root_vel)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.step()
        cur_time += sim_dt
        scene.update(sim_dt)
        # print(torch.min(robot.data.body_state_w[:, :, 2]))

        while time.time() < wait_refresh_time:
            continue


def main():
    data = torch.load(args_cli.pkl, map_location=args_cli.device)
    data["root_orient"] = angle_axis_to_quaternion(data["root_orient"])


    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1/50)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([4.5, 1.0, 3.0], [0.0, 0.0, 2.0])

    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    action_targ_order = scene.articulations["robot"].joint_names
    urdf_path = args_cli.urdf
    rk = RK(urdf_path)
    action_raw_order = rk.get_dof_names()
    action_targ_order_idx = [action_raw_order.index(joint) for joint in action_targ_order]
    data["actions"] = data["actions"][:, action_targ_order_idx]

    print(action_targ_order)
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene, data["fps"], data["trans"], data["root_orient"],
                #   data["lin_vel"], data["ang_vel"],
                  data["actions"],
                #   cyclic_subseq=data["cyclic_subseq"],
                  feet_contact=data["feet_contact"],
                  feet_contact_links=("l.*ankle.*roll.*", "r.*ankle.*roll.*")
    )


if __name__ == "__main__":
    main()
    simulation_app.close()