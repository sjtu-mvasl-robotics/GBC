from collections.abc import Sequence
from isaaclab.envs import ManagerBasedRLEnv

from GBC.gyms.isaaclab_45.managers import RefObservationManager, ReferenceCommandManager, PhysicsModifierManager

from .manager_based_ref_rl_env_cfg import ManagerBasedRefRLEnvCfg
import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg, DEFORMABLE_TARGET_MARKER_CFG, CONTACT_SENSOR_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG
from isaaclab.managers import SceneEntityCfg
from GBC_MUJICA.diffusion_planner.data_loaders.humanoid_utils import HML_LINK_NAMES, HML_EE_LINK_NAMES, HML_ROOT_LINK_NAME
import isaaclab.sim as sim_utils
import GBC.gyms.isaaclab_45.lab_tasks.mdp as gbc_mdp

class DebugVisualizer:
    def __init__(self, env):
        self.env = env        
        self._init_markers()
        
    def _init_markers(self):
        links = HML_LINK_NAMES  # exclude root link
        # base link markers
        marker_base_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/World/base_link_marker")
        self.base_link_marker = VisualizationMarkers(marker_base_cfg)
        
        marker_link_cfg = VisualizationMarkersCfg(
            prim_path="/World/link_markers",
            markers={
                f'marker_{link}': 
                    sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.467, 0.6, 0.8)), # #7799CC for Togawa Sakiko
                    )
                for link in links
                }
        )
        self.link_markers = VisualizationMarkers(marker_link_cfg)
            
    def step(self, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        pose, mask = self.env.ref_observation_manager.get_term("target_base_pose")
        pose = pose[mask]
        r_pos, r_quat = pose[:, :3], pose[:, 3:7]
        self.base_link_marker.visualize(r_pos, r_quat)
        target_link_poses, mask = self.env.ref_observation_manager.get_term("target_link_poses")
        target_link_poses = target_link_poses[mask]
        global_link_pos, global_link_quat = gbc_mdp.recover_body_pos_robot_frame_to_global(r_pos, r_quat, target_link_poses[..., :3], target_link_poses[..., 3:7])
        self.link_markers.visualize(global_link_pos, global_link_quat)
        
        
class ManagerBasedRefRLEnv(ManagerBasedRLEnv):
    cfg: ManagerBasedRefRLEnvCfg

    def __init__(self, cfg: ManagerBasedRefRLEnvCfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        self.cur_time = torch.zeros(self.num_envs, device=self.device)
        # self.debug_vis = DebugVisualizer(self)

    def load_managers(self):
        self.ref_observation_manager = RefObservationManager(self.cfg.ref_observation, self)
        print("[INFO] Reference Observation Manager:", self.ref_observation_manager)

        super().load_managers()
        
        self.command_manager = ReferenceCommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager:", self.command_manager)
        if hasattr(self.cfg, "physics_modifiers"):
            self.physics_modifier_manager = PhysicsModifierManager(self.cfg.physics_modifiers, self)
        else:
            self.physics_modifier_manager = None
        print("[INFO] Physics Modifier Manager:", self.physics_modifier_manager)


    def step(self, action: torch.Tensor):
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update reference observations first.
        5. Update the environment counters and compute the rewards and terminations.
        6. Reset the environments that terminated.
        7. Compute the observations.
        8. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
            if self.physics_modifier_manager is not None:
                self.physics_modifier_manager.apply()

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        # -- update reference observations
        self.cur_time = self.episode_length_buf.to(torch.float32) * self.step_dt
        self.ref_obs_buf = self.ref_observation_manager.compute(self.cur_time)

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()
        
        # always call physics_modifier_manager.update **AFTER** computing observations
        if self.physics_modifier_manager is not None:
            self.physics_modifier_manager.update()
            
        # self.debug_vis.step()
        # if self.use_custom_obs:
        #     self.custom_obs_buf = self.observation_manager.custom_compute()

        # return observations, ref_observations, rewards, resets and extras
        return self.obs_buf, self.ref_obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        info = self.ref_observation_manager.reset(self, env_ids)
        self.extras["log"].update(info)