# convert wham output to GBC poses
from GBC.utils.data_preparation.amass_action_converter import *
import joblib

class WHAMConverter:
    
    def __init__(self, cfg: BaseCfg, **kwargs):
        self.cfg = cfg
        self.load_hands = cfg.load_hands
        self.fk = RobotKinematics(cfg.urdf_path, device=cfg.device)
        self.model = PoseTransformer(num_actions=self.fk.num_dofs, load_hands=cfg.load_hands).to(cfg.device)
        self.model.load_state_dict(torch.load(cfg.pose_transformer_path, map_location=torch.device(cfg.device)))
        self.post_process = TrackingDataPostProcess(filter_cfg=cfg.filter).to(cfg.device)
        self.batch_size = cfg.batch_size
        self.device = cfg.device
        self.export_path = cfg.export_path
        self.visualize = cfg.visualize
        self.default_body_map = None
        self.smplh_body_names = ['Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist']
        self.smplh_extend_names = ['L_Index1', 'L_Index2', 'L_Index3', 'L_Middle1', 'L_Middle2', 'L_Middle3', 'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 'L_Ring1', 'L_Ring2', 'L_Ring3', 'L_Thumb1', 'L_Thumb2', 'L_Thumb3', 'R_Index1', 'R_Index2', 'R_Index3', 'R_Middle1', 'R_Middle2', 'R_Middle3', 'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 'R_Ring1', 'R_Ring2', 'R_Ring3', 'R_Thumb1', 'R_Thumb2', 'R_Thumb3']
        self.action_flip_left_right = None
        self.set_body_model(cfg.smplh_model_path, cfg.dmpls_model_path, cfg.smpl_fits_dir)
        
    def set_body_model(self, smplh_model_path: str, dmpls_model_path: str, smpl_fits_dir: str):
        self.body_model = BodyModel(bm_fname=smplh_model_path, dmpl_fname=dmpls_model_path, num_betas=16, num_dmpls=8).to(self.device)
        try:
            self.fit_data = torch.load(smpl_fits_dir, map_location=self.device)
        except FileNotFoundError:
            print(f"Warning: {smpl_fits_dir} not found")
            self.fit_data = None

    @torch.no_grad()
    def pose_to_action(self, pose: torch.Tensor):
        '''
            Convert pose to action using the PoseTransformer model

            Args:
                pose (torch.Tensor): The input pose to be converted (T, 63 if not load_hands else 63+90)

            Returns:
                torch.Tensor: The converted action (T, 63 if not load_hands else 63+90)
        '''
        num_splits = pose.shape[0] // self.batch_size
        if pose.shape[0] % self.batch_size != 0:
            num_splits += 1

        actions = []

        for i in range(num_splits):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, pose.shape[0])
            pose_batch = pose[start:end].to(self.device)
            action_batch = self.model(pose_batch)
            if self.default_body_map is not None:
                pose_body_indices, action_indices = self.default_body_map
                pose_body_indices = torch.tensor(pose_body_indices, dtype=torch.long, device=self.device)
                action_indices = torch.tensor(action_indices, dtype=torch.long, device=self.device)
                pose_body = pose_batch[:, :63].reshape(-1, 21, 3)
                ypr = batch_angle_axis_to_ypr(pose_body)
                # ypr[:,:,2] *= 0.01

    
                action_batch[:, action_indices] = ypr[:, pose_body_indices[:, 0], pose_body_indices[:, 1]]

            actions.append(action_batch)

        actions = torch.cat(actions, dim=0)
        actions = adjust_shape_0(actions, pose.shape[0])
        return actions
    
    def load_wham_data(self, wham_path):
        """
        Load WHAM data from the specified path.
        """
        with open(wham_path, 'rb') as f:
            wham_data = joblib.load(f)
        wham_keys = list(wham_data.keys())
        if len(wham_keys) < 1:
            return None # No available data
        
        wham_rtn = []
        for key in wham_keys:
            wham_rtn.append(wham_data[key])
        return wham_rtn
    
    def load_tracking_data(self, tracking_path):
        """
        Load tracking data from .pth file.
        """
        with open(tracking_path, 'rb') as f:
            tracking_data = joblib.load(f)
        if not isinstance(tracking_data, dict):
            print(f"Tracking data is not a dictionary, got {type(tracking_data)}")
            return None
        tracking_group = []
        for key, value in tracking_data.items():
            tracking_group.append(value)
        
        return tracking_group

        

    def extract_wham_data(self, wham_data):
        wham_raw = {}
        wham_world = {}
        wham_raw['trans'] = wham_data['trans']
        wham_world['trans'] = wham_data['trans_world']
        wham_raw["poses"] = wham_data['pose']
        wham_world["poses"] = wham_data['pose_world']
        return wham_raw, wham_world
        

    def adjust_trans_height(self, trans: torch.Tensor, poses: torch.Tensor):
            body_parms = {
                "betas": self.fit_data["beta"],
                "dmpls": self.fit_data["dmpls"],
                "pose_body": poses[:, 3:66],
                "trans": trans,
                "root_orient": poses[:, :3],
            }
            positions = self.body_model(**body_parms).Jtr.detach()
            height = positions[:, :, 1]
            trans[:, 1] -= height.min()
            return trans
        
    def adjust_trans_height_with_fk(self, trans: torch.Tensor, poses: torch.Tensor, orient: torch.Tensor, foot_names: list):
        """
        Adjust the height of the translation based on the forward kinematics of the robot.
        """
        from GBC.utils.base.base_fk import RobotKinematics as FK
        fk = FK(self.cfg.urdf_path, device=self.device)
        fk.set_target_links(foot_names)
        link_positions = fk(poses, root_trans_offset=trans,root_rot=orient) # shape: (T, 2, 3)
        link_height = link_positions[:, :, 2]
        link_height_min = link_height.min(dim=1, keepdim=True)[0] # shape: (T, 1)
        trans[:, 2] -= link_height_min.squeeze(1) # adjust the height of the translation
        trans[:, 2] + 0.05
        return trans
        
    
    def convert_dict_to_gbc(self, data_dict: dict, tracking_data: dict, foot_names: list, export_path: str, save_name: str = "converted_data.pkl"):
        """
        Convert WHAM data dictionary to GBC format.
        """

        trans = torch.Tensor(data_dict['trans']).to(self.device)
        poses = torch.Tensor(data_dict['poses']).to(self.device)

        root_orient = poses[:, :3]
        
        init_global_orient = tracking_data["init_global_orient"] # shape (1,1,3,3)
        init_global_orient = torch.tensor(init_global_orient, dtype=torch.float32, device=self.device).squeeze(0).squeeze(0) # shape (3,3)
        root_rot_mat = rot_vec_to_mat(root_orient)
        root_rot_mat = torch.einsum("ij, tjk -> tik", init_global_orient, root_rot_mat) # (T, 3, 3)
        root_rot_mat_0 = root_rot_mat[0, :, :].permute(1, 0) # (1, 3, 3)
        root_rot_mat = torch.einsum("ij, tjk -> tik", root_rot_mat_0, root_rot_mat) # (T, 3, 3)
        additional_rot_mat = torch.tensor(
            [
                [1., 0, 0.],
                [0.0, 0., -1.],
                [0., 1., 0.]
            ], dtype=torch.float32, device=self.device
        )
        root_rot_mat = torch.einsum("ij, tjk -> tik", additional_rot_mat, root_rot_mat) # (T, 3, 3)
        
        root_orient = rot_mat_to_vec(root_rot_mat) # (T, 3)
        trans = torch.einsum("ij, tj -> ti", init_global_orient, trans)
        trans = torch.einsum("ij, tj -> ti", root_rot_mat_0, trans)
        trans = torch.einsum("ij, tj -> ti", additional_rot_mat, trans)
        
        
        root_orient = self.post_process.adjust_root_orient(root_orient)
        root_orient = adjust_shape_0(root_orient, poses.shape[0])
        trans = self.adjust_trans_height(trans, poses)

        pose = poses[:, 3:66].to(self.device) if not self.load_hands else poses[:, 3:].to(self.device)
        actions = self.pose_to_action(pose)
        actions_filtered = self.post_process.filt(actions)
        save_dict = {
            "trans": trans,
            "root_orient": root_orient,
            "actions_orig": actions,
            "actions": actions_filtered,
            "cyclic_subseq": None,
            "fps": 30,
        }
        save_dict = self.post_process(save_dict, contact_tf_mats=None, filter_pose=True)
        save_dict["trans"] = self.adjust_trans_height_with_fk(save_dict["trans"], actions_filtered, save_dict["root_orient"],  foot_names)
        
        feet_contact_indices, filtered_root_poses = filt_feet_contact(
            actions=actions_filtered.unsqueeze(0), # shape (1, num_frames, num_dofs)
            root_pos=save_dict["trans"].unsqueeze(0), # shape (1, num_frames, 3)
            root_rot=save_dict["root_orient"].unsqueeze(0), # shape (1, num_frames, 3)
            fk=RobotKinematics(self.cfg.urdf_path, device=self.device),
            foot_names=foot_names,
            threshold=0.065,
            debug_visualize=False,
            disable_height=False,
            debug_save_dir=export_path,
        )
        
        filtered_root_poses = self.post_process.filt(filtered_root_poses.squeeze(0))

        save_dict["trans"] = filtered_root_poses.squeeze(0)
        
        
        
        feet_contact_indices = feet_contact_indices.squeeze(0)
        save_dict["feet_contact"] = feet_contact_indices
        
        
        
        
        self.fk.set_target_links(self.cfg.mapping_table.values())
        link_positions = self.fk.forward(actions_filtered, root_trans_offset=save_dict["trans"], root_rot=save_dict["root_orient"]) # (T, N_joints, 3)
        link_velocities = torch.diff(link_positions[:, :, :], dim=0) * 30
        link_velocities = torch.cat([torch.zeros_like(link_positions[0:1, :, :]), link_velocities], dim=0)
        rot_mat = rot_vec_to_mat(save_dict["root_orient"])
        link_velocities_local = torch.einsum("ijk, ilj -> ilk", rot_mat, link_velocities)
        save_dict["link_positions"] = link_positions # global positions
        link_positions_local = link_positions
        # link_positions_local[:, :, 0] -= link_positions[0:1, :, 0]
        # link_positions_local[:, :, 1] -= link_positions[0:1, :, 1]
        # Lines above assume the root is the first link which is now deprecated. We calculate by using root translation instead.
        link_positions_local[:, :, 0] -= save_dict["trans"][:, 0:1]
        link_positions_local[:, :, 1] -= save_dict["trans"][:, 1:2]
        
        link_positions_local = torch.einsum("ijk, ilj -> ilk", rot_mat, link_positions_local)
        save_dict["link_positions_local"] = link_positions_local # local positions
        link_velocities_local = torch.einsum("ijk, ilj -> ilk", rot_mat, link_velocities_local)
        save_dict["link_velocities"] = link_velocities_local #
        
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        save_path = os.path.join(export_path, save_name)
        
        torch.save(save_dict, save_path)
        
        

if __name__ == "__main__":
    import argparse
    import yaml
    # get parser args
    parser = argparse.ArgumentParser(description="Convert WHAM data to GBC format")
    parser.add_argument("--wham_path", type=str, default="/home/yifei/codespace/WHAM/vdo/zhiyin", help="Path to WHAM data folder")
    # parser.add_argument("--tracking_path", type=str, default="/home/yifei/codespace/WHAM/vdo/test/test/tracking_results.pth", help="Path to WHAM tracking data file")
    parser.add_argument("--export_path", type=str, default="export/zhiyin", help="Path to export converted GBC data")
    parser.add_argument("--config_path", type=str, default="/home/yifei/codespace/GBC_DP/GBC_MUJICA/data_preparation/humanoid_cfg/unitree_g1.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    mapping_table = config["smplh"]["mapping_table"]
    urdf_path = config["base"]["urdf_path"]
    pose_transformer_path = config["base"]["ik_model_path"]
    smpl_fits_dir = config["base"]["smpl_fits_dir"]

    cfg = AMASSActionConverterCfg(
        urdf_path=urdf_path,
        pose_transformer_path=pose_transformer_path,
        export_path=args.export_path,
        mapping_table=mapping_table,
        smplh_model_path=DATA_PATHS.smplh_model_path,
        dmpls_model_path=DATA_PATHS.dmpls_model_path,
        smpl_fits_dir=smpl_fits_dir,
    )
    
    foot_names = config["base"]["foot_names"]
    
    converter = WHAMConverter(cfg)
    converter.visualize = False
    
    wham_data = converter.load_wham_data(args.wham_path + "/wham_output.pkl")
    if wham_data is None:
        print(f"No available data in {args.wham_path}")
        exit(0)

    tracking_data = converter.load_tracking_data(args.wham_path + "/tracking_results.pth")

    for i, data in enumerate(wham_data):
        wham_raw, wham_world = converter.extract_wham_data(data)
        converter.convert_dict_to_gbc(
            wham_raw,
            tracking_data=tracking_data[i],
            foot_names=foot_names,
            export_path=args.export_path,
            save_name=f"converted_data_{i}.pkl"
        )

        converter.convert_dict_to_gbc(
            wham_world,
            tracking_data=tracking_data[i],
            foot_names=foot_names,
            export_path=args.export_path,
            save_name=f"converted_world_data_{i}.pkl"
        )
        print(f"Converted WHAM data {i} to GBC format and saved to {args.export_path}")
    print("Conversion completed.")