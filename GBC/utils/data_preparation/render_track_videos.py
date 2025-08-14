# load amass dataset and render tracking videos
from GBC.utils.data_preparation.pose_transformer import PoseTransformer
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from human_body_prior.body_model.body_model import BodyModel
import trimesh
from body_visualizer.tools.vis_tools import colors, show_image
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Optional
from GBC.utils.base.base_fk import RobotKinematics
from GBC.utils.data_preparation.amass_loader import AMASSDataset, AMASSDatasetInterpolate
from GBC.utils.data_preparation.robot_visualizer import RobotVisualizer
import cv2
from scipy.spatial.transform import Rotation as R
from glob import glob
import shlex

from GBC.utils.base.math_utils import symmetry_smplh_pose
from GBC.utils.data_preparation.data_preparation_cfg import BaseCfg, FilterCfg
from GBC.utils.data_preparation.amass_action_converter import TrackingDataPostProcess



class PoseRenderer:
    def __init__(self,
                 urdf_path: str,
                 dataset_path: str,
                 mapping_table: Dict[str, str],
                 smplh_model_path: str,
                 dmpls_model_path: str,
                 poseformer_model_path: str,
                 save_path: str,
                 secondary_dir: Optional[str] = None,
                 device: str = "cuda",
                 max_single_batch: int = 128,
                 without_model = False,
                 smpl_fits_dir = None
                 ):
        self.fk = RobotKinematics(urdf_path, device)
        # self.dataset = AMASSDataset(root_dir=dataset_path, num_betas=16, num_dmpls=8, secondary_dir=secondary_dir)
        self.dataset = AMASSDataset(root_dir=dataset_path, num_betas=16, num_dmpls=8, secondary_dir=secondary_dir)
        self.mapping_table = mapping_table
        self.fk.set_target_links(list(mapping_table.values()))
        self.save_path = save_path
        self.bm = BodyModel(bm_fname=smplh_model_path, dmpl_fname=dmpls_model_path, num_betas=16, num_dmpls=8).to(device)
        if not without_model:
            self.load_poseformer(poseformer_model_path, device)
        else:
            self.poseformer = None
        self.max_single_batch = max_single_batch

        self.robot_vis = RobotVisualizer(urdf_path, width=800, height=600, device=device)

        if smpl_fits_dir is not None:
            self.smpl_fits = torch.load(smpl_fits_dir, map_location=torch.device(device))
        else:
            self.smpl_fits = None

    @classmethod
    def from_cfg(cls, cfg: BaseCfg):
        return cls(
            urdf_path=cfg.urdf_path,
            dataset_path=cfg.root_dir,
            mapping_table=cfg.mapping_table,
            smplh_model_path=cfg.smplh_model_path,
            dmpls_model_path=cfg.dmpls_model_path,
            poseformer_model_path=cfg.poseformer_model_path,
            save_path=cfg.save_path,
            secondary_dir=cfg.secondary_dir,
            device=cfg.device
        )
    
    def set_smpl_fits(self, smpl_fits: Dict[str, torch.Tensor]):
        self.smpl_fits = smpl_fits


    def load_poseformer(self, poseformer_model_path: str, device: str):
        self.poseformer = PoseTransformer(num_actions=self.fk.num_dofs)
        self.poseformer.to(device)
        self.poseformer.load_state_dict(torch.load(poseformer_model_path, map_location=device))

    def get_pose_amass(self, pose_body: torch.Tensor, pose_hand: Optional[torch.Tensor] = None, add_smpl_fits: bool = True):
        root_orient = R.from_quat([0, 0, 0, 1]).as_rotvec()
        root_orient = torch.tensor(root_orient, dtype=pose_body.dtype).to(pose_body.device)
        root_orient = root_orient.unsqueeze(0).repeat(pose_body.shape[0], 1)

        # if pose_hand is None:
        #     pose_hand = torch.zeros((pose_body.shape[0], 30 * 3), dtype=pose_body.dtype).to(pose_body.device)
        body_parms = {
            'root_orient': root_orient,
            'pose_body': pose_body,
            # 'pose_hand': pose_hand
        }
        if self.smpl_fits is not None and add_smpl_fits:
            body_parms['betas'] = self.smpl_fits['beta']
            body_parms['dmpls'] = self.smpl_fits['dmpls']
        if pose_hand is not None:
            body_parms['pose_hand'] = pose_hand
        return self.bm(**body_parms)
    
    def get_pose_former(self, pose: torch.Tensor):
        actions = self.poseformer(pose)
        return self.fk(actions)
    
    def render_pose_amass(self, pose: torch.Tensor, face: torch.Tensor, save_path: str, width: int = 800, height: int = 600, update_mv: bool = False):
        if not hasattr(self, 'mv') or update_mv:
            self.mv = MeshViewer(width=width, height=height, use_offscreen=True)
        
        mv = self.mv
        pose = pose.detach().cpu().numpy()
        face = face.detach().cpu().numpy()
        body_mesh = trimesh.Trimesh(vertices=pose, faces=face, vertex_colors=np.tile(colors['grey'], (6890, 1)))
        mv.set_static_meshes([body_mesh])
        body_image = mv.render(render_wireframe=False)
        write_success = cv2.imwrite(save_path, body_image)
        if not write_success:
            print(f"Failed to write image to {save_path}")
        
    def render_pose_robot(self, action: torch.Tensor, save_path: str, root_tf: Optional[torch.Tensor] = None):
        img = self.robot_vis(action, root_tf)
        write_success = cv2.imwrite(save_path, img)
        if not write_success:
            print(f"Failed to write image to {save_path}")

    def render_pose_amass_with_bm(self, body_parms: Dict[str, torch.Tensor], save_path: str, add_smpl_fits: bool = True, update_mv: bool = True):
        bm = self.get_pose_amass(body_parms['pose_body'][0:1], add_smpl_fits=add_smpl_fits)
        self.render_pose_amass(bm.v[0], bm.f, save_path, update_mv=update_mv)
        


    @torch.no_grad()
    def render_pose(self, pose: torch.Tensor, save_path: str, fps: torch.Tensor, name: str):
        filter = TrackingDataPostProcess(
            filter_cfg=FilterCfg(
                filter_sample_rate=fps.item(),
                filter_cutoff=fps.item() / 10.0,
                filter_order=2,
                device=self.fk.device
            )
        )
        # pose: (B, N, 66)
        assert pose.dim() == 3, "pose should be 3D tensor"
        assert pose.shape[0] == 1, "batch size should be 1 for rendering"
        # reshape pose to (B*N, 66)
        pose = pose.squeeze(0)
        num_splits = pose.shape[0] // self.max_single_batch
        if pose.shape[0] % self.max_single_batch != 0:
            num_splits += 1

        robot_vis_tf = torch.zeros((4, 4))
        robot_vis_tf[range(4), [1, 2, 0, 3]] = 1

        bm_save_path = os.path.join(save_path, "bm")
        fk_save_path = os.path.join(save_path, "fk")
        for i in range(num_splits):
            # print(f"Saving {i}/{num_splits} batches")
            start = i * self.max_single_batch
            end = min((i + 1) * self.max_single_batch, pose.shape[0])
            poses = pose[start:end][:, :66]
            pose_body = poses[:, 3:66]
            body_parms = self.get_pose_amass(pose_body)
            actions = self.poseformer(pose_body)
            actions = filter.filt(actions)
            for j, _ in enumerate(tqdm(range(end - start), desc = "Rendering to path: {}, batch: {}/{}".format(save_path, i, num_splits))):
                
                if not os.path.exists(bm_save_path):
                    os.makedirs(bm_save_path)
                
                if not os.path.exists(fk_save_path):
                    os.makedirs(fk_save_path)
                    
                save_idx = start + j
                self.render_pose_amass(body_parms.v[j], body_parms.f, os.path.join(bm_save_path, "%.5d.png" % save_idx), update_mv=True)
                self.render_pose_robot(actions[j], os.path.join(fk_save_path, "%.5d.png" % save_idx), root_tf=robot_vis_tf)

        # concatentate all images to a video
        video_name = os.path.join(save_path, "video.mp4")
        bm_images = glob(os.path.join(bm_save_path, "*.png"))
        fk_images = glob(os.path.join(fk_save_path, "*.png"))
        assert len(bm_images) == len(fk_images), "number of images should be equal"
        bm_images.sort()
        fk_images.sort()
        assert len(bm_images) > 0, "no images found"
        assert len(fk_images) > 0, "no images found"

        # calculate framesize
        bm_img = cv2.imread(bm_images[0])
        fk_img = cv2.imread(fk_images[0])
        h, w, _ = bm_img.shape
        h_fk, w_fk, _ = fk_img.shape
        w_new = int(w * h_fk / h)
        framesize = (w_new + w_fk, h_fk)

        # set cv2 video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename=video_name, fourcc=fourcc, fps=fps.item(), frameSize=framesize)

        for i, _ in enumerate(tqdm(range(len(bm_images)), desc = "Creating video")):
            bm_img = cv2.imread(bm_images[i])
            fk_img = cv2.imread(fk_images[i])
            # reshape to same height
            h, w, _ = bm_img.shape
            targ_h = fk_img.shape[0]
            bm_img = cv2.resize(bm_img, (int(w * targ_h / h), targ_h))
            img = np.hstack((bm_img, fk_img))
            out.write(img)

        # release video writer
        out.release()

        # remove bm and fk directories
        os.system(f"rm -r {shlex.quote(bm_save_path)}")
        os.system(f"rm -r {shlex.quote(fk_save_path)}")

    def render_pose_without_model(self, pose: torch.Tensor, action: torch.Tensor, save_path: str, fps: int):
        bm_save_path = os.path.join(save_path, "bm")
        fk_save_path = os.path.join(save_path, "fk")
        if not os.path.exists(bm_save_path):
            os.makedirs(bm_save_path)
        if not os.path.exists(fk_save_path):
            os.makedirs(fk_save_path)
        
        robot_vis_tf = torch.zeros((4, 4))
        robot_vis_tf[range(4), [1, 2, 0, 3]] = 1

        num_splits = pose.shape[0] // self.max_single_batch
        if pose.shape[0] % self.max_single_batch != 0:
            num_splits += 1

        for i in range(num_splits):
            start = i * self.max_single_batch
            end = min((i + 1) * self.max_single_batch, pose.shape[0])
            body_parms = self.get_pose_amass(pose_body=pose[start:end, 3:66])
            actions = action[start:end]
            for j, _ in enumerate(tqdm(range(end - start), desc = "Rendering to path{}, batch: {}/{}".format(save_path, i + 1, num_splits))):
                save_idx = start + j
                self.render_pose_amass(body_parms.v[j], body_parms.f, os.path.join(bm_save_path, "%.5d.png" % save_idx))
                self.render_pose_robot(actions[j], os.path.join(fk_save_path, "%.5d.png" % save_idx), root_tf=robot_vis_tf)

        # concatentate all images to a video
        video_name = os.path.join(save_path, "video.mp4")
        bm_images = glob(os.path.join(bm_save_path, "*.png"))
        fk_images = glob(os.path.join(fk_save_path, "*.png"))
        assert len(bm_images) == len(fk_images), "number of images should be equal"
        bm_images.sort()
        fk_images.sort()
        assert len(bm_images) > 0, "no images found"
        assert len(fk_images) > 0, "no images found"

        # calculate framesize
        bm_img = cv2.imread(bm_images[0])
        fk_img = cv2.imread(fk_images[0])
        h, w, _ = bm_img.shape
        h_fk, w_fk, _ = fk_img.shape
        w_new = int(w * h_fk / h)
        framesize = (w_new + w_fk, h_fk)

        # set cv2 video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename=video_name, fourcc=fourcc, fps=fps, frameSize=framesize)

        for i, _ in enumerate(tqdm(range(len(bm_images)), desc = "Creating video")):
            bm_img = cv2.imread(bm_images[i])
            fk_img = cv2.imread(fk_images[i])
            # reshape to same height
            h, w, _ = bm_img.shape
            targ_h = fk_img.shape[0]
            bm_img = cv2.resize(bm_img, (int(w * targ_h / h), targ_h))
            img = np.hstack((bm_img, fk_img))
            out.write(img)

        # release video writer
        out.release()

        # remove bm and fk directories
        os.system(f"rm -r {shlex.quote(bm_save_path)}")
        os.system(f"rm -r {shlex.quote(fk_save_path)}")

    def render_pose_without_model_urdf(self, action: torch.Tensor, save_path: str, fps: int, name: str = "video"):
        fk_save_path = os.path.join(save_path, "temp")
        if not os.path.exists(fk_save_path):
            os.makedirs(fk_save_path)
        
        robot_vis_tf = torch.zeros((4, 4))
        robot_vis_tf[range(4), [1, 2, 0, 3]] = 1

        num_splits = action.shape[0] // self.max_single_batch
        if action.shape[0] % self.max_single_batch != 0:
            num_splits += 1

        for i in range(num_splits):
            start = i * self.max_single_batch
            end = min((i + 1) * self.max_single_batch, action.shape[0])

            actions = action[start:end]
            for j, _ in enumerate(tqdm(range(end - start), desc = "Rendering to path{}, batch: {}/{}".format(save_path, i + 1, num_splits))):
                save_idx = start + j
                # self.render_pose_amass(body_parms.v[j], body_parms.f, os.path.join(bm_save_path, "%.5d.png" % save_idx))
                self.render_pose_robot(actions[j], os.path.join(fk_save_path, "%.5d.png" % save_idx), root_tf=robot_vis_tf)

        # concatentate all images to a video
        video_name = os.path.join(save_path, f"{name}.mp4")
        # bm_images = glob(os.path.join(bm_save_path, "*.png"))
        fk_images = glob(os.path.join(fk_save_path, "*.png"))
        # assert len(bm_images) == len(fk_images), "number of images should be equal"
        # bm_images.sort()
        fk_images.sort()
        # assert len(bm_images) > 0, "no images found"
        assert len(fk_images) > 0, "no images found"

        # calculate framesize
        # bm_img = cv2.imread(bm_images[0])
        fk_img = cv2.imread(fk_images[0])
        framesize = (fk_img.shape[1], fk_img.shape[0])

        # set cv2 video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename=video_name, fourcc=fourcc, fps=fps, frameSize=framesize)

        for i, _ in enumerate(tqdm(range(len(fk_images)), desc = "Creating video")):
            # bm_img = cv2.imread(bm_images[i])
            fk_img = cv2.imread(fk_images[i])
            # reshape to same height
            # h, w, _ = bm_img.shape
            # targ_h = fk_img.shape[0]
            # bm_img = cv2.resize(bm_img, (int(w * targ_h / h), targ_h))
            # img = np.hstack((bm_img, fk_img))
            out.write(fk_img)

        # release video writer
        out.release()

        # remove bm and fk directories
        # os.system(f"rm -r {bm_save_path}")
        os.system(f"rm -r {shlex.quote(fk_save_path)}")

    def main(self):
        if os.path.exists(self.save_path):
            os.system(f"rm -r {self.save_path}")
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=4)
        for i, data in enumerate(tqdm(dataloader, desc = "Rendering")):
            pose = data['poses'].to(self.fk.device)
            fps = data['fps']
            name = data['title'][0]
            self.render_pose(pose, os.path.join(self.save_path, name), fps, name)
