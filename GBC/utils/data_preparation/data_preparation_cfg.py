import sys
import os
# PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# sys.path.append(PROJECT_DIR)

# from utils.base import configclass
from GBC.utils.base import configclass
from typing import List, Optional, Callable
import torch

@configclass
class BaseCfg:
    """Base configuration."""
    smplh_model_path="/home/turin/sjtu/dataset/smplh/male/model.npz"
    dmpls_model_path="/home/turin/sjtu/dataset/dmpls/male/model.npz"
    urdf_path="/home/rl/TurinHumanoid/TurinHumanoidV2/robot_model/full_dof/urdf/full_dof_al_v2.urdf"
    # root_dir='/home/yyf/dataset'
    device="cuda"


@configclass
class AMASSDatasetCfg(BaseCfg):
    """AMASS dataset configuration."""
    root_dir: str = "/home/turin/sjtu/dataset/amass"
    num_betas: int = 16
    num_dmpls: int = 8
    load_hands: bool = False
    secondary_dir: Optional[str] = None

@configclass
class AMASSDatasetInterpolateCfg(AMASSDatasetCfg):
    """AMASS dataset interpolation configuration."""
    interpolate_fps: int = 50
    seq_min_duration: Optional[int] = None
    specialize_dir: Optional[List[str]] = None

@configclass
class AMASSDatasetSingleFrameCfg(AMASSDatasetCfg):
    """AMASS dataset single frame configuration."""
    transform: Optional[Callable] = None

@configclass
class FilterCfg(BaseCfg):
    """Actions and velocities filter configuration."""
    filter_cutoff: float = 5
    filter_sample_rate: float = 50
    filter_order: int = 2



@configclass
class RobotKinematicsCfg(BaseCfg):
    mapping_table = {
        'Pelvis': 'base_link',
        'L_Hip': 'Link_hip_l_yaw',
        'R_Hip': 'Link_hip_r_yaw',
        'L_Knee': 'Link_knee_l_pitch',
        'R_Knee': 'Link_knee_r_pitch',
        'L_Ankle': 'Link_ankle_l_pitch',
        'R_Ankle': 'Link_ankle_r_pitch',
        'L_Toe': 'Link_ankle_l_roll',
        'R_Toe': 'Link_ankle_r_roll',
        'L_Shoulder': 'Link_arm_l_01',
        'R_Shoulder': 'Link_arm_r_01',
        'L_Elbow': 'Link_arm_l_04',
        'R_Elbow': 'Link_arm_r_04',
        'L_Wrist': 'Link_arm_l_06',
        'R_Wrist': 'Link_arm_r_06',
        'Head': 'Link_head_yaw'
    }

    offset_map = {
            'Link_ankle_l_roll': torch.Tensor([0.15, 0, -0.07]),
            'Link_ankle_r_roll': torch.Tensor([0.15, 0, -0.07]),
        }

    def __init__(self):
        if hasattr(self, 'offset_map'):
            for key, val in self.offset_map.items():
                self.offset_map[key] = val.to(self.device)

@configclass
class AMASSActionConverterCfg(AMASSDatasetInterpolateCfg, RobotKinematicsCfg):
    """AMASS action converter configuration."""
    pose_transformer_path: str = ""
    batch_size: int = 512
    export_path: str = ""
    filter: FilterCfg = FilterCfg()
    visualize: bool = True
    smpl_fits_dir: str = ""

@configclass
class BodyModelCfg(BaseCfg):
    pass

@configclass
class PoseRendererCfg(RobotKinematicsCfg, BodyModelCfg, AMASSDatasetCfg):
    poseformer_model_path: str = "models/15_08_55_epoch_235.pt"
    save_path: str = "outputs/pose_render"
    max_single_batch: int = 512
    