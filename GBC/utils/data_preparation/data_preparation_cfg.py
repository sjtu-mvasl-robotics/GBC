import sys
import os

from omegaconf import MISSING
# PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# sys.path.append(PROJECT_DIR)

# from utils.base import configclass
from GBC.utils.base import configclass
from typing import List, Optional, Callable
import torch

@configclass
class BaseCfg:
    """Base configuration."""
    smplh_model_path: str = MISSING
    dmpls_model_path: str = MISSING
    urdf_path: str = MISSING
    device: str = "cuda"


@configclass
class AMASSDatasetCfg(BaseCfg):
    """AMASS dataset configuration."""
    root_dir: str = MISSING
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
    mapping_table = MISSING

    offset_map = None
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
    poseformer_model_path: str = MISSING
    save_path: str = MISSING
    max_single_batch: int = 512
    