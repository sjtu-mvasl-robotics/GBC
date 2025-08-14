from GBC.utils.base.base_fk import RobotKinematics
from human_body_prior.body_model.body_model import BodyModel
from typing import List, Tuple, Dict, Any, Union, Optional
from smplx.body_models import SMPLH
from GBC.utils.data_preparation.amass_loader import AMASSDatasetSingleFrame, OptimizedAMASSDatasetSingleFrame, InMemoryAMASSDatasetSingleFrame
from tqdm import tqdm
from copy import deepcopy
import os
import time
import numpy as np
from GBC.utils.base.math_utils import swap_order, symmetry_smplh_pose, batch_angle_axis_to_ypr
from GBC.utils.data_preparation.data_preparation_cfg import BaseCfg
from GBC.utils.data_preparation.render_track_videos import PoseRenderer
from GBC.utils.data_preparation.pose_transformer import PoseTransformer, PoseTransformerV2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad
from GBC.utils.data_preparation.robot_flip_left_right import RobotFlipLeftRight
from torch.nn.functional import huber_loss
import wandb
from dataclasses import dataclass


@dataclass
class LossCoefficients:
    """
    Class to manage all loss coefficients in the training process
    """
    # Main loss coefficients
    main_loss: float = 5000.0
    aux_loss: float = 1.0
    
    # Regularization losses
    out_of_range_loss: float = 1000.0
    high_value_action_loss: float = 1000.0
    
    # Task-specific losses
    direct_mapping_loss: float = 1000.0
    symmetry_loss: float = 1000.0
    reference_action_loss: float = 5000.0
    bone_direction_loss: float = 1000.0
    
    # Progressive disturbance loss (will be multiplied by progressive coefficient)
    disturbance_base: float = 1.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging"""
        return {
            'main_loss': self.main_loss,
            'aux_loss': self.aux_loss,
            'out_of_range_loss': self.out_of_range_loss,
            'high_value_action_loss': self.high_value_action_loss,
            'direct_mapping_loss': self.direct_mapping_loss,
            'symmetry_loss': self.symmetry_loss,
            'reference_action_loss': self.reference_action_loss,
            'bone_direction_loss': self.bone_direction_loss,
            'disturbance_base': self.disturbance_base,
        }


class TrainingLogger:
    """
    Training logger with wandb integration and improved formatting
    """
    def __init__(self, use_wandb: bool = True, project_name: str = "pose_transformer_training"):
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project=project_name)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics to wandb and format for console output"""
        if self.use_wandb:
            wandb.log({f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}, step=step)
    
    def log_hyperparameters(self, config: Dict[str, Any]):
        """Log hyperparameters"""
        if self.use_wandb:
            wandb.config.update(config)
    
    def finish(self):
        """Finish logging"""
        if self.use_wandb:
            wandb.finish()
    
    @staticmethod
    def format_metrics_table(metrics: Dict[str, float], title: str = "", width: int = 80) -> str:
        """Format metrics in a nice table format"""
        lines = []
        lines.append("=" * width)
        if title:
            lines.append(f"{title:^{width}}")
            lines.append("=" * width)
        
        # Group metrics by category
        main_metrics = {}
        loss_metrics = {}
        other_metrics = {}
        
        for key, value in metrics.items():
            if 'loss' in key.lower():
                loss_metrics[key] = value
            elif key in ['lr', 'epoch', 'disturbance_coeff']:
                main_metrics[key] = value
            else:
                other_metrics[key] = value
        
        # Print main metrics first
        if main_metrics:
            for key, value in main_metrics.items():
                if key == 'lr':
                    lines.append(f"│ {key.upper():<20} │ {value:.2e} │")
                elif key == 'disturbance_coeff':
                    lines.append(f"│ {key:<20} │ {value:.2f} │")
                else:
                    lines.append(f"│ {key.upper():<20} │ {value} │")
            lines.append("─" * width)
        
        # Print loss metrics
        if loss_metrics:
            for key, value in loss_metrics.items():
                formatted_key = key.replace('_', ' ').title()
                lines.append(f"│ {formatted_key:<20} │ {value:.4f} │")
        
        # Print other metrics
        if other_metrics:
            lines.append("─" * width)
            for key, value in other_metrics.items():
                formatted_key = key.replace('_', ' ').title()
                lines.append(f"│ {formatted_key:<20} │ {value:.4f} │")
        
        lines.append("=" * width)
        return "\n".join(lines)


class DatasetWithIK(Dataset):
    def __init__(self, subset, ik_tensor):
        """
        Initialize the dataset with IK results.
        
        Args:
            subset: The subset dataset (e.g., from random_split)
            ik_tensor: The full IK tensor for the entire dataset
        """
        self.subset = subset
        self.ik_tensor = ik_tensor
        self.indices = subset.indices  # Global indices from the original dataset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        """
        Fetch a data sample and its corresponding IK result.
        
        Args:
            idx: Local index within the subset
            
        Returns:
            tuple: (original_data, ik_action)
        """
        data = self.subset[idx]
        global_idx = self.indices[idx]
        ik_action = self.ik_tensor[global_idx]
        return data, ik_action


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup + Cosine Annealing Learning Rate Scheduler
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress)) 
                    for base_lr in self.base_lrs]


class PoseFormerTrainer:
    '''
    PoseFormer Trainer

    Trainer for PoseFormer

    Args:
        urdf_path (str): Path to the URDF file
        dataset_path (str): Path to the dataset
        mapping_table (Dict[str, str]): Mapping table for smplh -> humanoid links
        load_hands (bool): Load hands or not
        smplh_model_path (str): Path to the SMPLH model
        dmpls_model_path (str): Path to the DMPLs model
        smpl_fits_dir (str): Path to the SMPL fits directory
        batch_size (int): Batch size
        train_size (float): Train size (default: 0.8)
        num_workers (int): Number of workers
        save_dir (str): Path to the save directory
        device (str): Device
        use_renderer (bool): Use renderer or not (default: False). Using renderer for visualization may slow down the training process.
        loss_coefficients (LossCoefficients): Loss coefficients configuration
        use_wandb (bool): Whether to use wandb for logging
        wandb_project (str): Wandb project name

    '''
    def __init__(self,
                 urdf_path: str,
                 dataset_path: str,
                 mapping_table: Dict[str, str],
                 smplh_model_path: str,
                 dmpls_model_path: str,
                 smpl_fits_dir: str,
                 load_hands: bool = False,
                 secondary_dir: Optional[str] = None,
                 sample_steps: int = 20,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 train_size: float = 0.8,
                 save_dir: str = '.',
                 device: str = 'cuda',
                 use_renderer: bool = False,
                 use_reference_actions: bool = False,
                 loss_coefficients: Optional[LossCoefficients] = None,
                 use_wandb: bool = True,
                 wandb_project: str = "pose_transformer_training",
                ):
        assert 0 < train_size < 1, 'Train size must be between 0 and 1'
        self.use_reference_actions = use_reference_actions
        self.device = device
        self.num_workers = num_workers
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.fk = RobotKinematics(urdf_path, device=device)
        self.num_dofs = self.fk.num_dofs
        self.mapping_table = mapping_table
        self.load_hands = load_hands
        
        # Initialize loss coefficients
        self.loss_coeffs = loss_coefficients if loss_coefficients is not None else LossCoefficients()
        
        # Initialize logger
        self.logger = TrainingLogger(use_wandb=use_wandb, project_name=wandb_project)
        
        # dataset = AMASSDatasetSingleFrame(dataset_path, load_hands=load_hands, secondary_dir=secondary_dir, sample_steps=sample_steps)
        # dataset = OptimizedAMASSDatasetSingleFrame(dataset_path, load_hands=load_hands, secondary_dir=secondary_dir, sample_steps=sample_steps)
        dataset = InMemoryAMASSDatasetSingleFrame(dataset_path, load_hands=load_hands, secondary_dir=secondary_dir, sample_steps=sample_steps)
        train_size = int(train_size * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # self.model = PoseTransformer(num_actions=self.num_dofs, load_hands=load_hands).to(device)
        # self.model = SpatialMoETransformer(load_hands=load_hands, robot_actions=self.num_dofs).to(device)
        self.model = PoseTransformer(load_hands=load_hands, num_actions=self.num_dofs).to(device)
        self.flip_left_right = None
        # self.model = PoseMLP(load_hands=load_hands,num_actions=self.num_dofs).to(device)

        # Check if model supports auxiliary loss
        self.model_supports_aux_loss = False

        self.smplh_body_names = ['Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist']
        self.smplh_extend_names = ['L_Index1', 'L_Index2', 'L_Index3', 'L_Middle1', 'L_Middle2', 'L_Middle3', 'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 'L_Ring1', 'L_Ring2', 'L_Ring3', 'L_Thumb1', 'L_Thumb2', 'L_Thumb3', 'R_Index1', 'R_Index2', 'R_Index3', 'R_Middle1', 'R_Middle2', 'R_Middle3', 'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 'R_Ring1', 'R_Ring2', 'R_Ring3', 'R_Thumb1', 'R_Thumb2', 'R_Thumb3']
        self.load_body_model(smplh_model_path, dmpls_model_path, smpl_fits_dir)

        self.fk.set_target_links(list(self.mapping_table.values()))

        # Initialize bone connections for efficient bone direction loss computation
        self._init_bone_connections()

        self.last_checkpoint = None
        self.last_epoch = 0

        self.set_root_orient()

        self.direct_joint_map = {}

        if use_renderer:
            self.renderer = PoseRenderer(
                urdf_path=urdf_path,
                dataset_path=dataset_path,
                mapping_table=mapping_table,
                smplh_model_path=smplh_model_path,
                dmpls_model_path=dmpls_model_path,
                poseformer_model_path=None,
                save_path=None,
                device=device,
                without_model=True
            )

            self.renderer.set_smpl_fits(self.fit_data)

        else:
            self.renderer = None

    def set_root_orient(self):
        '''
        Set root orientation
        '''
        from scipy.spatial.transform import Rotation as R
        root_orient = R.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec()
        self.root_orient = torch.tensor(root_orient, dtype=torch.float32, requires_grad=False).to(self.device)


    def load_body_model(self, smplh_model_path: str, dmpls_model_path: str, smpl_fits_dir: str):
        '''
        Load body model

        Args:
            smplh_model_path (str): Path to the SMPLH model
            dmpls_model_path (str): Path to the DMPLs model
            smpl_fits_dir (str): Path to the SMPL fits directory
        '''
        assert smpl_fits_dir.endswith('.pt'), 'SMPL fits directory must be a .pt file'

        self.fit_data = torch.load(smpl_fits_dir, map_location=self.device)
        for key in self.fit_data.keys():
            if isinstance(self.fit_data[key], torch.Tensor):
                self.fit_data[key] = self.fit_data[key].requires_grad_(False)

        self.body_model = BodyModel(bm_fname=smplh_model_path, dmpl_fname=dmpls_model_path, num_betas=16, num_dmpls=8).to(self.device)

    def set_direct_joint_map_body(self, direct_joint_map: Dict[str, Dict[str, str]]):
        '''
        Set direct joint map_body

        Args:
            direct_joint_map (Dict[str, Dict[str, str]]): Direct joint map


        Format:
        Dict{
            SMPLH joint name: Dict{
                'yaw'/ 'pitch'/ 'roll': Humanoid joint name
                }
            }
        '''
        # order: yaw, pitch, roll
        angles = ['yaw', 'pitch', 'roll']
        pose_body_indices = []
        action_indices = []
        for smplh_joint, humanoid_joint in direct_joint_map.items():
            for angle_name, humanoid_joint_name in humanoid_joint.items():
                pose_body_indices.append([self.smplh_body_names.index(smplh_joint) - 1, angles.index(angle_name)])
                action_indices.append(self.fk.dof_names.index(humanoid_joint_name))

        if "body" not in self.direct_joint_map.keys():
            self.direct_joint_map["body"] = (pose_body_indices, action_indices)
        else:
            self.direct_joint_map["body"] = (self.direct_joint_map["body"][0] + pose_body_indices, self.direct_joint_map["body"][1] + action_indices)

    def set_flip_left_right(self, flip_left_right: RobotFlipLeftRight):
        '''
        Set flip left right
        '''
        self.flip_left_right = flip_left_right

    def _model_forward(self, data):
        """
        Forward pass through model with proper output handling
        
        Args:
            data: Input data tensor
            
        Returns:
            tuple: (actions, aux_loss) where aux_loss is 0 if not supported
        """
        if self.model_supports_aux_loss:
            return self.model(data)
        else:
            actions = self.model(data)
            aux_loss = torch.tensor(0.0, device=self.device)
            return actions, aux_loss

    def _get_progressive_disturbance_coeff(self, epoch, total_epochs, min_coeff=1.0, max_coeff=50.0):
        """
        Get progressive disturbance loss coefficient that increases during training
        
        Args:
            epoch: Current epoch
            total_epochs: Total training epochs
            min_coeff: Minimum coefficient
            max_coeff: Maximum coefficient
            
        Returns:
            float: Current disturbance coefficient
        """
        progress = min(epoch / total_epochs, 1.0)
        return min_coeff + (max_coeff - min_coeff) * progress

    def direct_mapping_loss(self, actions: torch.Tensor, pose_body: torch.Tensor, pose_hand: torch.Tensor = None):
        '''
        Direct mapping loss

        Args:
            actions (torch.Tensor): Actions
            pose_body (torch.Tensor): Body pose
            pose_hand (torch.Tensor): Hand pose

        Returns:
            Direct mapping loss (torch.Tensor)
        '''
        if "body" not in self.direct_joint_map.keys():
            return torch.tensor(0.0).to(self.device)
        pose_body_indices, action_indices = self.direct_joint_map["body"]
        pose_body_indices = torch.tensor(pose_body_indices).to(self.device)
        action_indices = torch.tensor(action_indices).to(self.device)
        pose_body = pose_body.reshape(-1, 21, 3)
        ypr = batch_angle_axis_to_ypr(pose_body)
        ypr = ypr[:, pose_body_indices[:, 0], pose_body_indices[:, 1]]
        ypr = ypr.reshape(-1, len(pose_body_indices))
        actions = actions[:, action_indices]
        loss = F.mse_loss(actions, ypr.detach())
        if self.load_hands and pose_hand is not None:
            pose_hand_indices, action_indices = self.direct_joint_map["hand"]
            pose_hand_indices = torch.tensor(pose_hand_indices).to(self.device)
            action_indices = torch.tensor(action_indices).to(self.device)
            pose_hand = pose_hand.reshape(-1, 30, 3)
            ypr = batch_angle_axis_to_ypr(pose_hand)
            ypr = ypr[:, pose_hand_indices[:, 0], pose_hand_indices[:, 1]]
            ypr = ypr.reshape(-1, len(pose_hand_indices))
            actions = actions[:, action_indices]
            loss += F.mse_loss(actions, ypr.detach())
        return loss


    def save(self, path: str, optimizer: torch.optim.Optimizer):
        '''
        Save the model

        Args:
            path (str): Path to save the model
            optimizer (torch.optim.Optimizer): Optimizer
        '''
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': self.last_epoch
        }, path)

    def load(self, path: str, optimizer: torch.optim.Optimizer):
        '''
        Load the model

        Args:
            path (str): Path to load the model
            optimizer (torch.optim.Optimizer): Optimizer
        '''
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        self.last_epoch = checkpoint['epoch']
        self.last_checkpoint = path

    def create_model_input(self, pose_body: torch.Tensor, pose_hand: torch.Tensor = None, trans: torch.Tensor = None, override_root_orient: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        '''
        Create model input

        Args:
            pose_body (torch.Tensor): Body pose
            pose_hand (torch.Tensor): Hand pose
            trans (torch.Tensor): Translation
            override_root_orient (Optional[torch.Tensor]): Override root orientation

        Returns:
            Model input (Dict[str, torch.Tensor])
        '''
        if override_root_orient is None:
            root_orient = deepcopy(self.root_orient)
            root_orient = root_orient.repeat(pose_body.shape[0], 1)
        else:
            root_orient = override_root_orient
        
        body_parms = {
            'root_orient': root_orient,
            'pose_body': pose_body,
            'betas': self.fit_data['beta'],
            'dmpls': self.fit_data['dmpls']
        }
        if pose_hand is not None:
            body_parms['pose_hand'] = pose_hand
        if trans is not None:
            body_parms['trans'] = trans
        return body_parms
    
    @staticmethod
    def disturbance_loss(actions: torch.Tensor, actions_disturbed: torch.Tensor, disturbance: torch.Tensor):
        '''
        Disturbance loss

        Args:
            actions (torch.Tensor): Actions (Bxdim_actions)
            actions_disturbed (torch.Tensor): Disturbed actions (Bxdim_actions)
            disturbance (torch.Tensor): Disturbance (Bxdim_input)

        Returns:
            Disturbance loss (torch.Tensor)
        '''
        disturb_norm = torch.norm(disturbance) # shape: (B)
        actions_norm = torch.norm(actions - actions_disturbed) # shape: (B)
        return actions_norm / disturb_norm
    
    @staticmethod
    def out_of_range_loss(actions: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor):
        '''
        Out of range loss

        Args:
            actions (torch.Tensor): Actions (Bxdim_actions)
            min_val (torch.Tensor): Minimum value (dim_actions)
            max_val (torch.Tensor): Maximum value (dim_actions)

        Returns:
            Out of range loss (torch.Tensor)

        '''
        out_range_actions = torch.where(actions < min_val, min_val - actions, torch.zeros_like(actions)) + torch.where(actions > max_val, actions - max_val, torch.zeros_like(actions))
        return torch.norm(out_range_actions)
    
    @staticmethod
    def high_value_action_loss(actions: torch.Tensor, max_val: torch.Tensor):
        '''
        High value action loss

        Args:
            actions (torch.Tensor): Actions (Bxdim_actions)
            max_val (torch.Tensor): Maximum value (dim_actions)

        Returns:
            High value action loss (torch.Tensor)
        '''
        return torch.norm(torch.where(torch.abs(actions) > max_val, actions - max_val, torch.zeros_like(actions)))
    
    def _init_bone_connections(self):
        '''
        Initialize bone connections for efficient bone direction loss computation
        '''
        mapping_keys = list(self.mapping_table.keys())
        
        bone_pairs = [
            ('L_Hip', 'L_Knee'),
            ('R_Hip', 'R_Knee'),
            ('L_Knee', 'L_Ankle'),
            ('R_Knee', 'R_Ankle'),
            ('L_Shoulder', 'L_Elbow'),
            ('R_Shoulder', 'R_Elbow'),
            ('L_Elbow', 'L_Wrist'),
            ('R_Elbow', 'R_Wrist'),
        ]
        
        start_indices = []
        end_indices = []
        
        for start_joint, end_joint in bone_pairs:
            if start_joint in mapping_keys and end_joint in mapping_keys:
                start_indices.append(mapping_keys.index(start_joint))
                end_indices.append(mapping_keys.index(end_joint))
        
        # Convert to tensors and store as class members
        if start_indices:
            self.bone_start_indices = torch.tensor(start_indices, device=self.device, dtype=torch.long)
            self.bone_end_indices = torch.tensor(end_indices, device=self.device, dtype=torch.long)
            self.num_bones = len(start_indices)
        else:
            self.bone_start_indices = None
            self.bone_end_indices = None
            self.num_bones = 0

    def bone_direction_loss(self, robot_joints: torch.Tensor, gt_joints: torch.Tensor):
        '''
        Optimized bone direction loss - enforces that bone directions match between robot and ground truth
        
        Args:
            robot_joints (torch.Tensor): Robot joint positions (B, num_joints, 3)
            gt_joints (torch.Tensor): Ground truth joint positions (B, num_joints, 3)
            
        Returns:
            Bone direction loss (torch.Tensor)
        '''
        if self.num_bones == 0:
            return torch.tensor(0.0, device=robot_joints.device)
        
        # Vectorized bone vector computation
        # robot_start_joints: (B, num_bones, 3)
        # robot_end_joints: (B, num_bones, 3)
        robot_start_joints = robot_joints[:, self.bone_start_indices, :]  # (B, num_bones, 3)
        robot_end_joints = robot_joints[:, self.bone_end_indices, :]      # (B, num_bones, 3)
        gt_start_joints = gt_joints[:, self.bone_start_indices, :]        # (B, num_bones, 3)
        gt_end_joints = gt_joints[:, self.bone_end_indices, :]            # (B, num_bones, 3)
        
        # Calculate bone vectors for all bones at once
        robot_bone_vecs = robot_end_joints - robot_start_joints  # (B, num_bones, 3)
        gt_bone_vecs = gt_end_joints - gt_start_joints          # (B, num_bones, 3)
        
        # Compute norms with tolerance to avoid division by zero
        robot_bone_norms = torch.norm(robot_bone_vecs, dim=-1, keepdim=True)  # (B, num_bones, 1)
        gt_bone_norms = torch.norm(gt_bone_vecs, dim=-1, keepdim=True)        # (B, num_bones, 1)
        
        # Add tolerance to avoid division by zero
        eps = 1e-8
        robot_bone_norms = torch.clamp(robot_bone_norms, min=eps)
        gt_bone_norms = torch.clamp(gt_bone_norms, min=eps)
        
        # Normalize to unit vectors
        robot_bone_units = robot_bone_vecs / robot_bone_norms  # (B, num_bones, 3)
        gt_bone_units = gt_bone_vecs / gt_bone_norms          # (B, num_bones, 3)
        
        # Calculate cosine similarity (dot product of unit vectors)
        cos_sim = torch.sum(robot_bone_units * gt_bone_units, dim=-1)  # (B, num_bones)
        
        # Clamp to avoid numerical issues
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Loss is 1 - cos_sim, averaged over all bones and batch
        bone_loss = torch.mean(1.0 - cos_sim)
        
        return bone_loss

    def precompute_and_save_full_ik(self, dataset, save_path, batch_size=128, max_iter=500):
        """
        Precompute and save IK results for the entire dataset in a single file.
        
        Args:
            dataset: The full dataset
            save_path: Path to save the IK results tensor
            batch_size: Batch size for IK computation
            max_iter: Maximum iterations for IK solver
        """
        if os.path.exists(save_path):
            print(f"IK results already exist at {save_path}. Skipping computation.")
            return
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_ik_results = []
        for batch in tqdm(loader, desc="Precomputing IK"):
            data = batch.to(self.device)
            pose_body = data[:, 3:66]
            pose_hand = data[:, 66:] if self.load_hands else None
            body_parms = self.create_model_input(pose_body=pose_body, pose_hand=pose_hand)
            gt_joints = self.body_model(**body_parms).Jtr.detach()
            reshape_idx = [self.smplh_body_names.index(name) for name in self.mapping_table.keys()]
            if self.load_hands:
                reshape_idx += [self.smplh_extend_names.index(name) + len(self.smplh_body_names) for name in self.mapping_table.keys()]
            gt_joints = gt_joints[:, reshape_idx, :]
            gt_joints -= gt_joints[:, :1, :]
            actions = self.fk.inverse_kinematics(gt_joints, max_iter=max_iter, verbose=True, log_interval=100).cpu()
            all_ik_results.append(actions)
        all_ik_results = torch.cat(all_ik_results, dim=0)
        torch.save(all_ik_results, save_path)
        print(f"Saved IK results to {save_path}")
        torch.cuda.empty_cache()
        del all_ik_results

    import random

    def validate_ik_dataset(self, dataset, val_nums: int = 5):
        """
        Validate the IK dataset by checking if the FK results for precomputed actions match the global positions.

        Args:
            dataset: The dataset to validate (instance of DatasetWithIK).
            val_nums: Number of random samples to validate (default: 5).
        """
        import random
        with torch.no_grad():  # Disable gradient computation for validation
            # Ensure we don't request more samples than available
            num_samples = min(val_nums, len(dataset))
            selected_indices = random.sample(range(len(dataset)), num_samples)

            for idx in selected_indices:
                # Retrieve sample data and add batch dimension
                original_data, ik_action = dataset[idx]
                original_data = original_data.unsqueeze(0).to(self.device)
                ik_action = ik_action.unsqueeze(0).to(self.device)

                # Extract pose parameters and compute ground truth joints
                pose_body = original_data[:, 3:66]  # Body pose parameters
                pose_hand = original_data[:, 66:] if self.load_hands else None  # Hand pose if applicable
                body_parms = self.create_model_input(pose_body=pose_body, pose_hand=pose_hand)
                gt_joints = self.body_model(**body_parms).Jtr  # Shape: (1, num_joints, 3)

                # Select relevant joints based on mapping table
                reshape_idx = [self.smplh_body_names.index(name) for name in self.mapping_table.keys()]
                if self.load_hands:
                    reshape_idx += [self.smplh_extend_names.index(name) + len(self.smplh_body_names) 
                                for name in self.mapping_table.keys()]
                gt_joints = gt_joints[:, reshape_idx, :]
                gt_joints = gt_joints - gt_joints[:, :1, :]  # Center joints relative to root

                # Compute FK positions from IK actions
                fk_output = self.fk(ik_action)  # Shape: (1, num_target_links, 3)
                fk_output = fk_output - fk_output[:, :1, :]  # Center FK positions

                # Calculate differences between FK and ground truth positions
                difference = torch.norm(fk_output - gt_joints, dim=-1)  # Shape: (1, num_joints)
                mean_diff = difference.mean().item()
                max_diff = difference.max().item()

                # Print validation results for this sample
                print(f"Sample {idx}: Mean difference: {mean_diff:.6f}, Max difference: {max_diff:.6f}")
                if self.renderer is not None:
                    robot_vis_tf = torch.zeros((4, 4)).to(self.device)
                    robot_vis_tf[range(4), [1, 2, 0, 3]] = 1
                    save_dir = f'{self.save_dir}/figs'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    robot_save_dir = f'{save_dir}/robot_ikval_{idx}.png'
                    amass_save_dir = f'{save_dir}/amass_ikval_{idx}.png'
                    self.renderer.render_pose_robot(action = ik_action[0].detach(), save_path=robot_save_dir, root_tf=robot_vis_tf)
                    print("render robot to", robot_save_dir)
                    self.renderer.render_pose_amass_with_bm(body_parms=body_parms, save_path=amass_save_dir)
                    print("render amass to", amass_save_dir)
                    self.renderer.render_pose_amass(pose=gt_joints[0], face=self.body_model.faces, save_path=amass_save_dir, update_mv=True)

                    # concatenate the images
                    os.system(f'convert {robot_save_dir} {amass_save_dir} +append {save_dir}/robot_amass_ikval_{idx}.png')
                    os.system(f'rm {robot_save_dir} {amass_save_dir}')



    def train(self, 
              epochs: int = 100,
              lr: float = 1.0e-3,
              min_lr: float = 1.0e-5,
              warmup_epochs: int = 10,
              lr_scheduler: torch.optim.lr_scheduler = None,  # Will be overridden
              validation_interval: int = 1,
              save_interval: int = 100,
              optimizer: torch.optim.Optimizer = torch.optim.AdamW,
              criterion: torch.nn.Module = torch.nn.MSELoss(),
              loss: nn.Module = nn.MSELoss(),
              load: bool = False,
              load_path: str = '',
              joint_weights: Optional[torch.Tensor] = None,
              visualize: bool = False,
              save_figs: bool = False,
              apply_noise: bool = True,
              apply_symmetry: bool = False,
              update_existing_ik: bool = False,
              ik_batch_size: int = 256,
              apply_noise_after_epoch: int = 0,
              disturbance_min_coeff: float = 1.0,
              disturbance_max_coeff: float = 50.0,
              **kwargs
                ):
        '''
        Train the model

        Args:
            epochs (int): Number of epochs
            lr (float): Learning rate
            min_lr (float): Minimum learning rate
            warmup_epochs (int): Number of warmup epochs
            lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler
            validation_interval (int): Validation interval
            save_interval (int): Save interval
            optimizer (torch.optim.Optimizer): Optimizer
            criterion (torch.nn.Module): Criterion
            loss (nn.Module): Loss function
            load (bool): Load model or not
            load_path (str): Load path
            joint_weights (Optional[torch.Tensor]): Joint weights
            visualize (bool): Visualize or not
            save_figs (bool): Save figures or not
            apply_noise (bool): Apply noise or not
            apply_symmetry (bool): Apply symmetry or not
            update_existing_ik (bool): Update existing IK or not
            disturbance_min_coeff (float): Minimum disturbance loss coefficient
            disturbance_max_coeff (float): Maximum disturbance loss coefficient
        '''

        # Log hyperparameters
        config = {
            'epochs': epochs,
            'lr': lr,
            'min_lr': min_lr,
            'warmup_epochs': warmup_epochs,
            'batch_size': self.batch_size,
            'apply_noise': apply_noise,
            'apply_symmetry': apply_symmetry,
            'disturbance_min_coeff': disturbance_min_coeff,
            'disturbance_max_coeff': disturbance_max_coeff,
            'model_type': type(self.model).__name__,
            'loss_coefficients': self.loss_coeffs.to_dict(),
        }
        self.logger.log_hyperparameters(config)

        if self.use_reference_actions:
            full_dataset = self.train_dataset.dataset  # Access the full dataset
            ik_save_path = os.path.join(self.save_dir, 'ik_results.pt')
            if not os.path.exists(ik_save_path) or update_existing_ik:
                if os.path.exists(ik_save_path):
                    print("Existing IK results found, but update_existing_ik is set. Recomputing IK...")
                    os.remove(ik_save_path)
                print("Computing IK for the full dataset...")
                self.precompute_and_save_full_ik(full_dataset, ik_save_path, batch_size=ik_batch_size, max_iter=1500)
            print("Loading IK results...")
            ik_tensor = torch.load(ik_save_path, map_location='cpu')
            self.train_dataset = DatasetWithIK(self.train_dataset, ik_tensor)
            self.test_dataset = DatasetWithIK(self.test_dataset, ik_tensor)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            print("Validating the IK results...")
            print("Training dataset:")
            self.validate_ik_dataset(self.train_dataset)
            print("Validation dataset:")
            self.validate_ik_dataset(self.test_dataset)
        
        # Optimized AdamW configuration
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False
        )
        
        # Warmup + Cosine Annealing scheduler
        scheduler = WarmupCosineAnnealingLR(
            optimizer, 
            warmup_epochs=warmup_epochs, 
            total_epochs=epochs, 
            min_lr=min_lr
        )

        if load:
            self.load(load_path, optimizer)

        if joint_weights is not None:
            assert joint_weights.shape[0] == len(self.mapping_table), 'Joint weights must have the same length as the mapping table'
            if joint_weights.device != self.device:
                joint_weights = joint_weights.to(self.device)

        
        
        for epoch in range(self.last_epoch, epochs):
            current_apply_noise = apply_noise and (epoch >= apply_noise_after_epoch)
            
            # Get progressive disturbance coefficient
            disturbance_coeff = self._get_progressive_disturbance_coeff(
                epoch, epochs, disturbance_min_coeff, disturbance_max_coeff
            )

            # train
            self.model.train()
            total_loss = 0.0
            total_main_loss = 0.0  # Add main loss tracking
            total_out_of_range_loss = 0.0
            total_high_value_action_loss = 0.0 # penalize actions over pi
            total_ref_action_loss = 0.0
            total_aux_loss = 0.0
            total_disturbance_loss = 0.0
            total_symmetry_loss = 0.0
            total_direct_mapping_loss = 0.0
            total_position_error = 0.0  # Add position error tracking
            total_max_position_error = 0.0  # Add max position error tracking
            total_grad_norm = 0.0  # Add gradient norm tracking
            
            for i, data in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training')):
                if self.use_reference_actions:
                    data, ref_actions = data
                    ref_actions = ref_actions.to(self.device)
                else:
                   ref_actions = None
                data = data.to(self.device)
                optimizer.zero_grad()
                root_orient = data[:, :3]
                pose_body = data[:, 3:66]
                pose_hand = data[:, 66:] if self.load_hands else None

                body_parms = self.create_model_input(pose_body=pose_body, pose_hand=pose_hand)
                data = data[:, 3:] # remove root orientation
                
                actions, aux_loss = self._model_forward(data)
                total_aux_loss += aux_loss.item()

                
                output = self.fk(actions, **kwargs) # shape: (N, num_joints, 3)
                output = output - output[:, 0:1, :]

                gt = self.body_model(**body_parms).Jtr # shape: (N, num_total_joints, 3)
                # remove gradient from gt
                gt = gt.detach()
                # reshape gt to match output
                reshape_idx = [self.smplh_body_names.index(name) for name in self.mapping_table.keys()]
                if self.load_hands:
                    reshape_idx += [self.smplh_extend_names.index(name) + len(self.smplh_body_names) for name in self.mapping_table.keys()]
            
                gt = gt[:, reshape_idx, :]
                gt = gt - gt[:, 0:1, :]
                scale = self.fit_data.get('scale', 1.0)
                gt = gt * scale.item()

                if visualize:
                    gt_copy = gt.clone()

                # Calculate position errors for logging
                position_error = torch.norm(output - gt, dim=-1).mean()
                max_position_error = torch.norm(output - gt, dim=-1).max()
                total_position_error += position_error.item()
                total_max_position_error += max_position_error.item()

                # Track action statistics
                action_stats = {}  # Initialize empty dict
                if i == 0:  # Only track for first batch to avoid too much computation
                    action_mean = actions.mean().item()
                    action_std = actions.std().item()
                    action_min = actions.min().item()
                    action_max = actions.max().item()
                    action_stats = {
                        'action_mean': action_mean,
                        'action_std': action_std,
                        'action_min': action_min,
                        'action_max': action_max,
                    }

                output_weighted = torch.einsum('ijk, j -> ijk', output, joint_weights) if joint_weights is not None else output
                gt_weighted = torch.einsum('ijk, j -> ijk', gt, joint_weights) if joint_weights is not None else gt
                main_loss = criterion(output_weighted, gt_weighted)
                total_main_loss += main_loss.item()  # Track unweighted main loss
                
                loss = main_loss * self.loss_coeffs.main_loss
                loss += aux_loss * self.loss_coeffs.aux_loss
                
                direct_mapping_loss = self.direct_mapping_loss(actions, pose_body, pose_hand) * self.loss_coeffs.direct_mapping_loss
                loss += direct_mapping_loss
                total_direct_mapping_loss += direct_mapping_loss.item()

                if current_apply_noise:
                    noise = torch.randn_like(data) * 0.1
                    actions_disturbed, _ = self._model_forward(data + noise)
                    action_disturbance_loss = self.disturbance_loss(actions, actions_disturbed, noise) * (self.loss_coeffs.disturbance_base * disturbance_coeff)
                    loss += action_disturbance_loss
                    total_disturbance_loss += action_disturbance_loss.item()

                if apply_symmetry:
                    assert self.flip_left_right is not None, "robot flipper is not set, call set_flip_left_right to set it"
                    sym_data = symmetry_smplh_pose(data)
                    sym_actions, _ = self._model_forward(sym_data)
                    sym_sym_actions = self.flip_left_right.flip(sym_actions)
                    sym_loss = criterion(actions, sym_sym_actions) * self.loss_coeffs.symmetry_loss
                    loss += sym_loss
                    total_symmetry_loss += sym_loss.item()

                if self.use_reference_actions:
                    ref_loss = criterion(actions, ref_actions) * self.loss_coeffs.reference_action_loss
                    loss += ref_loss
                    total_ref_action_loss += ref_loss.item()

                

                action_lim_min, action_lim_max = self.fk.get_dof_limits()
                out_of_range_loss = self.out_of_range_loss(actions, action_lim_min, action_lim_max)
                loss += out_of_range_loss * self.loss_coeffs.out_of_range_loss
                total_out_of_range_loss += out_of_range_loss.item() * self.loss_coeffs.out_of_range_loss

                high_value_action_loss = self.high_value_action_loss(actions, np.pi * 0.85)
                loss += high_value_action_loss * self.loss_coeffs.high_value_action_loss
                total_high_value_action_loss += high_value_action_loss.item() * self.loss_coeffs.high_value_action_loss

                total_loss += loss.item()
                loss.backward()

                # Calculate gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))
                total_grad_norm += grad_norm.item()

                optimizer.step()

            # Calculate average losses
            num_batches = len(self.train_loader)
            
            # Calculate loss component ratios for analysis
            total_unweighted = (total_main_loss + total_aux_loss + total_out_of_range_loss/self.loss_coeffs.out_of_range_loss + 
                              total_high_value_action_loss/self.loss_coeffs.high_value_action_loss + 
                              total_direct_mapping_loss/self.loss_coeffs.direct_mapping_loss) / num_batches
            
            avg_losses = {
                'epoch': epoch + 1,
                'total_loss': total_loss / num_batches,
                'main_loss': total_main_loss / num_batches,  # Add main loss
                'position_error': total_position_error / num_batches,  # Add position error
                'max_position_error': total_max_position_error / num_batches,  # Add max position error
                'grad_norm': total_grad_norm / num_batches,  # Add gradient norm
                'out_of_range_loss': total_out_of_range_loss / num_batches,
                'high_value_action_loss': total_high_value_action_loss / num_batches,
                'direct_mapping_loss': total_direct_mapping_loss / num_batches,
                'aux_loss': total_aux_loss / num_batches,
                'lr': scheduler.get_last_lr()[0],
                'disturbance_coeff': disturbance_coeff,  # Dynamic disturbance coefficient
                'apply_noise_enabled': current_apply_noise,  # Whether noise is currently applied
                'apply_symmetry_enabled': apply_symmetry,  # Whether symmetry is enabled
                'epoch_progress': (epoch + 1) / epochs,  # Training progress
                # Loss component ratios (for understanding relative importance)
                'main_loss_ratio': (total_main_loss / num_batches) / max(total_unweighted, 1e-8),
                'aux_loss_ratio': (total_aux_loss / num_batches) / max(total_unweighted, 1e-8),
                'out_of_range_loss_ratio': (total_out_of_range_loss/self.loss_coeffs.out_of_range_loss / num_batches) / max(total_unweighted, 1e-8),
                'high_value_action_loss_ratio': (total_high_value_action_loss/self.loss_coeffs.high_value_action_loss / num_batches) / max(total_unweighted, 1e-8),
                'direct_mapping_loss_ratio': (total_direct_mapping_loss/self.loss_coeffs.direct_mapping_loss / num_batches) / max(total_unweighted, 1e-8),
            }
            
            # Add loss coefficients for tracking
            loss_coeff_metrics = {
                f'loss_coeff_{k}': v for k, v in self.loss_coeffs.to_dict().items()
            }
            avg_losses.update(loss_coeff_metrics)
            
            # Add action statistics if available
            if action_stats:  # Check if dict is not empty
                avg_losses.update(action_stats)
            
            if current_apply_noise:
                avg_losses['disturbance_loss'] = total_disturbance_loss / num_batches
                avg_losses['disturbance_loss_ratio'] = (total_disturbance_loss/self.loss_coeffs.disturbance_base/disturbance_coeff / num_batches) / max(total_unweighted, 1e-8)
            if apply_symmetry:
                avg_losses['symmetry_loss'] = total_symmetry_loss / num_batches
                avg_losses['symmetry_loss_ratio'] = (total_symmetry_loss/self.loss_coeffs.symmetry_loss / num_batches) / max(total_unweighted, 1e-8)
            if self.use_reference_actions:
                avg_losses['reference_action_loss'] = total_ref_action_loss / num_batches
                avg_losses['reference_action_loss_ratio'] = (total_ref_action_loss/self.loss_coeffs.reference_action_loss / num_batches) / max(total_unweighted, 1e-8)

            # Log training metrics
            self.logger.log_metrics(avg_losses, step=epoch, prefix="train")
            
            # Print formatted training statistics
            print(self.logger.format_metrics_table(avg_losses, f"TRAINING - EPOCH {epoch + 1}"))

            if visualize:
                self.fk.visualize_joints(axis_length=0.05, bm=gt_copy, title_prefix="(Iter {} Train)".format(epoch))

            if save_figs:
                save_dir = f'{self.save_dir}/figs'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                self.fk.visualize_joints(show=False, axis_length=0.05, bm=gt, title_prefix="(Iter {} Train)".format(epoch), save_path=save_dir)

                if self.renderer is not None:
                    # print("render robot to", robot_save_dir)
                    robot_vis_tf = torch.zeros((4, 4)).to(self.device)
                    robot_vis_tf[range(4), [1, 2, 0, 3]] = 1
                    robot_save_dir = f'{save_dir}/robot_{epoch}_train.png'
                    amass_save_dir = f'{save_dir}/amass_{epoch}_train.png'
                    amass_bm_save_dir = f'{save_dir}/amass_bm_{epoch}_train.png'
                    self.renderer.render_pose_robot(action = actions[0].detach(), save_path=robot_save_dir, root_tf=robot_vis_tf)
                    self.renderer.render_pose_amass_with_bm(body_parms=body_parms, save_path=amass_save_dir)
                    self.renderer.render_pose_amass_with_bm(body_parms=body_parms, save_path=amass_bm_save_dir, add_smpl_fits=False, update_mv=False)

                    # concatenate the images
                    os.system(f'convert {robot_save_dir} {amass_save_dir} {amass_bm_save_dir} +append {save_dir}/concat_{epoch}_train.png')
                    os.system(f'rm {robot_save_dir} {amass_save_dir} {amass_bm_save_dir}')


            if epoch % validation_interval == 0:
                val_metrics = self.validate(
                    criterion=criterion, 
                    joint_weights=joint_weights, 
                    visualize=visualize, 
                    epoch=epoch, 
                    save_figs=save_figs, 
                    apply_noise=current_apply_noise, 
                    apply_symmetry=apply_symmetry,
                    disturbance_coeff=disturbance_coeff,
                    **kwargs
                )

            if epoch % save_interval == 0:
                save_dir = f'{self.save_dir}/models'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                date = time.strftime("%d_%H_%M")
                self.save(f'{save_dir}/{date}_epoch_{epoch}.pt', optimizer=optimizer)

            scheduler.step()
            
        # Finish logging
        self.logger.finish()

    def validate(self, 
                 criterion: torch.nn.Module = torch.nn.MSELoss(), 
                 joint_weights: Optional[torch.Tensor] = None, 
                 visualize: bool = False, 
                 epoch: int = 0, 
                 save_figs: bool = False, 
                 apply_noise: bool = True, 
                 apply_symmetry: bool = False,
                 disturbance_coeff: float = 10.0,
                 **kwargs):
        '''
        Validate the model
        '''
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_main_loss = 0.0  # Add main loss tracking
            total_out_of_range_loss = 0.0
            total_high_value_action_loss = 0.0
            total_ref_action_loss = 0.0
            total_aux_loss = 0.0
            total_disturbance_loss = 0.0
            total_symmetry_loss = 0.0
            total_direct_mapping_loss = 0.0
            total_position_error = 0.0  # Add position error tracking
            total_max_position_error = 0.0  # Add max position error tracking
            
            for i, data in enumerate(tqdm(self.test_loader, desc=f'Epoch {epoch+1} - Validation')):
                if self.use_reference_actions:
                    data, ref_actions = data
                    ref_actions = ref_actions.to(self.device)
                else:
                   ref_actions = None
                data = data.to(self.device)
                # root_orient = data[:, :3]
                pose_body = data[:, 3:66]
                pose_hand = data[:, 66:] if self.load_hands else None

                body_parms = self.create_model_input(pose_body=pose_body, pose_hand=pose_hand)
                data = data[:, 3:]

                actions, aux_loss = self._model_forward(data)
                output = self.fk(actions, **kwargs) # shape: (N, num_joints, 3)
                output = output - output[:, 0:1, :]
                total_aux_loss += aux_loss.item()
                gt = self.body_model(**body_parms).Jtr
                # remove gradient from gt
                gt = gt.detach()
                # reshape gt to match output
                reshape_idx = [self.smplh_body_names.index(name) for name in self.mapping_table.keys()]
                if self.load_hands:
                    reshape_idx += [self.smplh_extend_names.index(name) + len(self.smplh_body_names) for name in self.mapping_table.keys()]

                gt = gt[:, reshape_idx, :]
                gt = gt - gt[:, 0:1, :]
                scale = self.fit_data.get('scale', 1.0)
                gt = gt * scale.item()
                
                if visualize:
                    gt_copy = gt.clone()

                # Calculate position errors for logging
                position_error = torch.norm(output - gt, dim=-1).mean()
                max_position_error = torch.norm(output - gt, dim=-1).max()
                total_position_error += position_error.item()
                total_max_position_error += max_position_error.item()

                # Track action statistics
                action_stats = {}  # Initialize empty dict
                if i == 0:  # Only track for first batch to avoid too much computation
                    val_action_mean = actions.mean().item()
                    val_action_std = actions.std().item()
                    val_action_min = actions.min().item()
                    val_action_max = actions.max().item()
                    action_stats = {
                        'action_mean': val_action_mean,
                        'action_std': val_action_std,
                        'action_min': val_action_min,
                        'action_max': val_action_max,
                    }

                output_weighted = torch.einsum('ijk, j -> ijk', output, joint_weights) if joint_weights is not None else output
                gt_weighted = torch.einsum('ijk, j -> ijk', gt, joint_weights) if joint_weights is not None else gt
                main_loss = criterion(output_weighted, gt_weighted)
                total_main_loss += main_loss.item()  # Track unweighted main loss
                
                loss = main_loss * self.loss_coeffs.main_loss
                loss += aux_loss * self.loss_coeffs.aux_loss
                
                direct_mapping_loss = self.direct_mapping_loss(actions, pose_body, pose_hand) * self.loss_coeffs.direct_mapping_loss
                loss += direct_mapping_loss
                total_direct_mapping_loss += direct_mapping_loss.item()
                
                high_value_action_loss = self.high_value_action_loss(actions, np.pi * 0.85)
                loss += high_value_action_loss * self.loss_coeffs.high_value_action_loss
                total_high_value_action_loss += high_value_action_loss.item() * self.loss_coeffs.high_value_action_loss
                
                if apply_noise:
                    noise = torch.randn_like(data) * 0.1
                    actions_disturbed, _ = self._model_forward(data + noise)
                    action_disturbance_loss = self.disturbance_loss(actions, actions_disturbed, noise) * (self.loss_coeffs.disturbance_base * disturbance_coeff)
                    loss += action_disturbance_loss
                    total_disturbance_loss += action_disturbance_loss.item()

                if apply_symmetry:
                    sym_data = symmetry_smplh_pose(data)
                    sym_actions, _ = self._model_forward(sym_data)
                    sym_sym_actions = self.flip_left_right.flip(sym_actions)
                    sym_loss = criterion(actions, sym_sym_actions) * self.loss_coeffs.symmetry_loss
                    loss += sym_loss
                    total_symmetry_loss += sym_loss.item()

                if self.use_reference_actions:
                    ref_loss = criterion(actions, ref_actions) * self.loss_coeffs.reference_action_loss
                    loss += ref_loss
                    total_ref_action_loss += ref_loss.item()

                action_lim_min, action_lim_max = self.fk.get_dof_limits()
                out_of_range_loss = self.out_of_range_loss(actions, action_lim_min, action_lim_max)
                loss += out_of_range_loss * self.loss_coeffs.out_of_range_loss
                total_out_of_range_loss += out_of_range_loss.item() * self.loss_coeffs.out_of_range_loss

                total_loss += loss.item()

            # Calculate average validation losses
            num_batches = len(self.test_loader)
            
            # Calculate loss component ratios for analysis
            total_unweighted = (total_main_loss + total_aux_loss + total_out_of_range_loss/self.loss_coeffs.out_of_range_loss + 
                              total_high_value_action_loss/self.loss_coeffs.high_value_action_loss + 
                              total_direct_mapping_loss/self.loss_coeffs.direct_mapping_loss) / num_batches
            
            val_metrics = {
                'total_loss': total_loss / num_batches,
                'main_loss': total_main_loss / num_batches,  # Add main loss
                'position_error': total_position_error / num_batches,  # Add position error
                'max_position_error': total_max_position_error / num_batches,  # Add max position error
                'out_of_range_loss': total_out_of_range_loss / num_batches,
                'high_value_action_loss': total_high_value_action_loss / num_batches,
                'direct_mapping_loss': total_direct_mapping_loss / num_batches,
                'aux_loss': total_aux_loss / num_batches,
                'disturbance_coeff': disturbance_coeff,  # Dynamic disturbance coefficient
                'apply_noise_enabled': apply_noise,  # Whether noise is applied in validation
                'apply_symmetry_enabled': apply_symmetry,  # Whether symmetry is enabled
                # Loss component ratios (for understanding relative importance)
                'main_loss_ratio': (total_main_loss / num_batches) / max(total_unweighted, 1e-8),
                'aux_loss_ratio': (total_aux_loss / num_batches) / max(total_unweighted, 1e-8),
                'out_of_range_loss_ratio': (total_out_of_range_loss/self.loss_coeffs.out_of_range_loss / num_batches) / max(total_unweighted, 1e-8),
                'high_value_action_loss_ratio': (total_high_value_action_loss/self.loss_coeffs.high_value_action_loss / num_batches) / max(total_unweighted, 1e-8),
                'direct_mapping_loss_ratio': (total_direct_mapping_loss/self.loss_coeffs.direct_mapping_loss / num_batches) / max(total_unweighted, 1e-8),
            }
            
            # Add loss coefficients for tracking in validation
            loss_coeff_metrics = {
                f'loss_coeff_{k}': v for k, v in self.loss_coeffs.to_dict().items()
            }
            val_metrics.update(loss_coeff_metrics)
            
            # Add action statistics if available
            if action_stats:  # Check if dict is not empty
                val_metrics.update(action_stats)
            
            if apply_noise:
                val_metrics['disturbance_loss'] = total_disturbance_loss / num_batches
                val_metrics['disturbance_loss_ratio'] = (total_disturbance_loss/self.loss_coeffs.disturbance_base/disturbance_coeff / num_batches) / max(total_unweighted, 1e-8)
            if apply_symmetry:
                val_metrics['symmetry_loss'] = total_symmetry_loss / num_batches
                val_metrics['symmetry_loss_ratio'] = (total_symmetry_loss/self.loss_coeffs.symmetry_loss / num_batches) / max(total_unweighted, 1e-8)
            if self.use_reference_actions:
                val_metrics['reference_action_loss'] = total_ref_action_loss / num_batches
                val_metrics['reference_action_loss_ratio'] = (total_ref_action_loss/self.loss_coeffs.reference_action_loss / num_batches) / max(total_unweighted, 1e-8)

            # Log validation metrics
            self.logger.log_metrics(val_metrics, step=epoch, prefix="val")
            
            # Print formatted validation statistics
            print(self.logger.format_metrics_table(val_metrics, f"VALIDATION - EPOCH {epoch + 1}"))

            if visualize:
                self.fk.visualize_joints(axis_length=0.05, bm=gt_copy, title_prefix="(Iter {} Val)".format(epoch))

            if save_figs:
                save_dir = f'{self.save_dir}/figs'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                self.fk.visualize_joints(show=False, axis_length=0.05, bm=gt, title_prefix="(Iter {} Val)".format(epoch), save_path=save_dir)

                if self.renderer is not None:
                    robot_vis_tf = torch.zeros((4, 4)).to(self.device)
                    robot_vis_tf[range(4), [1, 2, 0, 3]] = 1
                    robot_save_dir = f'{save_dir}/robot_{epoch}_eval.png'
                    amass_save_dir = f'{save_dir}/amass_{epoch}_eval.png'
                    amass_bm_save_dir = f'{save_dir}/amass_bm_{epoch}_eval.png'
                    self.renderer.render_pose_robot(action = actions[0].detach(), save_path=robot_save_dir, root_tf=robot_vis_tf)
                    self.renderer.render_pose_amass_with_bm(body_parms=body_parms, save_path=amass_save_dir, add_smpl_fits=False)
                    self.renderer.render_pose_amass_with_bm(body_parms=body_parms, save_path=amass_bm_save_dir, add_smpl_fits=True, update_mv=False)

                    # concatenate the images
                    os.system(f'convert {robot_save_dir} {amass_save_dir} {amass_bm_save_dir} +append {save_dir}/concat_{epoch}_eval.png')
                    os.system(f'rm {robot_save_dir} {amass_save_dir} {amass_bm_save_dir}')
                
            return val_metrics
