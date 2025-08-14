import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import numpy as np
from GBC.utils.base.math_utils import interpolate_angle_axis, interpolate_trans
from typing import List, Optional
from GBC.utils.data_preparation.data_preparation_cfg import BaseCfg
import os
import pickle
from functools import lru_cache
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool

class AMASSDataset(Dataset):
    def __init__(self, root_dir, num_betas, num_dmpls, load_hands=False, secondary_dir=None):
        self.root_dir = root_dir
        self.num_betas = num_betas
        self.num_dmpls = num_dmpls
        if secondary_dir is not None:
            self.files = glob.glob(root_dir + '/' + secondary_dir + "/*.npz")
        else:
            self.files = glob.glob(root_dir + "/*/*/*.npz")
        self.load_hands = load_hands

        # check if the files are valid
        for file in self.files:
            data = np.load(file)
            if 'poses' not in data.files:
                self.files.remove(file)
                continue

    @classmethod
    def from_cfg(cls, cfg: BaseCfg):
        return cls(
            root_dir=cfg.dataset_path,
            num_betas=cfg.num_betas,
            num_dmpls=cfg.num_dmpls,
            load_hands=cfg.load_hands,
            secondary_dir=cfg.secondary_dir
        )
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        betas = data['betas'][:self.num_betas]
        # dmpls = data['dmpls'][:self.num_dmpls]
        if self.load_hands:
            pose = data['poses']
        else:
            pose = data['poses'][:, :66]
        fps = data['mocap_framerate']
        trans = data['trans']

        return {
            "poses": torch.tensor(pose, dtype=torch.float32),
            "betas": torch.tensor(betas, dtype=torch.float32),
            # "dmpls": dmpls,
            "fps": torch.tensor(fps, dtype=torch.float32),
            "trans": torch.tensor(trans, dtype=torch.float32)
        }
    
class AMASSDatasetInterpolate(Dataset):
    def __init__(self, root_dir, num_betas, num_dmpls, load_hands=False, specialize_dir=None, interpolate_fps=1000, seq_min_duration: Optional[int] = None):
        self.root_dir = root_dir
        self.num_betas = num_betas
        self.num_dmpls = num_dmpls
        self.interpolate_fps = interpolate_fps
        self.seq_min_duration = seq_min_duration
        if specialize_dir is not None:
            # self.files = glob.glob(root_dir + '/' + specialize_dir + "/*.npz")
            # if specialize_dir is str:
            if isinstance(specialize_dir, str):
                self.files = glob.glob(root_dir + '/' + specialize_dir + "/*.npz")
            # elif specialize_dir is list:
            elif isinstance(specialize_dir, list):
                self.files = []
                for dir in specialize_dir:
                    self.files += glob.glob(root_dir + '/' + dir + "/*.npz")
            else:
                raise ValueError("specialize_dir must be a string or a list of strings")
        else:
            self.files = glob.glob(root_dir + "/*/*/*.npz")
        self.load_hands = load_hands

        # check if the files are valid
        for file in self.files:
            data = np.load(file)
            if 'poses' not in data.files:
                self.files.remove(file)
            
            if self.seq_min_duration is not None:
                fps = data['mocap_framerate'] if 'mocap_framerate' in data.files else data['mocap_frame_rate'] # change the key name to mocap_framerate for npzfiles
                if data['poses'].shape[0] / fps < self.seq_min_duration:
                    self.files.remove(file)
    
    @classmethod
    def from_cfg(cls, cfg: BaseCfg):
        return cls(
            root_dir=cfg.root_dir,
            num_betas=cfg.num_betas,
            num_dmpls=cfg.num_dmpls,
            load_hands=cfg.load_hands,
            specialize_dir=cfg.specialize_dir,
            interpolate_fps=cfg.interpolate_fps,
            seq_min_duration=cfg.seq_min_duration
        )
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
            
        title = self.files[idx].split('/')[-1].replace('.npz', '')
        fpath = '/'.join(self.files[idx].split('/')[-3:-1]) # the path should at least be root_dir/specialize_dir which means len(self.files[idx].split('/')) >= 3
        betas = data['betas'][:self.num_betas]
        trans = data['trans']
        # dmpls = data['dmpls'][:self.num_dmpls]
        if self.load_hands:
            pose = data['poses']
        else:
            pose = data['poses'][:, :66]
        fps = data['mocap_framerate'] if 'mocap_framerate' in data.files else data['mocap_frame_rate'] # change the key name to mocap_framerate for npzfiles

        pose = torch.tensor(pose, dtype=torch.float32)
        betas = torch.tensor(betas, dtype=torch.float32)
        # dmpls = torch.tensor(dmpls, dtype=torch.float32)
        fps = torch.tensor(fps, dtype=torch.float32)
        trans = torch.tensor(trans, dtype=torch.float32)
        # interpolate the translations
        trans = interpolate_trans(trans, target_fps=self.interpolate_fps,
        source_fps=fps)
        # interpolate the poses
        pose = interpolate_angle_axis(pose, target_fps=self.interpolate_fps, 
        source_fps=fps)

        return {
            "poses": pose,
            "betas": betas,
            # "dmpls": dmpls,
            "fps": fps,
            "trans": trans,
            "title": title,
            "fpath": fpath
        }
    
class AMASSDatasetSingleFrame(Dataset):
    def __init__(self, root_dir, transform=None, load_hands=False, secondary_dir="ACCAD", sample_steps=1):
        """
        Args:
            root_dir (str): Root directory of the dataset, containing all `.npz` files
            transform (callable, optional): Optional transform function for data augmentation
        """
        self.root_dir = root_dir
        self.transform = transform
        self.load_hands = load_hands
        
        self.files = glob.glob(root_dir + "/" + secondary_dir + "/*/*.npz") if secondary_dir is not None else glob.glob(root_dir + "/*/*/*.npz")
        self.sample_steps = sample_steps
        self.frames = []
        self.cached_data = {}
        self.files_progress = {}

        for file in self.files:
            data = np.load(file)
            if 'poses' not in data.files:
                continue
            num_frames = data['poses'].shape[0]
            
            for frame_idx in range(0, num_frames, self.sample_steps):
                self.frames.append((file, frame_idx))
            
            self.files_progress[file] = 0

    @classmethod
    def from_cfg(cls, cfg: BaseCfg):
        return cls(
            root_dir=cfg.dataset_path,
            load_hands=cfg.load_hands,
            transform=cfg.transform
        )

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        file_path, frame_idx = self.frames[idx]
        
        if file_path not in self.cached_data:
            data = np.load(file_path)
            self.cached_data[file_path] = data
        
        data = self.cached_data[file_path]
        pose = data['poses'][frame_idx] 
        if not self.load_hands:
            pose = pose[:66]
        # beta = data['beta']
        # dmpls = data['dmpls']
        
        # mocap_framerate = torch.tensor(mocap_framerate, dtype=torch.float32)
        pose = torch.tensor(pose, dtype=torch.float32)
        # beta = torch.tensor(beta, dtype=torch.float32)
        # dmpls = torch.tensor(dmpls, dtype=torch.float32)
        
        if self.transform:
            pose = self.transform(pose)
            beta = self.transform(beta)
            dmpls = self.transform(dmpls)
        
        self.files_progress[file_path] = frame_idx + 1

        if self.files_progress[file_path] >= data['poses'].shape[0]:
            del self.cached_data[file_path]
            del self.files_progress[file_path]

        return pose

    def clear_cache(self):
        self.cached_data.clear()
        self.files_progress.clear()


def _read_metadata_worker(file_path: str) -> tuple | None:
    """
    Worker function that reads metadata from a single .npz file.
    Returns a tuple (file_path, num_frames) or None if the file is invalid.
    """
    try:
        # Use np.load with memory mapping to only read metadata, which is faster
        with np.load(file_path, mmap_mode='r') as data:
            if 'poses' in data:
                return (file_path, data['poses'].shape[0])
    except Exception:
        # Silently ignore corrupted or unreadable files
        return None
    return None


class OptimizedAMASSDatasetSingleFrame(Dataset):
    """
    An optimized Dataset for AMASS that uses one-time, parallel indexing and an LRU cache
    to handle large datasets and random access patterns efficiently.

    Args:
        root_dir (str): Root directory of the dataset.
        load_hands (bool): Whether to load hand pose data.
        secondary_dir (str, optional): Sub-directory to scan (e.g., "ACCAD"). 
                                       If None, scans all sub-directories.
        sample_steps (int): The step used to sample frames from each sequence.
        transform (callable, optional): Optional transform function for data augmentation.
        cache_size (int): The number of .npz files to keep in memory using an LRU cache.
    """
    def __init__(self, root_dir, load_hands=False, secondary_dir="ACCAD", 
                 sample_steps=1, transform=None, cache_size=128):
        super().__init__()
        self.root_dir = root_dir
        self.load_hands = load_hands
        self.sample_steps = sample_steps
        self.transform = transform

        # Define path for the pre-computed index file, now with a 'v2' for the new format
        index_filename = f"amass_index_v2_{secondary_dir}_{'hands' if load_hands else 'nohands'}.pkl"
        index_path = os.path.join(self.root_dir, index_filename)

        # 1. --- ONE-TIME PARALLEL INDEXING ---
        if os.path.exists(index_path):
            print(f"Loading pre-computed index from: {index_path}")
            with open(index_path, 'rb') as f:
                self.file_metadata = pickle.load(f)
        else:
            print("No index file found. Building index in parallel for the first time...")
            print("This may take a moment, but it's a one-time process.")
            self.file_metadata = self._build_index_parallel(secondary_dir)
            print(f"Saving index to: {index_path}")
            with open(index_path, 'wb') as f:
                pickle.dump(self.file_metadata, f)

        # Create the flat list of all available frames (file_path, frame_index)
        self.frames = []
        for file_path, num_frames in self.file_metadata:
            for frame_idx in range(0, num_frames, self.sample_steps):
                self.frames.append((file_path, frame_idx))
        
        # 2. --- DYNAMIC CACHE CREATION ---
        self._get_data_from_file = lru_cache(maxsize=cache_size)(self._load_file)
    
    def _build_index_parallel(self, secondary_dir):
        """ Scans the dataset directory in parallel to get metadata. """
        pattern = os.path.join(self.root_dir, secondary_dir, "*", "*.npz") if secondary_dir else os.path.join(self.root_dir, "*", "*", "*.npz")
        file_paths = glob.glob(pattern)

        if not file_paths:
            print("Warning: No .npz files found at the specified path.")
            return []

        # Use all available CPU cores for maximum speed
        num_processes = multiprocessing.cpu_count()
        print(f"Starting parallel index build with {num_processes} processes...")

        metadata = []
        with Pool(processes=num_processes) as pool:
            # Use imap_unordered for efficiency, it returns results as they are completed.
            # tqdm shows a progress bar for the long-running process.
            results_iterator = pool.imap_unordered(_read_metadata_worker, file_paths)
            
            for result in tqdm(results_iterator, total=len(file_paths), desc="Building dataset index"):
                if result is not None:
                    metadata.append(result)
        
        return metadata

    def _load_file(self, file_path: str):
        """ 
        Loads a .npz file from disk. This raw method will be wrapped by lru_cache
        in the __init__ method.
        """
        return np.load(file_path)

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        file_path, frame_idx = self.frames[idx]
        
        data = self._get_data_from_file(file_path)
        
        pose = data['poses'][frame_idx]
        
        if not self.load_hands:
            pose = pose[:66]
            
        pose = torch.tensor(pose, dtype=torch.float32)
        
        if self.transform:
            pose = self.transform(pose)
            
        return pose


# This function is now designed to load the actual data, not just metadata.
def _load_data_worker(file_path: str) -> tuple | None:
    """
    Worker function that loads the 'poses' array from a single .npz file.
    """
    try:
        with np.load(file_path) as data:
            if 'poses' in data:
                # Return the file path and the actual pose data
                return (file_path, data['poses'])
    except Exception:
        return None
    return None

class InMemoryAMASSDatasetSingleFrame(Dataset):
    """
    A high-performance Dataset for AMASS that preloads the entire dataset 
    (or a large subset) into memory to eliminate I/O bottlenecks during training.
    This is ideal when the dataset subset can fit into RAM.

    Args:
        root_dir (str): Root directory of the dataset.
        load_hands (bool): Whether to load hand pose data.
        secondary_dir (str, optional): Sub-directory to scan (e.g., "ACCAD").
        sample_steps (int): The step used to sample frames from each sequence.
        transform (callable, optional): Optional transform function for data augmentation.
    """
    def __init__(self, root_dir, load_hands=False, secondary_dir="ACCAD", 
                 sample_steps=1, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.load_hands = load_hands
        self.sample_steps = sample_steps
        self.transform = transform

        # --- PRELOAD ENTIRE DATASET INTO RAM ---
        print("Initializing InMemoryAMASSDataset...")
        all_data = self._preload_data_parallel(secondary_dir)

        print(f"Preloading complete. Processing {len(all_data)} files into frames...")
        self.frames = []
        for file_path, poses_array in tqdm(all_data, desc="Flattening frames"):
            # If not loading hands, slice the array once here for efficiency
            if not self.load_hands:
                poses_array = poses_array[:, :66]

            num_frames = poses_array.shape[0]
            for frame_idx in range(0, num_frames, self.sample_steps):
                # Instead of storing indices, we store the actual data slice.
                self.frames.append(torch.tensor(poses_array[frame_idx], dtype=torch.float32))

        # Calculate and print memory usage to inform the user
        total_size_gb = sum(t.nelement() * t.element_size() for t in self.frames) / (1024**3)
        print(f"Dataset successfully loaded into memory. Number of frames: {len(self.frames)}")
        print(f"Estimated RAM usage for this dataset: {total_size_gb:.2f} GB")

    def _preload_data_parallel(self, secondary_dir):
        """ Loads all .npz pose arrays into a list in parallel. """
        pattern = os.path.join(self.root_dir, secondary_dir, "*", "*.npz") if secondary_dir else os.path.join(self.root_dir, "*", "*", "*.npz")
        file_paths = glob.glob(pattern)

        if not file_paths:
            raise FileNotFoundError(f"No .npz files found at the specified path: {pattern}")

        num_processes = max(1, multiprocessing.cpu_count() - 1)
        print(f"Starting parallel data preload with {num_processes} processes...")

        all_data = []
        with Pool(processes=num_processes) as pool:
            results_iterator = pool.imap_unordered(_load_data_worker, file_paths)
            
            for result in tqdm(results_iterator, total=len(file_paths), desc="Preloading data files"):
                if result is not None:
                    all_data.append(result)
        return all_data

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        # The logic is now lightning-fast: just return the pre-loaded tensor.
        pose = self.frames[idx]
        
        if self.transform:
            pose = self.transform(pose)
            
        return pose
        


if __name__ == "__main__":

    import time
    import os

    # dataset = AMASSDatasetInterpolate(root_dir="/home/yyf/dataset", num_betas=16, num_dmpls=8, specialize_dir="/ACCAD/Male1Walking_c3d")
    dataset = AMASSDatasetSingleFrame(root_dir=os.path.expanduser("~/sjtu/dataset/amass"), sample_steps=25, secondary_dir="CMU")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(dataloader):
        # print(data['poses'].shape)
        # print(data['trans'].shape)
        # print(data['betas'].shape)
        # print(data['fps'].shape)
        print(data.shape)
        break
    # for i, data in enumerate(train_loader):
    #     print(data.shape)
    #     break
