import torch
from torch.utils.data import Dataset
import h5py

class DiffusionSorptionDataset(Dataset):
    def __init__(self, h5_path, split="train", train_ratio=0.8):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            # groups are integer strings like '0', '1', ...
            keys = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
            self.keys = sorted(keys, key=lambda x: int(x))
            
        n_total = len(self.keys)
        n_train = int(n_total * train_ratio)
        
        if split == "train":
            self.keys = self.keys[:n_train]
        elif split == "val":
            self.keys = self.keys[n_train:]
        else:
            raise ValueError("Split must be 'train' or 'val'")
            
        # Spatial coordinate grid [0, 1]
        self.grid = torch.linspace(0, 1, 1024, dtype=torch.float32).unsqueeze(0)
            
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with h5py.File(self.h5_path, 'r') as f:
            # data shape: (101, 1024, 1) -> (time, spatial, channel)
            data = f[key]['data'][:]
            data = torch.from_numpy(data).float()
            
            # Input: initial condition t=0 -> shape (1, 1024)
            # data[0] is (1024, 1), we permute to (1, 1024)
            x = data[0].permute(1, 0)
            
            # Target: subsequent states t=1 to 100 -> shape (100, 1024)
            # data[1:] is (100, 1024, 1), squeeze the last channel
            y = data[1:].squeeze(-1)
            
            # Extract BCs and broadcast to full spatial dimension
            left_bc_val = data[0, 0, 0].item()    # scalar BC value at x=0
            right_bc_val = data[0, -1, 0].item()  # scalar BC value at x=1
            
            left_bc_channel = torch.full((1, 1024), left_bc_val, dtype=torch.float32)
            right_bc_channel = torch.full((1, 1024), right_bc_val, dtype=torch.float32)
            
            # Inject BCs alongside Initial Condition and Grid
            x_with_grid_and_bcs = torch.cat([x, self.grid, left_bc_channel, right_bc_channel], dim=0)
            
        return x_with_grid_and_bcs, y

class ReactionDiffusionDataset(Dataset):
    def __init__(self, h5_path, split="train", train_ratio=0.8):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            n_total = f['tensor'].shape[0]
            
        n_train = int(n_total * train_ratio)
        
        if split == "train":
            self.start_idx = 0
            self.end_idx = n_train
        elif split == "val":
            self.start_idx = n_train
            self.end_idx = n_total
        else:
            raise ValueError("Split must be 'train' or 'val'")
            
        self.length = self.end_idx - self.start_idx
        
        # Spatial coordinate grid [0, 1]
        self.grid = torch.linspace(0, 1, 1024, dtype=torch.float32).unsqueeze(0)
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        with h5py.File(self.h5_path, 'r') as f:
            # data shape: (10000, 101, 1024) -> (samples, time, spatial)
            # We take time 0 to 100
            data = f['tensor'][actual_idx, :101, :]
            data = torch.from_numpy(data).float()
            
            # Input: initial condition t=0 -> shape (1, 1024)
            x = data[0].unsqueeze(0)
            
            # Target: subsequent states t=1 to 100 -> shape (100, 1024)
            y = data[1:]
            
            # Extract BCs and broadcast to full spatial dimension
            left_bc_val = data[0, 0].item()    # scalar BC value at x=0
            right_bc_val = data[0, -1].item()  # scalar BC value at x=1
            
            left_bc_channel = torch.full((1, 1024), left_bc_val, dtype=torch.float32)
            right_bc_channel = torch.full((1, 1024), right_bc_val, dtype=torch.float32)
            
            # Inject BCs alongside Initial Condition and Grid
            x_with_grid_and_bcs = torch.cat([x, self.grid, left_bc_channel, right_bc_channel], dim=0)
            
        return x_with_grid_and_bcs, y

