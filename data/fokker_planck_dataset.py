"""
Dataset loader for the p-adic Fokker-Planck HDF5 files produced by
data/fokker_planck_generator.py.

HDF5 layout (written by the generator):
    f['p_density']  float32  (N_samples, N_save, M)   probability density
    f['V']          float32  (N_samples, M)            trapping landscape

Model interface:
    x_in  : (4, M)       = [p_0, grid, V(x), zeros]
    y_out : (out_t, M)   = [p(t_1), ..., p(t_out_t)]

    Channels:
      0  p_0(x)  — initial probability density (non-negative, sum = M)
      1  grid    — spatial index normalised to [0, 1]
      2  V(x)    — trapping potential (zero for free diffusion case)
      3  zeros   — padding channel (keeps in_channels = 4 across all tasks)

    Target: the first `out_t` saved snapshots after t=0, i.e. indices 1..out_t
    from the (N_save,) time axis.  N_save is typically 101, out_t=100.

Normalisation:
    UnitGaussianNormalizer in train.py handles input/output normalisation.
    No manual scaling is applied here.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class FokkerPlanckDataset(Dataset):
    """
    Dataset for p-adic Fokker-Planck trajectory prediction.

    Parameters
    ----------
    h5_path     : path to HDF5 file produced by fokker_planck_generator.py
    split       : 'train' or 'val'
    train_ratio : fraction of samples used for training
    n_x         : number of spatial points (must equal M = p^n in the file)
    out_t       : number of output time steps (snapshots after t=0)
    """

    def __init__(self, h5_path: str, split: str = 'train',
                 train_ratio: float = 0.8,
                 n_x: int = 1024, out_t: int = 100):
        super().__init__()
        self.n_x   = n_x
        self.out_t = out_t

        with h5py.File(h5_path, 'r') as f:
            p_density = f['p_density'][:]   # (N, N_save, M)
            V         = f['V'][:]           # (N, M)

        N = p_density.shape[0]
        n_train = int(N * train_ratio)

        if split == 'train':
            idx = slice(0, n_train)
        else:
            idx = slice(n_train, N)

        self.p_density = torch.from_numpy(p_density[idx]).float()  # (S, N_save, M)
        self.V         = torch.from_numpy(V[idx]).float()           # (S, M)

        grid = torch.linspace(0.0, 1.0, n_x)                       # (M,)
        self.grid = grid

    def __len__(self) -> int:
        return self.p_density.shape[0]

    def __getitem__(self, idx: int):
        p_traj = self.p_density[idx]        # (N_save, M)
        V_i    = self.V[idx]                # (M,)

        # Input: t=0 snapshot
        p0   = p_traj[0]                    # (M,)
        grid = self.grid                    # (M,)
        pad  = torch.zeros_like(grid)       # (M,) — 4th channel padding

        x_in = torch.stack([p0, grid, V_i, pad], dim=0)   # (4, M)

        # Target: snapshots t_1 ... t_{out_t}
        y = p_traj[1:1 + self.out_t]        # (out_t, M)

        return x_in, y
