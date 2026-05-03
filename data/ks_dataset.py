"""
Dataset loader for the Kuramoto-Sivashinsky (KS) HDF5 dataset.

HDF5 layout (KS_ML_DATASET.h5):
    f['train']    float32  (40000, 768, 512)   training windows
    f['test']     float64  (10000, 768, 512)   test windows
    f['metadata'] attrs: window=256, horizon=512

Model interface (single-shot):
    x_in  : (in_t + 1, M)     = [u(t_0), ..., u(t_{in_t-1}), grid]
    y_out : (out_t, M)         = [u(t_{in_t}), ..., u(t_{in_t + out_t - 1})]

    The first `in_t` timesteps are the input context window.
    The next `out_t` timesteps are the prediction horizon.
    An additional grid channel is appended to the input.

Normalisation:
    UnitGaussianNormalizer in train.py handles input/output normalisation.
    No manual scaling is applied here.

NOTE: This dataset uses lazy HDF5 loading (reads from disk per __getitem__)
because the full dataset (~88 GB) does not fit in RAM.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class KuramotoSivashinskyDataset(Dataset):
    """
    Lazy-loading dataset for Kuramoto-Sivashinsky trajectory prediction.

    Parameters
    ----------
    h5_path   : path to KS_ML_DATASET.h5
    split     : 'train' or 'val' (val uses 'test' key in HDF5)
    in_t      : number of input timesteps (context window)
    out_t     : number of output timesteps (prediction horizon)
    subsample : use every n-th sample (useful for normalizer fitting)
    """

    def __init__(self, h5_path: str, split: str = 'train',
                 in_t: int = 256, out_t: int = 512,
                 subsample: int = 1):
        super().__init__()
        self.h5_path   = h5_path
        self.in_t      = in_t
        self.out_t     = out_t
        self.split     = split

        # Determine the HDF5 key and dataset length
        self._key = 'train' if split == 'train' else 'test'

        with h5py.File(h5_path, 'r') as f:
            self._total_len = f[self._key].shape[0]
            self._n_x = f[self._key].shape[2]   # spatial points (512)

        # Build index map (for subsampling)
        self._indices = list(range(0, self._total_len, subsample))

        # Precompute grid channel
        self.grid = torch.linspace(0.0, 1.0, self._n_x)  # (M,)

        # Keep file handle closed; open lazily per worker
        self._file = None

    def _open_file(self):
        """Open HDF5 file lazily (safe for DataLoader workers)."""
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r', swmr=True)
        return self._file

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        real_idx = self._indices[idx]
        f = self._open_file()
        
        # Read single trajectory window: (768, 512)
        traj = f[self._key][real_idx]  # numpy array (768, 512)

        # Input: first in_t timesteps
        x_data = torch.from_numpy(traj[:self.in_t].astype(np.float32))  # (in_t, M)

        # Append grid channel
        grid_expanded = self.grid.unsqueeze(0)  # (1, M)
        x_in = torch.cat([x_data, grid_expanded], dim=0)  # (in_t + 1, M)

        # Target: next out_t timesteps
        y = torch.from_numpy(
            traj[self.in_t:self.in_t + self.out_t].astype(np.float32)
        )  # (out_t, M)

        return x_in, y

    def __del__(self):
        if self._file is not None:
            self._file.close()
