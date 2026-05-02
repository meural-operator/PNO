"""
Dataset loader for the p-adic Schrodinger equation experiments.

HDF5 file layout (produced by data/schrodinger_generator.py):
    f['psi_re']  : float32, shape (N_samples, N_save, M)   Re(psi(x,t))
    f['psi_im']  : float32, shape (N_samples, N_save, M)   Im(psi(x,t))
    f['V']       : float32, shape (N_samples, M)           potential V(x)

Each sample is returned as:

    x_in : (4, M)    = [ Re(psi_0), Im(psi_0), spatial_grid, V(x) ]
    y    : (2*out_t, M)
               first  out_t channels : Re(psi(x, t_1 .. t_{out_t}))
               second out_t channels : Im(psi(x, t_1 .. t_{out_t}))

The 200-channel output (out_t=100) is consistent with out_channels=200
in the model config.  The existing LpLoss computes relative L^2 error
on the full (B, 200, M) tensor, which equals the complex L^2 error:

    ||Re_pred - Re_true||^2 + ||Im_pred - Im_true||^2
    ─────────────────────────────────────────────────
          ||Re_true||^2 + ||Im_true||^2
"""

import torch
from torch.utils.data import Dataset
import h5py


class PAdicSchrodingerDataset(Dataset):
    """
    Dataset for the p-adic Schrodinger equation.

    Parameters
    ----------
    h5_path     : str    Path to HDF5 file from schrodinger_generator.py.
    split       : str    'train' or 'val'.
    train_ratio : float  Fraction of samples for training (default 0.8).
    n_x         : int    Spatial resolution, must equal p^n (default 1024).
    out_t       : int    Future time steps to predict (default 100).
    ic_index    : int    Time index used as IC (default 0).
    """

    def __init__(
        self,
        h5_path:     str,
        split:       str   = 'train',
        train_ratio: float = 0.8,
        n_x:         int   = 1024,
        out_t:       int   = 100,
        ic_index:    int   = 0,
    ):
        self.h5_path   = h5_path
        self.n_x       = n_x
        self.out_t     = out_t
        self.ic_index  = ic_index

        with h5py.File(h5_path, 'r') as f:
            shape     = f['psi_re'].shape      # (N, N_save, M)
            n_total   = shape[0]
            n_t_avail = shape[1]
            n_x_file  = shape[2]

        assert n_x_file >= n_x, (
            f"File has {n_x_file} spatial points but n_x={n_x} requested.")
        assert n_t_avail >= ic_index + out_t + 1, (
            f"File has only {n_t_avail} time steps; need at least "
            f"{ic_index + out_t + 1}.")

        n_train = int(n_total * train_ratio)
        if split == 'train':
            self.start_idx, self.end_idx = 0, n_train
        elif split == 'val':
            self.start_idx, self.end_idx = n_train, n_total
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self.length = self.end_idx - self.start_idx
        self.grid   = torch.linspace(0, 1, n_x, dtype=torch.float32).unsqueeze(0)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        actual_idx = self.start_idx + idx
        ic  = self.ic_index
        end = ic + self.out_t + 1

        with h5py.File(self.h5_path, 'r') as f:
            # Re(psi) and Im(psi): slice [ic_index .. ic_index + out_t] inclusive
            psi_re = torch.from_numpy(
                f['psi_re'][actual_idx, ic:end, :self.n_x]).float()   # (out_t+1, M)
            psi_im = torch.from_numpy(
                f['psi_im'][actual_idx, ic:end, :self.n_x]).float()
            V      = torch.from_numpy(
                f['V'][actual_idx, :self.n_x]).float()                 # (M,)

        # ── Input: 4 channels (M,) ──────────────────────────────────────
        re0 = psi_re[0].unsqueeze(0)   # Re(psi_0), shape (1, M)
        im0 = psi_im[0].unsqueeze(0)   # Im(psi_0), shape (1, M)
        V_c = V.unsqueeze(0)           # potential,  shape (1, M)
        x_in = torch.cat([re0, im0, self.grid, V_c], dim=0)  # (4, M)

        # ── Target: 2*out_t channels ────────────────────────────────────
        # Channels  0 ..   out_t-1 :  Re(psi) at t_1 .. t_{out_t}
        # Channels out_t .. 2*out_t-1: Im(psi) at t_1 .. t_{out_t}
        y_re = psi_re[1:self.out_t + 1]   # (out_t, M)
        y_im = psi_im[1:self.out_t + 1]   # (out_t, M)
        y    = torch.cat([y_re, y_im], dim=0)   # (2*out_t, M)

        return x_in, y
