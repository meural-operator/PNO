"""
PDE Dataset Classes
====================

Dataset loaders for various PDE benchmarks. Each dataset maps an input field
(initial condition, coefficient, or parameters) to an output solution field.

All datasets normalize spatial coordinates to [0, 1) for p-adic addressing.
"""

import os
import json
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class PDEDatasetBase(Dataset):
    """Base class handling train/val/test splitting."""

    def __init__(self, file_path, n_samples=None, split="train",
                 train_ratio=0.8, val_ratio=0.1):
        super().__init__()
        self.file_path = file_path
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.n_samples = n_samples

    def _compute_split_indices(self, total):
        n = min(self.n_samples, total) if self.n_samples else total
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        if self.split == "train":
            return 0, n_train
        elif self.split == "val":
            return n_train, n_train + n_val
        else:  # test
            return n_train + n_val, n

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class PDE1DDataset(PDEDatasetBase):
    pass

class PDE2DDataset(PDEDatasetBase):
    pass

class PDE3DDataset(PDEDatasetBase):
    pass


# ---------------------------------------------------------------------------
# 1D Burgers Equation
# ---------------------------------------------------------------------------

class Burgers1DDataset(PDE1DDataset):
    """
    1D Burgers equation from HDF5.
    Maps initial condition u(x, t=0) to solution u(x, t_target).

    HDF5 keys: 'tensor' (samples, time, x), 'x-coordinate' (x,)

    Input:  (N_x, 2) = [u_init, x_coord]
    Output: (N_x, 1) = u at t_target
    """

    def __init__(self, file_path, t_target_idx=-1, sub=1, **kwargs):
        super().__init__(file_path, **kwargs)
        self.sub = sub
        self.t_target_idx = t_target_idx

        with h5py.File(self.file_path, "r") as f:
            total_samples = f["tensor"].shape[0]

            if self.n_samples is None:
                self.n_samples = total_samples
            else:
                self.n_samples = min(self.n_samples, total_samples)

            split_idx = int(self.n_samples * self.train_ratio)

            if self.split == "train":
                self.start_idx = 0
                self.end_idx = split_idx
            else:
                self.start_idx = split_idx
                self.end_idx = self.n_samples

            self.length = self.end_idx - self.start_idx
            self.grid = torch.tensor(f["x-coordinate"][:], dtype=torch.float32)[::self.sub]
            
            # Compute stable Normalization constants solely from Training Split
            # This ensures no data leakage into the validation framework
            train_tensor = torch.tensor(f["tensor"][0:split_idx, :, ::self.sub], dtype=torch.float32)
            self.u_mean = train_tensor.mean()
            self.u_std = train_tensor.std() + 1e-6

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        with h5py.File(self.file_path, "r") as f:
            u_init = torch.tensor(f["tensor"][actual_idx, 0, ::self.sub], dtype=torch.float32)
            u_target = torch.tensor(f["tensor"][actual_idx, self.t_target_idx, ::self.sub], dtype=torch.float32)

        # Standardize strictly the physical states using pre-computed normalisers
        u_init = (u_init - self.u_mean) / self.u_std
        u_target = (u_target - self.u_mean) / self.u_std

        x_in = torch.stack([u_init, self.grid], dim=-1)   # (N_x, 2)
        y_out = u_target.unsqueeze(-1)                     # (N_x, 1)
        return x_in, y_out


# ---------------------------------------------------------------------------
# 2D Darcy Flow
# ---------------------------------------------------------------------------

class DarcyFlow2DDataset(PDE2DDataset):
    """
    2D Darcy Flow from HDF5.
    Maps coefficient field nu(x,y) to solution u(x,y).

    HDF5 keys: 'nu' (N, 128, 128), 'tensor' (N, 1, 128, 128),
               'x-coordinate' (128,), 'y-coordinate' (128,)

    Input:  (N_x*N_y, 3) = [nu, x_coord, y_coord]  (flattened grid)
    Output: (N_x*N_y, 1) = u(x,y)
    """

    def __init__(self, file_path, sub=1, **kwargs):
        super().__init__(file_path, **kwargs)
        self.sub = sub

        with h5py.File(self.file_path, "r") as f:
            total = f["nu"].shape[0]
            self.start_idx, self.end_idx = self._compute_split_indices(total)
            self.length = self.end_idx - self.start_idx

            # Build 2D grid, normalised to [0, 1)
            x = torch.tensor(f["x-coordinate"][:], dtype=torch.float32)[::sub]
            y = torch.tensor(f["y-coordinate"][:], dtype=torch.float32)[::sub]
            # Normalise to [0,1)
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)
            gx, gy = torch.meshgrid(x, y, indexing="ij")
            self.grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (N_x*N_y, 2)

        self.res = len(x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        with h5py.File(self.file_path, "r") as f:
            nu = torch.tensor(f["nu"][actual_idx, ::self.sub, ::self.sub], dtype=torch.float32)
            u = torch.tensor(f["tensor"][actual_idx, 0, ::self.sub, ::self.sub], dtype=torch.float32)

        nu_flat = nu.flatten().unsqueeze(-1)               # (N, 1)
        u_flat = u.flatten().unsqueeze(-1)                  # (N, 1)
        x_in = torch.cat([nu_flat, self.grid], dim=-1)      # (N, 3)
        return x_in, u_flat


# ---------------------------------------------------------------------------
# 3D Navier-Stokes (CFD Turbulence)
# ---------------------------------------------------------------------------

class NS3DDataset(PDE3DDataset):
    """
    3D Navier-Stokes CFD Turbulence from HDF5.
    Maps state at t=0 to state at t_target (autoregressive-compatible).

    HDF5 keys: 'Vx','Vy','Vz','density','pressure' — each (N, T, 64, 64, 64)

    This dataset is very large — subsampling is strongly recommended.

    Input:  (N_pts, d_in) where d_in = 5 (Vx,Vy,Vz,density,pressure) + 3 (coords)
    Output: (N_pts, 5) at target time
    """

    def __init__(self, file_path, sub=4, t_input_idx=0, t_target_idx=-1, **kwargs):
        super().__init__(file_path, **kwargs)
        self.sub = sub
        self.t_input_idx = t_input_idx
        self.t_target_idx = t_target_idx

        with h5py.File(self.file_path, "r") as f:
            total = f["Vx"].shape[0]
            self.start_idx, self.end_idx = self._compute_split_indices(total)
            self.length = self.end_idx - self.start_idx

            # Build 3D grid normalised to [0,1)
            x = torch.tensor(f["x-coordinate"][:], dtype=torch.float32)[::sub]
            y = torch.tensor(f["y-coordinate"][:], dtype=torch.float32)[::sub]
            z = torch.tensor(f["z-coordinate"][:], dtype=torch.float32)[::sub]
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)
            z = (z - z.min()) / (z.max() - z.min() + 1e-8)
            gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
            self.grid = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)

        self.res = len(x)
        self.fields = ["Vx", "Vy", "Vz", "density", "pressure"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        s = self.sub
        inp_fields, tgt_fields = [], []
        with h5py.File(self.file_path, "r") as f:
            for field in self.fields:
                inp = f[field][actual_idx, self.t_input_idx, ::s, ::s, ::s]
                tgt = f[field][actual_idx, self.t_target_idx, ::s, ::s, ::s]
                inp_fields.append(torch.tensor(inp, dtype=torch.float32).flatten())
                tgt_fields.append(torch.tensor(tgt, dtype=torch.float32).flatten())

        inp_stack = torch.stack(inp_fields, dim=-1)    # (N_pts, 5)
        tgt_stack = torch.stack(tgt_fields, dim=-1)    # (N_pts, 5)
        x_in = torch.cat([inp_stack, self.grid], dim=-1)  # (N_pts, 8)
        return x_in, tgt_stack


# ---------------------------------------------------------------------------
# Temporal NetCDF datasets (Boussinesq, Cylinder2D)
# ---------------------------------------------------------------------------

class TemporalNetCDFDataset(PDEDatasetBase):
    """
    Temporal prediction from NetCDF files (single simulation).

    Creates samples by windowing: input is state at t, output is state at t+dt.
    Variables 'u' and 'v' are velocity components with shape (T, Ny, Nx).

    Supports subsampling in both space and time.

    Input:  (N_pts, d_in) = [u, v, x_coord, y_coord]
    Output: (N_pts, 2) = [u, v] at target time
    """

    def __init__(self, file_path, dt_steps=10, sub_spatial=1, sub_temporal=1,
                 split="train", train_ratio=0.8, val_ratio=0.1, n_samples=None):
        super().__init__(file_path, n_samples=n_samples, split=split,
                         train_ratio=train_ratio, val_ratio=val_ratio)
        import xarray as xr

        self.dt_steps = dt_steps
        self.sub_s = sub_spatial
        self.sub_t = sub_temporal

        ds = xr.open_dataset(file_path)
        self.u = torch.tensor(ds["u"].values[::sub_temporal, ::sub_spatial, ::sub_spatial],
                              dtype=torch.float32)
        self.v = torch.tensor(ds["v"].values[::sub_temporal, ::sub_spatial, ::sub_spatial],
                              dtype=torch.float32)
        ds.close()

        T, Ny, Nx = self.u.shape
        total_windows = T - dt_steps
        self.start_idx, self.end_idx = self._compute_split_indices(total_windows)
        self.length = self.end_idx - self.start_idx

        # Build normalised 2D grid
        x = torch.linspace(0, 1 - 1e-6, Nx)
        y = torch.linspace(0, 1 - 1e-6, Ny)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        self.grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (Ny*Nx, 2)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        t = self.start_idx + idx
        u_in = self.u[t].flatten().unsqueeze(-1)
        v_in = self.v[t].flatten().unsqueeze(-1)
        u_out = self.u[t + self.dt_steps].flatten().unsqueeze(-1)
        v_out = self.v[t + self.dt_steps].flatten().unsqueeze(-1)

        x_in = torch.cat([u_in, v_in, self.grid], dim=-1)   # (N, 4)
        y_out = torch.cat([u_out, v_out], dim=-1)            # (N, 2)
        return x_in, y_out


class BoussinesqDataset(TemporalNetCDFDataset):
    """Boussinesq convection (boussinesq.nc)."""
    pass


class Cylinder2DDataset(TemporalNetCDFDataset):
    """2D cylinder flow (cylinder2d.nc)."""
    pass


# ---------------------------------------------------------------------------
# BC Cylinder (multi-case, parameter-conditioned)
# ---------------------------------------------------------------------------

class BCCylinderDataset(PDEDatasetBase):
    """
    Boundary-condition cylinder flow dataset (multi-case).
    Each case has parameters (json) and velocity fields u.npy, v.npy.

    Creates temporal windows from each case. Parameters are appended to input.

    Input:  (N_pts, d_in) = [u, v, x, y, vel_in, density, viscosity, radius]
    Output: (N_pts, 2) = [u, v] at target time
    """

    def __init__(self, data_dir, dt_steps=10, sub_spatial=1, sub_temporal=1,
                 split="train", train_ratio=0.8, val_ratio=0.1, n_samples=None):
        super().__init__(data_dir, n_samples=n_samples, split=split,
                         train_ratio=train_ratio, val_ratio=val_ratio)
        self.dt_steps = dt_steps
        self.sub_s = sub_spatial
        self.sub_t = sub_temporal

        # Discover all cases
        case_dirs = sorted(glob.glob(os.path.join(data_dir, "case*")))

        # Build index: list of (case_idx, time_idx)
        self.cases = []
        self.params_list = []
        self.index = []

        for ci, cdir in enumerate(case_dirs):
            json_path = os.path.join(cdir, "case.json")
            u_path = os.path.join(cdir, "u.npy")
            v_path = os.path.join(cdir, "v.npy")
            if not (os.path.exists(json_path) and os.path.exists(u_path)):
                continue

            with open(json_path) as f:
                params = json.load(f)

            u = np.load(u_path).astype(np.float32)[::sub_temporal, ::sub_spatial, ::sub_spatial]
            v = np.load(v_path).astype(np.float32)[::sub_temporal, ::sub_spatial, ::sub_spatial]
            T = u.shape[0]

            self.cases.append({
                "u": torch.from_numpy(u),
                "v": torch.from_numpy(v),
                "params": params,
            })
            param_vec = [
                params.get("vel_in", 0),
                params.get("density", 0),
                params.get("viscosity", 0),
                params.get("radius", 0),
            ]
            self.params_list.append(torch.tensor(param_vec, dtype=torch.float32))

            for t in range(T - dt_steps):
                self.index.append((ci, t))

        total = len(self.index)
        self.start_idx, self.end_idx = self._compute_split_indices(total)
        self.length = self.end_idx - self.start_idx

        # Build normalised 2D grid from first case
        _, Ny, Nx = self.cases[0]["u"].shape
        x = torch.linspace(0, 1 - 1e-6, Nx)
        y = torch.linspace(0, 1 - 1e-6, Ny)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        self.grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ci, t = self.index[self.start_idx + idx]
        case = self.cases[ci]
        params = self.params_list[ci]

        u_in = case["u"][t].flatten().unsqueeze(-1)
        v_in = case["v"][t].flatten().unsqueeze(-1)
        u_out = case["u"][t + self.dt_steps].flatten().unsqueeze(-1)
        v_out = case["v"][t + self.dt_steps].flatten().unsqueeze(-1)

        N = u_in.shape[0]
        params_expanded = params.unsqueeze(0).expand(N, -1)  # (N, 4)

        x_in = torch.cat([u_in, v_in, self.grid, params_expanded], dim=-1)  # (N, 8)
        y_out = torch.cat([u_out, v_out], dim=-1)                           # (N, 2)
        return x_in, y_out
