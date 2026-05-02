"""
Synthetic 1D PDE data generators for the PNO project.

Generates HDF5 datasets with 'tensor' key of shape (N_samples, N_t, N_x=1024),
compatible with PDEBench1DDataset and the existing PNO training pipeline.

Supported PDEs
--------------
burgers    :  ∂u/∂t + u ∂u/∂x  = ν ∂²u/∂x²          (shocks for small ν)
allencahn  :  ∂u/∂t             = ε² ∂²u/∂x² + u(1−u²)  (sharp interfaces)
advection  :  ∂u/∂t + β ∂u/∂x  = 0                       (exact spectral solution)

All PDEs use:
  • Periodic boundary conditions on [0, 1]
  • Spatial resolution  N_x = 1024  (= 2^10, native to PNO with p=2, n=10)
  • Pseudo-spectral methods with 2/3 dealiasing

Quick start
-----------
    conda activate turbo_nigo
    cd C:/Users/DIAT/ashish/neural_operators/PNO

    # Generate Burgers datasets (three viscosities for viscosity-sweep experiment)
    python -m data.pde_generators --pde burgers --nu 0.1   --N 1000
    python -m data.pde_generators --pde burgers --nu 0.01  --N 1000
    python -m data.pde_generators --pde burgers --nu 0.001 --N 1000

    # Generate Allen-Cahn datasets (two interface widths)
    python -m data.pde_generators --pde allencahn --eps 0.01  --N 1000
    python -m data.pde_generators --pde allencahn --eps 0.005 --N 1000

    # Generate Advection dataset
    python -m data.pde_generators --pde advection --beta 1.0 --N 1000
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm


# =============================================================================
# Initial condition generators
# =============================================================================

def ic_fourier_random(Nx: int, n_modes: int = 10, seed: int | None = None) -> np.ndarray:
    """
    Smooth random IC: weighted sum of sin/cos modes with 1/k amplitude decay.
    Normalised so that max(|u|) ≈ 1.
    """
    rng = np.random.default_rng(seed)
    x   = np.linspace(0, 1, Nx, endpoint=False)
    u   = np.zeros(Nx)
    for k in range(1, n_modes + 1):
        amp   = rng.standard_normal() / k          # 1/k decay → smooth
        phase = rng.uniform(0, 2 * np.pi)
        u    += amp * np.sin(2 * np.pi * k * x + phase)
    scale = np.max(np.abs(u))
    return u / (scale + 1e-8)


def ic_tanh_interfaces(
    Nx: int,
    n_interfaces: int = 3,
    eps: float = 0.01,
    seed: int | None = None,
) -> np.ndarray:
    """
    Allen-Cahn IC: smooth ±1 regions separated by tanh interfaces.
    Interface width ≈ eps.  Values always in (−1, +1).
    """
    rng  = np.random.default_rng(seed)
    x    = np.linspace(0, 1, Nx, endpoint=False)
    cuts = np.sort(rng.uniform(0.15, 0.85, size=n_interfaces))
    sign = rng.choice([-1.0, 1.0])

    u = np.full(Nx, sign)
    for j, cut in enumerate(cuts):
        # Each interface flips sign via a tanh transition
        u += -2.0 * sign * ((-1) ** j) * 0.5 * (np.tanh((x - cut) / eps) + 1.0)

    return np.clip(u, -1.0, 1.0).astype(np.float64)


# =============================================================================
# Burgers equation  —  CNAB2 pseudo-spectral solver
# =============================================================================
# Scheme: Crank-Nicolson (diffusion, implicit) + Adams-Bashforth 2nd order
#         (nonlinear advection, explicit) in spectral space.
#
# Stability: CN handles stiff diffusion exactly for any dt.
#            AB2 for nonlinear requires  dt < dx / max|u|.
#            With N_sub=50 and T=1.0, N_save=101 → dt ≈ 2e-4 → CFL ≈ 0.2  ✓
# =============================================================================

def solve_burgers(
    u0: np.ndarray,
    nu: float,
    T: float = 1.0,
    N_save: int = 101,
    N_sub: int = 50,
) -> np.ndarray:
    """
    Solve 1D Burgers equation on periodic [0,1].

    Parameters
    ----------
    u0     : Initial condition, shape (Nx,).
    nu     : Kinematic viscosity.
    T      : Final time.
    N_save : Number of time snapshots saved (including t=0).
    N_sub  : Internal sub-steps between consecutive saved snapshots.

    Returns
    -------
    trajectory : float32 array, shape (N_save, Nx).
    """
    Nx      = len(u0)
    N_steps = N_sub * (N_save - 1)
    dt      = T / N_steps

    # Wavenumbers for periodic domain [0, 1]
    k  = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=1.0 / Nx)
    k2 = k ** 2
    k_max = Nx // 3        # 2/3 dealiasing cutoff index

    # Crank-Nicolson factors for diffusion
    cn_num   = 1.0 - 0.5 * nu * k2 * dt
    cn_denom = 1.0 + 0.5 * nu * k2 * dt

    def nonlinear_hat(u_hat: np.ndarray) -> np.ndarray:
        """Spectral nonlinear term  −∂(u²/2)/∂x  with 2/3 dealiasing."""
        uh = u_hat.copy()
        uh[k_max:] = 0.0
        u_phys     = np.fft.irfft(uh, Nx)
        uu2_hat    = np.fft.rfft(0.5 * u_phys ** 2)
        uu2_hat[k_max:] = 0.0
        return -1j * k * uu2_hat

    u_hat  = np.fft.rfft(u0.astype(np.float64))
    traj   = [np.fft.irfft(u_hat, Nx).astype(np.float32)]
    N_prev = nonlinear_hat(u_hat)

    for step in range(N_steps):
        N_curr = nonlinear_hat(u_hat)
        # AB2 for nonlinear + CN for diffusion
        u_hat  = (cn_num * u_hat + dt * (1.5 * N_curr - 0.5 * N_prev)) / cn_denom
        N_prev = N_curr

        if (step + 1) % N_sub == 0:
            traj.append(np.fft.irfft(u_hat, Nx).astype(np.float32))

    return np.array(traj)   # (N_save, Nx)


# =============================================================================
# Allen-Cahn equation  —  semi-implicit CN spectral solver
# =============================================================================
# Scheme: Crank-Nicolson for ε² ∂²u/∂x²  (implicit, handles stiffness for
#         small ε). Explicit Euler for reaction  u(1−u²).
#
# With N_sub=100, T=0.5, N_save=101 → dt = 5e-5, stable for ε down to 0.001.
# =============================================================================

def solve_allen_cahn(
    u0: np.ndarray,
    eps: float,
    T: float = 0.5,
    N_save: int = 101,
    N_sub: int = 100,
) -> np.ndarray:
    """
    Solve 1D Allen-Cahn equation on periodic [0,1].

    Parameters
    ----------
    u0     : Initial condition in (−1, +1), shape (Nx,).
    eps    : Interface width parameter (ε).  Typical: 0.005 – 0.02.
    T      : Final time.
    N_save : Number of saved snapshots (including t=0).
    N_sub  : Sub-steps between saved snapshots.

    Returns
    -------
    trajectory : float32 array, shape (N_save, Nx).
    """
    Nx      = len(u0)
    N_steps = N_sub * (N_save - 1)
    dt      = T / N_steps

    k  = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=1.0 / Nx)
    k2 = k ** 2

    # Crank-Nicolson factors for diffusion term  ε² ∂²u/∂x²
    cn_num   = 1.0 - 0.5 * eps ** 2 * k2 * dt
    cn_denom = 1.0 + 0.5 * eps ** 2 * k2 * dt

    u    = u0.astype(np.float64).copy()
    traj = [u.astype(np.float32)]

    for step in range(N_steps):
        u_hat  = np.fft.rfft(u)
        nl_hat = np.fft.rfft(u * (1.0 - u ** 2))        # explicit reaction
        u_hat  = (cn_num * u_hat + dt * nl_hat) / cn_denom
        u      = np.fft.irfft(u_hat, Nx)

        if (step + 1) % N_sub == 0:
            traj.append(u.astype(np.float32))

    return np.array(traj)   # (N_save, Nx)


# =============================================================================
# Advection equation  —  exact spectral solution (Fourier phase shift)
# =============================================================================
# The exact solution is  u(x,t) = u(x − βt)  on the periodic domain, which
# in Fourier space is simply  û(k,t) = û(k,0) · exp(−iβkt).
# No time-stepping error whatsoever — a perfect clean-room baseline.
# =============================================================================

def solve_advection(
    u0: np.ndarray,
    beta: float,
    T: float = 1.0,
    N_save: int = 101,
) -> np.ndarray:
    """
    Solve 1D Advection equation on periodic [0,1].
    Exact solution via Fourier phase multiplication.

    Parameters
    ----------
    u0     : Initial condition, shape (Nx,).
    beta   : Advection speed (can be negative).
    T      : Final time.
    N_save : Number of saved snapshots (including t=0).

    Returns
    -------
    trajectory : float32 array, shape (N_save, Nx).
    """
    Nx      = len(u0)
    k       = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=1.0 / Nx)
    t_saves = np.linspace(0.0, T, N_save)
    u0_hat  = np.fft.rfft(u0.astype(np.float64))

    traj = []
    for t in t_saves:
        u_hat_t = u0_hat * np.exp(-1j * beta * k * t)
        traj.append(np.fft.irfft(u_hat_t, Nx).astype(np.float32))

    return np.array(traj)   # (N_save, Nx)


# =============================================================================
# Dataset writer
# =============================================================================

def generate_dataset(
    pde: str,
    save_path: str,
    N_samples: int = 1000,
    Nx: int = 1024,
    N_save: int = 101,
    **solver_kwargs,
) -> str:
    """
    Generate a synthetic dataset and save to HDF5 with ``tensor`` key.
    Output shape: (N_samples, N_save, Nx).  Compatible with PDEBench1DDataset.

    Parameters
    ----------
    pde          : One of ``'burgers'``, ``'allencahn'``, ``'advection'``.
    save_path    : Output HDF5 file path.
    N_samples    : Number of trajectories to generate.
    Nx           : Spatial resolution (must be 1024 for PNO).
    N_save       : Number of time snapshots per trajectory (including IC).
    **solver_kwargs : Passed directly to the corresponding solver function.
    """
    _solvers = {
        "burgers":   solve_burgers,
        "allencahn": solve_allen_cahn,
        "advection": solve_advection,
    }
    _ic_fns = {
        "burgers":   lambda s: ic_fourier_random(Nx, n_modes=10, seed=s),
        "allencahn": lambda s: ic_tanh_interfaces(
                         Nx, n_interfaces=3,
                         eps=solver_kwargs.get("eps", 0.01), seed=s),
        "advection": lambda s: ic_fourier_random(Nx, n_modes=8, seed=s),
    }

    assert pde in _solvers, f"Unknown PDE '{pde}'. Choose from {list(_solvers)}."
    solver = _solvers[pde]
    ic_fn  = _ic_fns[pde]

    data = np.zeros((N_samples, N_save, Nx), dtype=np.float32)

    for i in tqdm(range(N_samples), desc=f"[{pde}]  generating"):
        u0         = ic_fn(i)
        data[i, :] = solver(u0, N_save=N_save, **solver_kwargs)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with h5py.File(save_path, "w") as f:
        ds = f.create_dataset("tensor", data=data,
                               compression="gzip", compression_opts=4)
        f.attrs["pde"]       = pde
        f.attrs["N_samples"] = N_samples
        f.attrs["Nx"]        = Nx
        f.attrs["N_save"]    = N_save
        for key, val in solver_kwargs.items():
            f.attrs[key] = val

    print(f"\nSaved {N_samples} samples -> {save_path}")
    return save_path


# =============================================================================
# CLI
# =============================================================================

_DATASETS_DIR = r"C:\Users\DIAT\ashish\neural_operators\datasets"

_FNAME_TEMPLATES = {
    "burgers":   "1D_Burgers_Nu{nu}.hdf5",
    "allencahn": "1D_AllenCahn_Eps{eps}.hdf5",
    "advection": "1D_Advection_Beta{beta}.hdf5",
}

_DEFAULT_SOLVER_KWARGS = {
    "burgers":   dict(N_sub=50),
    "allencahn": dict(N_sub=100),
    "advection": dict(),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic 1D PDE datasets for PNO experiments."
    )
    parser.add_argument("--pde",      type=str,   required=True,
                        choices=["burgers", "allencahn", "advection"])
    parser.add_argument("--N",        type=int,   default=1000,
                        help="Number of samples  (default 1000)")
    parser.add_argument("--N_save",   type=int,   default=101,
                        help="Saved time snapshots per sample  (default 101, includes t=0)")
    parser.add_argument("--Nx",       type=int,   default=1024,
                        help="Spatial resolution  (must be 1024 for PNO)")
    parser.add_argument("--save_dir", type=str,   default=_DATASETS_DIR)
    parser.add_argument("--T",        type=float, default=None,
                        help="Final time  (defaults: Burgers=1.0, AC=0.5, Adv=1.0)")
    # PDE-specific parameters
    parser.add_argument("--nu",   type=float, default=0.01,
                        help="[burgers]   kinematic viscosity  (default 0.01)")
    parser.add_argument("--eps",  type=float, default=0.01,
                        help="[allencahn] interface width ε     (default 0.01)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="[advection] advection speed β     (default 1.0)")
    parser.add_argument("--N_sub", type=int,  default=None,
                        help="Sub-steps between snapshots  (auto if not set)")
    args = parser.parse_args()

    # Build solver kwargs
    T_defaults    = {"burgers": 1.0, "allencahn": 0.5, "advection": 1.0}
    pde_params    = {
        "burgers":   {"nu":   args.nu},
        "allencahn": {"eps":  args.eps},
        "advection": {"beta": args.beta},
    }
    kwargs = _DEFAULT_SOLVER_KWARGS[args.pde].copy()
    kwargs.update(pde_params[args.pde])
    kwargs["T"]     = args.T if args.T is not None else T_defaults[args.pde]
    if args.N_sub is not None:
        kwargs["N_sub"] = args.N_sub
    if args.pde == "advection" and "N_sub" in kwargs:
        del kwargs["N_sub"]   # advection solver has no N_sub

    param_val = {"burgers": args.nu, "allencahn": args.eps, "advection": args.beta}
    fname     = _FNAME_TEMPLATES[args.pde].format(**{
        {"burgers": "nu", "allencahn": "eps", "advection": "beta"}[args.pde]: param_val[args.pde]
    })
    save_path = os.path.join(args.save_dir, fname)

    generate_dataset(
        pde=args.pde,
        save_path=save_path,
        N_samples=args.N,
        Nx=args.Nx,
        N_save=args.N_save,
        **kwargs,
    )
