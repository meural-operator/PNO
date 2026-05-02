"""
P-adic Schrodinger equation data generator for the PNO project.

PDE (p-adic quantum mechanics, Vladimirov-Volovich formulation):

    i hbar d/dt psi(x,t) = ( D^alpha + V(|x|_p) ) psi(x,t)

where:
  - psi in L^2(Z/p^n Z),  M = p^n = 1024 spatial points
  - D^alpha is the Vladimirov pseudo-differential operator
  - V(|x|_p) is a potential depending on the p-adic absolute value

Key theoretical property exploited here:

    D^alpha Psi_{gamma,j,a} = p^{(gamma+1)*alpha} * Psi_{gamma,j,a}

So Kozyrev wavelets are EXACT eigenfunctions of D^alpha. Time evolution under
the free Hamiltonian H = D^alpha is therefore exact to machine precision:

    psi(t) = IDFT[ exp(-i * t * |xi|_p^alpha) * DFT[psi_0] ]

No time-stepping error, no spectral leakage. This is the theoretical
differentiator vs FNO, which must approximate D^alpha in the Fourier basis.

Eigenvalue structure (p=2, n=10):
  - gamma=0 (finest spatial scale): lambda_0 = 2^alpha   (smallest)
  - gamma=9 (coarsest spatial scale): lambda_9 = 2^{10 alpha} (largest)

Initial conditions are generated with energy concentrated at FINE scales
(small gamma), so the dominant eigenvalues are small and the dynamics are
well-resolved at T=1.0 with 100 saved snapshots.

Two experiment cases:
  'free'      - V=0, free Schrodinger, exact closed-form solution
  'potential' - V(x) a random p-adic hierarchical potential, split-operator

HDF5 output layout (compatible with PAdicSchrodingerDataset):
    f['psi_re']  : float32, shape (N_samples, N_save, M)
    f['psi_im']  : float32, shape (N_samples, N_save, M)
    f['V']       : float32, shape (N_samples, M)   [zeros for free case]

Usage
-----
    conda activate turbo_nigo
    cd C:/Users/DIAT/ashish/neural_operators/PNO

    python -m data.schrodinger_generator --case free      --alpha 0.5  --N 1000
    python -m data.schrodinger_generator --case free      --alpha 1.0  --N 1000
    python -m data.schrodinger_generator --case free      --alpha 1.5  --N 1000
    python -m data.schrodinger_generator --case potential --alpha 1.0  --N 1000
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

# Make adicton importable when called as a module from PNO root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'adicton'))

from adicton.transforms.vladimirov import vladimirov_eigenvalues
from adicton.transforms.fourier import padic_dft, padic_idft
from adicton.transforms.wavelet import wavelet_decompose, wavelet_reconstruct


# ---------------------------------------------------------------------------
# Initial condition generation  (wavelet-domain superposition)
# ---------------------------------------------------------------------------

def generate_ic(p: int, n: int, rng: np.random.Generator,
                scale_decay: float = 2.0) -> np.ndarray:
    """
    Generate a normalised complex IC as a weighted sum of Kozyrev wavelets.

    Eigenvalue ordering on Z/p^nZ (confirmed from vladimirov_eigenvalues):
        gamma=0   finest spatial scale  ->  Fourier xi ~ p^{n-1}  ->  |xi|_p^alpha SMALL
        gamma=n-1 coarsest spatial scale -> Fourier xi ~ 1         ->  |xi|_p^alpha LARGE (=1)

    Strategy: use a flat wavelet spectrum (equal energy per wavelet coefficient)
    so all scales are excited.  The coarse scales (gamma=n-1) then dominate the
    observable dynamics at T ~ 2pi because they have the largest eigenvalues.
    The resulting training data shows clear multi-scale evolution: coarse
    components oscillate rapidly, fine components are nearly frozen.

    'scale_decay' modulates a mild coarse-scale bias on top of the flat
    spectrum:  sigma_gamma = exp(+(gamma) / scale_decay) so that
    gamma = n-1 gets a small boost.  scale_decay=inf recovers the flat spectrum.

    Parameters
    ----------
    p, n        : p-adic parameters (p=2, n=10 -> M=1024).
    rng         : NumPy random generator (seeded for reproducibility).
    scale_decay : Coarse-scale bias strength.  Default 2.0 gives a mild
                  tilt; larger values approach flat.

    Returns
    -------
    psi0 : complex128 numpy array, shape (M,), normalised so
           mean(|psi0|^2) = 1.
    """
    M = p ** n

    coeffs: dict = {}
    coeffs['scaling'] = torch.zeros(1, dtype=torch.complex128).squeeze()

    for gamma in range(n):
        # Flat base + mild coarse-scale tilt
        # gamma=0 (fine, small eigenvalue) gets sigma ~ 1
        # gamma=n-1 (coarse, eigenvalue=1) gets sigma ~ exp((n-1)/scale_decay)
        sigma      = np.exp(gamma / scale_decay)
        num_blocks = p ** (n - gamma - 1)
        for j in range(1, p):
            for a in range(num_blocks):
                re = rng.normal(0.0, sigma)
                im = rng.normal(0.0, sigma)
                coeffs[f'w_{gamma}_{j}_{a}'] = torch.tensor(
                    re + 1j * im, dtype=torch.complex128)

    psi = wavelet_reconstruct(coeffs, p, n)          # (M,) complex128
    norm = torch.sqrt((psi.abs() ** 2).mean())
    return (psi / (norm + 1e-12)).numpy()            # complex128 numpy


# ---------------------------------------------------------------------------
# Random p-adic potential  (constant on p-adic balls)
# ---------------------------------------------------------------------------

def generate_potential(p: int, n: int, rng: np.random.Generator,
                       n_levels: int = 5) -> np.ndarray:
    """
    Generate a random potential that is constant on p-adic balls.

    V(x) is a step function with p^n_levels constant blocks.
    The p-adic ball structure means V encodes hierarchical "barriers"
    at scale p^{n_levels}.

    Parameters
    ----------
    n_levels : int
        Number of non-trivial refinement levels (1=coarsest, n=finest).
        Default 5 -> 2^5=32 constant blocks on the 1024-point grid.

    Returns
    -------
    V : float32 numpy array, shape (M,).
    """
    M = p ** n
    block_size = p ** (n - n_levels)   # size of each constant block
    n_blocks   = M // block_size       # total number of blocks

    values = rng.normal(0.0, 1.0, n_blocks).astype(np.float32)
    return np.repeat(values, block_size)   # (M,)


# ---------------------------------------------------------------------------
# Exact free-particle evolution  (vectorised over all save times at once)
# ---------------------------------------------------------------------------

def evolve_free(psi0: np.ndarray, alpha: float, t_saves: np.ndarray,
                p: int = 2, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute psi(t) for all t in t_saves EXACTLY via spectral phase rotation.

        psi(t) = IDFT[ exp(-i * t * |xi|_p^alpha) * DFT[psi_0] ]

    Zero time-stepping error. Vectorised: one DFT + one IDFT regardless
    of the number of save times.

    Parameters
    ----------
    psi0    : complex128 numpy array, shape (M,).
    alpha   : Vladimirov exponent.
    t_saves : float64 numpy array, shape (N_save,).

    Returns
    -------
    psi_re, psi_im : float32 numpy arrays, each shape (N_save, M).
                     Include t=0 (psi_0) as the first row.
    """
    M   = p ** n
    dev = torch.device('cpu')

    # Eigenvalues |xi|_p^alpha for all xi in Z/p^nZ
    eigenvals = vladimirov_eigenvalues(p, n, alpha, dev)   # (M,) float64

    # Forward DFT of psi0: shape (1, M)
    psi0_t = torch.from_numpy(psi0).to(torch.complex128).unsqueeze(0)
    psi0_hat = padic_dft(psi0_t, p, n)                    # (1, M)

    # Phase multipliers for all save times: (N_save, M)
    t_tensor = torch.from_numpy(t_saves).double()          # (N_save,)
    exponents = eigenvals.unsqueeze(0) * t_tensor.unsqueeze(1)   # (N_save, M)
    multipliers = torch.exp(-1j * exponents.to(torch.complex128))  # (N_save, M)

    # Apply multipliers (psi0_hat broadcasts from (1,M) to (N_save,M))
    psi_hat_all = psi0_hat * multipliers                   # (N_save, M)

    # Batch inverse DFT
    psi_all = padic_idft(psi_hat_all, p, n)                # (N_save, M)

    return psi_all.real.float().numpy(), psi_all.imag.float().numpy()


# ---------------------------------------------------------------------------
# Split-operator evolution with potential (2nd-order, unitary)
# ---------------------------------------------------------------------------

def evolve_with_potential(psi0: np.ndarray, V: np.ndarray,
                          alpha: float, T: float,
                          N_save: int, N_sub: int = 100,
                          p: int = 2, n: int = 10
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve i d/dt psi = (D^alpha + V) psi using the split-operator method.

    Scheme (Strang splitting, 2nd-order, unitary):
        psi -> exp(-i V dt/2) psi            [V half-step, position space]
             -> exp(-i D^alpha dt) psi        [D^alpha full step, Fourier]
             -> exp(-i V dt/2) psi            [V half-step, position space]

    The D^alpha step is EXACT (spectral multiplication by exp(-i dt |xi|^alpha)).
    The V step is exact for constant dt (diagonal in position space).

    Parameters
    ----------
    psi0    : complex128 numpy array, shape (M,).
    V       : float32 numpy array, shape (M,). The potential at each grid point.
    alpha   : Vladimirov exponent.
    T       : Final time.
    N_save  : Number of saved snapshots including t=0.
    N_sub   : Sub-steps per saved interval (controls accuracy of V term).

    Returns
    -------
    psi_re, psi_im : float32 numpy arrays, each shape (N_save, M).
    """
    M       = p ** n
    N_steps = N_sub * (N_save - 1)
    dt      = T / N_steps
    dev     = torch.device('cpu')

    # Precompute D^alpha full-step multiplier (cached, reused every substep)
    eigenvals      = vladimirov_eigenvalues(p, n, alpha, dev)           # (M,)
    dalpha_mult    = torch.exp(-1j * dt * eigenvals.to(torch.complex128))  # (M,)

    # Precompute V half-step multiplier
    V_t     = torch.from_numpy(V.astype(np.float64))
    V_half  = torch.exp(-1j * (dt / 2) * V_t.to(torch.complex128))    # (M,)

    # Initialise
    psi = torch.from_numpy(psi0).to(torch.complex128)                  # (M,)

    psi_re_list = [psi.real.float().numpy()]
    psi_im_list = [psi.imag.float().numpy()]

    for step in range(N_steps):
        # --- half V step ---
        psi = psi * V_half

        # --- full D^alpha step  (exact via Fourier) ---
        psi_hat = padic_dft(psi.unsqueeze(0), p, n)                    # (1,M)
        psi_hat = (psi_hat * dalpha_mult).squeeze(0)                   # (M,)
        psi     = padic_idft(psi_hat.unsqueeze(0), p, n).squeeze(0)   # (M,)

        # --- half V step ---
        psi = psi * V_half

        if (step + 1) % N_sub == 0:
            psi_re_list.append(psi.real.float().numpy())
            psi_im_list.append(psi.imag.float().numpy())

    return np.array(psi_re_list), np.array(psi_im_list)   # each (N_save, M)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    case:      str,           # 'free' or 'potential'
    alpha:     float,
    save_path: str,
    N_samples: int   = 1000,
    N_save:    int   = 101,
    T:         float = 1.0,
    p:         int   = 2,
    n:         int   = 10,
    N_sub:     int   = 100,   # only used for 'potential'
    n_levels:  int   = 5,     # potential coarseness for 'potential'
    scale_decay: float = 2.0, # IC spectral width
) -> str:
    """
    Generate a p-adic Schrodinger dataset and save to HDF5.

    Output file layout
    ------------------
    f['psi_re']  : float32  (N_samples, N_save, M)  Re(psi)
    f['psi_im']  : float32  (N_samples, N_save, M)  Im(psi)
    f['V']       : float32  (N_samples, M)           potential  (zeros for free)
    f.attrs      : metadata (case, alpha, T, p, n, ...)
    """
    M       = p ** n
    t_saves = np.linspace(0.0, T, N_save)

    psi_re_all = np.zeros((N_samples, N_save, M), dtype=np.float32)
    psi_im_all = np.zeros((N_samples, N_save, M), dtype=np.float32)
    V_all      = np.zeros((N_samples, M),         dtype=np.float32)

    desc = f"[schrodinger/{case}/alpha={alpha}]"

    for i in tqdm(range(N_samples), desc=desc):
        rng  = np.random.default_rng(seed=i)

        # Initial condition
        psi0 = generate_ic(p, n, rng, scale_decay=scale_decay)

        if case == 'free':
            # Exact closed-form solution
            V_i = np.zeros(M, dtype=np.float32)
            psi_re, psi_im = evolve_free(psi0, alpha, t_saves, p, n)

        elif case == 'potential':
            # Random p-adic hierarchical potential
            rng2 = np.random.default_rng(seed=N_samples + i)
            V_i  = generate_potential(p, n, rng2, n_levels=n_levels)
            psi_re, psi_im = evolve_with_potential(
                psi0, V_i, alpha, T, N_save, N_sub, p, n)

        else:
            raise ValueError(f"case must be 'free' or 'potential', got '{case}'")

        psi_re_all[i] = psi_re   # (N_save, M)
        psi_im_all[i] = psi_im
        V_all[i]      = V_i

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('psi_re', data=psi_re_all, compression='gzip', compression_opts=4)
        f.create_dataset('psi_im', data=psi_im_all, compression='gzip', compression_opts=4)
        f.create_dataset('V',      data=V_all,      compression='gzip', compression_opts=4)
        f.attrs['case']        = case
        f.attrs['alpha']       = alpha
        f.attrs['T']           = T
        f.attrs['N_save']      = N_save
        f.attrs['p']           = p
        f.attrs['n']           = n
        f.attrs['scale_decay'] = scale_decay
        if case == 'potential':
            f.attrs['n_levels'] = n_levels
            f.attrs['N_sub']    = N_sub

    print(f"Saved {N_samples} samples -> {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DATASETS_DIR = r"C:\Users\DIAT\ashish\neural_operators\datasets"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate p-adic Schrodinger equation datasets for PNO.')
    parser.add_argument('--case',  type=str,   required=True,
                        choices=['free', 'potential'],
                        help="'free' (exact, V=0) or 'potential' (split-operator)")
    parser.add_argument('--alpha', type=float, required=True,
                        help='Vladimirov fractional exponent (e.g. 0.5, 1.0, 1.5)')
    parser.add_argument('--N',     type=int,   default=1000, help='Number of samples')
    parser.add_argument('--N_save',type=int,   default=101,  help='Saved time snapshots')
    parser.add_argument('--T',     type=float, default=1.0,  help='Final time')
    parser.add_argument('--N_sub', type=int,   default=100,
                        help='[potential] sub-steps per saved interval')
    parser.add_argument('--n_levels', type=int, default=5,
                        help='[potential] coarseness of potential (1-10)')
    parser.add_argument('--scale_decay', type=float, default=2.0,
                        help='IC spectral width parameter')
    parser.add_argument('--save_dir', type=str, default=_DATASETS_DIR)
    args = parser.parse_args()

    fname = f"PAdicSchrodinger_{args.case}_alpha{args.alpha}.hdf5"
    save_path = os.path.join(args.save_dir, fname)

    generate_dataset(
        case=args.case, alpha=args.alpha,
        save_path=save_path,
        N_samples=args.N, N_save=args.N_save, T=args.T,
        N_sub=args.N_sub, n_levels=args.n_levels,
        scale_decay=args.scale_decay,
    )
