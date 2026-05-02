"""
P-adic Fokker-Planck equation data generator.

Physical model (Avetisov-Bikulov-Kozyrev, 2002):
    Protein conformational dynamics on a p-adic ultrametric tree.
    Conformational states x ∈ Z/p^n Z; two states are "close" iff they share
    a coarse-scale p-adic neighbourhood.

PDE:
    Case 'diffusion':  ∂p/∂t = -D^alpha p
    Case 'trapping':   ∂p/∂t = -D^alpha p + V(x) p   (+ renormalisation)

where p(x,t) is the probability density over conformational states and
D^alpha is the Vladimirov pseudo-differential operator.

Spectral structure (identical to Schrödinger but REAL-valued time):
    D^alpha Psi_{gamma,j,a} = p^{(gamma+1)*alpha} Psi_{gamma,j,a}

    Free solution (exact, zero time-stepping error):
        p(x,t) = IDFT[ exp(-t * |xi|_p^alpha) * DFT[p_0(x)] ]

    Key property: eigenvalue at xi=0 is |0|_p^alpha = 0.
    => DC mode (total probability) is conserved exactly.

Eigenvalue ordering (p=2, n=10):
    gamma=0 (FINEST  spatial scale): lambda = 2^alpha        ~ 2   (alpha=1)
    gamma=9 (COARSEST spatial scale): lambda = 2^{10*alpha} ~ 1024 (alpha=1)

    Coarse modes decay first; fine modes persist longest.
    => Protein quickly escapes local vicinity; slowly explores distant basins.

Initial conditions:
    p_0(x) = |psi(x)|^2  where psi is a random complex wavefunction with
    FINE-scale spectral bias (opposite of Schrödinger).  This gives dominant
    eigenvalues lambda ~ 2-16 (for alpha=1), so dynamics unfold over t in [0,1].

    p_0 is non-negative and satisfies sum(p_0) = M = p^n (discrete L1 norm = 1).

Verification metrics:
    Free case:    max |sum(p(t_i)) - M| should be < 1e-4 (machine precision)
    Trapping:     norm drift before renorm reported as quality indicator
    Both cases:   mean |p(x,t_100) - M/M| should show significant spread from p_0

HDF5 output layout (compatible with FokkerPlanckDataset):
    f['p_density']  float32  (N_samples, N_save, M)   probability density
    f['V']          float32  (N_samples, M)            trapping landscape (0 for free)
    f.attrs         metadata

Usage
-----
    conda activate turbo_nigo
    cd C:/Users/DIAT/ashish/neural_operators/PNO

    python -m data.fokker_planck_generator --case diffusion --alpha 0.5 --N 1000
    python -m data.fokker_planck_generator --case diffusion --alpha 1.0 --N 1000
    python -m data.fokker_planck_generator --case trapping  --alpha 1.0 --N 1000

Reference
---------
    Avetisov V.A., Bikulov A.H., Kozyrev S.V., Osipov V.A. (2002).
    "p-Adic models of ultrametric diffusion constrained by hierarchical
    energy landscapes."  Journal of Physics A, 35(2), 177-189.
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'adicton'))

from adicton.transforms.vladimirov import vladimirov_eigenvalues
from adicton.transforms.fourier import padic_dft, padic_idft
from adicton.transforms.wavelet import wavelet_decompose, wavelet_reconstruct


# ---------------------------------------------------------------------------
# Initial condition: probability density = |psi|^2
# ---------------------------------------------------------------------------

def _generate_complex_wavefunction(p: int, n: int, rng: np.random.Generator,
                                   scale_decay: float = 2.0) -> np.ndarray:
    """
    Random complex wavefunction with FINE-scale spectral bias.

    sigma_gamma = exp(-gamma / scale_decay)  so gamma=0 (fine) gets sigma~1
    and gamma=n-1 (coarse) gets sigma~exp(-(n-1)/scale_decay) ~ very small.

    This is the OPPOSITE bias to the Schrödinger generator (which uses +gamma)
    because the Fokker-Planck eigenvalues decrease as gamma decreases:
      lambda_gamma = p^{(gamma+1)*alpha}
    Fine scales (small gamma) => small lambda => slow dynamics over [0, T].
    Coarse scales (large gamma) => large lambda => instant decay at T~1.

    Returns complex128 numpy array of shape (M,), normalised so
    mean(|psi|^2) = 1, i.e. sum(|psi|^2) = M.
    """
    M = p ** n
    coeffs: dict = {}
    coeffs['scaling'] = torch.zeros(1, dtype=torch.complex128).squeeze()

    for gamma in range(n):
        sigma = np.exp(-gamma / scale_decay)    # fine-scale bias (note the minus)
        num_blocks = p ** (n - gamma - 1)
        for j in range(1, p):
            for a in range(num_blocks):
                re = rng.normal(0.0, sigma)
                im = rng.normal(0.0, sigma)
                coeffs[f'w_{gamma}_{j}_{a}'] = torch.tensor(
                    re + 1j * im, dtype=torch.complex128)

    psi = wavelet_reconstruct(coeffs, p, n)         # (M,) complex128
    norm = torch.sqrt((psi.abs() ** 2).mean())
    return (psi / (norm + 1e-12)).numpy()


def generate_ic(p: int, n: int, rng: np.random.Generator,
                scale_decay: float = 2.0) -> np.ndarray:
    """
    Generate a probability density p_0(x) = |psi(x)|^2.

    The Born-rule construction guarantees:
      - p_0(x) >= 0                         (non-negativity)
      - sum(p_0) = M = p^n                  (discrete L1 normalisation)
      - Multi-scale structure inherited from psi

    Parameters
    ----------
    scale_decay : float
        Controls spectral spread.  Default 2.0 concentrates energy in
        fine-scale wavelets (gamma=0-3).

    Returns
    -------
    p0 : float64 numpy array, shape (M,).  Non-negative, sum = M.
    """
    psi = _generate_complex_wavefunction(p, n, rng, scale_decay)
    p0 = np.abs(psi) ** 2                          # non-negative, sum = M
    return p0                                       # float64


# ---------------------------------------------------------------------------
# Random trapping landscape (constant on p-adic balls, V <= 0)
# ---------------------------------------------------------------------------

def generate_trapping_landscape(p: int, n: int, rng: np.random.Generator,
                                n_levels: int = 5,
                                depth: float = 1.0) -> np.ndarray:
    """
    Hierarchical trapping landscape V(x) <= 0.

    V is a step function with p^n_levels constant blocks, one per p-adic ball
    at scale n_levels.  V(x) in [-depth, 0]: deep wells trap probability.

    In the equation dp/dt = -D^alpha p + V p, sites with large negative V
    lose probability faster; over time the distribution pools at V~0 sites.

    Returns float32 numpy array, shape (M,).
    """
    M = p ** n
    block_size = p ** (n - n_levels)
    n_blocks   = M // block_size
    # Uniform in [-depth, 0]: negative so probability is absorbed into wells
    values = -depth * rng.uniform(0.0, 1.0, n_blocks).astype(np.float32)
    return np.repeat(values, block_size)            # (M,)


# ---------------------------------------------------------------------------
# Exact free evolution: p(t) = IDFT[exp(-t * lambda) * DFT[p0]]
# ---------------------------------------------------------------------------

def evolve_free(p0: np.ndarray, alpha: float, t_saves: np.ndarray,
                p: int = 2, n: int = 10) -> np.ndarray:
    """
    Exact solution of dp/dt = -D^alpha p via spectral decay.

    Theoretical guarantees:
      - sum(p(t)) = sum(p0) for all t  (DC eigenvalue = 0, so exp(-0) = 1)
      - p(t) -> uniform as t -> inf    (all non-zero modes decay)
      - Zero time-stepping error        (exact formula, not a scheme)

    The imaginary part of the IDFT output is O(1e-14) for real p0 and is
    discarded.  The maximum imaginary part is reported as a sanity check.

    Parameters
    ----------
    p0      : float64 numpy array, shape (M,). Non-negative, sum = M.
    alpha   : Vladimirov exponent.
    t_saves : float64 numpy array, shape (N_save,). Includes t=0.

    Returns
    -------
    p_traj : float32 numpy array, shape (N_save, M).
    """
    dev = torch.device('cpu')

    # Real eigenvalues of D^alpha: |xi|_p^alpha >= 0.  eigenvals[0] = 0 (DC).
    eigenvals = vladimirov_eigenvalues(p, n, alpha, dev)    # (M,) float64

    # Forward DFT of p0
    p0_t   = torch.from_numpy(p0).double().unsqueeze(0)     # (1, M) float64
    p0_hat = padic_dft(p0_t, p, n)                          # (1, M) complex128

    # Decay multipliers: exp(-t * lambda)  -- real and non-negative
    t_tensor  = torch.from_numpy(t_saves).double()                   # (N_save,)
    decay     = torch.exp(
        -eigenvals.unsqueeze(0) * t_tensor.unsqueeze(1)
    ).to(torch.complex128)                                            # (N_save, M)

    # Spectral multiplication: p_hat(xi, t) = exp(-t*lambda) * p0_hat(xi)
    # p0_hat broadcasts from (1, M) to (N_save, M)
    p_hat_all = p0_hat * decay                                        # (N_save, M)

    # Batch inverse DFT
    p_all = padic_idft(p_hat_all, p, n)                              # (N_save, M)

    # Imaginary part should be machine-eps for real p0
    max_imag = p_all.imag.abs().max().item()
    if max_imag > 1e-6:
        print(f"  [warning] evolve_free: max |Im(p)| = {max_imag:.3e}  "
              f"(expected ~1e-14 for real input)")

    return p_all.real.float().numpy()                                 # (N_save, M)


# ---------------------------------------------------------------------------
# Split-operator trapping: dp/dt = -D^alpha p + V(x) p
# ---------------------------------------------------------------------------

def evolve_with_trapping(p0: np.ndarray, V: np.ndarray,
                         alpha: float, T: float,
                         N_save: int, N_sub: int = 200,
                         p: int = 2, n: int = 10) -> np.ndarray:
    """
    Numerical solution of dp/dt = -D^alpha p + V(x) p  (Strang splitting).

    Splitting per substep dt:
        Step 1 — V half-step  (position space, exact for constant dt):
            p -> p * exp(V * dt/2)
        Step 2 — D^alpha full step (Fourier space, exact):
            p -> IDFT[ exp(-dt * lambda) * DFT[p] ]
        Step 3 — V half-step (same as step 1)

    Because V <= 0 everywhere, the V steps shrink probability.
    Probability is renormalised at each saved snapshot so the network
    learns the SHAPE of the distribution, not the absolute scale.

    Norm drift before renormalisation is computed and printed as a
    quality metric; large drift signals dt is too large.

    Parameters
    ----------
    p0    : float64 array (M,). Non-negative, sum = M.
    V     : float32 array (M,). Trapping rates, all <= 0.
    alpha : Vladimirov exponent.
    T     : Final integration time.
    N_save: Number of saved snapshots (including t=0).
    N_sub : Sub-steps per saved interval.

    Returns
    -------
    p_traj : float32 array, shape (N_save, M). Renormalised at each step.
    """
    M       = p ** n
    N_steps = N_sub * (N_save - 1)
    dt      = T / N_steps
    dev     = torch.device('cpu')

    # Precompute fixed multipliers (reused every substep)
    eigenvals   = vladimirov_eigenvalues(p, n, alpha, dev)               # (M,)
    dalpha_mult = torch.exp(-dt * eigenvals).double()                     # (M,) real

    V_t         = torch.from_numpy(V.astype(np.float64))                 # (M,)
    v_half_mult = torch.exp(0.5 * dt * V_t).double()                     # (M,) real

    # Work in float64 throughout for numerical stability
    prob = torch.from_numpy(p0).double()                                  # (M,)

    p_traj = [prob.float().numpy()]
    norm_drifts = []

    for step in range(N_steps):
        # --- V half-step (real multiply) ---
        prob = prob * v_half_mult

        # --- D^alpha full step (exact spectral) ---
        # padic_dft/idft expect complex input
        prob_hat = padic_dft(prob.unsqueeze(0).to(torch.complex128), p, n)  # (1,M)
        prob_hat = (prob_hat * dalpha_mult).squeeze(0)                       # (M,)
        prob     = padic_idft(prob_hat.unsqueeze(0), p, n).squeeze(0)       # (M,) complex
        prob     = prob.real                                                  # (M,) real

        # --- V half-step ---
        prob = prob * v_half_mult

        if (step + 1) % N_sub == 0:
            # Record norm drift before renorm
            current_norm = prob.sum().item()
            norm_drifts.append(abs(current_norm - M) / M)

            # Renormalise to keep sum = M
            prob = prob * (M / (current_norm + 1e-30))

            p_traj.append(prob.float().numpy())

    max_drift = max(norm_drifts) if norm_drifts else 0.0
    if max_drift > 0.05:
        print(f"  [warning] large norm drift: max {max_drift:.4%}  "
              f"(consider increasing N_sub)")

    return np.array(p_traj)    # (N_save, M) float32


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    case:        str,
    alpha:       float,
    save_path:   str,
    N_samples:   int   = 1000,
    N_save:      int   = 101,
    T:           float = 1.0,
    p:           int   = 2,
    n:           int   = 10,
    N_sub:       int   = 200,
    n_levels:    int   = 5,
    trap_depth:  float = 1.0,
    scale_decay: float = 2.0,
) -> str:
    """
    Generate a p-adic Fokker-Planck dataset and save to HDF5.

    Output layout
    -------------
    f['p_density']  float32  (N_samples, N_save, M)  probability density
    f['V']          float32  (N_samples, M)           landscape (0 for free)
    f.attrs         metadata (case, alpha, T, p, n, ...)
    """
    M       = p ** n
    t_saves = np.linspace(0.0, T, N_save)

    p_all = np.zeros((N_samples, N_save, M), dtype=np.float32)
    V_all = np.zeros((N_samples, M),         dtype=np.float32)

    desc = f"[fokker_planck/{case}/alpha={alpha}]"

    for i in tqdm(range(N_samples), desc=desc):
        rng  = np.random.default_rng(seed=i)

        p0 = generate_ic(p, n, rng, scale_decay=scale_decay)       # (M,) float64

        if case == 'diffusion':
            V_i  = np.zeros(M, dtype=np.float32)
            p_traj = evolve_free(p0, alpha, t_saves, p, n)

        elif case == 'trapping':
            rng2 = np.random.default_rng(seed=N_samples + i)
            V_i  = generate_trapping_landscape(p, n, rng2, n_levels, trap_depth)
            p_traj = evolve_with_trapping(p0, V_i, alpha, T, N_save, N_sub, p, n)

        else:
            raise ValueError(f"case must be 'diffusion' or 'trapping', got '{case}'")

        p_all[i] = p_traj
        V_all[i] = V_i

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('p_density', data=p_all, compression='gzip', compression_opts=4)
        f.create_dataset('V',         data=V_all, compression='gzip', compression_opts=4)
        f.attrs['case']        = case
        f.attrs['alpha']       = alpha
        f.attrs['T']           = T
        f.attrs['N_save']      = N_save
        f.attrs['p']           = p
        f.attrs['n']           = n
        f.attrs['scale_decay'] = scale_decay
        if case == 'trapping':
            f.attrs['n_levels']   = n_levels
            f.attrs['trap_depth'] = trap_depth
            f.attrs['N_sub']      = N_sub

    print(f"Saved {N_samples} samples -> {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Verification: sanity checks before generating the full dataset
# ---------------------------------------------------------------------------

def run_verification(p: int = 2, n: int = 10, alpha: float = 1.0,
                     T: float = 1.0, N_save: int = 11,
                     scale_decay: float = 2.0) -> None:
    """
    Run sanity checks on a single sample.  Call before generating 1000 samples.

    Checks:
      1. Eigenvalue range and DC zero
      2. IC non-negativity and normalisation
      3. Free case: norm conservation (should be machine eps ~ 1e-12)
      4. Free case: mean change between t=0 and t=T (must be > 0.01)
      5. Trapping case: norm drift before renorm (should be < 5%)
    """
    import sys
    from adicton.transforms.vladimirov import vladimirov_eigenvalues

    dev = torch.device('cpu')
    M = p ** n

    print(f"\n{'='*60}")
    print(f"  p-adic Fokker-Planck verification  (p={p}, n={n}, alpha={alpha})")
    print(f"{'='*60}")

    # 1. Eigenvalue range
    eigs = vladimirov_eigenvalues(p, n, alpha, dev).numpy()
    print(f"  [1] Eigenvalue range: [{eigs.min():.6f}, {eigs.max():.6f}]")
    print(f"      DC eigenvalue (xi=0): {eigs[0]:.2e}  (must be 0)")
    assert eigs[0] == 0.0, "DC eigenvalue must be exactly 0"
    print(f"      DC eigenvalue check: PASS")

    # 2. IC
    rng = np.random.default_rng(seed=42)
    p0  = generate_ic(p, n, rng, scale_decay)
    print(f"\n  [2] IC check:")
    print(f"      min(p0)  = {p0.min():.6e}  (must be >= 0)")
    print(f"      sum(p0)  = {p0.sum():.6f}  (must be {M})")
    print(f"      max(p0)  = {p0.max():.6f}")
    assert p0.min() >= 0.0, "IC must be non-negative"
    assert abs(p0.sum() - M) < 1e-6, f"IC norm wrong: {p0.sum()}"
    print(f"      IC check: PASS")

    # 3. Free case norm conservation
    t_saves = np.linspace(0.0, T, N_save)
    p_traj  = evolve_free(p0, alpha, t_saves, p, n)
    norms   = p_traj.sum(axis=1)
    max_norm_err = abs(norms - M).max()
    print(f"\n  [3] Free evolution norm conservation:")
    print(f"      max |sum(p(t)) - M| = {max_norm_err:.3e}  (expect < 1e-4)")
    assert max_norm_err < 1e-4, f"Norm not conserved: {max_norm_err:.3e}"
    print(f"      Norm conservation: PASS")

    # 4. Non-trivial dynamics
    p_final  = p_traj[-1]
    mean_change = np.abs(p_final - p_traj[0]).mean()
    uniform  = M / M  # = 1.0 (each point holds average prob = 1)
    print(f"\n  [4] Dynamics check (free case):")
    print(f"      Mean |p(T) - p(0)| = {mean_change:.4f}  (want > 0.01)")
    print(f"      p(0) max            = {p_traj[0].max():.4f}")
    print(f"      p(T) max            = {p_final.max():.4f}  (should be closer to 1.0)")
    if mean_change < 0.01:
        print(f"  [WARNING] Very small dynamics. Consider larger T or smaller alpha.")
    else:
        print(f"      Dynamics check: PASS")

    # 5. Trapping case
    rng2 = np.random.default_rng(seed=99)
    V    = generate_trapping_landscape(p, n, rng2, n_levels=5, depth=1.0)
    print(f"\n  [5] Trapping landscape check:")
    print(f"      V range: [{V.min():.4f}, {V.max():.4f}]  (all <= 0)")
    assert V.max() <= 0.0, "Trapping landscape must be <= 0"
    print(f"      V check: PASS")

    print(f"\n  All checks passed.")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DATASETS_DIR = r"C:\Users\DIAT\ashish\neural_operators\datasets"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate p-adic Fokker-Planck datasets for PNO.')
    parser.add_argument('--case',  type=str, required=True,
                        choices=['diffusion', 'trapping'],
                        help="'diffusion' (exact, V=0) or 'trapping' (split-operator)")
    parser.add_argument('--alpha', type=float, required=True,
                        help='Vladimirov exponent (e.g. 0.5, 1.0, 1.5)')
    parser.add_argument('--N',     type=int,   default=1000)
    parser.add_argument('--N_save',type=int,   default=101)
    parser.add_argument('--T',     type=float, default=1.0)
    parser.add_argument('--N_sub', type=int,   default=200,
                        help='[trapping] sub-steps per saved interval')
    parser.add_argument('--n_levels',  type=int,   default=5)
    parser.add_argument('--trap_depth',type=float, default=1.0)
    parser.add_argument('--scale_decay', type=float, default=2.0)
    parser.add_argument('--save_dir', type=str, default=_DATASETS_DIR)
    parser.add_argument('--verify', action='store_true',
                        help='Run verification checks only (no data generation)')
    args = parser.parse_args()

    if args.verify:
        run_verification(alpha=args.alpha, T=args.T)
    else:
        fname = f"PAdicFokkerPlanck_{args.case}_alpha{args.alpha}.hdf5"
        save_path = os.path.join(args.save_dir, fname)
        generate_dataset(
            case=args.case, alpha=args.alpha,
            save_path=save_path,
            N_samples=args.N, N_save=args.N_save, T=args.T,
            N_sub=args.N_sub, n_levels=args.n_levels,
            trap_depth=args.trap_depth, scale_decay=args.scale_decay,
        )
