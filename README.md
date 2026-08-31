# PNO: P-adic Wavelet Neural Operator for PDE Dynamics

PNO is an experimental **P-adic Neural Operator** for learning solution operators of time-dependent partial differential equations using a **Kozyrev p-adic wavelet representation** rather than a conventional Euclidean Fourier basis.

The central idea is to represent a spatial field on the finite p-adic domain

\[
\mathbb{Z}/p^n\mathbb{Z}, \qquad M=p^n,
\]

apply learnable channel-mixing operators directly to its multiscale p-adic wavelet coefficients, and reconstruct the field back to physical space. The resulting architecture provides a hierarchical/ultrametric alternative to Fourier-based neural operators and supports experiments on both classical PDEs and equations whose dynamics are themselves formulated using p-adic operators.

> **Research status:** this repository is an experimental research implementation. It contains the model, training pipeline, dataset interfaces, synthetic PDE generators, p-adic PDE generators, and experiment configurations. Trained checkpoints and consolidated benchmark tables are not currently committed to the repository.

---

## What PNO does

For most experiments, PNO learns a single-shot solution map

\[
\mathcal{G}_\theta:\; a(x) \longmapsto \{u(x,t_1),\ldots,u(x,t_T)\}.
\]

Rather than recursively advancing one time step at a time, the current implementation predicts the requested future trajectory simultaneously as output channels.

PNO combines:

- **Kozyrev p-adic wavelet decomposition** on \(\mathbb{Z}/p^n\mathbb{Z}\)
- **Scale-dependent learnable wavelet multipliers**
- **Weight sharing across translations at a fixed p-adic scale**
- **P-adic-aware pointwise residual channel mixing**
- **GELU nonlinearities** between operator blocks
- **Gaussian input/output normalization**
- **Relative \(L^2\) training loss**
- **Single-shot long-horizon trajectory prediction**

The p-adic transforms and p-adic-aware layers are supplied by the [`adicton`](https://github.com/meural-operator/adicton) Git submodule.

---

## Core operator

Let

\[
x\in\mathbb{R}^{C_{\mathrm{in}}\times M},\qquad M=p^n.
\]

For each input channel, PNO computes its p-adic wavelet expansion

\[
x \;\longleftrightarrow\; \left(c_{\mathrm{sc}},\;c_{\gamma,j,a}\right),
\]

where \(\gamma\) is scale, \(j\in\{1,\ldots,p-1\}\) is the oscillation index, and \(a\) indexes translations / p-adic balls.

A `PAdicWaveletConv` then applies channel-mixing matrices in coefficient space:

\[
\widetilde c_{\mathrm{sc}} = W_{\mathrm{sc}}c_{\mathrm{sc}},
\]

\[
\widetilde c_{\gamma,j,a}=W_{\gamma,j}c_{\gamma,j,a}.
\]

Importantly, \(W_{\gamma,j}\) depends on **scale and oscillation but not on translation \(a\)**. Wavelet coefficients at the same hierarchical scale therefore share the same learned transformation.

The transformed coefficients are reconstructed to physical space and combined with a residual pointwise branch:

\[
y = \operatorname{GELU}\!\left(
\mathcal{W}^{-1}_{p}\,W\,\mathcal{W}_{p}x
+
P(x)
\right),
\]

where \(\mathcal{W}_p\) denotes the p-adic wavelet transform and \(P\) is a learnable `PAdicLinear` map.

The complete model is

```text
Input field / trajectory context
          │
          ▼
   PAdicLinear lifting
          │
          ▼
 ┌─────────────────────────────┐
 │  P-adic Wavelet Block × L   │
 │                             │
 │  Kozyrev decomposition      │
 │          │                  │
 │  scale-shared multipliers   │
 │          │                  │
 │  wavelet reconstruction     │
 │          │                  │
 │     + pointwise skip        │
 │          │                  │
 │        GELU                 │
 └─────────────────────────────┘
          │
          ▼
  PAdicLinear projection
          │
          ▼
 Full predicted trajectory
```

### What “p-adic” means in the current implementation

The operator topology is p-adic: spatial resolution is organized as \(M=p^n\), and the main nonlocal operator branch uses a discrete Kozyrev wavelet basis on \(\mathbb{Z}/p^n\mathbb{Z}\). The trainable tensors themselves are standard real/complex PyTorch parameters, and the current blocks use GELU nonlinearities.

Accordingly, this repository is best described as a **p-adic-wavelet neural operator**, rather than as a network in which every learnable operation is carried out with exact arithmetic in \(\mathbb{Q}_p\).

---

## Why p-adic wavelets?

Fourier neural operators exploit global spectral structure associated with Euclidean translation symmetry. PNO explores a different inductive bias: **hierarchical locality**.

In a p-adic metric, closeness is determined by shared digits/ancestry rather than ordinary Euclidean distance. The space therefore has an intrinsic tree-like or ultrametric organization. Kozyrev wavelets encode this hierarchy through nested p-adic balls and multiresolution scales.

Instead of only asking which Euclidean frequencies are important, PNO can learn which **hierarchical scales** of the p-adic domain are important.

The research question behind the project is:

> **Can an operator architecture whose spectral representation is aligned with ultrametric hierarchy provide a useful inductive bias for multiscale PDE dynamics, particularly when the underlying process is naturally hierarchical or non-Archimedean?**

---

## Supported experiments

| System | Dataset interface | Typical domain | Prediction setup |
|---|---|---:|---|
| Diffusion–Sorption | `DiffusionSorptionDataset` | \(2^{10}=1024\) points | IC + grid + boundary values → 100 future states |
| Reaction–Diffusion | `ReactionDiffusionDataset` | \(2^{10}=1024\) | IC + grid + boundary values → 100 future states |
| Burgers equation | `PDEBench1DDataset` | \(2^{10}=1024\) | IC + grid + boundary values → 100 future states |
| Allen–Cahn equation | `PDEBench1DDataset` | \(2^{10}=1024\) | IC + grid + boundary values → 100 future states |
| Linear Advection | `PDEBench1DDataset` | \(2^{10}=1024\) | IC + grid + boundary values → 100 future states |
| Kuramoto–Sivashinsky | `KuramotoSivashinskyDataset` | \(2^9=512\) | 256-step context + grid → next 512 states |
| p-adic Schrödinger | `PAdicSchrodingerDataset` | \(2^{10}=1024\) | complex IC + grid + potential → complex future trajectory |
| p-adic Fokker–Planck | `FokkerPlanckDataset` | \(2^{10}=1024\) | initial density + grid + potential → future probability densities |

Example parameter sweeps are provided under `configs/`, including:

- Burgers: \(\nu\in\{0.1,0.01,0.001\}\)
- Allen–Cahn: \(\epsilon\in\{0.01,0.005\}\)
- Reaction–Diffusion parameter variations
- p-adic Schrödinger: multiple Vladimirov exponents and free/potential cases
- p-adic Fokker–Planck: diffusion and hierarchical trapping cases
- Kuramoto–Sivashinsky: 256 → 512 single-shot forecasting

---

## Native p-adic dynamical systems

### p-adic Schrödinger equation

The repository includes experiments of the form

\[
i\,\partial_t\psi = \left(D^\alpha + V\right)\psi,
\]

where \(D^\alpha\) is the Vladimirov pseudo-differential operator. The free case is generated spectrally, while the potential case uses a hierarchical p-adic potential.

A key theoretical relation is that Kozyrev wavelets diagonalize the Vladimirov operator:

\[
D^\alpha\psi_{\gamma,j,a}=\lambda_\gamma\psi_{\gamma,j,a}.
\]

Example:

```bash
python -m data.schrodinger_generator \
  --case free \
  --alpha 1.0 \
  --N 1000 \
  --save_dir ./datasets
```

Potential case:

```bash
python -m data.schrodinger_generator \
  --case potential \
  --alpha 1.0 \
  --N 1000 \
  --save_dir ./datasets
```

### p-adic Fokker–Planck dynamics

The repository also studies ultrametric diffusion

\[
\partial_t \rho = -D^\alpha \rho
\]

and a potential-modified hierarchical trapping variant.

Example:

```bash
python -m data.fokker_planck_generator \
  --case diffusion \
  --alpha 1.0 \
  --N 1000 \
  --save_dir ./datasets
```

Hierarchical trapping:

```bash
python -m data.fokker_planck_generator \
  --case trapping \
  --alpha 1.0 \
  --N 1000 \
  --save_dir ./datasets
```

A verification mode is available before generating a full Fokker–Planck dataset:

```bash
python -m data.fokker_planck_generator \
  --case diffusion \
  --alpha 1.0 \
  --verify
```

---

## Synthetic classical PDE data

`data/pde_generators.py` generates compatible HDF5 datasets for Burgers, Allen–Cahn, and linear advection on a periodic one-dimensional domain.

```bash
# Burgers
python -m data.pde_generators \
  --pde burgers \
  --nu 0.01 \
  --N 1000 \
  --save_dir ./datasets

# Allen–Cahn
python -m data.pde_generators \
  --pde allencahn \
  --eps 0.01 \
  --N 1000 \
  --save_dir ./datasets

# Advection
python -m data.pde_generators \
  --pde advection \
  --beta 1.0 \
  --N 1000 \
  --save_dir ./datasets
```

The generated files use an HDF5 `tensor` layout compatible with `PDEBench1DDataset`.

---

## Installation

PNO uses `adicton` as a Git submodule, so clone recursively:

```bash
git clone --recurse-submodules https://github.com/meural-operator/PNO.git
cd PNO
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

Create a Python environment and install the minimal dependencies:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

pip install torch numpy h5py tqdm matplotlib
pip install -e ./adicton
```

`adicton` targets Python 3.10+ and PyTorch 2.0+. CUDA is optional but recommended for larger experiments.

---

## Dataset paths

The JSON configurations currently contain the original development-machine dataset paths. Before running an experiment, edit

```json
"h5_path": "..."
```

to point to the corresponding dataset on your system.

For example:

```json
"dataset": {
  "name": "PDEBench1D",
  "h5_path": "./datasets/1D_Burgers_Nu0.01.hdf5",
  "train_ratio": 0.8,
  "n_x": 1024,
  "out_t": 100,
  "ic_index": 0
}
```

---

## Training

Train from any configuration with:

```bash
python train.py --config configs/burgers/nu0.01.json
```

Other examples:

```bash
python train.py --config configs/allen_cahn/eps0.01.json
python train.py --config configs/reaction_diffusion/rho1.0.json
python train.py --config configs/ks/singleshot.json
python train.py --config configs/schrodinger/free_alpha1.0.json
python train.py --config configs/fokker_planck/diffusion_alpha1.0.json
```

The training pipeline uses:

- `AdamW`
- cosine annealing learning-rate scheduling
- channel-wise Gaussian normalization
- relative \(L^2\) loss
- validation after every epoch
- best/latest/periodic checkpoints
- source snapshots for reproducibility

Each new run is stored as

```text
runs/<experiment_name>_<timestamp>/
├── config.json
├── metrics.jsonl
├── checkpoints/
│   ├── best.pth
│   ├── latest.pth
│   └── epoch_*.pth
└── src_mirror/
```

### Resume training

```bash
python train.py \
  --resume runs/<run-name>/checkpoints/latest.pth
```

When resuming, the saved run configuration is reused automatically.

---

## Configuration

A typical model configuration is:

```json
"model": {
  "in_channels": 4,
  "out_channels": 100,
  "hidden_channels": 64,
  "num_blocks": 4,
  "p": 2,
  "n": 10
}
```

The key p-adic constraint is

\[
M=p^n,
\]

where \(M\) is the spatial resolution. For example:

- `p=2, n=10` → 1024 spatial points
- `p=2, n=9` → 512 spatial points
- `p=3, n=6` → 729 spatial points, provided the dataset is prepared at that resolution

---

## Kuramoto–Sivashinsky experiment

The KS configuration provides a long single-shot forecasting task. The input consists of 256 previous states plus a spatial-grid channel, and the network predicts the next 512 states simultaneously:

\[
[u_1,\ldots,u_{256}]
\longmapsto
[u_{257},\ldots,u_{768}].
\]

This tests whether the hierarchical wavelet representation can encode long temporal evolution of chaotic PDE dynamics without autoregressive rollout.

---

## Evaluation

`train.py` records physical-space relative \(L^2\) validation error in `metrics.jsonl` and stores the best validation checkpoint.

The current `eval.py` is a **legacy Diffusion–Sorption-specific visualization utility**, not a general evaluation CLI for every supported dataset/configuration. For comparisons across the full experiment suite, use the per-epoch validation metrics or adapt `eval.py` to the desired dataset interface.

---

## Repository structure

```text
PNO/
├── adicton/                       # Git submodule: p-adic computation library
├── configs/
│   ├── advection/
│   ├── allen_cahn/
│   ├── burgers/
│   ├── diffusion_sorption/
│   ├── fokker_planck/
│   ├── ks/
│   ├── reaction_diffusion/
│   └── schrodinger/
├── data/
│   ├── dataset.py                 # Diffusion-Sorption / Reaction-Diffusion
│   ├── pdebench_dataset.py        # Generic 1D PDEBench-style loader
│   ├── pde_generators.py          # Burgers / Allen-Cahn / Advection generator
│   ├── ks_dataset.py              # Kuramoto-Sivashinsky loader
│   ├── schrodinger_dataset.py     # p-adic Schrödinger loader
│   ├── schrodinger_generator.py   # p-adic Schrödinger data generation
│   ├── fokker_planck_dataset.py   # p-adic Fokker-Planck loader
│   └── fokker_planck_generator.py # p-adic Fokker-Planck generation
├── models/
│   └── pno.py                     # P-adic wavelet neural operator
├── train.py                       # Config-driven training pipeline
├── eval.py                        # Diffusion-Sorption visualization utility
└── utils.py                       # Normalization, relative Lp loss, logging
```

---

## Current limitations

- Spatial resolution must currently match \(M=p^n\).
- The primary operator/data interface is currently one-dimensional.
- Trainable parameters are standard PyTorch real/complex tensors; the p-adic structure enters through the domain, Kozyrev transform, and p-adic-aware layers.
- Experiment JSON files still contain local dataset paths and must be edited before use on another system.
- The checked-in repository does not contain trained checkpoints or consolidated benchmark tables.
- `eval.py` is not yet a general evaluation CLI for all supported datasets.

---

## Theoretical background

The implementation is motivated by work on:

- S. V. Kozyrev, **Wavelet theory as p-adic spectral analysis** (2002)
- V. S. Vladimirov, I. V. Volovich, E. I. Zelenov, **p-Adic Analysis and Mathematical Physics**
- p-adic pseudo-differential / Vladimirov operators
- neural operators and spectral operator learning
- ultrametric models of hierarchical diffusion and dynamical systems

The p-adic transform and numerical infrastructure used by PNO is implemented in [`adicton`](https://github.com/meural-operator/adicton).

---

## Related repository

- **Adicton** — GPU-accelerated p-adic computation, transforms, numerical operators, and neural-network utilities: https://github.com/meural-operator/adicton
