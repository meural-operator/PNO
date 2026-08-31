# PNO: P-adic Wavelet Neural Operator for PDE Dynamics

PNO is an experimental **P-adic Neural Operator (PNO)** for learning solution operators of time-dependent partial differential equations using a **Kozyrev p-adic wavelet representation** instead of a conventional Euclidean Fourier basis.

The core idea is to represent a spatial field on the finite p-adic domain

$$
\mathbb{Z}/p^n\mathbb{Z}, \qquad M=p^n,
$$

transform the field into a multiscale p-adic wavelet basis, learn scale-dependent transformations of the wavelet coefficients, and reconstruct the resulting field back in physical space.

This gives PNO a **hierarchical and ultrametric inductive bias**. The repository contains experiments on both conventional PDEs and dynamical systems that are themselves formulated using p-adic operators.

> **Research status:** This repository is an experimental research implementation. It contains the model architecture, training pipeline, dataset interfaces, PDE generators, p-adic dynamical-system generators, and experiment configurations. Trained checkpoints and consolidated benchmark tables are not currently committed to the repository.

---

## Overview

For most experiments, PNO learns a single-shot solution operator

$$
\mathcal{G}_{\theta}: a(x) \mapsto \{u(x,t_1),u(x,t_2),\ldots,u(x,t_T)\}.
$$

Instead of recursively predicting one time step and feeding that prediction back into the model, the present implementation predicts the requested future trajectory **simultaneously as output channels**.

PNO combines:

- Kozyrev p-adic wavelet decomposition on $\mathbb{Z}/p^n\mathbb{Z}$
- learnable scale-dependent wavelet multipliers
- parameter sharing across translations at a fixed p-adic scale
- p-adic-aware pointwise residual channel mixing
- GELU nonlinearities between operator blocks
- Gaussian input/output normalization
- relative $L^2$ training loss
- single-shot trajectory prediction

The p-adic transforms and p-adic-aware neural-network layers are provided by the [`adicton`](https://github.com/meural-operator/adicton) Git submodule.

---

## Core operator

Let the input field be

$$
x \in \mathbb{R}^{C_{\mathrm{in}} \times M}, \qquad M=p^n.
$$

For each channel, PNO computes a discrete p-adic wavelet expansion

$$
x \longleftrightarrow \left(c_{\mathrm{sc}}, c_{\gamma,j,a}\right),
$$

where:

- $\gamma$ denotes the p-adic scale,
- $j \in \{1,\ldots,p-1\}$ denotes the oscillation index,
- $a$ indexes translations, or equivalently p-adic balls at that scale.

The `PAdicWaveletConv` layer applies channel-mixing transformations directly in wavelet space.

For the scaling coefficient,

$$
\widetilde{c}_{\mathrm{sc}} = W_{\mathrm{sc}} c_{\mathrm{sc}}.
$$

For the wavelet coefficients,

$$
\widetilde{c}_{\gamma,j,a} = W_{\gamma,j} c_{\gamma,j,a}.
$$

A key structural property is that $W_{\gamma,j}$ depends on **scale and oscillation but not on translation $a$**. Therefore, all wavelet coefficients at the same hierarchical scale use the same learned transformation.

The transformed coefficients are reconstructed into physical space and combined with a pointwise residual branch:

$$
y = \operatorname{GELU}\left(
\mathcal{W}_p^{-1} W \mathcal{W}_p x + P(x)
\right),
$$

where $\mathcal{W}_p$ is the p-adic wavelet transform and $P$ is a learnable `PAdicLinear` map.

### Architecture

```text
Input field / trajectory context
            |
            v
     PAdicLinear lifting
            |
            v
  +---------------------------+
  |   P-adic Wavelet Block    |
  |                           |
  | Kozyrev decomposition     |
  |            |              |
  |            v              |
  | Scale-shared multipliers  |
  |            |              |
  |            v              |
  | Wavelet reconstruction    |
  |            |              |
  |      + pointwise skip     |
  |            |              |
  |          GELU             |
  +---------------------------+
            |
         repeat L times
            |
            v
    PAdicLinear projection
            |
            v
    Predicted trajectory
```

---

## What does "p-adic" mean here?

The present implementation is most accurately described as a **p-adic-wavelet neural operator**.

Its spatial representation is organized on the finite domain

$$
\mathbb{Z}/p^n\mathbb{Z},
$$

and its principal nonlocal operator branch uses a discrete **Kozyrev wavelet basis** adapted to this hierarchical geometry.

The trainable model parameters themselves are standard real- and complex-valued PyTorch tensors, and the current operator blocks use GELU nonlinearities. Therefore, PNO does **not** claim that every neural-network operation is evaluated using exact arithmetic in $\mathbb{Q}_p$.

Exact p-adic arithmetic and additional non-Archimedean numerical primitives are developed in the associated [`adicton`](https://github.com/meural-operator/adicton) framework.

---

## Why p-adic wavelets?

Fourier neural operators represent global interactions using sinusoidal modes associated with Euclidean translation symmetry. PNO explores a different inductive bias: **hierarchical locality**.

In a p-adic metric, two points are close when they share a sufficiently long common p-adic prefix. This creates a naturally tree-like, ultrametric organization of the domain.

Kozyrev wavelets are adapted to this hierarchy. They separate a function into components associated with nested p-adic scales and balls.

The resulting operator therefore learns not only which spectral components matter, but which **hierarchical scales** matter.

The main research question behind this repository is:

> **Can an operator architecture whose representation is aligned with ultrametric hierarchy provide a useful inductive bias for multiscale PDE dynamics, especially when the underlying process is hierarchical or non-Archimedean?**

---

## Supported experiments

The current repository contains data interfaces and configurations for the following systems.

| System | Dataset interface | Spatial resolution | Prediction task |
|---|---|---:|---|
| Diffusion-Sorption | `DiffusionSorptionDataset` | 1024 | Initial state + grid + BCs to 100 future states |
| Reaction-Diffusion | `ReactionDiffusionDataset` | 1024 | Initial state + grid + BCs to 100 future states |
| Burgers equation | `PDEBench1DDataset` | 1024 | Initial state + grid + BCs to 100 future states |
| Allen-Cahn equation | `PDEBench1DDataset` | 1024 | Initial state + grid + BCs to 100 future states |
| Linear Advection | `PDEBench1DDataset` | 1024 | Initial state + grid + BCs to 100 future states |
| Kuramoto-Sivashinsky | `KuramotoSivashinskyDataset` | 512 | 256 context states to 512 future states |
| p-adic Schrodinger | `PAdicSchrodingerDataset` | 1024 | Complex initial state + grid + potential to complex trajectory |
| p-adic Fokker-Planck | `FokkerPlanckDataset` | 1024 | Initial density + grid + potential to future densities |

For the default binary p-adic experiments:

- 1024 spatial points correspond to $p=2$, $n=10$.
- 512 spatial points correspond to $p=2$, $n=9$.

The repository contains parameter sweeps for several systems, including Burgers viscosities, Allen-Cahn interface widths, p-adic Schrodinger Vladimirov exponents, p-adic Fokker-Planck diffusion/trapping settings, and a long-horizon Kuramoto-Sivashinsky experiment.

---

## Native p-adic dynamical systems

### p-adic Schrodinger equation

The repository includes experiments for dynamics of the form

$$
i\,\partial_t \psi = \left(D^\alpha + V\right)\psi,
$$

where $D^\alpha$ is the Vladimirov pseudo-differential operator and $V$ is an optional hierarchical p-adic potential.

A key relation is that Kozyrev wavelets are eigenfunctions of the Vladimirov operator:

$$
D^\alpha \psi_{\gamma,j,a}
= \lambda_\gamma \psi_{\gamma,j,a}.
$$

The repository contains both:

- **free evolution**, with $V=0$,
- **potential-driven evolution**, with a hierarchical p-adic potential.

Generate a free-particle dataset with:

```bash
python -m data.schrodinger_generator \
  --case free \
  --alpha 1.0 \
  --N 1000 \
  --save_dir ./datasets
```

Generate a potential case with:

```bash
python -m data.schrodinger_generator \
  --case potential \
  --alpha 1.0 \
  --N 1000 \
  --save_dir ./datasets
```

---

### p-adic Fokker-Planck dynamics

The repository also studies ultrametric diffusion of the form

$$
\partial_t \rho = -D^\alpha \rho,
$$

as well as a hierarchical trapping variant involving a potential landscape.

Generate a free diffusion dataset with:

```bash
python -m data.fokker_planck_generator \
  --case diffusion \
  --alpha 1.0 \
  --N 1000 \
  --save_dir ./datasets
```

Generate a trapping dataset with:

```bash
python -m data.fokker_planck_generator \
  --case trapping \
  --alpha 1.0 \
  --N 1000 \
  --save_dir ./datasets
```

A verification mode is also available:

```bash
python -m data.fokker_planck_generator \
  --case diffusion \
  --alpha 1.0 \
  --verify
```

---

## Synthetic classical PDE datasets

`data/pde_generators.py` can generate HDF5 datasets for several one-dimensional PDEs on a periodic domain.

### Burgers equation

$$
u_t + u u_x = \nu u_{xx}.
$$

```bash
python -m data.pde_generators \
  --pde burgers \
  --nu 0.01 \
  --N 1000 \
  --save_dir ./datasets
```

### Allen-Cahn equation

$$
u_t = \epsilon^2 u_{xx} + u(1-u^2).
$$

```bash
python -m data.pde_generators \
  --pde allencahn \
  --eps 0.01 \
  --N 1000 \
  --save_dir ./datasets
```

### Linear advection

$$
u_t + \beta u_x = 0.
$$

```bash
python -m data.pde_generators \
  --pde advection \
  --beta 1.0 \
  --N 1000 \
  --save_dir ./datasets
```

The generated HDF5 files use a `tensor` layout compatible with `PDEBench1DDataset`.

---

## Installation

PNO uses [`adicton`](https://github.com/meural-operator/adicton) as a Git submodule. Clone the repository recursively:

```bash
git clone --recurse-submodules https://github.com/meural-operator/PNO.git
cd PNO
```

If PNO has already been cloned without its submodule, run:

```bash
git submodule update --init --recursive
```

Create a Python environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

or on Windows:

```powershell
.venv\Scripts\activate
```

Install the minimal dependencies:

```bash
pip install torch numpy h5py tqdm matplotlib
pip install -e ./adicton
```

`adicton` targets Python 3.10+ and PyTorch 2.0+. CUDA is optional but recommended for larger experiments.

---

## Dataset paths

The experiment JSON files currently contain the machine-local dataset paths used during development.

Before running an experiment, update the `h5_path` field to point to your local dataset.

For example:

```json
{
  "dataset": {
    "name": "PDEBench1D",
    "h5_path": "./datasets/1D_Burgers_Nu0.01.hdf5",
    "train_ratio": 0.8,
    "n_x": 1024,
    "out_t": 100,
    "ic_index": 0
  }
}
```

---

## Training

Train PNO from any configuration file with:

```bash
python train.py --config configs/burgers/nu0.01.json
```

Additional examples:

```bash
python train.py --config configs/allen_cahn/eps0.01.json
python train.py --config configs/reaction_diffusion/rho1.0.json
python train.py --config configs/ks/singleshot.json
python train.py --config configs/schrodinger/free_alpha1.0.json
python train.py --config configs/fokker_planck/diffusion_alpha1.0.json
```

The training pipeline currently uses:

- AdamW optimization
- cosine-annealing learning-rate scheduling
- channel-wise Gaussian normalization
- relative $L^2$ loss
- validation after each epoch
- best, latest, and periodic checkpoints
- source snapshots for reproducibility

Each run is stored under:

```text
runs/<experiment_name>_<timestamp>/
```

with the following structure:

```text
runs/<experiment_name>_<timestamp>/
|-- config.json
|-- metrics.jsonl
|-- checkpoints/
|   |-- best.pth
|   |-- latest.pth
|   `-- epoch_*.pth
`-- src_mirror/
```

### Resume training

```bash
python train.py \
  --resume runs/<run-name>/checkpoints/latest.pth
```

When resuming, the configuration stored in the original run directory is reused automatically.

---

## Model configuration

A typical PNO model configuration is:

```json
{
  "model": {
    "in_channels": 4,
    "out_channels": 100,
    "hidden_channels": 64,
    "num_blocks": 4,
    "p": 2,
    "n": 10
  }
}
```

The fundamental spatial-resolution constraint is

$$
M=p^n.
$$

Examples:

- `p = 2, n = 10` gives 1024 spatial points.
- `p = 2, n = 9` gives 512 spatial points.
- `p = 3, n = 6` gives 729 spatial points, provided the dataset is prepared at that resolution.

---

## Kuramoto-Sivashinsky experiment

The Kuramoto-Sivashinsky configuration provides a particularly long single-shot forecasting problem.

The model receives 256 trajectory states as context and predicts the next 512 states simultaneously:

$$
[u_1,\ldots,u_{256}]
\mapsto
[u_{257},\ldots,u_{768}].
$$

A spatial-grid channel is appended to the input context.

This setup tests whether the hierarchical wavelet representation can encode long temporal evolution of chaotic PDE dynamics without autoregressive rollout.

---

## Evaluation

During training, PNO reports the physical-space relative $L^2$ error

$$
\frac{\|\widehat{u}-u\|_2}{\|u\|_2}.
$$

Per-epoch training and validation metrics are written to:

```text
metrics.jsonl
```

and the best validation checkpoint is stored as:

```text
checkpoints/best.pth
```

The current `eval.py` is a **Diffusion-Sorption-specific visualization utility**, not yet a general evaluator for every supported experiment. For broader comparisons, use the validation metrics produced by `train.py` or adapt `eval.py` to the relevant dataset class.

---

## Repository structure

```text
PNO/
|-- adicton/                       # Git submodule: p-adic computation framework
|-- configs/
|   |-- advection/
|   |-- allen_cahn/
|   |-- burgers/
|   |-- diffusion_sorption/
|   |-- fokker_planck/
|   |-- ks/
|   |-- reaction_diffusion/
|   `-- schrodinger/
|-- data/
|   |-- dataset.py                 # Diffusion-Sorption / Reaction-Diffusion
|   |-- pdebench_dataset.py        # Generic 1D PDEBench-style loader
|   |-- pde_generators.py          # Burgers / Allen-Cahn / Advection generator
|   |-- ks_dataset.py              # Kuramoto-Sivashinsky loader
|   |-- schrodinger_dataset.py     # p-adic Schrodinger loader
|   |-- schrodinger_generator.py   # p-adic Schrodinger generator
|   |-- fokker_planck_dataset.py   # p-adic Fokker-Planck loader
|   `-- fokker_planck_generator.py # p-adic Fokker-Planck generator
|-- models/
|   `-- pno.py                     # P-adic wavelet neural operator
|-- train.py                       # Config-driven training pipeline
|-- eval.py                        # Diffusion-Sorption visualization utility
`-- utils.py                       # Normalization, relative Lp loss, logging
```

---

## Current limitations

- The main operator and data interfaces are currently one-dimensional.
- Spatial resolution must match $M=p^n$.
- Trainable parameters are standard PyTorch real/complex tensors; p-adic structure enters through the domain organization, Kozyrev wavelet transform, and p-adic-aware layers.
- Experiment configuration files contain local dataset paths and must be edited before use on another system.
- Trained checkpoints and consolidated benchmark tables are not currently committed.
- `eval.py` is not yet a general evaluation CLI for every dataset.

---

## Theoretical background

The project is motivated by ideas from:

- p-adic analysis and ultrametric spaces
- Kozyrev wavelets
- Vladimirov pseudo-differential operators
- non-Archimedean mathematical physics
- spectral neural operators
- operator learning for PDEs
- hierarchical diffusion and dynamical systems

Relevant foundational references include:

1. S. V. Kozyrev, *Wavelet Theory as p-Adic Spectral Analysis*, 2002.
2. V. S. Vladimirov, I. V. Volovich, and E. I. Zelenov, *p-Adic Analysis and Mathematical Physics*.
3. Literature on Vladimirov operators, ultrametric diffusion, and neural operator learning.

---

## Related project

### Adicton

PNO uses **Adicton**, a GPU-oriented framework for p-adic numerical computation, transforms, pseudo-differential operators, and machine-learning primitives.

Repository: https://github.com/meural-operator/adicton
