"""
P-Adic Attention Mechanism
===========================

Replaces the Euclidean inner product in standard attention with a p-adic
proximity score:

    Standard attention:  A(Q, K) = softmax(Q K^T / sqrt(d))
    P-adic attention:    A_p(x_i, x_j) = Phi(|x_i - x_j|_p)

where |.|_p is the p-adic norm and Phi is a learnable proximity kernel.

Key insight: tokens that share a fine-scale p-adic ball (same branch of the
tree at high depth) attend strongly; tokens sharing only a coarse-scale
ancestor attend weakly. This naturally encodes the multiscale hierarchical
structure of PDE solutions.

References:
    - Vladimirov et al. (1994), "P-Adic Analysis and Mathematical Physics"
    - Kozyrev (2004), "P-adic pseudodifferential operators and p-adic wavelets"
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.padic import padic_distance_matrix
from .base import PAdicLayerBase, ProximityKernel


# ---------------------------------------------------------------------------
# Proximity Kernels
# ---------------------------------------------------------------------------

class ExponentialKernel(ProximityKernel):
    """Phi(d) = exp(-alpha * d). Learnable alpha controls sharpness."""

    def __init__(self, alpha_init: float = 1.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init)))

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def forward(self, distances: Tensor) -> Tensor:
        return torch.exp(-self.alpha * distances)


class PowerLawKernel(ProximityKernel):
    """Phi(d) = (d + eps)^{-alpha}. Vladimirov-style heavier tails."""

    def __init__(self, alpha_init: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init)))
        self.eps = eps

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def forward(self, distances: Tensor) -> Tensor:
        return (distances + self.eps) ** (-self.alpha)


class LearnedKernel(ProximityKernel):
    """Fully learned proximity kernel via small MLP on log-distance."""

    def __init__(self, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = 1
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Softplus())
        self.mlp = nn.Sequential(*layers)

    def forward(self, distances: Tensor) -> Tensor:
        shape = distances.shape
        log_d = torch.log(distances.clamp(min=1e-10)).reshape(-1, 1)
        weights = self.mlp(log_d).reshape(shape)
        return weights


class KernelType(str, Enum):
    EXPONENTIAL = "exponential"
    POWER_LAW = "power_law"
    LEARNED = "learned"


def build_kernel(kernel_type: str | KernelType, **kwargs) -> ProximityKernel:
    kernel_type = KernelType(kernel_type)
    if kernel_type == KernelType.EXPONENTIAL:
        return ExponentialKernel(**kwargs)
    elif kernel_type == KernelType.POWER_LAW:
        return PowerLawKernel(**kwargs)
    elif kernel_type == KernelType.LEARNED:
        return LearnedKernel(**kwargs)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


# ---------------------------------------------------------------------------
# P-Adic Attention Layer
# ---------------------------------------------------------------------------

class PAdicAttention(PAdicLayerBase):
    """
    P-Adic Attention Layer.

    Computes attention scores based on p-adic proximity in the spatial domain,
    then applies those scores to transform function values.

    Architecture:
        1. Project v into queries Q, keys K, values V via linear maps.
        2. Compute pairwise p-adic distance matrix D[i,j] = |x_i - x_j|_p.
        3. Map distances to attention weights: W[i,j] = Phi(D[i,j]).
        4. Optionally combine with content-based (dot-product) attention.
        5. Apply weights: output_i = sum_j W[i,j] * V_j.

    Args:
        d_model: Dimension of input/output function values.
        n_heads: Number of attention heads.
        p: Prime base for p-adic distances.
        L: Depth of p-adic tree.
        kernel_type: Type of proximity kernel.
        content_blend: If > 0, blends p-adic attention with standard dot-product
                       attention using a learnable gate initialized to this value.
        dropout: Dropout rate on attention weights.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 1,
        p: int = 2,
        L: int = 10,
        kernel_type: str | KernelType = KernelType.EXPONENTIAL,
        content_blend: float = 0.0,
        dropout: float = 0.0,
        **kernel_kwargs,
    ):
        super().__init__(p=p, L=L)

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.content_blend = content_blend

        # Per-head proximity kernels
        self.kernels = nn.ModuleList([
            build_kernel(kernel_type, **kernel_kwargs) for _ in range(n_heads)
        ])

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

        # Optional content-based blending
        if content_blend > 0:
            init_val = math.log(content_blend / (1.0 - content_blend + 1e-8))
            self.content_gate = nn.Parameter(torch.full((n_heads,), init_val))
        else:
            self.content_gate = None

    def forward(
        self,
        v: Tensor,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of p-adic attention.

        Args:
            v: Function values (B, N, d_model).
            x: Spatial locations (B, N, d_x) in [0,1)^d.
            mask: Optional boolean mask (B, N, N).

        Returns:
            Updated function values (B, N, d_model).
        """
        B, N, _ = v.shape

        # 1. Multi-head projections: (B, n_heads, N, d_head)
        Q = self._reshape_to_heads(self.W_q(v))
        K = self._reshape_to_heads(self.W_k(v))
        V = self._reshape_to_heads(self.W_v(v))

        # 2. Fully-vectorized batched p-adic distance: (B, N, N)
        dist_matrix = padic_distance_matrix(x, L=self.L, p=self.p)

        # 3. Per-head proximity kernels → (B, n_heads, N, N)
        attn_weights = self._compute_padic_weights(dist_matrix)

        # 4. Optional content-based blending
        if self.content_gate is not None:
            content_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
            content_attn = F.softmax(content_scores, dim=-1)
            gate = torch.sigmoid(self.content_gate).view(1, self.n_heads, 1, 1)
            attn_weights = (1 - gate) * attn_weights + gate * content_attn

        # 5. Mask
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask.unsqueeze(1), 0.0)

        # 6. Normalize (row-stochastic) and dropout
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-10)
        attn_weights = self.attn_dropout(attn_weights)

        # 7. Weighted aggregation
        out = torch.matmul(attn_weights, V)

        # 8. Reshape back and project
        out = self._reshape_from_heads(out)
        return self.W_o(out)

    def _compute_padic_weights(self, dist_matrix: Tensor) -> Tensor:
        """Apply per-head proximity kernels to the distance matrix."""
        weights = torch.stack([kernel(dist_matrix) for kernel in self.kernels], dim=1)
        return weights  # (B, n_heads, N, N)

    def _reshape_to_heads(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

    def _reshape_from_heads(self, x: Tensor) -> Tensor:
        B, _, N, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, self.d_model)

    def get_attention_weights(self, v: Tensor, x: Tensor) -> Tensor:
        """Return raw p-adic attention weights for visualization."""
        dist_matrix = padic_distance_matrix(x, L=self.L, p=self.p)
        weights = self._compute_padic_weights(dist_matrix)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-10)
        return weights
