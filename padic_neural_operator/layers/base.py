"""
Abstract Base Classes for P-Adic Neural Operator Layers
========================================================

Defines the interface contracts that all layers in the framework must follow.
This ensures consistency and composability across the p-adic attention,
integral operator, and future layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class PAdicLayerBase(nn.Module, ABC):
    """
    Abstract base class for all p-adic neural operator layers.

    Every layer in the PNO framework:
        1. Is parameterized by a prime p and a tree depth L.
        2. Operates on function representations v(x) at spatial locations x.
        3. Produces updated representations using p-adic geometry.

    Args:
        p: Prime base for the p-adic number system.
        L: Depth of the p-adic tree (number of address levels).
    """

    def __init__(self, p: int = 2, L: int = 10):
        super().__init__()
        if p < 2:
            raise ValueError(f"Prime p must be >= 2, got {p}")
        if L < 1:
            raise ValueError(f"Tree depth L must be >= 1, got {L}")
        self._p = p
        self._L = L

    @property
    def p(self) -> int:
        """Prime base."""
        return self._p

    @property
    def L(self) -> int:
        """Tree depth."""
        return self._L

    @abstractmethod
    def forward(self, v: Tensor, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            v: Function values of shape (batch, N, d_v).
            x: Spatial locations of shape (batch, N, d_x) in [0,1)^d.

        Returns:
            Updated function values of shape (batch, N, d_out).
        """
        ...


class ProximityKernel(nn.Module, ABC):
    """
    Abstract base class for p-adic proximity kernels.

    A proximity kernel maps a p-adic distance to a scalar attention weight.
    This is the function Phi in: A_p(x, y) = Phi(|x - y|_p).

    Different choices of Phi give different inductive biases:
        - Phi(d) = exp(-alpha * d): soft exponential decay
        - Phi(d) = d^{-alpha}: power-law (Vladimirov-style)
        - Phi(d) = learnable MLP on log(d)
    """

    @abstractmethod
    def forward(self, distances: Tensor) -> Tensor:
        """
        Map p-adic distances to attention weights.

        Args:
            distances: Tensor of p-adic distances, shape arbitrary.

        Returns:
            Attention weights of the same shape, non-negative.
        """
        ...
