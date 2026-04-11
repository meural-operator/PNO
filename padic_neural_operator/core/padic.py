"""
P-Adic Number Theory Primitives
================================

Core mathematical operations for p-adic numbers, parameterized by a prime p.

Key concepts:
    - p-adic valuation v_p(n): the exponent of the highest power of p dividing n
    - p-adic norm |n|_p = p^{-v_p(n)}: small when p divides n many times
    - p-adic address: base-p digit expansion mapping [0,1)^d points to a p-adic tree
    - p-adic distance: ultrametric distance derived from shared ancestry depth

The ultrametric inequality |x - z|_p <= max(|x - y|_p, |y - z|_p) gives the
tree structure: points cluster into nested balls, and membership in the same
ball at a given scale corresponds to shared "ancestry."
"""

from __future__ import annotations

import torch
from torch import Tensor


def padic_valuation(n: int, p: int = 2) -> int:
    """
    Compute the p-adic valuation v_p(n).

    The p-adic valuation is the largest exponent k such that p^k divides n.
    By convention, v_p(0) = infinity; we return -1 as a sentinel for 0.

    Args:
        n: Integer whose valuation to compute.
        p: Prime base (must be >= 2).

    Returns:
        The p-adic valuation of n, or -1 if n == 0.
    """
    if p < 2:
        raise ValueError(f"Prime p must be >= 2, got {p}")
    if n == 0:
        return -1  # Convention: v_p(0) = infinity
    n = abs(n)
    v = 0
    while n % p == 0:
        v += 1
        n //= p
    return v


def padic_norm(n: int, p: int = 2) -> float:
    """
    Compute the p-adic norm |n|_p = p^{-v_p(n)}.

    Args:
        n: Integer whose norm to compute.
        p: Prime base (must be >= 2).

    Returns:
        The p-adic norm. Returns 0.0 for n == 0 (convention: |0|_p = 0).
    """
    v = padic_valuation(n, p)
    if v == -1:  # n == 0
        return 0.0
    return float(p) ** (-v)


def padic_address(x: Tensor, L: int, p: int = 2) -> Tensor:
    """
    Compute the p-adic address of points in [0, 1)^d.

    Maps each coordinate of each point to its base-p digit expansion up to
    L levels. This assigns each point a position in the p-adic tree of depth L.

    For a scalar x in [0,1):
        digit_1 = floor(x * p)
        digit_2 = floor((x * p - digit_1) * p)
        ...etc.

    Args:
        x: Tensor of shape (..., d) with values in [0, 1).
        L: Number of levels (depth of the p-adic tree).
        p: Prime base.

    Returns:
        Tensor of shape (..., d, L) containing integer digits in {0, 1, ..., p-1}.
    """
    if p < 2:
        raise ValueError(f"Prime p must be >= 2, got {p}")
    if L < 1:
        raise ValueError(f"Number of levels L must be >= 1, got {L}")

    digits = []
    remainder = x.clone()
    for _ in range(L):
        remainder = remainder * p
        digit = remainder.floor().long()
        digit = digit.clamp(0, p - 1)
        remainder = remainder - digit.float()
        digits.append(digit)

    return torch.stack(digits, dim=-1)


def padic_distance_from_addresses(addr_x: Tensor, addr_y: Tensor, p: int = 2) -> Tensor:
    """
    Compute the p-adic distance between points given their p-adic addresses.

    The p-adic distance is determined by the depth of the first disagreement
    in the digit expansions. If two addresses agree on the first L_match levels
    and disagree at level L_match + 1, then:

        |x - y|_p = p^{-L_match}

    If they agree on all L levels, the distance is p^{-L} (resolution limit).

    Args:
        addr_x: Tensor of shape (..., d, L) — p-adic addresses.
        addr_y: Tensor of shape (..., d, L) — p-adic addresses.
        p: Prime base.

    Returns:
        Tensor of shape (...,) — p-adic distances (one per point pair).
        For multi-dimensional inputs, returns the maximum distance across
        coordinates (the l-infinity p-adic distance).
    """
    match = (addr_x == addr_y)
    cum_match = match.cumprod(dim=-1)
    match_depth = cum_match.sum(dim=-1).float()
    dist_per_coord = float(p) ** (-match_depth)
    distance = dist_per_coord.max(dim=-1).values
    return distance


def padic_distance(x: Tensor, y: Tensor, L: int, p: int = 2) -> Tensor:
    """
    Compute the p-adic distance between points in [0, 1)^d.

    Args:
        x: Tensor of shape (..., d) with values in [0, 1).
        y: Tensor of shape (..., d) with values in [0, 1).
        L: Number of levels for the p-adic address.
        p: Prime base.

    Returns:
        Tensor of shape (...,) — p-adic distances.
    """
    addr_x = padic_address(x, L=L, p=p)
    addr_y = padic_address(y, L=L, p=p)
    return padic_distance_from_addresses(addr_x, addr_y, p=p)


def padic_distance_matrix(x: Tensor, L: int, p: int = 2, y: Tensor | None = None) -> Tensor:
    """
    Compute the pairwise p-adic distance matrix — fully vectorized.

    Supports both unbatched (N, d) and batched (B, N, d) inputs.

    Args:
        x: Tensor of shape (N, d) or (B, N, d) — query points.
        L: Number of levels for the p-adic address.
        p: Prime base.
        y: Tensor of shape (M, d) or (B, M, d) — key points. If None, uses x.

    Returns:
        Tensor of shape (N, M) or (B, N, M) — pairwise p-adic distance matrix.
    """
    if y is None:
        y = x

    # Compute addresses: (..., N, d, L) and (..., M, d, L)
    addr_x = padic_address(x, L=L, p=p)
    addr_y = padic_address(y, L=L, p=p)

    # Expand for broadcasting: 
    # addr_x: (..., N, d, L) -> (..., N, 1, d, L)
    # addr_y: (..., M, d, L) -> (..., 1, M, d, L)
    addr_x_exp = addr_x.unsqueeze(-3)
    addr_y_exp = addr_y.unsqueeze(-4)

    return padic_distance_from_addresses(addr_x_exp, addr_y_exp, p=p)
