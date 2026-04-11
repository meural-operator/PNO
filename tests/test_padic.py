"""
Tests for p-adic core math primitives.

Validates:
    - p-adic valuation and norm for integer inputs
    - p-adic addressing for general primes
    - Ultrametric inequality: d(x,z) <= max(d(x,y), d(y,z))
    - Distance consistency across different primes
    - Distance matrix symmetry and non-negativity
"""

import pytest
import torch

from padic_neural_operator.core.padic import (
    padic_valuation,
    padic_norm,
    padic_address,
    padic_distance,
    padic_distance_from_addresses,
    padic_distance_matrix,
)


class TestPadicValuation:
    """Tests for padic_valuation."""

    def test_v2_of_powers_of_2(self):
        assert padic_valuation(1, 2) == 0
        assert padic_valuation(2, 2) == 1
        assert padic_valuation(4, 2) == 2
        assert padic_valuation(8, 2) == 3
        assert padic_valuation(16, 2) == 4

    def test_v2_of_odd(self):
        assert padic_valuation(3, 2) == 0
        assert padic_valuation(7, 2) == 0
        assert padic_valuation(15, 2) == 0

    def test_v2_of_mixed(self):
        assert padic_valuation(6, 2) == 1   # 6 = 2 * 3
        assert padic_valuation(12, 2) == 2  # 12 = 4 * 3
        assert padic_valuation(24, 2) == 3  # 24 = 8 * 3

    def test_v3(self):
        assert padic_valuation(9, 3) == 2
        assert padic_valuation(27, 3) == 3
        assert padic_valuation(6, 3) == 1   # 6 = 2 * 3

    def test_v5(self):
        assert padic_valuation(25, 5) == 2
        assert padic_valuation(50, 5) == 2  # 50 = 2 * 25

    def test_zero(self):
        assert padic_valuation(0, 2) == -1

    def test_negative(self):
        assert padic_valuation(-8, 2) == 3

    def test_invalid_prime(self):
        with pytest.raises(ValueError):
            padic_valuation(10, 1)


class TestPadicNorm:
    """Tests for padic_norm."""

    def test_norm_of_powers(self):
        assert padic_norm(1, 2) == 1.0
        assert padic_norm(2, 2) == 0.5
        assert padic_norm(4, 2) == 0.25

    def test_norm_of_zero(self):
        assert padic_norm(0, 2) == 0.0

    def test_norm_p3(self):
        assert padic_norm(9, 3) == pytest.approx(1.0 / 9.0)
        assert padic_norm(27, 3) == pytest.approx(1.0 / 27.0)


class TestPadicAddress:
    """Tests for padic_address."""

    def test_dyadic_basic(self):
        """x=0.75 = 0.11 in binary => digits [1, 1]."""
        x = torch.tensor([0.75])
        addr = padic_address(x, L=2, p=2)
        assert addr.shape == (1, 2)
        assert addr[0, 0].item() == 1
        assert addr[0, 1].item() == 1

    def test_ternary_basic(self):
        """x=1/3 = 0.1 in base 3 => first digit is 1."""
        x = torch.tensor([1.0 / 3.0])
        addr = padic_address(x, L=3, p=3)
        assert addr.shape == (1, 3)
        assert addr[0, 0].item() == 1  # first ternary digit

    def test_address_shape_multidim(self):
        """Multi-dimensional input: (B, d) -> (B, d, L)."""
        x = torch.rand(5, 3)
        addr = padic_address(x, L=8, p=2)
        assert addr.shape == (5, 3, 8)

    def test_digits_in_valid_range(self):
        """All digits should be in {0, 1, ..., p-1}."""
        for p in [2, 3, 5, 7]:
            x = torch.rand(100)
            addr = padic_address(x, L=10, p=p)
            assert addr.min().item() >= 0
            assert addr.max().item() <= p - 1

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            padic_address(torch.tensor([0.5]), L=5, p=1)

    def test_invalid_L(self):
        with pytest.raises(ValueError):
            padic_address(torch.tensor([0.5]), L=0, p=2)


class TestPadicDistance:
    """Tests for p-adic distance computation."""

    def test_zero_distance_same_point(self):
        """Distance from a point to itself should be minimal (p^{-L})."""
        x = torch.tensor([0.3])
        d = padic_distance(x, x, L=10, p=2)
        expected_min = 2.0 ** (-10)
        assert d.item() == pytest.approx(expected_min)

    def test_max_distance_far_points(self):
        """Points in different halves: d(0.1, 0.9) should be 1.0 for p=2."""
        x = torch.tensor([0.1])
        y = torch.tensor([0.9])
        d = padic_distance(x, y, L=10, p=2)
        assert d.item() == pytest.approx(1.0)

    def test_ultrametric_inequality(self):
        """
        The ultrametric inequality: d(x,z) <= max(d(x,y), d(y,z)).

        This is the DEFINING property that distinguishes p-adic distances
        from Euclidean — it must hold for all triples.
        """
        torch.manual_seed(123)
        for p in [2, 3, 5]:
            x = torch.rand(100, 2)
            y = torch.rand(100, 2)
            z = torch.rand(100, 2)

            dxy = padic_distance(x, y, L=12, p=p)
            dyz = padic_distance(y, z, L=12, p=p)
            dxz = padic_distance(x, z, L=12, p=p)

            # Ultrametric: d(x,z) <= max(d(x,y), d(y,z))
            max_side = torch.max(dxy, dyz)
            violation = (dxz > max_side + 1e-7).any()
            assert not violation, f"Ultrametric violated for p={p}"

    def test_symmetry(self):
        """d(x, y) == d(y, x)."""
        x = torch.rand(50, 2)
        y = torch.rand(50, 2)
        dxy = padic_distance(x, y, L=10, p=3)
        dyx = padic_distance(y, x, L=10, p=3)
        assert torch.allclose(dxy, dyx)

    def test_distance_is_power_of_p(self):
        """P-adic distances are always of the form p^{-k}."""
        for p in [2, 3, 5]:
            x = torch.rand(200, 1)
            y = torch.rand(200, 1)
            d = padic_distance(x, y, L=10, p=p)
            # Each distance should be p^{-k} for some k in {0, 1, ..., L}
            valid_dists = {float(p) ** (-k) for k in range(11)}
            for di in d.tolist():
                assert any(abs(di - vd) < 1e-6 for vd in valid_dists), \
                    f"Distance {di} is not a power of p={p}"


class TestPadicDistanceMatrix:
    """Tests for padic_distance_matrix."""

    def test_shape(self):
        x = torch.rand(10, 2)
        D = padic_distance_matrix(x, L=8, p=2)
        assert D.shape == (10, 10)

    def test_symmetric(self):
        x = torch.rand(20, 3)
        D = padic_distance_matrix(x, L=8, p=3)
        assert torch.allclose(D, D.T)

    def test_diagonal_minimal(self):
        """Diagonal entries should be p^{-L} (self-distance at resolution limit)."""
        x = torch.rand(15, 2)
        L = 8
        p = 2
        D = padic_distance_matrix(x, L=L, p=p)
        expected_diag = float(p) ** (-L)
        assert torch.allclose(D.diag(), torch.full((15,), expected_diag))

    def test_non_square(self):
        """Distance matrix between different sets of points."""
        x = torch.rand(10, 2)
        y = torch.rand(7, 2)
        D = padic_distance_matrix(x, L=8, p=2, y=y)
        assert D.shape == (10, 7)

    def test_non_negative(self):
        x = torch.rand(20, 2)
        D = padic_distance_matrix(x, L=10, p=5)
        assert (D >= 0).all()
