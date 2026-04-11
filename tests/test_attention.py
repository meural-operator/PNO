"""
Tests for the P-Adic Attention layer.

Validates:
    - Output shapes for various configurations
    - Multi-head attention splitting
    - Proximity kernels produce valid weights
    - Content blending mode
    - Gradient flow through the layer
    - Equivariance: permuting inputs permutes outputs consistently
"""

import pytest
import torch

from padic_neural_operator.layers.attention import (
    PAdicAttention,
    ExponentialKernel,
    PowerLawKernel,
    LearnedKernel,
    build_kernel,
)


class TestProximityKernels:
    """Tests for proximity kernel implementations."""

    def test_exponential_kernel_positive(self):
        k = ExponentialKernel(alpha_init=2.0)
        d = torch.tensor([0.0, 0.5, 1.0, 2.0])
        w = k(d)
        assert (w > 0).all()
        # Monotonically decreasing
        assert (w[:-1] >= w[1:]).all()

    def test_power_law_kernel_positive(self):
        k = PowerLawKernel(alpha_init=1.0)
        d = torch.tensor([0.01, 0.5, 1.0, 2.0])
        w = k(d)
        assert (w > 0).all()
        # Monotonically decreasing
        assert (w[:-1] >= w[1:]).all()

    def test_learned_kernel_positive(self):
        k = LearnedKernel(hidden_dim=16, num_layers=2)
        d = torch.rand(10)
        w = k(d)
        assert (w >= 0).all()  # Softplus ensures non-negative
        assert w.shape == d.shape

    def test_build_kernel_factory(self):
        k1 = build_kernel("exponential")
        k2 = build_kernel("power_law")
        k3 = build_kernel("learned")
        assert isinstance(k1, ExponentialKernel)
        assert isinstance(k2, PowerLawKernel)
        assert isinstance(k3, LearnedKernel)

    def test_build_kernel_invalid(self):
        with pytest.raises(ValueError):
            build_kernel("nonexistent")


class TestPAdicAttention:
    """Tests for the PAdicAttention layer."""

    @pytest.fixture
    def default_layer(self):
        return PAdicAttention(d_model=32, n_heads=4, p=2, L=8)

    @pytest.fixture
    def sample_input(self):
        B, N, d_model, d_x = 2, 16, 32, 2
        v = torch.randn(B, N, d_model)
        x = torch.rand(B, N, d_x)
        return v, x

    def test_output_shape(self, default_layer, sample_input):
        v, x = sample_input
        out = default_layer(v, x)
        assert out.shape == v.shape

    def test_output_shape_single_head(self, sample_input):
        layer = PAdicAttention(d_model=32, n_heads=1, p=3, L=6)
        v, x = sample_input
        out = layer(v, x)
        assert out.shape == v.shape

    def test_different_primes(self, sample_input):
        v, x = sample_input
        for p in [2, 3, 5, 7]:
            layer = PAdicAttention(d_model=32, n_heads=2, p=p, L=6)
            out = layer(v, x)
            assert out.shape == v.shape
            assert torch.isfinite(out).all(), f"Non-finite output for p={p}"

    def test_different_kernels(self, sample_input):
        v, x = sample_input
        for kernel in ["exponential", "power_law", "learned"]:
            layer = PAdicAttention(d_model=32, n_heads=2, p=2, L=8, kernel_type=kernel)
            out = layer(v, x)
            assert out.shape == v.shape
            assert torch.isfinite(out).all(), f"Non-finite output for kernel={kernel}"

    def test_content_blend(self, sample_input):
        """Test that content blending runs without error."""
        layer = PAdicAttention(d_model=32, n_heads=2, p=2, L=8, content_blend=0.3)
        v, x = sample_input
        out = layer(v, x)
        assert out.shape == v.shape

    def test_gradient_flow(self, sample_input):
        """Verify gradients flow through the entire layer.

        Uses content_blend > 0 so that Q/K projections participate in the
        forward pass (pure p-adic path only routes through V and output).
        """
        layer = PAdicAttention(d_model=32, n_heads=2, p=2, L=8, content_blend=0.5)
        v, x = sample_input
        v.requires_grad_(True)
        out = layer(v, x)
        loss = out.sum()
        loss.backward()
        assert v.grad is not None
        assert v.grad.shape == v.shape
        # Check that parameter gradients exist
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_gradient_flow_pure_padic(self, sample_input):
        """Verify gradients flow in pure p-adic mode (no content blend).

        W_q and W_k have no gradient here because the pure p-adic path
        computes attention weights from spatial distances, not Q·K^T.
        Only W_v, W_o, and kernel parameters receive gradients.
        """
        layer = PAdicAttention(d_model=32, n_heads=2, p=2, L=8)
        v, x = sample_input
        v.requires_grad_(True)
        out = layer(v, x)
        loss = out.sum()
        loss.backward()
        assert v.grad is not None
        # V and output projections must have gradients
        assert layer.W_v.weight.grad is not None, "No gradient for W_v"
        assert layer.W_o.weight.grad is not None, "No gradient for W_o"
        # Q and K are unused in pure p-adic mode — no gradient expected
        assert layer.W_q.weight.grad is None
        assert layer.W_k.weight.grad is None

    def test_attention_weights_shape(self, default_layer, sample_input):
        v, x = sample_input
        weights = default_layer.get_attention_weights(v, x)
        B, N = v.shape[0], v.shape[1]
        assert weights.shape == (B, default_layer.n_heads, N, N)

    def test_attention_weights_row_stochastic(self, default_layer, sample_input):
        v, x = sample_input
        weights = default_layer.get_attention_weights(v, x)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_attention_weights_non_negative(self, default_layer, sample_input):
        v, x = sample_input
        weights = default_layer.get_attention_weights(v, x)
        assert (weights >= 0).all()

    def test_permutation_equivariance(self):
        """
        If we permute the input tokens, the output should be permuted in
        the same way (attention is permutation-equivariant).
        """
        layer = PAdicAttention(d_model=16, n_heads=2, p=2, L=8)
        layer.eval()

        B, N, d_model, d_x = 1, 8, 16, 1
        v = torch.randn(B, N, d_model)
        x = torch.rand(B, N, d_x)

        # Forward on original
        with torch.no_grad():
            out1 = layer(v, x)

        # Random permutation
        perm = torch.randperm(N)
        v_perm = v[:, perm, :]
        x_perm = x[:, perm, :]

        with torch.no_grad():
            out2 = layer(v_perm, x_perm)

        # Output should be permuted the same way
        out1_perm = out1[:, perm, :]
        assert torch.allclose(out1_perm, out2, atol=1e-5)

    def test_invalid_d_model(self):
        with pytest.raises(ValueError):
            PAdicAttention(d_model=33, n_heads=4, p=2, L=8)

    def test_mask(self, sample_input):
        """Masking out tokens should zero their contribution."""
        layer = PAdicAttention(d_model=32, n_heads=2, p=2, L=8)
        v, x = sample_input
        B, N = v.shape[:2]
        # Mask: only attend to first half
        mask = torch.zeros(B, N, N, dtype=torch.bool)
        mask[:, :, :N // 2] = True
        out = layer(v, x, mask=mask)
        assert out.shape == v.shape
        assert torch.isfinite(out).all()
