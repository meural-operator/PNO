"""
Demo: P-Adic Attention on a 1D Grid
=====================================

Shows that p-adic attention produces hierarchical, tree-structured attention
patterns — tokens at the same branch of the p-adic tree attend strongly,
while tokens at distant branches attend weakly.

Usage:
    python examples/demo_attention.py
"""

import torch

from padic_neural_operator.core.padic import padic_distance_matrix
from padic_neural_operator.layers.attention import PAdicAttention


def demo_distance_matrix():
    """Visualize p-adic distance matrices for different primes."""
    N = 16
    x = torch.linspace(0.0, 1.0 - 1e-6, N).unsqueeze(-1)  # (N, 1)

    print("=" * 60)
    print("P-Adic Distance Matrices")
    print("=" * 60)

    for p in [2, 3, 5]:
        D = padic_distance_matrix(x, L=8, p=p)
        print(f"\np={p}: Distance matrix for {N} uniform points on [0,1)")
        print(f"  Shape: {D.shape}")
        print(f"  Unique distances: {sorted(D.unique().tolist())}")
        print(f"  Min (non-self): {D[D > D.min()].min().item():.6f}")
        print(f"  Max: {D.max().item():.6f}")


def demo_attention_forward():
    """Run p-adic attention on a simple 1D grid."""
    B, N, d_model, d_x = 1, 32, 16, 1
    L = 10

    # Uniform grid on [0, 1)
    x = torch.linspace(0.0, 1.0 - 1e-6, N).view(1, N, 1)
    v = torch.randn(B, N, d_model)

    print("\n" + "=" * 60)
    print("P-Adic Attention Forward Pass")
    print("=" * 60)

    for p in [2, 3]:
        for kernel in ["exponential", "power_law"]:
            layer = PAdicAttention(
                d_model=d_model,
                n_heads=2,
                p=p,
                L=L,
                kernel_type=kernel,
            )
            layer.eval()

            with torch.no_grad():
                out = layer(v, x)

            weights = layer.get_attention_weights(v, x)

            print(f"\np={p}, kernel={kernel}:")
            print(f"  Input:  {v.shape}")
            print(f"  Output: {out.shape}")
            print(f"  Attention weights shape: {weights.shape}")
            print(f"  Attention sparsity (max weight): {weights.max():.4f}")
            print(f"  Attention entropy (head 0): {-(weights[0, 0] * torch.log(weights[0, 0] + 1e-10)).sum(-1).mean():.4f}")


def demo_gradient_flow():
    """Verify gradients flow through p-adic attention."""
    B, N, d_model, d_x = 2, 16, 8, 2
    layer = PAdicAttention(d_model=d_model, n_heads=2, p=3, L=8)

    v = torch.randn(B, N, d_model, requires_grad=True)
    x = torch.rand(B, N, d_x)

    out = layer(v, x)
    loss = out.sum()
    loss.backward()

    print("\n" + "=" * 60)
    print("Gradient Flow Check")
    print("=" * 60)
    print(f"  Input grad norm: {v.grad.norm().item():.6f}")
    for name, param in layer.named_parameters():
        if param.grad is not None:
            print(f"  {name}: grad norm = {param.grad.norm().item():.6f}")
        else:
            print(f"  {name}: NO GRADIENT")


if __name__ == "__main__":
    demo_distance_matrix()
    demo_attention_forward()
    demo_gradient_flow()
    print("\nAll demos completed successfully!")
