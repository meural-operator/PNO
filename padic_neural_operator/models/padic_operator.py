import math
import torch
from torch import Tensor
import torch.nn as nn
from padic_neural_operator.layers.attention import PAdicAttention


class FourierPositionalEncoding(nn.Module):
    """
    Fourier feature encoding for spatial coordinates.

    Maps coordinates x in [0,1)^d to a higher-dimensional space using
    random Fourier features: [sin(2*pi*B*x), cos(2*pi*B*x)] where B
    is drawn from N(0, sigma^2).

    This helps the model learn high-frequency functions — critical for
    PDE solutions with sharp gradients (e.g., Burgers shocks).
    """

    def __init__(self, d_coord: int, n_frequencies: int = 32, sigma: float = 10.0):
        super().__init__()
        B = torch.randn(d_coord, n_frequencies) * sigma
        self.register_buffer("B", B)
        self.out_dim = n_frequencies * 2

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., d_coord) -> (..., n_frequencies * 2)"""
        proj = 2 * math.pi * x @ self.B  # (..., n_frequencies)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PAdicBlock(nn.Module):
    """
    A single block combining PAdic Attention and an MLP with residual connections.
    """

    def __init__(self, d_model, n_heads, p, L, kernel_type="exponential", mlp_ratio=2.0, dropout=0.0, layer_scale_init_value=1e-4, content_blend=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PAdicAttention(
            d_model=d_model,
            n_heads=n_heads,
            p=p,
            L=L,
            kernel_type=kernel_type,
            content_blend=content_blend,
            dropout=dropout
        )
        self.gamma_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((d_model)), requires_grad=True
        ) if layer_scale_init_value > 0 else None

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(
            in_features=d_model,
            hidden_features=int(d_model * mlp_ratio),
            out_features=d_model,
            dropout=dropout,
        )
        self.gamma_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((d_model)), requires_grad=True
        ) if layer_scale_init_value > 0 else None

    def forward(self, v, x):
        # Attention path
        res = v
        v = self.norm1(v)
        v = self.attn(v, x)
        if self.gamma_1 is not None:
            v = self.gamma_1 * v
        v = v + res

        # MLP path
        res = v
        v = self.norm2(v)
        v = self.mlp(v)
        if self.gamma_2 is not None:
            v = self.gamma_2 * v
        v = v + res

        return v


class PAdicNeuralOperator(nn.Module):
    """
    P-Adic Neural Operator (PNO).

    Uses P-Adic Attention to compute global and multi-scale interactions
    without transforming into frequency domain. Includes Fourier positional
    encoding for spatial coordinates and optional content-based attention blending.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_model: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        p: int = 2,
        L: int = 10,
        kernel_type: str = "exponential",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        content_blend: float = 0.0,
        d_coord: int = 1,
        n_fourier_freq: int = 32,
        fourier_sigma: float = 10.0,
    ):
        super().__init__()

        # Fourier positional encoding for grid coordinates
        self.d_coord = d_coord
        self.pos_enc = FourierPositionalEncoding(
            d_coord=d_coord, n_frequencies=n_fourier_freq, sigma=fourier_sigma
        )
        # Lifting: input features + positional encoding → d_model
        self.lifting = nn.Linear(d_in + self.pos_enc.out_dim, d_model)

        self.blocks = nn.ModuleList([
            PAdicBlock(
                d_model=d_model,
                n_heads=n_heads,
                p=p,
                L=L,
                kernel_type=kernel_type,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                content_blend=content_blend,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Projection: two linear layers with activation
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_out),
        )

    def forward(self, v, x):
        """
        Args:
            v: Input features (Batch, N_points, d_in)
            x: Grid coordinates (Batch, N_points, d_x). Values must be in [0, 1).

        Returns:
            Output features (Batch, N_points, d_out)
        """
        # Ensure x is strictly < 1.0 for p-adic address safety
        x = x.clamp(0.0, 1.0 - 1e-6)

        # Fourier positional encoding of coordinates
        pos_features = self.pos_enc(x)  # (B, N, n_freq*2)

        # Concatenate input features with positional encoding before lifting
        v = torch.cat([v, pos_features], dim=-1)  # (B, N, d_in + n_freq*2)

        # 1. Lift to model dimension
        v = self.lifting(v)

        # 2. Iterative P-Adic processing
        for block in self.blocks:
            v = block(v, x)

        v = self.norm(v)

        # 3. Project to output dimension
        out = self.projection(v)

        return out
