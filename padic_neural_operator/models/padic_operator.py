import math
import torch
from torch import Tensor
import torch.nn as nn
from padic_neural_operator.layers.attention import PAdicAttention
from padic_neural_operator.layers.integral import PAdicIntegralLayer
from padic_neural_operator.core.padic import padic_address
class PAdicPositionalEncoding(nn.Module):
    """
    P-Adic Positional Encoding for spatial coordinates.

    Maps coordinates x in [0,1)^d via their structural base-p digit expansion
    to represent their exact coordinate address inside the Haar/P-Adic wavelet tree.
    This entirely abolishes Euclidean space mappings (Gibbs artifacts).
    """

    def __init__(self, p: int, L: int, d_coord: int = 1, embed_dim: int = 16):
        super().__init__()
        self.p = p
        self.L = L
        self.d_coord = d_coord
        self.embed_dim = embed_dim
        # A unique categorical lookup for each branching layer avoids representation collapse
        self.level_embeddings = nn.ModuleList([
            nn.Embedding(p, embed_dim) for _ in range(L * d_coord)
        ])
        self.out_dim = L * d_coord * embed_dim

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., d_coord) -> (..., out_dim)"""
        # (..., d_coord, L)
        addr = padic_address(x, L=self.L, p=self.p).long()
        
        # Safely cap values inside the prime base (categorical safety constraint)
        addr = torch.clamp(addr, 0, self.p - 1)
        
        embeddings = []
        idx = 0
        for d in range(self.d_coord):
            for l in range(self.L):
                digit = addr[..., d, l]
                emb = self.level_embeddings[idx](digit)
                embeddings.append(emb)
                idx += 1
                
        return torch.cat(embeddings, dim=-1)


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
        # Spectral Kozyrev/Vladimirov Layer (O(N) Haar-basis evaluation)
        self.spectral_layer = PAdicIntegralLayer(d_model=d_model)
        
        # Learnable topological blend between Global Attention and Spectral Wavelet integration
        self.spectral_blend = nn.Parameter(torch.tensor(0.5))

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
        # Neural Operator Dual Pathways
        res = v
        v_norm = self.norm1(v)
        
        v_attn = self.attn(v_norm, x)
        v_spec = self.spectral_layer(v_norm, x)
        
        b = torch.sigmoid(self.spectral_blend)
        v_mixed = b * v_attn + (1 - b) * v_spec
        
        if self.gamma_1 is not None:
            v_mixed = self.gamma_1 * v_mixed
        v = v_mixed + res

        # MLP Component
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
        self.pos_enc = PAdicPositionalEncoding(
            p=p, L=L, d_coord=d_coord, embed_dim=16
        )
        # Lifting: input features + positional embedding → d_model
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

        # Parse physical coordinates through discrete Haar wavelet tree
        pos_features = self.pos_enc(x)  

        # Concatenate input physics features with discrete positional embedding before lifting
        v = torch.cat([v, pos_features], dim=-1)  

        # 1. Lift to model dimension
        v = self.lifting(v)

        # 2. Iterative P-Adic processing
        for block in self.blocks:
            v = block(v, x)

        v = self.norm(v)

        # 3. Project to output dimension
        out = self.projection(v)

        return out
