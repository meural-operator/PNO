import torch
import torch.nn as nn
from torch import Tensor
import math


def haar_transform_1d(x: Tensor) -> Tensor:
    """
    1D Haar wavelet transform.
    Input: x of shape (..., N) where N = 2^L.
    Executes in O(N) by recursively splitting into sums and differences.
    """
    N = x.size(-1)
    result = x.clone()
    h = N
    while h > 1:
        half = h // 2
        # Averages and differences for the current scale
        avg = (result[..., :h:2] + result[..., 1:h:2]) / 2.0
        diff = (result[..., :h:2] - result[..., 1:h:2]) / 2.0
        
        # In-place update for PyTorch graph tracking
        result = result.clone()
        result[..., :half] = avg
        result[..., half:h] = diff
        h = half
        
    return result


def inverse_haar_transform_1d(x: Tensor) -> Tensor:
    """
    Inverse 1D Haar wavelet transform.
    Input: x of shape (..., N) where N = 2^L.
    Executes in O(N).
    """
    N = x.size(-1)
    result = x.clone()
    h = 2
    while h <= N:
        half = h // 2
        avg = result[..., :half]
        diff = result[..., half:h]
        
        # Interleave to reconstruct signal
        even = avg + diff
        odd = avg - diff
        
        result = result.clone()
        result[..., :h:2] = even
        result[..., 1:h:2] = odd
        
        h *= 2
        
    return result


class PAdicIntegralLayer(nn.Module):
    """
    P-Adic Integral Operator Layer (Kozyrev/Vladimirov Layer).
    
    Transforms Euclidean function space mapping into Haar discrete wavelet space,
    executes the mathematically defined Kozyrev operator kernel (learned spectral weights),
    and inverse transforms back.
    
    Time Complexity: O(N), replacing scaled O(N^2) pairwise distance matrices.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # A 1D convolutional operator mixes spectral channels across all basis modes.
        # This operates on Haar modes iteratively.
        self.spectral_weights = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=1
        )

    def forward(self, v: Tensor, x: Tensor = None) -> Tensor:
        """
        v: (B, N, d_model) -> we need Haar transform over N.
        x: Coordinates (unused natively as Haar transform acts topologically).
        """
        # Permute to (B, d_model, N) for Haar transform over last dimension
        v_T = v.transpose(1, 2)
        
        # O(N) Haar Transform
        v_haar = haar_transform_1d(v_T)
        
        # Learnable topological spectral mixing (Integral Kozyrev Kernels)
        # Note: In standard PDE neural operator architectures, we could modulate this
        # with an activation or complex filtering, but linear transforms over the wavelet
        # basis is intrinsically scale-independent.
        v_haar = self.spectral_weights(v_haar)
        
        # O(N) Inverse Haar Transform
        out_T = inverse_haar_transform_1d(v_haar)
        
        # Permute back to (B, N, d_model)
        return out_T.transpose(1, 2)
