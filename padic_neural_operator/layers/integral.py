import torch
import torch.nn as nn
from torch import Tensor
import math
import torch.nn.functional as F

def create_dct_matrix(p: int) -> Tensor:
    """Returns a purely orthogonal (real-valued) DCT-II matrix of size p x p."""
    idx = torch.arange(p, dtype=torch.float32)
    k, n = torch.meshgrid(idx, idx, indexing="ij")
    
    matrix = torch.cos(math.pi * k * (2 * n + 1) / (2 * p))
    matrix[0, :] *= 1 / math.sqrt(2)
    matrix *= math.sqrt(2 / p)
    
    return matrix

def base_p_wavelet_transform(x: Tensor, L: int, dct_mat: Tensor) -> tuple[list[Tensor], Tensor]:
    """
    Forward Base-p wavelet transform.
    Args:
        x: Tensor of shape (B, C, p^L)
        L: Tree depth
        dct_mat: (p, p) orthogonal basis matrix (first row is uniform average).
    Returns:
        List containing detail coefficients (from finest to coarsest),
        and the final root scale average tensor.
    """
    p = dct_mat.size(0)
    current = x
    coeffs = []
    
    for l in range(L):
        B, C, H = current.shape
        current_g = current.view(B, C, H // p, p)
        
        # Apply orthogonal projection over the local p-group
        transformed = torch.matmul(current_g, dct_mat.t())
        
        # The first component is the normalized average (coarse signal for next step)
        coarse = transformed[..., 0]
        
        # The remaining p-1 components securely hold the orthogonal details
        details = transformed[..., 1:]
        
        coeffs.append(details)
        current = coarse
    
    # coeffs[0] is finest scale, coeffs[-1] is coarsest details before root.
    return coeffs, current

def inverse_base_p_wavelet_transform(coeffs: list[Tensor], root: Tensor, L: int, dct_mat: Tensor) -> Tensor:
    """
    Inverse Base-p wavelet transform via exact deterministic inversion.
    """
    p = dct_mat.size(0)
    current = root
    
    # Reconstruct from coarsest down to finest
    for l in reversed(range(L)):
        details = coeffs[l]
        B, C, H_over_p, p_minus_1 = details.shape
        
        combined = torch.cat([current.unsqueeze(-1), details], dim=-1)
        reconstructed = torch.matmul(combined, dct_mat)
        
        current = reconstructed.view(B, C, -1)
        
    return current


class PAdicIntegralLayer(nn.Module):
    """
    P-Adic Integral Operator Layer (Kozyrev/Vladimirov Layer).
    
    Mathematically upgrades the architecture by strictly segregating
    the spectral responses across distinct topological tree scales (defined by
    Kozyrev's theorem), instead of flattening all scales into a generic convolution.
    
    Fully dynamically aligns boundary grids to perfect p^L scaling spaces using replicate padding.
    """
    def __init__(self, d_model: int, p: int, L: int):
        super().__init__()
        self.d_model = d_model
        self.p = p
        self.L = L
        
        self.register_buffer("dct_mat", create_dct_matrix(p))
        
        # Isolated spectral operators correctly parameterized depth-wise
        self.scale_weights = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=1) for _ in range(L)
        ])
        
        # Root average integrator
        self.root_weight = nn.Conv1d(d_model, d_model, kernel_size=1)

    def _pad_to_power_of_p(self, v: Tensor) -> tuple[Tensor, int]:
        """Dynamically pads the spatial length to the nearest structural capacity."""
        B, C, N = v.shape
        target_N = self.p ** self.L
        
        if N == target_N:
            return v, 0
            
        if N > target_N:
            raise ValueError(f"Input spatial length {N} exceeds tree capacity {target_N} (p={self.p}, L={self.L}). Increase depth L.")
            
        pad_size = target_N - N
        # We use purely physical replicate padding bridging the boundary conditions safely
        v_padded = F.pad(v, (0, pad_size), mode='replicate')
        return v_padded, pad_size

    def forward(self, v: Tensor, x: Tensor = None) -> Tensor:
        # permute channels for 1D convolutions natively
        v_T = v.transpose(1, 2)
        
        # 1. Base-P strict topological grid alignment
        v_padded, pad_len = self._pad_to_power_of_p(v_T)
        
        # 2. Project into Haar-like discrete scaling space
        coeffs, root = base_p_wavelet_transform(v_padded, self.L, self.dct_mat)
        
        # 3. Modulate eigenvalues via independent linear transforms bound by Kozyrev structure
        new_coeffs = []
        for l in range(self.L):
            details = coeffs[l] 
            B, C, H_over_p, p_minus_1 = details.shape
            
            details_flat = details.reshape(B, C, -1)
            out = self.scale_weights[l](details_flat)
            new_coeffs.append(out.reshape(B, C, H_over_p, p_minus_1))
            
        B, C, root_H = root.shape 
        root_out = self.root_weight(root.reshape(B, C, -1)).reshape(B, C, root_H)
        
        # 4. Invert projection mapping
        out_padded = inverse_base_p_wavelet_transform(new_coeffs, root_out, self.L, self.dct_mat)
        
        # 5. Safe clipping of boundary projection
        if pad_len > 0:
            out_T = out_padded[..., :-pad_len]
        else:
            out_T = out_padded
            
        return out_T.transpose(1, 2)
