import torch
import torch.nn as nn
from adicton.transforms.wavelet import wavelet_decompose, wavelet_reconstruct
from adicton.nn.linear import PAdicLinear
from adicton.nn.activations import padic_ball_exact

class PAdicWaveletConv(nn.Module):
    def __init__(self, in_channels, out_channels, p=2, n=10):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.p = p
        self.n = n
        
        import math
        self.weight_scaling = nn.Parameter(
            torch.randn(out_channels, in_channels, dtype=torch.float64) / math.sqrt(in_channels)
        )
        
        self.weight_wavelet = nn.ParameterDict()
        for gamma in range(n):
            # Enforcing translation invariance: weights depend ONLY on scale (gamma) and oscillation (p-1)
            # Shape: (out_channels, in_channels, p-1)
            w_real = torch.randn(out_channels, in_channels, p - 1, dtype=torch.float64) / math.sqrt(in_channels)
            w_imag = torch.randn(out_channels, in_channels, p - 1, dtype=torch.float64) / math.sqrt(in_channels)
            self.weight_wavelet[f"real_{gamma}"] = nn.Parameter(w_real)
            self.weight_wavelet[f"imag_{gamma}"] = nn.Parameter(w_imag)

    def forward(self, x):
        B, C_in, M = x.shape
        assert C_in == self.in_channels
        
        # wavelet_decompose per channel
        x_flat = x.reshape(B * C_in, M)
        coeffs = wavelet_decompose(x_flat, self.p, self.n)
        
        out_coeffs = {}
        
        # Scaling branch
        scaling_c = coeffs['scaling'].view(B, C_in)
        scaling_out = torch.einsum("bc,oc->bo", scaling_c, self.weight_scaling.to(scaling_c.dtype))
        out_coeffs['scaling'] = scaling_out.view(B * self.out_channels)
        
        # Wavelet branches
        for gamma in range(self.n):
            num_blocks = self.p ** (self.n - gamma - 1)
            weight_complex = torch.complex(
                self.weight_wavelet[f"real_{gamma}"], 
                self.weight_wavelet[f"imag_{gamma}"]
            )
            for j_idx in range(1, self.p):
                w_list = [coeffs[f"w_{gamma}_{j_idx}_{a}"] for a in range(num_blocks)]
                w_tensor = torch.stack(w_list, dim=-1) # (B * C_in, num_blocks)
                w_tensor = w_tensor.view(B, C_in, num_blocks)
                
                w_weight = weight_complex[:, :, j_idx - 1]
                w_out = torch.einsum("bcn,oc->bon", w_tensor, w_weight)
                
                w_out_flat = w_out.reshape(B * self.out_channels, num_blocks)
                for a in range(num_blocks):
                    out_coeffs[f"w_{gamma}_{j_idx}_{a}"] = w_out_flat[:, a]
                    
        # Reconstruct
        out_flat = wavelet_reconstruct(out_coeffs, self.p, self.n)
        out = out_flat.reshape(B, self.out_channels, M)
        
        return out.real

class PAdicWaveletBlock(nn.Module):
    def __init__(self, in_channels, out_channels, p=2, n=10):
        super().__init__()
        self.wavelet_conv = PAdicWaveletConv(in_channels, out_channels, p, n)
        self.skip = PAdicLinear(in_channels, out_channels, p=p, bias=True)
        self.activation = nn.GELU()

    def forward(self, x):
        x_wavelet = self.wavelet_conv(x)
        x_skip = self.skip(x.transpose(-2, -1)).transpose(-2, -1)
        out = x_wavelet + x_skip
        return self.activation(out)

class PAdicNeuralOperator(nn.Module):
    def __init__(self, in_channels=1, out_channels=100, hidden_channels=32, num_blocks=4, p=2, n=10):
        super().__init__()
        self.lifting = PAdicLinear(in_channels, hidden_channels, p=p, bias=True)
        self.blocks = nn.ModuleList([
            PAdicWaveletBlock(hidden_channels, hidden_channels, p=p, n=n)
            for _ in range(num_blocks)
        ])
        self.projection = PAdicLinear(hidden_channels, out_channels, p=p, bias=True)

    def forward(self, x):
        # x: (B, C_in, M) -> PAdicLinear operates on (B, M, C_in)
        x = x.transpose(-2, -1)
        x = self.lifting(x)
        x = x.transpose(-2, -1)
        
        for block in self.blocks:
            x = block(x)
            
        x = x.transpose(-2, -1)
        x = self.projection(x)
        x = x.transpose(-2, -1)
        return x
