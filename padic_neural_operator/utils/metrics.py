"""
Evaluation metrics for PDE neural operators.
"""

import torch
import torch.nn as nn
from torch import Tensor


class RelativeL2Loss(nn.Module):
    """
    Relative L2 error: ||pred - target||_2 / ||target||_2.
    Standard metric for PDE operator learning.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff_norms = torch.norm(pred - target, p=2, dim=1)
        target_norms = torch.norm(target, p=2, dim=1)
        loss = diff_norms / (target_norms + self.eps)
        return loss.mean()


class SobolevLoss(nn.Module):
    """
    H1 Sobolev Loss: Computes standard relative L2 error + relative L2 error of the spatial gradients.
    This natively enforces shock wave steepness constraints without relying on heuristic clamping.
    
    Formula: sqrt( (||u - pred||_2^2 + lambda * ||Du - Dpred||_2^2) / (||u||_2^2 + lambda * ||Du||_2^2) )
    """
    def __init__(self, eps=1e-8, lmbda=0.5):
        super().__init__()
        self.eps = eps
        self.lmbda = lmbda

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # Base pointwise L2 loss squared
        diff_sq = torch.sum((pred - target) ** 2, dim=1)
        target_sq = torch.sum(target ** 2, dim=1)
        
        # Spatial Gradient (Forward difference along the spatial sequence)
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        
        grad_diff_sq = torch.sum((pred_diff - target_diff) ** 2, dim=1)
        grad_target_sq = torch.sum(target_diff ** 2, dim=1)
        
        # Rigorous aggregate Relative H1 constraint. 
        # The denominator relies on the base energy field (target_sq), guaranteeing 
        # algorithmic stability globally without arbitrary constant clamps!
        numerator = diff_sq + self.lmbda * grad_diff_sq
        denominator = target_sq + self.lmbda * grad_target_sq
        
        loss = torch.sqrt(numerator / (denominator + self.eps))
        
        return loss.mean()


class LinfError(nn.Module):
    """Relative L-infinity error: max|pred - target| / max|target|."""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        B = pred.shape[0]
        pred_flat = pred.view(B, -1)
        target_flat = target.view(B, -1)
        linf = (pred_flat - target_flat).abs().max(dim=-1).values
        target_max = target_flat.abs().max(dim=-1).values
        return (linf / (target_max + self.eps)).mean()


@torch.no_grad()
def compute_all_metrics(model, dataloader, device="cuda", use_amp=False, grid_dims=1):
    """
    Evaluate model on a dataloader and return a dict of metrics.

    Returns:
        dict with keys: 'rel_l2', 'linf', 'mse', 'n_samples'
    """
    from torch.amp import autocast
    model.eval()

    rel_l2_fn = RelativeL2Loss()
    linf_fn = LinfError()
    total_rel_l2 = 0.0
    total_linf = 0.0
    total_mse = 0.0
    n_batches = 0

    for x, y in dataloader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        gd = grid_dims
        v = x[..., :-gd]
        grid_c = x[..., -gd:]

        if use_amp:
            with autocast("cuda"):
                out = model(v, grid_c)
        else:
            out = model(v, grid_c)

        total_rel_l2 += rel_l2_fn(out, y).item()
        total_linf += linf_fn(out, y).item()
        total_mse += torch.nn.functional.mse_loss(out, y).item()
        n_batches += 1

    return {
        "rel_l2": total_rel_l2 / n_batches,
        "linf": total_linf / n_batches,
        "mse": total_mse / n_batches,
        "n_samples": len(dataloader.dataset),
    }
