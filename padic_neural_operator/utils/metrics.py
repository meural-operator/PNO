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
