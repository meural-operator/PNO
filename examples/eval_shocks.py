import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from padic_neural_operator.data import Burgers1DDataset
from padic_neural_operator.models import PAdicNeuralOperator

def compute_shock_metrics(true_u, pred_u, dx):
    """Computes max gradient and shock width metrics."""
    # Compute numerical gradients
    grad_true = np.gradient(true_u, dx)
    grad_pred = np.gradient(pred_u, dx)
    
    max_grad_true = np.max(np.abs(grad_true))
    max_grad_pred = np.max(np.abs(grad_pred))
    
    grad_relative_error = abs(max_grad_true - max_grad_pred) / max_grad_true
    
    return max_grad_true, max_grad_pred, grad_relative_error, grad_true, grad_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=str, help="Directory of the training run")
    parser.add_argument("--idx", type=int, default=12, help="Sample index in validation set to visualize")
    args = parser.parse_args()

    config_path = os.path.join(args.run_dir, "config.yaml")
    ckpt_path = os.path.join(args.run_dir, "checkpoints", "best_model.pth")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing configs: {config_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
    data_cfg = config["data"]
    # Path relative or absolute as loaded from old config
    file_p = data_cfg["path"]
    if not os.path.isabs(file_p):
        file_p = os.path.normpath(os.path.join(args.run_dir, "..", "..", file_p))
        
    val_dataset = Burgers1DDataset(
        split="val",
        file_path=file_p,
        n_samples=data_cfg["n_samples"],
        train_ratio=data_cfg["train_ratio"],
        sub=data_cfg["sub"],
        t_target_idx=data_cfg["t_target_idx"]
    )
    
    # Load model
    model_cfg = config["model"]
    model = PAdicNeuralOperator(
        d_in=model_cfg["d_in"],
        d_out=model_cfg["d_out"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        p=model_cfg["p"],
        L=model_cfg["L"],
        kernel_type=model_cfg["kernel_type"],
        mlp_ratio=model_cfg.get("mlp_ratio", 2.0),
        dropout=model_cfg.get("dropout", 0.0),
        content_blend=model_cfg.get("content_blend", 0.0),
    ).to(device)

    # Load weights
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Get test sample
    x, y_true_norm = val_dataset[args.idx]
    x_input = x.unsqueeze(0).to(device)
    
    gd = data_cfg.get("grid_dims", 1)
    v = x_input[..., :-gd]
    grid_c = x_input[..., -gd:]
    
    with torch.no_grad():
        y_pred_norm = model(v, grid_c).squeeze(0).cpu()
    
    # Un-normalize to physical domain
    u_mean = val_dataset.u_mean.cpu()
    u_std = val_dataset.u_std.cpu()
    y_true = (y_true_norm * u_std + u_mean).numpy()
    y_pred = (y_pred_norm * u_std + u_mean).numpy()
    
    grid = val_dataset.grid.numpy()
    dx = grid[1] - grid[0]
    
    # Compute metrics
    metrics = compute_shock_metrics(y_true.flatten(), y_pred.flatten(), dx)
    max_grad_true, max_grad_pred, grad_rel_err, grad_true, grad_pred = metrics
    
    print("-" * 50)
    print("Numerical Shock Resolution Analysis:")
    print(f"Max absolute physical gradient (Ground Truth): {max_grad_true:.4f}")
    print(f"Max absolute physical gradient (PAdic Neural Operator pred): {max_grad_pred:.4f}")
    print(f"Gradient Steepness Relative Error: {grad_rel_err * 100:.2f}%")
    print("-" * 50)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute values plot
    ax1.plot(grid, y_true.flatten(), label="Ground Truth (Exact)", color='black', linewidth=2)
    ax1.plot(grid, y_pred.flatten(), label="P-Adic Prediction", color='red', linestyle='--', linewidth=2, alpha=0.9)
    ax1.set_title(f"Burgers 1D Equation - Validation Sample {args.idx}")
    ax1.set_xlabel("Spatial coordinate x")
    ax1.set_ylabel("Velocity u(x, t)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gradient plot
    ax2.plot(grid, grad_true, label="True Gradient du/dx", color='black', linewidth=2)
    ax2.plot(grid, grad_pred, label="Predicted Gradient du/dx", color='blue', linestyle='--', linewidth=2, alpha=0.9)
    ax2.set_title("Derivative Evaluation (Shock Sharpness / Gradient)")
    ax2.set_xlabel("Spatial coordinate x")
    ax2.set_ylabel("du/dx")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(args.run_dir, f"shock_evaluation_sample_{args.idx}.png")
    plt.savefig(save_path, dpi=150)
    print(f"Evaluation plot saved successfully to {save_path}")

if __name__ == "__main__":
    main()
