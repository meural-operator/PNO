import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_shock_metrics(true_u, pred_u, dx):
    grad_true = np.gradient(true_u, dx)
    grad_pred = np.gradient(pred_u, dx)
    max_grad_true = np.max(np.abs(grad_true))
    max_grad_pred = np.max(np.abs(grad_pred))
    grad_relative_error = abs(max_grad_true - max_grad_pred) / max_grad_true
    return max_grad_true, max_grad_pred, grad_relative_error, grad_true, grad_pred

def relative_l2(true_u, pred_u):
    return np.linalg.norm(true_u - pred_u) / np.linalg.norm(true_u)

def evaluate_run(run_dir):
    # Setup paths
    config_path = os.path.join(run_dir, "config.yaml")
    ckpt_path = os.path.join(run_dir, "checkpoints", "best_model.pth")
    src_dir = os.path.join(run_dir, "src")
    
    if not os.path.exists(config_path) or not os.path.exists(ckpt_path) or not os.path.exists(src_dir):
        raise FileNotFoundError(f"Run dir {run_dir} is missing config, checkpoint, or src mirror.")

    # 1. Inject mirrored source into python path to ensure exact architectural match
    sys.path.insert(0, src_dir)
    print(f"Injected mirror source from: {src_dir}")
    
    # Import from the mirror!
    from padic_neural_operator.data import Burgers1DDataset
    from padic_neural_operator.models import PAdicNeuralOperator

    # 2. Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 3. Load Dataset
    data_cfg = config["data"]
    file_p = data_cfg["path"]
    
    val_dataset = Burgers1DDataset(
        split="val",
        file_path=file_p,
        n_samples=data_cfg["n_samples"],
        train_ratio=data_cfg["train_ratio"],
        sub=data_cfg["sub"],
        t_target_idx=data_cfg["t_target_idx"]
    )
    print(f"Validation dataset loaded with {len(val_dataset)} samples.")

    # 4. Load Model
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
    print(f"Successfully loaded best_model.pth (Epoch {checkpoint.get('epoch', 'N/A')})")

    # 5. Evaluate and Visualize Samples
    samples_to_visualize = [12, 100, 200]
    
    u_mean = val_dataset.u_mean.cpu().numpy()
    u_std = val_dataset.u_std.cpu().numpy()
    grid = val_dataset.grid.numpy()
    dx = grid[1] - grid[0]
    gd = data_cfg.get("grid_dims", 1)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    for idx in samples_to_visualize:
        x, y_true_norm = val_dataset[idx]
        x_input = x.unsqueeze(0).to(device)
        
        v = x_input[..., :-gd]
        grid_c = x_input[..., -gd:]
        
        with torch.no_grad():
            y_pred_norm = model(v, grid_c).squeeze(0).cpu()
        
        # Un-normalize
        y_true = (y_true_norm.numpy() * u_std) + u_mean
        y_pred = (y_pred_norm.numpy() * u_std) + u_mean
        
        metrics = compute_shock_metrics(y_true.flatten(), y_pred.flatten(), dx)
        max_grad_true, max_grad_pred, grad_rel_err, grad_true, grad_pred = metrics
        rel_l2 = relative_l2(y_true.flatten(), y_pred.flatten())

        print(f"Sample {idx:3d} | Rel L2 Error: {rel_l2*100:6.2f}% | Gradient Error: {grad_rel_err * 100:6.2f}%")

        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(grid, y_true.flatten(), label="Ground Truth", color='black', linewidth=2)
        ax1.plot(grid, y_pred.flatten(), label="PNO Prediction", color='red', linestyle='--', linewidth=2, alpha=0.9)
        ax1.set_title(f"Sample {idx} - Real-Space Prediction")
        ax1.set_xlabel("x")
        ax1.set_ylabel("u(x)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(grid, grad_true, label="True du/dx", color='black', linewidth=2)
        ax2.plot(grid, grad_pred, label="Pred du/dx", color='blue', linestyle='--', linewidth=2, alpha=0.9)
        ax2.set_title(f"Sample {idx} - Gradient Shock Preservation")
        ax2.set_xlabel("x")
        ax2.set_ylabel("du/dx")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(run_dir, f"eval_sample_{idx}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    print("="*50)
    print(f"Plots saved to {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the runs/ folder")
    args = parser.parse_args()
    evaluate_run(args.run_dir)
