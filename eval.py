import os
import sys
import json
import glob
import torch
import matplotlib.pyplot as plt

# Ensure adicton is in the path
sys.path.append(os.path.abspath('adicton'))

from models.pno import PAdicNeuralOperator
from data.dataset import DiffusionSorptionDataset
from utils import LpLoss, UnitGaussianNormalizer

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    if not os.path.exists('config.json'):
        print("config.json not found!")
        return
        
    with open('config.json', 'r') as f:
        config = json.load(f)
        
    # Find latest run directory
    run_dirs = sorted(glob.glob("runs/PNO_Wavelet_DiffSorp_*"))
    if not run_dirs:
        print("No run directories found.")
        return
    latest_run = run_dirs[-1]
    best_ckpt = os.path.join(latest_run, "checkpoints", "best.pth")
    if not os.path.exists(best_ckpt):
        print(f"Checkpoint not found at {best_ckpt}. Training might not have saved it yet.")
        return
        
    print(f"Loading best checkpoint from: {best_ckpt}")
    
    # Load model
    model = PAdicNeuralOperator(
        in_channels=config['model']['in_channels'], 
        out_channels=config['model']['out_channels'], 
        hidden_channels=config['model']['hidden_channels'], 
        num_blocks=config['model']['num_blocks'], 
        p=config['model']['p'], 
        n=config['model']['n']
    ).to(device)
    
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    model.eval()

    # Load dataset to get normalizers and test sample
    print("Loading dataset for evaluation...")
    train_dataset = DiffusionSorptionDataset(config['dataset']['h5_path'], split="train", train_ratio=config['dataset']['train_ratio'])
    val_dataset = DiffusionSorptionDataset(config['dataset']['h5_path'], split="val", train_ratio=config['dataset']['train_ratio'])
    
    # Fit normalizers exactly as during training
    x_train_list, y_train_list = [], []
    for x, y in train_dataset:
        x_train_list.append(x)
        y_train_list.append(y)
    x_train = torch.stack(x_train_list)
    y_train = torch.stack(y_train_list)
    
    x_normalizer = UnitGaussianNormalizer(x_train)
    y_normalizer = UnitGaussianNormalizer(y_train)
    x_normalizer.to(device)
    y_normalizer.to(device)
    
    # Evaluate on a single validation sample
    sample_idx = 0
    x_val, y_val = val_dataset[sample_idx]
    x_val = x_val.unsqueeze(0).to(device) # (1, 1, 1024)
    y_val = y_val.unsqueeze(0).to(device) # (1, 100, 1024)
    
    print(f"Evaluating on Validation Sample {sample_idx}...")
    with torch.no_grad():
        x_enc = x_normalizer.encode(x_val).to(torch.float64)
        out = model(x_enc).to(torch.float32)
        out_decoded = y_normalizer.decode(out)
        
    criterion = LpLoss(d=1, p=2)
    error = criterion(out_decoded, y_val).item()
    print(f"Relative L2 Error on Sample {sample_idx}: {error:.6f}")
    
    # Plotting
    print("Generating plots...")
    pred = out_decoded[0].cpu().numpy() # (100, 1024)
    truth = y_val[0].cpu().numpy() # (100, 1024)
    
    timesteps = [10, 50, 90]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, t in enumerate(timesteps):
        axes[i].plot(truth[t], label="Ground Truth", color='black', linestyle='dashed', linewidth=2)
        axes[i].plot(pred[t], label="P-WNO Prediction", color='red', alpha=0.7, linewidth=2)
        axes[i].set_title(f"Time Step {t}")
        axes[i].set_xlabel("Spatial coordinate x")
        axes[i].set_ylabel("u(x, t)")
        axes[i].legend()
        
    plt.tight_layout()
    plot_path = "evaluation_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
