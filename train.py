import os
import time
import shutil
import json
import argparse
from datetime import datetime
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath('adicton'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import DiffusionSorptionDataset, ReactionDiffusionDataset
from data.pdebench_dataset import PDEBench1DDataset
from data.schrodinger_dataset import PAdicSchrodingerDataset
from models.pno import PAdicNeuralOperator
from utils import LpLoss, UnitGaussianNormalizer

def setup_run_dir(config_path, base_dir="runs"):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    exp_name = config.get("experiment_name", "experiment")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Intuitive run dir name
    run_dir = os.path.join(base_dir, f"{exp_name}_{timestamp}")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    
    # Source mirror (immutable backup of code)
    src_mirror = os.path.join(run_dir, "src_mirror")
    os.makedirs(src_mirror, exist_ok=True)
    
    # Copy essential source files
    for d in ["models", "data", "adicton"]:
        if os.path.exists(d):
            shutil.copytree(d, os.path.join(src_mirror, d), dirs_exist_ok=True)
    for f in ["train.py", "utils.py", config_path]:
        if os.path.exists(f):
            shutil.copy2(f, src_mirror)
            
    # Save config to run dir
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
        
    return run_dir, config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pth file to resume from")
    args = parser.parse_args()
    
    if args.resume:
        run_dir = os.path.dirname(os.path.dirname(args.resume))
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Resuming from run directory: {run_dir}")
    else:
        run_dir, config = setup_run_dir(args.config)
        
    log_file = os.path.join(run_dir, "metrics.jsonl")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{datetime.now()}] Starting Run in {run_dir}")
    print(f"[{datetime.now()}] Using device: {device}")
    
    print("Loading datasets...")
    dataset_name = config['dataset'].get('name', 'DiffusionSorption')
    
    h5_path     = config['dataset']['h5_path']
    train_ratio = config['dataset']['train_ratio']

    if dataset_name == 'DiffusionSorption':
        train_dataset = DiffusionSorptionDataset(h5_path, split="train", train_ratio=train_ratio)
        val_dataset   = DiffusionSorptionDataset(h5_path, split="val",   train_ratio=train_ratio)

    elif dataset_name == 'ReactionDiffusion':
        train_dataset = ReactionDiffusionDataset(h5_path, split="train", train_ratio=train_ratio)
        val_dataset   = ReactionDiffusionDataset(h5_path, split="val",   train_ratio=train_ratio)

    elif dataset_name == 'PDEBench1D':
        # Generic loader for Burgers, Allen-Cahn, Advection, and other
        # 1D PDEBench-format datasets (HDF5 'tensor' key, shape N×T×X).
        pdebench_kwargs = dict(
            n_x      = config['dataset'].get('n_x',      1024),
            out_t    = config['dataset'].get('out_t',    100),
            ic_index = config['dataset'].get('ic_index', 0),
        )
        train_dataset = PDEBench1DDataset(h5_path, split="train", train_ratio=train_ratio, **pdebench_kwargs)
        val_dataset   = PDEBench1DDataset(h5_path, split="val",   train_ratio=train_ratio, **pdebench_kwargs)

    elif dataset_name == 'PAdicSchrodinger':
        # P-adic Schrodinger equation (free particle or with potential).
        # Input:  (4, M)       = [Re(psi_0), Im(psi_0), grid, V(x)]
        # Output: (2*out_t, M) = [Re(psi(t_1..out_t)), Im(psi(t_1..out_t))]
        # Requires out_channels = 2 * out_t in the model config (e.g. 200).
        schro_kwargs = dict(
            n_x      = config['dataset'].get('n_x',      1024),
            out_t    = config['dataset'].get('out_t',    100),
            ic_index = config['dataset'].get('ic_index', 0),
        )
        train_dataset = PAdicSchrodingerDataset(h5_path, split='train', train_ratio=train_ratio, **schro_kwargs)
        val_dataset   = PAdicSchrodingerDataset(h5_path, split='val',   train_ratio=train_ratio, **schro_kwargs)

    else:
        raise ValueError(f"Unknown dataset name: '{dataset_name}'. "
                         f"Choose from: DiffusionSorption, ReactionDiffusion, "
                         f"PDEBench1D, PAdicSchrodinger.")
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    print("Fitting normalizers...")
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
    
    print("Initializing model...")
    model = PAdicNeuralOperator(
        in_channels=config['model']['in_channels'], 
        out_channels=config['model']['out_channels'], 
        hidden_channels=config['model']['hidden_channels'], 
        num_blocks=config['model']['num_blocks'], 
        p=config['model']['p'], 
        n=config['model']['n']
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    epochs = config['training']['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = LpLoss(d=1, p=2)
    
    best_loss = float('inf')
    start_epoch = 1
    
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"Resumed at epoch {start_epoch-1} with best loss {best_loss:.6f}")
    
    print("Starting training...")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        samples_processed = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            x_enc = x_normalizer.encode(x)
            y_enc = y_normalizer.encode(y)
            x_enc = x_enc.to(torch.float64)
            
            optimizer.zero_grad()
            out = model(x_enc)
            out = out.to(torch.float32)
            
            # Loss for backprop in normalized space
            loss = criterion(out, y_enc)
            loss.backward()
            optimizer.step()
            
            # True relative error in physical space
            out_decoded = y_normalizer.decode(out)
            true_loss = criterion(out_decoded, y)
            train_loss += true_loss.item() * x.size(0)
            samples_processed += x.size(0)
            pbar.set_postfix({"loss": f"{(train_loss / samples_processed):.4f}"})
            
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        val_samples_processed = 0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]  ")
        with torch.no_grad():
            for x, y in pbar_val:
                x, y = x.to(device), y.to(device)
                x_enc = x_normalizer.encode(x).to(torch.float64)
                out = model(x_enc)
                out = out.to(torch.float32)
                
                out_decoded = y_normalizer.decode(out)
                loss = criterion(out_decoded, y)
                val_loss += loss.item() * x.size(0)
                val_samples_processed += x.size(0)
                pbar_val.set_postfix({"loss": f"{(val_loss / val_samples_processed):.4f}"})
                
        val_loss /= len(val_loader.dataset)
        
        # Logging
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0]
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
            
        # Checkpointing
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'val_loss': val_loss
        }
        
        torch.save(checkpoint_state, os.path.join(ckpt_dir, "latest.pth"))
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint_state, os.path.join(ckpt_dir, "best.pth"))
            
        if epoch % config['training'].get('save_every', 10) == 0:
            torch.save(checkpoint_state, os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))
            
if __name__ == "__main__":
    main()
