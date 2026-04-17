import os
import sys
import yaml
import time
import datetime
import shutil
import csv
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from padic_neural_operator.models import PAdicNeuralOperator
from padic_neural_operator.data import NS3DDataset
from padic_neural_operator.training.trainer import SobolevLoss

def setup_immutable_run(config, config_path):
    """Orchestrates research-grade topological run directory isolation."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"NS3D_Run_{timestamp}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Checkpoint sub-directories
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 1. Source State Snapshotting
    src_dir = os.path.join(run_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    
    # Copy calling scripts natively
    shutil.copy2(__file__, os.path.join(src_dir, "train_ns3d.py"))
    shutil.copy2(config_path, os.path.join(src_dir, "config.yaml"))
    
    # Shallow copy the core library purely to lock versioning
    import padic_neural_operator as pno
    pno_root = os.path.dirname(pno.__file__)
    shutil.copytree(pno_root, os.path.join(src_dir, "padic_neural_operator"), dirs_exist_ok=True)
    
    print(f"\n[SYSTEM] Run initialized globally: {run_dir}")
    print("[SYSTEM] Immutable source state locked and preserved.")
    return run_dir, ckpt_dir

def main():
    parser = argparse.ArgumentParser(description="3D Navier-Stokes PNO Orchestrator")
    parser.add_argument("--config", default="configs/ns3d.yaml", type=str, help="Configuration YAML path")
    parser.add_argument("--resume", default=None, type=str, help="Path to run_dir to resume from (optional)")
    args = parser.parse_args()
    
    # Load parameters
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device acquired: {device}")
    
    # 1. Pipeline Environment
    if args.resume:
        run_dir = args.resume
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        print(f"[SYSTEM] Resuming formally from {run_dir}")
    else:
        run_dir, ckpt_dir = setup_immutable_run(config, args.config)
        
    csv_log_path = os.path.join(run_dir, "training_metrics.csv")
    if not args.resume:
        with open(csv_log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Time(s)", "Train_Loss_MSE", "Val_Rel_L2_Loss", "LR"])
    
    # 2. Instantiate Z-Ordered 3D Dataset 
    # The NS3DDataset internally manages HDF5 locks and calculates topological Z curve maps natively.
    data_cfg = config["data"]
    train_dataset = NS3DDataset(
        file_path=data_cfg["path"],
        split="train",
        sub=data_cfg["sub"],
        train_ratio=data_cfg.get("train_ratio", 0.8),
        t_input_idx=data_cfg["t_input_idx"],
        t_target_idx=data_cfg["t_target_idx"]
    )
    
    val_dataset = NS3DDataset(
        file_path=data_cfg["path"],
        split="val",
        sub=data_cfg["sub"],
        train_ratio=data_cfg.get("train_ratio", 0.8),
        t_input_idx=data_cfg["t_input_idx"],
        t_target_idx=data_cfg["t_target_idx"]
    )
    
    # Dataloader configurations strictly for memory management
    # Note: Using persistent workers because fetching sub-arrays from HDF5 lazily
    train_loader = DataLoader(
        train_dataset, 
        batch_size=data_cfg["batch_size"], 
        shuffle=True, 
        num_workers=data_cfg.get("num_workers", 0)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=data_cfg["batch_size"], 
        shuffle=False, 
        num_workers=data_cfg.get("num_workers", 0)
    )

    # 3. Model Definition
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
        content_blend=model_cfg["content_blend"],
        d_coord=3 # We are passing (x,y,z) physically across the 3D domain map!
    ).to(device)
    
    # 4. Engine Mechanics (Optimizers, Scalers, Metrics)
    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    scaler = GradScaler(enabled=train_cfg.get("mixed_precision", True))
    
    # Metrics
    mse_criterion = nn.MSELoss()
    sobolev_criterion = SobolevLoss(lambda_h1=train_cfg.get("loss_weight", 0.5))
    loss_fn_type = train_cfg.get("loss_fn", "mse")

    start_epoch = 0
    best_loss = float('inf')
    
    # 5. Fault-Tolerance Resumption
    if args.resume:
        snapshot_path = os.path.join(ckpt_dir, "latest_snapshot.pth")
        if os.path.exists(snapshot_path):
            checkpoint = torch.load(snapshot_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint["best_loss"]
            print(f"[SYSTEM] Engine successfully resumed from Epoch {start_epoch}")

    # 6. Primary Execution Loop
    epochs = train_cfg["epochs"]
    ckpt_interval = train_cfg.get("checkpoint_interval", 5)
    
    print("\n" + "="*60)
    print(" COMMENCING HIGH-FIDELITY PNO NAVIER-STOKES TRAINING PHASE")
    print("="*60)
    
    # Enable robust memory allocation tracking natively
    torch.backends.cudnn.benchmark = True 
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        
        # Training iteration wrapper configured aesthetically for research grade UI
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, dynamic_ncols=True, colour='green')
        
        for x_in, y_target in train_bar:
            x_in, y_target = x_in.to(device), y_target.to(device)
            
            # Peel off physical features vs spatial parameters internally logic handled by architecture
            v_in = x_in[..., :-3]
            grid_in = x_in[..., -3:]
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=train_cfg.get("mixed_precision", True)):
                preds = model(v_in, grid_in)
                
                if loss_fn_type == "sob_l1":
                    loss = sobolev_criterion(preds, y_target)
                else:
                    loss = mse_criterion(preds, y_target)
            
            # Massive dimension 3D backwards scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            train_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        
        # Validation Procedure (using absolute mathematical Rel L2 formulation)
        model.eval()
        total_val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]", leave=False, dynamic_ncols=True, colour='cyan')
        
        with torch.no_grad():
            for x_in, y_target in val_bar:
                x_in, y_target = x_in.to(device), y_target.to(device)
                
                v_in = x_in[..., :-3]
                grid_in = x_in[..., -3:]
                
                with autocast(enabled=train_cfg.get("mixed_precision", True)):
                    preds = model(v_in, grid_in)
                    
                    # Un-normalize using DataLoader exact global tracking definitions
                    u_mean, u_std = val_dataset.u_mean.to(device), val_dataset.u_std.to(device)
                    y_target_phys = (y_target * u_std) + u_mean
                    preds_phys = (preds * u_std) + u_mean
                    
                    diff_norms = torch.norm((preds_phys - y_target_phys).reshape(preds_phys.shape[0], -1), 2, dim=1)
                    target_norms = torch.norm(y_target_phys.reshape(y_target_phys.shape[0], -1), 2, dim=1)
                    
                    val_err = torch.mean(diff_norms / target_norms)
                    total_val_loss += val_err.item()
                    
        avg_val_loss = total_val_loss / len(val_loader)
        
        epoch_dur = time.time() - epoch_start
        
        # UI Terminal Logging
        print(f"Epoch {epoch+1:03d}/{epochs:03d} | Train Loss: {avg_train_loss:.5f} | Val Rel L2: {avg_val_loss*100:.2f}% | LR: {lr:.2e} | Time: {epoch_dur:.1f}s")
        
        # State Archival
        with open(csv_log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, f"{epoch_dur:.2f}", f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}", lr])
            
        snapshot_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
        }
        
        # Save explicit state snapshot 
        torch.save(snapshot_state, os.path.join(ckpt_dir, "latest_snapshot.pth"))
        
        # Save interval metric versions
        if (epoch + 1) % ckpt_interval == 0:
            torch.save(snapshot_state, os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth"))
            
        # Hard lock mathematical best version
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
            
if __name__ == "__main__":
    main()
