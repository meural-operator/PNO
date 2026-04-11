"""
Unified PDE training script.

Supports all datasets via config:
  dataset_type: burgers1d | darcy2d | ns3d | boussinesq | cylinder2d | bc_cylinder

Usage:
  python train_pde.py --config configs/darcy2d.yaml
"""

import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from padic_neural_operator.data import (
    Burgers1DDataset,
    DarcyFlow2DDataset,
    NS3DDataset,
    BoussinesqDataset,
    Cylinder2DDataset,
    BCCylinderDataset,
)
from padic_neural_operator.models import PAdicNeuralOperator
from padic_neural_operator.training import PNOTrainer
from padic_neural_operator.utils import parse_args_and_config, ExperimentManager


# ---------------------------------------------------------------------------
# Weight init
# ---------------------------------------------------------------------------

def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def build_warmup_cosine_schedule(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def build_datasets(data_cfg, split):
    """Build train or val dataset from config."""
    ds_type = data_cfg.get("dataset_type", "burgers1d")
    common = dict(
        split=split,
        train_ratio=data_cfg.get("train_ratio", 0.8),
        val_ratio=data_cfg.get("val_ratio", 0.1),
        n_samples=data_cfg.get("n_samples"),
    )

    if ds_type == "burgers1d":
        return Burgers1DDataset(
            file_path=data_cfg["path"],
            sub=data_cfg.get("sub", 1),
            t_target_idx=data_cfg.get("t_target_idx", -1),
            **common,
        )
    elif ds_type == "darcy2d":
        return DarcyFlow2DDataset(
            file_path=data_cfg["path"],
            sub=data_cfg.get("sub", 1),
            **common,
        )
    elif ds_type == "ns3d":
        return NS3DDataset(
            file_path=data_cfg["path"],
            sub=data_cfg.get("sub", 4),
            t_input_idx=data_cfg.get("t_input_idx", 0),
            t_target_idx=data_cfg.get("t_target_idx", -1),
            **common,
        )
    elif ds_type in ("boussinesq", "cylinder2d"):
        cls = BoussinesqDataset if ds_type == "boussinesq" else Cylinder2DDataset
        return cls(
            file_path=data_cfg["path"],
            dt_steps=data_cfg.get("dt_steps", 10),
            sub_spatial=data_cfg.get("sub_spatial", 1),
            sub_temporal=data_cfg.get("sub_temporal", 1),
            split=split,
            train_ratio=data_cfg.get("train_ratio", 0.8),
            val_ratio=data_cfg.get("val_ratio", 0.1),
            n_samples=data_cfg.get("n_samples"),
        )
    elif ds_type == "bc_cylinder":
        return BCCylinderDataset(
            data_dir=data_cfg["path"],
            dt_steps=data_cfg.get("dt_steps", 10),
            sub_spatial=data_cfg.get("sub_spatial", 1),
            sub_temporal=data_cfg.get("sub_temporal", 1),
            split=split,
            train_ratio=data_cfg.get("train_ratio", 0.8),
            val_ratio=data_cfg.get("val_ratio", 0.1),
            n_samples=data_cfg.get("n_samples"),
        )
    else:
        raise ValueError(f"Unknown dataset_type: {ds_type}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    config = parse_args_and_config()

    # Seed from env (benchmark runner) or config
    seed = int(os.environ.get("PNO_SEED", config.get("seed", 42)))
    set_seed(seed)
    print(f"Random seed: {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    tracker = ExperimentManager(config=config, base_log_dir="runs")
    print(f"Tracking experiment at: {tracker.run_dir}")

    # ---- Data ----
    print("Loading datasets...")
    train_dataset = build_datasets(data_cfg, "train")
    val_dataset = build_datasets(data_cfg, "val")

    num_workers = train_cfg.get("num_workers", 4)
    pin_memory = device == "cuda"
    loader_kwargs = dict(
        batch_size=train_cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ---- Model ----
    # Infer coordinate dimension from dataset type
    coord_dims_map = {
        "burgers1d": 1, "darcy2d": 2, "ns3d": 3,
        "boussinesq": 2, "cylinder2d": 2, "bc_cylinder": 2,
    }
    d_coord = model_cfg.get("d_coord", coord_dims_map.get(data_cfg.get("dataset_type", "burgers1d"), 1))

    model = PAdicNeuralOperator(
        d_in=model_cfg["d_in"],
        d_out=model_cfg["d_out"],
        d_model=model_cfg.get("d_model", 64),
        n_layers=model_cfg.get("n_layers", 4),
        n_heads=model_cfg.get("n_heads", 4),
        p=model_cfg.get("p", 3),
        L=model_cfg.get("L", 8),
        kernel_type=model_cfg.get("kernel_type", "exponential"),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        dropout=model_cfg.get("dropout", 0.0),
        content_blend=model_cfg.get("content_blend", 0.0),
        d_coord=d_coord,
        n_fourier_freq=model_cfg.get("n_fourier_freq", 32),
        fourier_sigma=model_cfg.get("fourier_sigma", 10.0),
    ).to(device)
    model.apply(_init_weights)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # ---- Optimizer & scheduler ----
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    warmup_epochs = train_cfg.get("warmup_epochs", int(0.05 * train_cfg["epochs"]))
    scheduler = build_warmup_cosine_schedule(optimizer, warmup_epochs, train_cfg["epochs"])

    # ---- Resume ----
    start_epoch = 1
    if config.get("resume"):
        start_epoch, _ = tracker.load_checkpoint(
            config["resume"], model, optimizer, scheduler, device
        )
        start_epoch += 1

    # ---- Train ----
    # Infer grid dimensionality from dataset type
    grid_dims_map = {
        "burgers1d": 1, "darcy2d": 2, "ns3d": 3,
        "boussinesq": 2, "cylinder2d": 2, "bc_cylinder": 2,
    }
    grid_dims = data_cfg.get("grid_dims", grid_dims_map.get(data_cfg.get("dataset_type", "burgers1d"), 1))

    trainer = PNOTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        experiment_manager=tracker,
        device=device,
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        use_amp=train_cfg.get("use_amp", True),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 0),
        grid_dims=grid_dims,
    )

    print("Starting training...")
    try:
        trainer.train(train_loader, val_loader, epochs=train_cfg["epochs"], start_epoch=start_epoch)
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    finally:
        tracker.close()
        print("Training finished.")


if __name__ == "__main__":
    main()
