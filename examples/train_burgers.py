import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from padic_neural_operator.data import Burgers1DDataset
from padic_neural_operator.models import PAdicNeuralOperator
from padic_neural_operator.training import PNOTrainer
from padic_neural_operator.utils import parse_args_and_config, ExperimentManager


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def _init_weights(module):
    """Xavier uniform for linear projections, small init for output layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup then cosine decay
# ---------------------------------------------------------------------------

def build_warmup_cosine_schedule(optimizer, warmup_epochs, total_epochs):
    """Linear warmup for `warmup_epochs`, then cosine anneal to 0."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    config = parse_args_and_config()

    seed = int(os.environ.get("PNO_SEED", config.get("seed", 42)))
    set_seed(seed)
    print(f"Random seed: {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    # ---- Experiment tracking ----
    tracker = ExperimentManager(config=config, base_log_dir="runs")
    print(f"Tracking experiment at: {tracker.run_dir}")

    # ---- Datasets & loaders ----
    print("Loading datasets...")
    common_ds_kwargs = dict(
        file_path=data_cfg["path"],
        n_samples=data_cfg["n_samples"],
        train_ratio=data_cfg["train_ratio"],
        sub=data_cfg["sub"],
        t_target_idx=data_cfg["t_target_idx"],
    )
    train_dataset = Burgers1DDataset(split="train", **common_ds_kwargs)
    val_dataset = Burgers1DDataset(split="val", **common_ds_kwargs)

    num_workers = train_cfg.get("num_workers", 4)
    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ---- Model ----
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

    # Apply weight initialisation
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

    # ---- Resume logic ----
    start_epoch = 1
    if config.get("resume"):
        start_epoch, best_val = tracker.load_checkpoint(
            checkpoint_path=config["resume"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch += 1

    # ---- Trainer ----
    trainer = PNOTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        experiment_manager=tracker,
        device=device,
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        use_amp=train_cfg.get("use_amp", True),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 0),
        grid_dims=data_cfg.get("grid_dims", 1),
        use_sobolev=train_cfg.get("use_sobolev", False),
        sobolev_lambda=train_cfg.get("sobolev_lambda", 0.5),
    )

    print("Starting training...")
    try:
        trainer.train(train_loader, val_loader, epochs=train_cfg["epochs"], start_epoch=start_epoch)
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    finally:
        tracker.close()
        print("Training finished. Checkpoints and traces archived.")


if __name__ == "__main__":
    main()
