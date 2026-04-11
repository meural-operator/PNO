import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from ..utils.metrics import RelativeL2Loss


class PNOTrainer:
    """
    Production-grade trainer for P-Adic Neural Operators.

    Features: gradient clipping, mixed precision (AMP), early stopping,
    configurable logging, and proper checkpoint management.
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        experiment_manager=None,
        device="cuda",
        grad_clip_norm=1.0,
        use_amp=True,
        early_stopping_patience=0,
        grid_dims=1,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = RelativeL2Loss()
        self.experiment_manager = experiment_manager
        self.grad_clip_norm = grad_clip_norm
        self.grid_dims = grid_dims  # number of trailing coordinate dimensions

        # Mixed precision
        self.use_amp = use_amp and device == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None

        # Early stopping
        self.patience = early_stopping_patience
        self._patience_counter = 0
        self._best_val_loss = float("inf")

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training", leave=False)
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            gd = self.grid_dims
            v = x[..., :-gd]
            grid_c = x[..., -gd:]

            # Forward pass with optional AMP
            if self.use_amp:
                with autocast("cuda"):
                    out = self.model(v, grid_c)
                    loss = self.criterion(out, y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(v, grid_c)
                loss = self.criterion(out, y)
                loss.backward()
                if self.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val
            pbar.set_postfix(loss=f"{loss_val:.4f}")

            if self.experiment_manager and step % 10 == 0:
                global_step = (epoch - 1) * len(dataloader) + step
                self.experiment_manager.log_metrics({"Train/Step_RelL2": loss_val}, global_step)

        if self.scheduler:
            self.scheduler.step()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def eval(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0.0
        
        # Check if dataset has normalization parameters hooked
        has_norm = hasattr(dataloader.dataset, 'u_mean') and hasattr(dataloader.dataset, 'u_std')
        if has_norm:
            u_mean = dataloader.dataset.u_mean.to(self.device)
            u_std = dataloader.dataset.u_std.to(self.device)

        for x, y in tqdm(dataloader, desc=f"Epoch {epoch} Evaluation", leave=False):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            
            gd = getattr(self, "grid_dims", 1)
            v = x[..., :-gd]
            grid_c = x[..., -gd:]

            if self.use_amp:
                with autocast("cuda"):
                    out = self.model(v, grid_c)
            else:
                out = self.model(v, grid_c)
                
            # Inverse transform to physical domain for absolute L2 benchmarking
            if has_norm:
                out_phys = (out * u_std) + u_mean
                y_phys = (y * u_std) + u_mean
                loss = self.criterion(out_phys, y_phys)
            else:
                loss = self.criterion(out, y)

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def _check_early_stopping(self, val_loss):
        """Returns True if training should stop."""
        if self.patience <= 0:
            return False
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._patience_counter = 0
            return False
        self._patience_counter += 1
        if self._patience_counter >= self.patience:
            return True
        return False

    def train(self, train_loader, val_loader=None, epochs=10, start_epoch=1):
        for epoch in range(start_epoch, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)

            val_loss = None
            val_loss_str = ""
            if val_loader:
                val_loss = self.eval(val_loader, epoch)
                val_loss_str = f" | Val RelL2: {val_loss:.4f}"

            lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}/{epochs} | LR: {lr:.2e} | Train RelL2: {train_loss:.4f}{val_loss_str}")

            # Experiment callbacks
            if self.experiment_manager:
                logs = {"Train/Epoch_RelL2": train_loss, "Settings/LR": lr}
                if val_loss is not None:
                    logs["Val/Epoch_RelL2"] = val_loss
                self.experiment_manager.log_metrics(logs, epoch)
                self.experiment_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch, val_loss
                )

            # Early stopping check
            if val_loss is not None and self._check_early_stopping(val_loss):
                print(f"Early stopping triggered at epoch {epoch} (patience={self.patience})")
                break
