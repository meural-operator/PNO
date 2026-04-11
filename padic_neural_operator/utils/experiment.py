import os
import shutil
import glob
import torch
import yaml
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class ExperimentManager:
    """
    Manages TensorBoard logging, Checkpointing, and Codebase Mirroring for reproducibility.
    """
    def __init__(self, config, base_log_dir="runs"):
        self.config = config
        
        # Format run name natively with date
        exp_name = config.get("experiment_name", "PNO_Experiment")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_log_dir, f"{exp_name}_{timestamp}")
        
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.src_dir = os.path.join(self.run_dir, "src")
        
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.src_dir, exist_ok=True)
        
        # 1. Initialize TensorBoard Writer
        self.writer = SummaryWriter(log_dir=self.run_dir)
        
        # 2. Mirror configuration into run directory
        with open(os.path.join(self.run_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        # 3. Mirror Source Code to guarantee reproducibility
        self._mirror_src()
        
        self.best_val_loss = float('inf')

    def _mirror_src(self):
        """
        Copies padic_neural_operator package and current training script to the run folder.
        """
        # Exclude pycache and data
        ignore_func = shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '*.hdf5', '*.nc')
        
        # Mirror core library
        lib_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        dest_lib = os.path.join(self.src_dir, "padic_neural_operator")
        if os.path.exists(lib_src):
            shutil.copytree(lib_src, dest_lib, ignore=ignore_func)
            
        # Try finding scripts ending in .py in the root or examples 
        project_root = os.path.abspath(os.path.join(lib_src, '..'))
        for py_file in glob.glob(os.path.join(project_root, 'examples', '*.py')):
            shutil.copy(py_file, self.src_dir)

    def log_metrics(self, metrics: dict, step: int):
        """
        Write scalar metrics continuously to TensorBoard
        """
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, val_loss=None):
        """
        Saves 'latest_model.pth', conditional 'best_model.pth', and 20-epoch snapshots.
        """
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_loss': val_loss
        }
        
        # 1. Latest
        latest_path = os.path.join(self.ckpt_dir, "latest_model.pth")
        torch.save(state, latest_path)
        
        # 2. Best
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.ckpt_dir, "best_model.pth")
            torch.save(state, best_path)
            
        # 3. Every 20 epochs
        if epoch % 20 == 0:
            periodic_path = os.path.join(self.ckpt_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(state, periodic_path)

    def load_checkpoint(self, checkpoint_path, model, optimizer, scheduler=None, device='cuda'):
        """
        Loads the states identically from the specified path.
        Returns the epoch to resume from and the best val loss at that time.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
            
        print(f"Loading checkpoint gracefully from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        epoch = checkpoint.get('epoch', 0)
        val_loss = checkpoint.get('val_loss', float('inf'))
        self.best_val_loss = val_loss
        print(f"Successfully resumed at Epoch {epoch} with Val Loss {val_loss}")
        return epoch, val_loss

    def close(self):
        self.writer.close()
