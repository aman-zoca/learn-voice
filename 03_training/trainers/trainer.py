"""
TTS Trainer Module
==================
Training loops and utilities for TTS models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
import time
import json
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainerConfig:
    """Configuration for training."""
    # Training
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    warmup_steps: int = 4000

    # Optimizer
    optimizer: str = 'adam'  # 'adam', 'adamw', 'sgd'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9

    # Scheduler
    scheduler: str = 'noam'  # 'noam', 'cosine', 'step', 'none'
    scheduler_step_size: int = 10000
    scheduler_gamma: float = 0.5

    # Mixed precision
    use_amp: bool = True

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every: int = 1000
    keep_last_n: int = 5

    # Logging
    log_every: int = 100
    eval_every: int = 1000
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = 'tts-training'

    # Device
    device: str = 'auto'


class TTSTrainer:
    """
    Trainer for TTS models.

    Handles:
    - Training loop
    - Validation
    - Checkpointing
    - Logging
    - Learning rate scheduling
    """

    def __init__(
        self,
        config: TrainerConfig,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None
    ):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Set device
        self.device = self._get_device()
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Initialize scaler for mixed precision
        self.scaler = GradScaler() if config.use_amp else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Setup logging
        self._setup_logging()

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(self.config.device)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        params = self.model.parameters()

        if self.config.optimizer == 'adam':
            return torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_eps,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_eps,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler."""
        if self.config.scheduler == 'noam':
            return NoamScheduler(
                self.optimizer,
                d_model=512,  # Default, should be overridden
                warmup_steps=self.config.warmup_steps
            )
        elif self.config.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma
            )
        elif self.config.scheduler == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def _setup_logging(self):
        """Setup logging backends."""
        self.writer = None
        self.wandb_run = None

        if self.config.use_tensorboard and TENSORBOARD_AVAILABLE:
            log_dir = self.checkpoint_dir / 'logs'
            self.writer = SummaryWriter(log_dir)

        if self.config.use_wandb and WANDB_AVAILABLE:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                config=self.config.__dict__
            )

    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch

            # Train epoch
            train_loss = self.train_epoch()

            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best')

            # Save checkpoint
            self.save_checkpoint(f'epoch_{epoch}')

            print(f"Epoch {epoch + 1}/{self.config.epochs} - "
                  f"Train Loss: {train_loss:.4f}")

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch in pbar:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss:.4f}'})

            # Log
            if self.global_step % self.config.log_every == 0:
                self._log_metrics({'train/loss': loss, 'train/lr': self._get_lr()})

            # Save checkpoint
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint(f'step_{self.global_step}')

            # Validate
            if self.global_step % self.config.eval_every == 0 and self.val_loader:
                val_loss = self.validate()
                self._log_metrics({'val/loss': val_loss})
                self.model.train()

        return total_loss / num_batches

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass with mixed precision
        if self.config.use_amp and self.scaler is not None:
            with autocast():
                outputs = self.model(
                    batch['text'],
                    batch['text_lengths'],
                    batch['mel']
                )
                loss = self._compute_loss(outputs, batch)

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training
            outputs = self.model(
                batch['text'],
                batch['text_lengths'],
                batch['mel']
            )
            loss = self._compute_loss(outputs, batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )

            # Optimizer step
            self.optimizer.step()

        # Scheduler step
        if self.scheduler is not None:
            if isinstance(self.scheduler, NoamScheduler):
                self.scheduler.step()
            # Other schedulers step per epoch

        return loss.item()

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute training loss."""
        if self.criterion is not None:
            return self.criterion(outputs, batch)

        # Default Tacotron2 loss
        mel_target = batch['mel']
        mel_lengths = batch['mel_lengths']

        mel_out = outputs['mel_outputs']
        mel_out_postnet = outputs['mel_outputs_postnet']
        gate_out = outputs['gate_outputs']

        # Mel loss
        mel_loss = nn.MSELoss()(mel_out, mel_target)
        mel_postnet_loss = nn.MSELoss()(mel_out_postnet, mel_target)

        # Gate loss
        gate_target = self._create_gate_target(mel_lengths, mel_target.size(2))
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        return mel_loss + mel_postnet_loss + gate_loss

    def _create_gate_target(
        self,
        mel_lengths: torch.Tensor,
        max_len: int
    ) -> torch.Tensor:
        """Create gate target (1 at end of sequence)."""
        batch_size = mel_lengths.size(0)
        gate_target = torch.zeros(batch_size, max_len, device=mel_lengths.device)

        for i, length in enumerate(mel_lengths):
            gate_target[i, length - 1:] = 1.0

        return gate_target

    @torch.no_grad()
    def validate(self) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = self.model(
                batch['text'],
                batch['text_lengths'],
                batch['mel']
            )
            loss = self._compute_loss(outputs, batch)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f'{name}.pt'

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def _cleanup_checkpoints(self):
        """Keep only the last N checkpoints."""
        checkpoints = sorted(
            self.checkpoint_dir.glob('step_*.pt'),
            key=lambda x: int(x.stem.split('_')[1])
        )

        # Keep best and last N
        for checkpoint in checkpoints[:-self.config.keep_last_n]:
            checkpoint.unlink()

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to backends."""
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.global_step)

        if self.wandb_run is not None:
            wandb.log(metrics, step=self.global_step)

    def _get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class NoamScheduler:
    """
    Noam learning rate scheduler from "Attention Is All You Need".

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int = 512,
        warmup_steps: int = 4000,
        factor: float = 1.0
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0

    def step(self):
        """Update learning rate."""
        self._step += 1
        lr = self._get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self) -> float:
        """Calculate learning rate."""
        step = max(self._step, 1)
        return self.factor * (
            self.d_model ** (-0.5) *
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )

    def state_dict(self) -> Dict[str, Any]:
        return {'step': self._step}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._step = state_dict['step']


class Tacotron2Loss(nn.Module):
    """Loss function for Tacotron2."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        mel_target = batch['mel']
        mel_lengths = batch['mel_lengths']

        mel_out = outputs['mel_outputs']
        mel_out_postnet = outputs['mel_outputs_postnet']
        gate_out = outputs['gate_outputs']

        # Truncate to match lengths
        max_len = min(mel_out.size(2), mel_target.size(2))
        mel_out = mel_out[:, :, :max_len]
        mel_out_postnet = mel_out_postnet[:, :, :max_len]
        mel_target = mel_target[:, :, :max_len]
        gate_out = gate_out[:, :max_len]

        # Mel loss
        mel_loss = self.mse(mel_out, mel_target)
        mel_postnet_loss = self.mse(mel_out_postnet, mel_target)

        # Gate loss
        batch_size = mel_lengths.size(0)
        gate_target = torch.zeros(batch_size, max_len, device=mel_lengths.device)
        for i, length in enumerate(mel_lengths):
            if length <= max_len:
                gate_target[i, length - 1:] = 1.0

        gate_loss = self.bce(gate_out, gate_target)

        return mel_loss + mel_postnet_loss + gate_loss


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description='TTS Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load config
    # ... implementation details

    print("Training script - use with config file")


if __name__ == '__main__':
    main()
