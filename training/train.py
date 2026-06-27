"""Training engine for CRNN OCR model."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from model.crnn import CRNN
from data.dataset import ocr_collate_fn


class Trainer:
    """CRNN model trainer with CTC loss."""

    def __init__(self, charset, settings):
        self.charset = charset
        self.settings = settings
        self.device = torch.device(settings.device)

        # Create model
        self.model = CRNN(
            num_classes=charset.num_classes,
            img_height=settings.img_height,
            img_channels=settings.img_channels,
            rnn_hidden=settings.rnn_hidden_size,
            rnn_layers=settings.rnn_num_layers,
        ).to(self.device)

        # CTC Loss
        self.criterion = nn.CTCLoss(blank=charset.blank_idx, zero_infinity=True)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )

        # LR Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=settings.lr_scheduler_patience,
            factor=settings.lr_scheduler_factor,
        )

        # Ensure checkpoint dir exists
        self.settings.model_dir.mkdir(parents=True, exist_ok=True)

        # GradScaler for AMP
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _train_one_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch, return average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="  Train Batch", leave=False)
        for images, labels, label_lengths, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            label_lengths = label_lengths.to(self.device)

            # Forward with AMP autocast
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                log_probs = self.model(images)  # (T, B, C)
                T = log_probs.shape[0]
                B = log_probs.shape[1]
                input_lengths = torch.full((B,), T, dtype=torch.int32, device=self.device)
                
                # Force CTC Loss computation in FP32 to prevent underflow/NaN issues
                loss = self.criterion(log_probs.float(), labels, input_lengths, label_lengths)

            # Backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Unscale for gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_val = loss.item()
            total_loss += loss_val
            num_batches += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}")

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> float:
        """Validate and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="  Val Batch", leave=False)
        for images, labels, label_lengths, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            label_lengths = label_lengths.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                log_probs = self.model(images)
                T = log_probs.shape[0]
                B = log_probs.shape[1]
                input_lengths = torch.full((B,), T, dtype=torch.int32, device=self.device)
                loss = self.criterion(log_probs.float(), labels, input_lengths, label_lengths)
            
            loss_val = loss.item()
            total_loss += loss_val
            num_batches += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}")

        return total_loss / max(num_batches, 1)

    def train(self, train_dataset, val_dataset=None) -> dict:
        """Run full training loop.

        Args:
            train_dataset: Training OCRDataset
            val_dataset: Validation OCRDataset (optional)

        Returns:
            History dict with 'train_loss', 'val_loss' lists
        """
        # Automatically use pin_memory on CUDA to speed up transfer to GPU
        pin_mem = (self.device.type == "cuda")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.settings.batch_size,
            shuffle=True,
            collate_fn=ocr_collate_fn,
            num_workers=self.settings.num_workers,
            pin_memory=pin_mem,
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.settings.batch_size,
                shuffle=False,
                collate_fn=ocr_collate_fn,
                num_workers=self.settings.num_workers,
                pin_memory=pin_mem,
            )

        history = {"train_loss": [], "val_loss": []}
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.settings.num_epochs):
            # Train
            train_loss = self._train_one_epoch(train_loader)
            history["train_loss"].append(train_loss)

            # Validate
            val_loss = None
            if val_loader:
                val_loss = self._validate(val_loader)
                history["val_loss"].append(val_loss)
                self.scheduler.step(val_loss)
                monitor_loss = val_loss
            else:
                self.scheduler.step(train_loss)
                monitor_loss = train_loss

            # Print progress
            msg = f"Epoch {epoch + 1}/{self.settings.num_epochs} - train_loss: {train_loss:.4f}"
            if val_loss is not None:
                msg += f" - val_loss: {val_loss:.4f}"
            lr = self.optimizer.param_groups[0]["lr"]
            msg += f" - lr: {lr:.6f}"
            print(msg)

            # Save best model
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                patience_counter = 0
                self._save_checkpoint("best_model.pt", epoch, best_loss)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.settings.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return history

    def _save_checkpoint(self, filename: str, epoch: int, loss: float):
        """Save model checkpoint."""
        path = self.settings.model_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "charset_size": self.charset.num_classes,
            "vocab": self.charset.vocab_list,
        }, path)
        print(f"Saved checkpoint: {path}")
