"""
Training utilities for BERT models
"""
import torch
import torch.nn as nn
from torch.optim import Adam
import tqdm
from typing import Optional, Dict, Any
import logging
from .optimizer import ScheduledOptim


logger = logging.getLogger(__name__)


class BERTTrainer:
    """Trainer for BERT pre-training with MLM and NSP tasks"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: Optional[torch.utils.data.DataLoader] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.999),
        warmup_steps: int = 10000,
        log_freq: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize BERT trainer.
        
        Args:
            model: BERT model to train
            train_dataloader: Training data loader
            test_dataloader: Test data loader
            lr: Learning rate
            weight_decay: Weight decay
            betas: Adam beta parameters
            warmup_steps: Warmup steps for learning rate
            log_freq: Logging frequency
            device: Device to train on
        """
        self.device = device
        self.model = model.to(device)
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.log_freq = log_freq

        # Setup optimizer
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.bert.d_model, n_warmup_steps=warmup_steps
        )

        # Loss function
        self.criterion = nn.NLLLoss(ignore_index=0)
        
        logger.info(f"Total Parameters: {sum(p.nelement() for p in self.model.parameters())}")

    def train(self, epoch: int) -> float:
        """Train for one epoch"""
        return self.iteration(epoch, self.train_data, train=True)

    def test(self, epoch: int) -> float:
        """Test for one epoch"""
        return self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch: int, data_loader: torch.utils.data.DataLoader, train: bool = True) -> float:
        """
        Run one epoch of training or testing.
        
        Args:
            epoch: Current epoch number
            data_loader: Data loader
            train: Whether this is training or testing
            
        Returns:
            Average loss for the epoch
        """
        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        mode = "train" if train else "test"
        
        # Set model mode
        if train:
            self.model.train()
        else:
            self.model.eval()

        # Progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc=f"EP_{mode}:{epoch}",
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )

        for i, data in data_iter:
            # Move data to device
            data = {key: value.to(self.device) for key, value in data.items()}

            # Forward pass
            with torch.set_grad_enabled(train):
                next_sent_output, mask_lm_output = self.model(data["bert_input"], data["segment_label"])

                # Calculate losses
                next_loss = self.criterion(next_sent_output, data["is_next"])
                mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
                loss = next_loss + mask_loss

            # Backward pass (only in training)
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # Calculate accuracy
            correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            # Logging
            if i % self.log_freq == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "avg_acc": total_correct / total_element * 100,
                    "loss": loss.item(),
                    "lr": self.optim_schedule.get_lr()
                }
                data_iter.set_postfix(post_fix)

        epoch_loss = avg_loss / len(data_loader)
        epoch_acc = total_correct * 100.0 / total_element
        
        logger.info(f"EP{epoch}, {mode}: avg_loss={epoch_loss:.4f}, total_acc={epoch_acc:.2f}%")
        
        return epoch_loss


class ClassificationTrainer:
    """Trainer for BERT classification tasks"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize classification trainer.
        
        Args:
            model: Classification model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            lr: Learning rate
            weight_decay: Weight decay
            device: Device to train on
        """
        self.device = device
        self.model = model.to(device)
        self.train_data = train_dataloader
        self.val_data = val_dataloader

        # Setup optimizer and loss
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        train_iter = tqdm.tqdm(
            self.train_data,
            desc=f"Training Epoch {epoch}",
            leave=False
        )

        for batch in train_iter:
            # Move to device
            inputs = batch["bert_input"].to(self.device)
            segments = batch["segment_label"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(inputs, segments)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            predictions = logits.argmax(dim=-1)
            correct = (predictions == labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)

            # Update progress bar
            train_iter.set_postfix({
                'loss': loss.item(),
                'acc': correct / labels.size(0)
            })

        return {
            'loss': total_loss / len(self.train_data),
            'accuracy': total_correct / total_samples
        }

    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if not self.val_data:
            return {}
            
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm.tqdm(self.val_data, desc="Validating", leave=False):
                inputs = batch["bert_input"].to(self.device)
                segments = batch["segment_label"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(inputs, segments)
                loss = self.criterion(logits, labels)

                predictions = logits.argmax(dim=-1)
                correct = (predictions == labels).sum().item()

                total_loss += loss.item()
                total_correct += correct
                total_samples += labels.size(0)

        return {
            'loss': total_loss / len(self.val_data),
            'accuracy': total_correct / total_samples
        }
