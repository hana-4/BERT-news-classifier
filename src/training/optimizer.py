"""
Optimizer utilities for BERT training
"""
import numpy as np
import torch


class ScheduledOptim:
    """Learning rate scheduler with warmup for BERT training"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, n_warmup_steps: int):
        """
        Initialize scheduled optimizer.
        
        Args:
            optimizer: PyTorch optimizer
            d_model: Model dimension
            n_warmup_steps: Number of warmup steps
        """
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        """Step with the inner optimizer and update learning rate"""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """Zero out the gradients"""
        self._optimizer.zero_grad()

    def _get_lr_scale(self) -> float:
        """Calculate learning rate scaling factor"""
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps
        ])

    def _update_learning_rate(self):
        """Update learning rate based on current step"""
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Get current learning rate"""
        return self._optimizer.param_groups[0]['lr']


class AdamWScheduled:
    """AdamW optimizer with warmup and cosine decay"""
    
    def __init__(self, model: torch.nn.Module, lr: float = 1e-4, 
                 weight_decay: float = 0.01, betas: tuple = (0.9, 0.999),
                 warmup_steps: int = 1000, total_steps: int = 10000):
        """
        Initialize AdamW with scheduling.
        
        Args:
            model: Model to optimize
            lr: Peak learning rate
            weight_decay: Weight decay coefficient
            betas: Adam beta parameters
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
        """
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_steps,
            eta_min=lr * 0.1
        )
        
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.base_lr = lr

    def step(self):
        """Step optimizer and scheduler"""
        self.current_step += 1
        
        # Warmup
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.scheduler.step()
        
        self.optimizer.step()

    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()

    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
