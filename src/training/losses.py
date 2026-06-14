# src/training/losses.py (add type safety)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AdaptiveBeliefLoss(nn.Module):
    """
    Composite loss for AdaptiveBelief training:
    L = L_task + λ1*L_uncertainty + λ2*L_sparsity + λ3*L_compression
    """

    def __init__(self, config: dict):
        super().__init__()
        # Ensure loss weights are floats
        self.lambda_task = float(config["loss_weights"]["task"])
        self.lambda_uncertainty = float(config["loss_weights"]["uncertainty"])
        self.lambda_sparsity = float(config["loss_weights"]["sparsity"])
        self.lambda_compression = float(config["loss_weights"]["compression"])

        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def task_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss for answer selection."""
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if targets.dim() > 0:
            targets = targets.squeeze()
        # Ensure targets are long type
        targets = targets.long()
        return self.ce_loss(logits, targets)

    def uncertainty_loss(self, uncertainty_pred: torch.Tensor, correctness: torch.Tensor) -> torch.Tensor:
        """
        Calibration loss - train uncertainty to predict reasoning failure.
        correctness: 1 if answer correct, 0 if incorrect
        """
        # Ensure tensors are float and have correct shape
        uncertainty_pred = uncertainty_pred.float().squeeze()
        correctness = correctness.float().squeeze()
        uncertainty_target = 1.0 - correctness
        return self.bce_loss(uncertainty_pred, uncertainty_target)

    def sparsity_loss(self, write_mask: torch.Tensor, total_steps: int) -> torch.Tensor:
        """Penalize excessive explicit writes."""
        write_rate = write_mask.sum() / float(total_steps)
        return write_rate ** 2  # Quadratic penalty

    def compression_loss(self, belief_state: torch.Tensor, trace_embedding: torch.Tensor) -> torch.Tensor:
        """Ensure belief state retains enough information."""
        return self.mse_loss(belief_state, trace_embedding)

    def forward(self, task_logits: torch.Tensor, task_targets: torch.Tensor,
                uncertainty_pred: torch.Tensor, correctness: torch.Tensor,
                write_mask: torch.Tensor, total_steps: int,
                belief_state: torch.Tensor, trace_embedding: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute composite loss.

        Returns:
            Total loss, Dictionary of individual losses
        """
        l_task = self.task_loss(task_logits, task_targets)
        l_uncertainty = self.uncertainty_loss(uncertainty_pred, correctness)
        l_sparsity = self.sparsity_loss(write_mask, total_steps)
        l_compression = self.compression_loss(belief_state, trace_embedding)

        total_loss = (self.lambda_task * l_task +
                      self.lambda_uncertainty * l_uncertainty +
                      self.lambda_sparsity * l_sparsity +
                      self.lambda_compression * l_compression)

        loss_dict = {
            "loss/total": total_loss.item(),
            "loss/task": l_task.item(),
            "loss/uncertainty": l_uncertainty.item(),
            "loss/sparsity": l_sparsity.item(),
            "loss/compression": l_compression.item()
        }

        return total_loss, loss_dict