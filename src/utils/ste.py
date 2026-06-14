# src/utils/ste.py (fixed)
import torch
import torch.nn as nn


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-Through Estimator for Hard Top-K selection.
    Forward: Hard selection (discrete)
    Backward: Identity (gradients flow through as if no selection)
    """

    @staticmethod
    def forward(ctx, inputs, mask):
        """
        Forward pass: Apply hard mask (zero out unselected positions).

        Args:
            inputs: Input tensor [..., dim]
            mask: Binary mask [..., 1]

        Returns:
            Masked inputs
        """
        ctx.save_for_backward(mask)
        # Ensure mask has same shape as inputs for broadcasting
        if mask.dim() < inputs.dim():
            mask = mask.unsqueeze(-1)
        return inputs * mask

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Pass gradients through as identity.

        Args:
            grad_output: Gradient from downstream

        Returns:
            Gradient w.r.t inputs (same as grad_output), None for mask
        """
        mask, = ctx.saved_tensors
        # STE: pass gradients through as if no selection
        # Only propagate through selected positions
        if mask.dim() < grad_output.dim():
            mask = mask.unsqueeze(-1)
        grad_input = grad_output * mask
        return grad_input, None


class HardTopKSTE(nn.Module):
    """Hard Top-K selection with Straight-Through Estimation."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, scores, values):
        """
        Select top-k based on scores using STE.

        Args:
            scores: Relevance scores [..., n]
            values: Values to select from [..., n, d]

        Returns:
            Selected values, Selection mask
        """
        # Get top-k indices
        _, top_indices = torch.topk(scores, self.k, dim=-1)

        # Create mask
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, top_indices, 1.0)

        # Apply STE
        selected = StraightThroughEstimator.apply(values, mask.unsqueeze(-1))

        return selected, mask