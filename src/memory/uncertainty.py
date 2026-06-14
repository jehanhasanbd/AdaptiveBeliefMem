# src/memory/uncertainty.py (add dtype handling)
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class UncertaintyEstimator(nn.Module):
    """
    Uncertainty estimator g_φ: Monitors belief reliability.
    Outputs u_t ∈ [0,1] indicating likelihood of reasoning failure.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.belief_dim = config.get("belief_dim", 768)
        self.hidden_dim = config.get("hidden_dim", 256)

        # 2-layer MLP with dropout for regularization
        self.network = nn.Sequential(
            nn.Linear(self.belief_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Optional: attention over belief history for better calibration
        self.use_history = config.get("use_history", False)
        if self.use_history:
            self.history_proj = nn.Linear(self.belief_dim, self.hidden_dim)
            self.attention = nn.MultiheadAttention(self.hidden_dim, num_heads=4, batch_first=True)

    def forward(self, belief_state: torch.Tensor, history: torch.Tensor = None) -> torch.Tensor:
        """
        Estimate uncertainty from belief state.

        Args:
            belief_state: Current belief b_t [batch_size, belief_dim]
            history: Optional history of belief states [batch_size, seq_len, belief_dim]

        Returns:
            Uncertainty score u_t ∈ [0,1] [batch_size, 1]
        """
        # Ensure input is float32 for the network (sigmoid works better)
        if belief_state.dtype != torch.float32:
            belief_state = belief_state.float()

        if self.use_history and history is not None:
            if history.dtype != torch.float32:
                history = history.float()
            # Attend over history to improve calibration
            hist_proj = self.history_proj(history)
            current_proj = self.history_proj(belief_state.unsqueeze(1))
            attended, _ = self.attention(current_proj, hist_proj, hist_proj)
            combined = attended.squeeze(1)
        else:
            combined = belief_state

        return self.network(combined)

    def get_uncertainty_score(self, belief_state: torch.Tensor) -> float:
        """Get scalar uncertainty score for decision making."""
        with torch.no_grad():
            u = self.forward(belief_state)
            return u.item()