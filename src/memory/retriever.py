# src/memory/retriever.py (fixed indexing)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime
import math

from ..utils.ste import StraightThroughEstimator


class BeliefRetriever(nn.Module):
    """
    Retriever: Externalizes belief state into structured decision trace.
    Uses Hard Top-K with Straight-Through Estimation for gradient routing.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.belief_dim = config.get("belief_dim", 768)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.base_k = config["top_k"]["base_k"]
        self.dynamic_k = config["top_k"]["dynamic"]

        # Projection layers for relevance scoring
        self.relevance_proj = nn.Linear(self.belief_dim, self.hidden_dim)
        self.score_proj = nn.Linear(self.hidden_dim, 1)

        # Decoder for generating structured traces
        self.trace_decoder = nn.Sequential(
            nn.Linear(self.belief_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
        )

        # Entity and relation prediction heads
        self.entity_head = nn.Linear(self.hidden_dim, 64)
        self.relation_head = nn.Linear(self.hidden_dim, 128)
        self.evidence_head = nn.Linear(self.hidden_dim, 256)

        # STE module
        self.ste = StraightThroughEstimator()

        self.step_counter = 0
        self.default_dtype = torch.float32

    def to(self, *args, **kwargs):
        """Override to to track dtype."""
        device = super().to(*args, **kwargs)
        for param in self.parameters():
            self.default_dtype = param.dtype
            break
        return device

    def compute_relevance_scores(self, belief_positions: torch.Tensor) -> torch.Tensor:
        """Compute relevance scores for belief positions."""
        if belief_positions.dtype != torch.float32:
            belief_positions = belief_positions.float()

        projected = F.relu(self.relevance_proj(belief_positions))
        scores = self.score_proj(projected).squeeze(-1)
        return torch.sigmoid(scores)

    def hard_top_k_selection(self, beliefs: torch.Tensor, uncertainty_scores: torch.Tensor,
                             num_sessions: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hard Top-K selection with Straight-Through Estimation.

        Args:
            beliefs: Stack of belief states [num_beliefs, belief_dim]
            uncertainty_scores: Uncertainty scores for each belief [num_beliefs]
            num_sessions: Total sessions for dynamic K calculation

        Returns:
            Selected beliefs [k, belief_dim], Selection mask [num_beliefs]
        """
        if uncertainty_scores.dtype != torch.float32:
            uncertainty_scores = uncertainty_scores.float()

        # Calculate dynamic K
        if self.dynamic_k:
            k = max(1, int(math.sqrt(num_sessions)) + self.base_k)
        else:
            k = self.base_k

        k = min(k, beliefs.size(0))

        # Weight scores by uncertainty
        weighted_scores = uncertainty_scores * self.compute_relevance_scores(beliefs)

        # Get top-k indices
        _, top_indices = torch.topk(weighted_scores, k)

        # Create hard selection mask (1D)
        mask = torch.zeros(beliefs.size(0), device=beliefs.device)
        mask[top_indices] = 1.0

        # Apply STE: forward uses hard mask, backward passes gradients
        # Expand mask to match beliefs dimensions for broadcasting
        mask_expanded = mask.unsqueeze(-1)  # [num_beliefs, 1]
        selected_beliefs = self.ste.apply(beliefs, mask_expanded)

        return selected_beliefs, mask

    def generate_trace(self, selected_beliefs: torch.Tensor, step: int,
                       uncertainty: float, trigger_reason: str) -> Dict[str, Any]:
        """Generate structured decision trace from selected beliefs."""
        if selected_beliefs.dim() == 2:
            aggregated = selected_beliefs.mean(dim=0)
        else:
            aggregated = selected_beliefs

        if aggregated.dtype != torch.float32:
            aggregated = aggregated.float()

        trace_features = self.trace_decoder(aggregated)

        # Generate entities
        entity_logits = self.entity_head(trace_features)
        entity_probs = torch.softmax(entity_logits, dim=-1)
        _, entity_indices = torch.topk(entity_probs, min(5, entity_probs.size(-1)))

        # Generate relations
        relation_logits = self.relation_head(trace_features)
        relation_probs = torch.sigmoid(relation_logits)

        # Generate evidence
        evidence_features = self.evidence_head(trace_features)

        # Construct structured trace
        trace = {
            "trace_id": f"step_{step}",
            "timestamp": datetime.now().isoformat(),
            "entities": [f"entity_{i.item()}" for i in entity_indices[:3]],
            "relations": [],
            "relevant_evidence": [],
            "uncertainty_score": float(uncertainty),
            "trigger_reason": trigger_reason
        }

        # Add relations
        top_relations = torch.topk(relation_probs, min(3, relation_probs.size(-1)))
        for idx in top_relations.indices:
            trace["relations"].append({
                "subject": trace["entities"][0] if trace["entities"] else "unknown",
                "predicate": f"relation_{idx.item()}",
                "object": trace["entities"][1] if len(trace["entities"]) > 1 else "unknown"
            })

        # Add evidence
        evidence_vals = torch.sigmoid(evidence_features)[:3]
        for i, val in enumerate(evidence_vals):
            trace["relevant_evidence"].append(f"Evidence {i + 1}: relevance {val.item():.2f}")

        return trace

    def forward(self, belief_states: torch.Tensor, uncertainty_scores: torch.Tensor,
                num_sessions: int, step: int, trigger_reason: str = "") -> Tuple[Dict, torch.Tensor]:
        """
        Retrieve decision trace from belief states.

        Returns:
            Structured trace dict, Selection mask
        """
        selected_beliefs, mask = self.hard_top_k_selection(
            belief_states, uncertainty_scores, num_sessions
        )

        # Filter out zeroed beliefs using the mask
        non_zero_indices = mask > 0
        if non_zero_indices.any():
            valid_beliefs = belief_states[non_zero_indices]
            avg_uncertainty = uncertainty_scores[non_zero_indices].mean().item()
            trace = self.generate_trace(valid_beliefs, step, avg_uncertainty, trigger_reason)
        else:
            trace = {
                "trace_id": f"step_{step}",
                "timestamp": datetime.now().isoformat(),
                "entities": [],
                "relations": [],
                "relevant_evidence": ["No relevant evidence selected"],
                "uncertainty_score": 0.0,
                "trigger_reason": "No selection"
            }

        return trace, mask