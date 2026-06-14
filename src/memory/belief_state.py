# src/memory/belief_state.py
import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

@dataclass
class BeliefSnapshot:
    """Snapshot of belief state with metadata."""
    belief: torch.Tensor
    step: int
    timestamp: datetime
    topic: Optional[str] = None
    uncertainty: Optional[float] = None


class BeliefState:
    """
    Manages the implicit belief state with versioning and temporal anchoring.
    """

    def __init__(self, belief_dim: int, device: str = "cuda"):
        self.belief_dim = belief_dim
        self.device = device
        self.current_belief = None
        self.snapshots: List[BeliefSnapshot] = []
        self.step_counter = 0

    def initialize(self) -> torch.Tensor:
        """Initialize zero belief state."""
        self.current_belief = torch.zeros(1, self.belief_dim, device=self.device)
        self._take_snapshot(topic="initialization")
        return self.current_belief

    def update(self, new_belief: torch.Tensor, topic: Optional[str] = None) -> torch.Tensor:
        """Update current belief state and optionally take snapshot."""
        self.current_belief = new_belief
        self.step_counter += 1

        # Take snapshot at topic shifts
        if topic is not None:
            self._take_snapshot(topic=topic)

        return self.current_belief

    def _take_snapshot(self, topic: Optional[str] = None, uncertainty: Optional[float] = None):
        """Store belief snapshot for versioning."""
        snapshot = BeliefSnapshot(
            belief=self.current_belief.clone(),
            step=self.step_counter,
            timestamp=datetime.now(),
            topic=topic,
            uncertainty=uncertainty
        )
        self.snapshots.append(snapshot)

    def get_belief_history(self, max_length: Optional[int] = None) -> torch.Tensor:
        """Get history of belief states."""
        if not self.snapshots:
            return torch.zeros(1, self.belief_dim, device=self.device)

        history = torch.stack([s.belief for s in self.snapshots])
        if max_length and history.size(0) > max_length:
            history = history[-max_length:]
        return history

    def get_snapshot_at_step(self, step: int) -> Optional[BeliefSnapshot]:
        """Get belief snapshot at specific step."""
        for snapshot in self.snapshots:
            if snapshot.step == step:
                return snapshot
        return None

    def get_temporal_snapshots(self, window_size: int = 5) -> List[BeliefSnapshot]:
        """Get recent temporal snapshots."""
        return self.snapshots[-window_size:] if self.snapshots else []

    def clear(self):
        """Reset belief state."""
        self.current_belief = None
        self.snapshots = []
        self.step_counter = 0