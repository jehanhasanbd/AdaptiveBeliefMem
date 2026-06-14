# src/utils/checkpoint.py
import torch
from pathlib import Path
from typing import Dict, Any, Optional


class CheckpointManager:
    """Manage model checkpoints."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: Dict[str, Any], filename: str):
        """Save checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(state, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load(self, filename: str, device: str = "cuda") -> Optional[Dict[str, Any]]:
        """Load checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        if not checkpoint_path.exists():
            print(f"Checkpoint {checkpoint_path} not found")
            return None

        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint

    def get_latest(self, pattern: str = "checkpoint_*.pt") -> Optional[str]:
        """Get latest checkpoint filename."""
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)