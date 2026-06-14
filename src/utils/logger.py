# src/utils/logger.py
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Logger:
    """Logging utility for training metrics."""

    def __init__(self, log_dir: str, use_wandb: bool = False, project_name: str = "adaptivebelief"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.project_name = project_name
        self.metrics_history = []

        if self.use_wandb:
            wandb.init(project=project_name)

    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics at a step."""
        metrics["step"] = step
        metrics["timestamp"] = datetime.now().isoformat()
        self.metrics_history.append(metrics)

        if self.use_wandb:
            wandb.log(metrics, step=step)

    def save(self, filename: str = "metrics.json"):
        """Save metrics history to file."""
        output_path = self.log_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def finish(self):
        """Finish logging."""
        if self.use_wandb:
            wandb.finish()