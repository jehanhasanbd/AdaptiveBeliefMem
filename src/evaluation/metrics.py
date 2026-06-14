# src/evaluation/metrics.py
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class MetricsCalculator:
    """Calculate evaluation metrics."""

    def __init__(self):
        pass

    def accuracy(self, predictions: List[bool]) -> float:
        """Calculate accuracy."""
        if not predictions:
            return 0.0
        return sum(predictions) / len(predictions)

    def write_rate(self, write_rates: List[float]) -> float:
        """Calculate average write rate."""
        if not write_rates:
            return 0.0
        return sum(write_rates) / len(write_rates)

    def ece(self, confidences: List[float], correctness: List[bool], n_bins: int = 10) -> float:
        """Expected Calibration Error."""
        if not confidences:
            return 0.0

        confidences = np.array(confidences)
        correctness = np.array(correctness)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(correctness[in_bin])
                confidence_in_bin = np.mean(confidences[in_bin])
                ece += prop_in_bin * np.abs(accuracy_in_bin - confidence_in_bin)

        return ece

    def compression_ratio(self, input_tokens: List[int], output_tokens: List[int]) -> float:
        """Calculate compression ratio."""
        if not output_tokens or sum(output_tokens) == 0:
            return 0.0
        return sum(input_tokens) / sum(output_tokens)

    def aggregate(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate metrics across evaluation results."""
        if not results:
            return {}

        correctness = [r["correct"] for r in results]
        write_rates = [r["write_rate"] for r in results]

        # Calculate per-type metrics
        metrics = {
            "accuracy": self.accuracy(correctness),
            "write_rate": self.write_rate(write_rates),
            "total_examples": len(results)
        }

        return metrics