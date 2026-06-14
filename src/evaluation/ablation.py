# src/evaluation/ablation.py
import json
from pathlib import Path
from typing import Dict, Any, List
import time

from ..training.data_loader import LoCoMoDataLoader
from ..reasoner.prompt_builder import PromptBuilder
from ..reasoner.output_parser import OutputParser


class AblationRunner:
    """Run ablation studies for different configurations."""

    def __init__(self, config: dict):
        self.config = config
        self.prompt_builder = PromptBuilder()
        self.output_parser = OutputParser()

    def run_baseline_full_context(self, reasoner, data_loader, split: str = "test") -> Dict:
        """Baseline: Full context (all sessions) to LLM."""
        loader = data_loader.get_dataloader(split, shuffle=False)

        correct = 0
        total = 0

        for batch in loader:
            session_texts = batch["session_texts"][0]
            question = batch["questions"][0]
            choices = batch["choices"][0]
            target_idx = batch["answer_indices"][0]

            # Build full context prompt
            full_context = "\n".join(session_texts)
            prompt = self.prompt_builder.build_without_trace(question, choices)
            prompt = f"## CONVERSATION HISTORY:\n{full_context}\n\n{prompt}"

            output = reasoner.generate(prompt)
            predicted_idx, _ = self.output_parser.parse(output, len(choices))

            if predicted_idx is not None and predicted_idx == target_idx:
                correct += 1
            total += 1

        return {"accuracy": correct / total if total > 0 else 0.0}

    def run_baseline_no_memory(self, reasoner, data_loader, split: str = "test") -> Dict:
        """Baseline: No memory (current session only)."""
        loader = data_loader.get_dataloader(split, shuffle=False)

        correct = 0
        total = 0

        for batch in loader:
            # Only use last session
            session_texts = batch["session_texts"][0]
            last_session = session_texts[-1] if session_texts else ""
            question = batch["questions"][0]
            choices = batch["choices"][0]
            target_idx = batch["answer_indices"][0]

            prompt = f"## CURRENT SESSION:\n{last_session}\n\n{self.prompt_builder.build_without_trace(question, choices)}"

            output = reasoner.generate(prompt)
            predicted_idx, _ = self.output_parser.parse(output, len(choices))

            if predicted_idx is not None and predicted_idx == target_idx:
                correct += 1
            total += 1

        return {"accuracy": correct / total if total > 0 else 0.0}

    def run_baseline_fixed_window(self, reasoner, data_loader, window_size: int = 5, split: str = "test") -> Dict:
        """Baseline: Fixed window (last N sessions)."""
        loader = data_loader.get_dataloader(split, shuffle=False)

        correct = 0
        total = 0

        for batch in loader:
            session_texts = batch["session_texts"][0]
            window = session_texts[-window_size:] if len(session_texts) > window_size else session_texts
            context = "\n".join(window)
            question = batch["questions"][0]
            choices = batch["choices"][0]
            target_idx = batch["answer_indices"][0]

            prompt = f"## LAST {len(window)} SESSIONS:\n{context}\n\n{self.prompt_builder.build_without_trace(question, choices)}"

            output = reasoner.generate(prompt)
            predicted_idx, _ = self.output_parser.parse(output, len(choices))

            if predicted_idx is not None and predicted_idx == target_idx:
                correct += 1
            total += 1

        return {"accuracy": correct / total if total > 0 else 0.0}

    def run_ablation_no_gating(self, adaptive_system, data_loader, split: str = "test") -> Dict:
        """Ablation: No uncertainty gating (write every step)."""
        # Similar to baseline but using adaptive components without gating
        return {"accuracy": 0.0}  # Placeholder

    def run_ablation_no_hard_topk(self, adaptive_system, data_loader, split: str = "test") -> Dict:
        """Ablation: No Hard Top-K (soft attention instead)."""
        return {"accuracy": 0.0}  # Placeholder

    def run_ablation_unfrozen_reasoner(self, adaptive_system, data_loader, split: str = "test") -> Dict:
        """Ablation: Reasoner unfrozen."""
        return {"accuracy": 0.0}  # Placeholder

    def run_all(self, reasoner, adaptive_system, data_loader) -> Dict[str, Dict]:
        """Run all baseline and ablation configurations."""
        results = {}

        print("Running baseline: Full Context...")
        results["baseline_full_context"] = self.run_baseline_full_context(reasoner, data_loader)

        print("Running baseline: No Memory...")
        results["baseline_no_memory"] = self.run_baseline_no_memory(reasoner, data_loader)

        print("Running baseline: Fixed Window...")
        results["baseline_fixed_window"] = self.run_baseline_fixed_window(reasoner, data_loader)

        print("Running ablation: No Gating...")
        results["ablation_no_gating"] = self.run_ablation_no_gating(adaptive_system, data_loader)

        print("Running ablation: No Hard Top-K...")
        results["ablation_no_hard_topk"] = self.run_ablation_no_hard_topk(adaptive_system, data_loader)

        print("Running ablation: Unfrozen Reasoner...")
        results["ablation_unfrozen_reasoner"] = self.run_ablation_unfrozen_reasoner(adaptive_system, data_loader)

        return results

    def save_results(self, results: Dict[str, Dict], output_path: str):
        """Save ablation results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)