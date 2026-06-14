# src/evaluation/evaluator.py
import torch
import time
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import json
from pathlib import Path

from ..memory import BeliefState
from ..training.data_loader import LoCoMoDataLoader
from ..training.train_3b import AdaptiveBelief3BTrainer
from ..evaluation.metrics import MetricsCalculator

class AdaptiveBeliefEvaluator:
    """Evaluator for AdaptiveBelief system."""

    def __init__(self, trainer: AdaptiveBelief3BTrainer):
        self.trainer = trainer
        self.metrics = MetricsCalculator()

    def evaluate(self, split: str = "test") -> Dict[str, Any]:
        """Run full evaluation on test set."""
        data_loader = LoCoMoDataLoader(
            self.trainer.config["data"]["train_file"].split('/')[0] + '/processed/',
            batch_size=1,
            max_sessions=self.trainer.config["training"]["max_sessions"]
        )
        loader = data_loader.get_dataloader(split, shuffle=False)

        results = []
        latencies = []

        for batch in tqdm(loader, desc=f"Evaluating {split}"):
            start_time = time.time()
            result = self.evaluate_batch(batch)
            latencies.append(time.time() - start_time)
            results.append(result)

        # Aggregate metrics
        aggregated = self.metrics.aggregate(results)
        aggregated["avg_latency"] = sum(latencies) / len(latencies)
        aggregated["p95_latency"] = sorted(latencies)[int(len(latencies) * 0.95)]

        return aggregated

    def evaluate_batch(self, batch: Dict) -> Dict:
        """Evaluate single batch."""
        session_texts = batch["session_texts"][0]
        question = batch["questions"][0]
        choices = batch["choices"][0]
        target_idx = batch["answer_indices"][0]

        # Process with AdaptiveBelief
        belief_state = BeliefState(self.trainer.config["belief_dim"], self.trainer.device)
        belief = belief_state.initialize()

        writes = 0
        trace = None

        for session in session_texts:
            belief, _ = self.trainer.builder(belief, session)
            belief = belief_state.update(belief)

            uncertainty = self.trainer.uncertainty(belief)
            if uncertainty.item() >= self.trainer.get_current_threshold():
                writes += 1

        if writes > 0:
            beliefs_stack = torch.stack(belief_state.get_belief_history())
            uncertainties_stack = torch.tensor([u.item() for u in uncertainties])
            trace, _ = self.trainer.retriever(beliefs_stack, uncertainties_stack,
                                              len(session_texts), 0, "")
            prompt = self.trainer.prompt_builder.build_prompt(trace, question, choices)
        else:
            prompt = self.trainer.prompt_builder.build_without_trace(question, choices)

        output = self.trainer.reasoner.generate(prompt)
        predicted_idx, is_insufficient = self.trainer.output_parser.parse(output, len(choices))

        correct = (predicted_idx == target_idx) if predicted_idx is not None else False

        return {
            "correct": correct,
            "write_rate": writes / len(session_texts) if session_texts else 0,
            "num_sessions": len(session_texts),
            "trace": trace
        }

    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)