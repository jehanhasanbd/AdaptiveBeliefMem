# scripts/evaluate_all.py
# !/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train_3b import AdaptiveBelief3BTrainer
from src.evaluation.evaluator import AdaptiveBeliefEvaluator
from src.evaluation.ablation import AblationRunner
from src.reasoner.frozen_llm_3b import QuantizedReasoner3B
from src.training.data_loader import LoCoMoDataLoader
import yaml
import json


def main():
    config_path = "config/rtx3060_3b.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize reasoner
    reasoner = QuantizedReasoner3B(config)
    data_loader = LoCoMoDataLoader(
        config["data"]["train_file"].split('/')[0] + '/processed/',
        batch_size=1,
        max_sessions=config["training"]["max_sessions"]
    )

    # Run ablations
    print("=" * 60)
    print("RUNNING ABLATION STUDIES")
    print("=" * 60)

    ablation_runner = AblationRunner(config)
    ablation_results = ablation_runner.run_all(reasoner, None, data_loader)

    # Save results
    output_path = Path("outputs/results/all_evaluations.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(ablation_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, results in ablation_results.items():
        print(f"{name:30}: {results.get('accuracy', 0):.3f}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()