# scripts/run_experiment.py
# !/usr/bin/env python
import argparse
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train_3b import AdaptiveBelief3BTrainer
from src.evaluation.evaluator import AdaptiveBeliefEvaluator
from src.evaluation.ablation import AblationRunner
from src.reasoner.frozen_llm_3b import QuantizedReasoner3B
from src.training.data_loader import LoCoMoDataLoader


def main():
    parser = argparse.ArgumentParser(description="Run AdaptiveBelief experiments")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "ablation"], default="train")
    parser.add_argument("--config", type=str, default="config/rtx3060_3b.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_sessions", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "train":
        print("Starting training...")
        trainer = AdaptiveBelief3BTrainer(args.config)
        trainer.config["training"]["max_sessions"] = args.max_sessions
        trainer.train(use_wandb=False)

    elif args.mode == "eval":
        print("Starting evaluation...")
        trainer = AdaptiveBelief3BTrainer(args.config)
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        evaluator = AdaptiveBeliefEvaluator(trainer)
        results = evaluator.evaluate("test")
        print(f"Results: {results}")

    elif args.mode == "ablation":
        print("Running ablation studies...")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Initialize components for ablation
        reasoner = QuantizedReasoner3B(config)
        data_loader = LoCoMoDataLoader(
            config["data"]["train_file"].split('/')[0] + '/processed/',
            batch_size=1,
            max_sessions=args.max_sessions
        )

        ablation_runner = AblationRunner(config)
        results = ablation_runner.run_all(reasoner, None, data_loader)
        ablation_runner.save_results(results, "outputs/results/ablation_results.json")
        print(f"Ablation results saved to outputs/results/ablation_results.json")


if __name__ == "__main__":
    main()