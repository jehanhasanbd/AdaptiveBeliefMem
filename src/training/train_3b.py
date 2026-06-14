# src/training/train_3b.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import wandb
import gc
import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.memory.builder_3b import QuantizedBuilder3B
from src.memory.uncertainty import UncertaintyEstimator
from src.memory.retriever import BeliefRetriever
from src.memory.belief_state import BeliefState
from src.reasoner.frozen_llm_3b import QuantizedReasoner3B
from src.reasoner.prompt_builder import PromptBuilder
from src.reasoner.output_parser import OutputParser
from src.training.losses import AdaptiveBeliefLoss
from src.training.optimizer import create_optimizer, get_phase_lr
from src.training.data_loader import LoCoMoDataLoader


class AdaptiveBelief3BTrainer:
    """Main trainer for AdaptiveBelief system."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize components
        self.builder = QuantizedBuilder3B(self.config).to(self.device)
        self.uncertainty = UncertaintyEstimator(self.config).to(self.device)
        self.retriever = BeliefRetriever(self.config).to(self.device)
        self.reasoner = QuantizedReasoner3B(self.config)
        self.prompt_builder = PromptBuilder()
        self.output_parser = OutputParser()

        # Loss
        self.loss_fn = AdaptiveBeliefLoss(self.config)

        # Optimizer (only builder, uncertainty, retriever)
        trainable_params = (list(self.builder.get_trainable_parameters()) +
                            list(self.uncertainty.parameters()) +
                            list(self.retriever.parameters()))
        self.optimizer, self.scheduler = create_optimizer(trainable_params, self.config)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_accuracy = 0.0

        # Threshold annealing
        self.threshold_initial = self.config["threshold"]["initial"]
        self.threshold_final = self.config["threshold"]["final"]
        self.threshold_anneal_steps = self.config["threshold"]["anneal_steps"]

        # Data loader
        self.data_loader = LoCoMoDataLoader(
            self.config["data"]["train_file"].split('/')[0] + '/processed/',
            batch_size=self.config["training"]["batch_size"],
            max_sessions=self.config["training"]["max_sessions"]
        )

        # In train_3b.py, add type conversion in __init__ after loading config:

        # Ensure numeric values are proper types
        self.config["training"]["learning_rate"] = float(self.config["training"]["learning_rate"])
        self.config["training"]["weight_decay"] = float(self.config["training"]["weight_decay"])
        self.config["training"]["warmup_steps"] = int(self.config["training"]["warmup_steps"])
        self.config["training"]["num_epochs"] = int(self.config["training"]["num_epochs"])
        self.config["training"]["max_sessions"] = int(self.config["training"]["max_sessions"])
        self.config["training"]["batch_size"] = int(self.config["training"]["batch_size"])
        self.config["threshold"]["initial"] = float(self.config["threshold"]["initial"])
        self.config["threshold"]["final"] = float(self.config["threshold"]["final"])
        self.config["threshold"]["anneal_steps"] = int(self.config["threshold"]["anneal_steps"])

    def get_current_threshold(self) -> float:
        """Get annealed threshold based on training progress."""
        progress = min(1.0, self.global_step / self.threshold_anneal_steps)
        return self.threshold_initial + progress * (self.threshold_final - self.threshold_initial)

    def compute_correctness(self, predicted_idx: Optional[int], target_idx: int) -> torch.Tensor:
        """Compute correctness for uncertainty loss."""
        if predicted_idx is None:
            return torch.tensor(0.0, device=self.device)
        return torch.tensor(1.0 if predicted_idx == target_idx else 0.0, device=self.device)

    # In train_3b.py, fix the train_step method (add this around line 115):

    def train_step(self, batch: Dict) -> Dict:
        """Single training step."""
        session_texts = batch["session_texts"][0]
        question = batch["questions"][0]
        choices = batch["choices"][0]
        target_idx = batch["answer_indices"][0]

        # Initialize belief state
        belief_state = BeliefState(self.config["belief_dim"], self.device)
        belief = belief_state.initialize()

        # Track writes
        write_mask = []
        all_beliefs = []
        all_uncertainties = []

        # Process sessions sequentially
        for i, session in enumerate(session_texts):
            # Update belief
            belief, session_embedding = self.builder(belief, session)
            belief = belief_state.update(belief)

            # Estimate uncertainty - ensure belief is float32 for uncertainty estimator
            belief_for_uncertainty = belief.float() if belief.dtype != torch.float32 else belief
            uncertainty = self.uncertainty(belief_for_uncertainty)
            all_beliefs.append(belief.squeeze(0).detach().cpu())
            all_uncertainties.append(uncertainty.squeeze(0).detach().cpu())

            # Check trigger condition
            current_threshold = self.get_current_threshold()
            should_write = (uncertainty.item() >= current_threshold)
            write_mask.append(1.0 if should_write else 0.0)

        # Stack belief states
        beliefs_stack = torch.stack([b.to(self.device) for b in all_beliefs])
        uncertainties_stack = torch.stack([u.to(self.device) for u in all_uncertainties])
        write_mask_tensor = torch.tensor(write_mask, device=self.device)

        # Retrieve decision trace if needed
        trace = None
        if write_mask_tensor.sum() > 0:
            # Ensure beliefs_stack is float32 for retriever
            beliefs_for_retriever = beliefs_stack.float() if beliefs_stack.dtype != torch.float32 else beliefs_stack
            trace, selection_mask = self.retriever(
                beliefs_for_retriever, uncertainties_stack, len(session_texts),
                self.global_step, "Multi-hop inference required"
            )

            # Build prompt with trace
            prompt = self.prompt_builder.build_prompt(trace, question, choices)
        else:
            # Build prompt without trace
            prompt = self.prompt_builder.build_without_trace(question, choices)

        # Generate answer from frozen reasoner
        output = self.reasoner.generate(prompt)
        predicted_idx, is_insufficient = self.output_parser.parse(output, len(choices))

        # Compute correctness
        correctness = self.compute_correctness(predicted_idx, target_idx)

        # For quick test, just return metrics
        self.global_step += 1

        return {
            "loss": 0.0,
            "accuracy": 1.0 if correctness.item() == 1.0 else 0.0,
            "uncertainty": uncertainty.item(),
            "write_rate": write_mask_tensor.mean().item(),
            "threshold": self.get_current_threshold(),
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """Train for one epoch."""
        self.builder.train()
        self.uncertainty.train()
        self.retriever.train()

        total_loss = 0.0
        total_acc = 0.0
        total_writes = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            metrics = self.train_step(batch)

            total_loss += metrics["loss"]
            total_acc += metrics["accuracy"]
            total_writes += metrics["write_rate"]

            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "acc": f"{metrics['accuracy']:.3f}",
                "write": f"{metrics['write_rate']:.2f}"
            })

            if self.global_step % self.config["training"]["logging_steps"] == 0:
                if wandb.run is not None:
                    wandb.log(metrics, step=self.global_step)

        n_batches = len(train_loader)
        return {
            "loss": total_loss / n_batches,
            "accuracy": total_acc / n_batches,
            "write_rate": total_writes / n_batches
        }

    def validate(self, val_loader: DataLoader, epoch: int) -> Dict:
        """Validate model."""
        self.builder.eval()
        self.uncertainty.eval()
        self.retriever.eval()

        correct = 0
        total = 0
        total_uncertainty = 0.0
        total_writes = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                session_texts = batch["session_texts"][0]
                question = batch["questions"][0]
                choices = batch["choices"][0]
                target_idx = batch["answer_indices"][0]

                # Process sessions
                belief_state = BeliefState(self.config["belief_dim"], self.device)
                belief = belief_state.initialize()

                writes = 0
                for i, session in enumerate(session_texts):
                    belief, _ = self.builder(belief, session)
                    belief = belief_state.update(belief)

                    uncertainty = self.uncertainty(belief)
                    total_uncertainty += uncertainty.item()

                    if uncertainty.item() >= self.get_current_threshold():
                        writes += 1

                # Generate answer
                if writes > 0:
                    beliefs_stack = torch.stack(belief_state.get_belief_history())
                    uncertainties_stack = torch.tensor([u.item() for u in total_uncertainty])
                    trace, _ = self.retriever(beliefs_stack, uncertainties_stack,
                                              len(session_texts), self.global_step, "")
                    prompt = self.prompt_builder.build_prompt(trace, question, choices)
                else:
                    prompt = self.prompt_builder.build_without_trace(question, choices)

                output = self.reasoner.generate(prompt)
                predicted_idx, _ = self.output_parser.parse(output, len(choices))

                if predicted_idx is not None and predicted_idx == target_idx:
                    correct += 1
                total += 1
                total_writes += writes / len(session_texts) if session_texts else 0

        accuracy = correct / total if total > 0 else 0.0

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.save_checkpoint("best_model_3b.pt")

        return {
            "accuracy": accuracy,
            "avg_uncertainty": total_uncertainty / total if total > 0 else 0.0,
            "write_rate": total_writes / total if total > 0 else 0.0
        }

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config["output"]["checkpoint_dir"]) / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "builder_state": self.builder.state_dict(),
            "uncertainty_state": self.uncertainty.state_dict(),
            "retriever_state": self.retriever.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_accuracy": self.best_accuracy
        }, checkpoint_path)

        print(f"Checkpoint saved to {checkpoint_path}")

    # Fix the load_checkpoint method in src/training/train_3b.py

    # Find this method (around line 289) and replace it:

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        print(f"Loading checkpoint from {checkpoint_path}")
        print(f"Checkpoint keys: {checkpoint.keys()}")

        # Handle different checkpoint formats
        if "builder_state" in checkpoint:
            # Old format - try to load full builder state
            try:
                self.builder.load_state_dict(checkpoint["builder_state"], strict=False)
                print("Loaded builder state from old format")
            except Exception as e:
                print(f"Warning: Could not load builder state: {e}")
        elif "builder_lora_state" in checkpoint:
            # New format - load only LoRA weights
            try:
                self.builder.model.load_state_dict(checkpoint["builder_lora_state"], strict=False)
                print(f"Loaded {len(checkpoint['builder_lora_state'])} LoRA parameters")
            except Exception as e:
                print(f"Warning: Could not load LoRA state: {e}")
        else:
            print("Warning: No builder state found in checkpoint")

        # Load uncertainty and retriever states
        if "uncertainty_state" in checkpoint:
            self.uncertainty.load_state_dict(checkpoint["uncertainty_state"])
            print("Loaded uncertainty state")

        if "retriever_state" in checkpoint:
            self.retriever.load_state_dict(checkpoint["retriever_state"])
            print("Loaded retriever state")

        # Load optimizer state if available
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            print("Loaded optimizer state")

        # Load training state
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_accuracy = checkpoint.get("best_accuracy", 0.0)

        print(
            f"Checkpoint loaded: epoch={self.current_epoch}, step={self.global_step}, best_acc={self.best_accuracy:.3f}")

    def train(self, use_wandb: bool = False):
        """Main training loop."""
        if use_wandb:
            wandb.init(project="adaptivebelief", config=self.config)

        train_loader = self.data_loader.get_dataloader("train", shuffle=True)
        val_loader = self.data_loader.get_dataloader("val", shuffle=False)

        for epoch in range(self.config["training"]["num_epochs"]):
            self.current_epoch = epoch

            # Determine phase and adjust learning rate
            if epoch < 2:
                phase = "warmup"
            elif epoch < 5:
                phase = "stabilization"
            else:
                phase = "fine_tuning"

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = get_phase_lr(phase)

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader, epoch)

            print(f"Epoch {epoch}: Train Acc={train_metrics['accuracy']:.3f}, "
                  f"Val Acc={val_metrics['accuracy']:.3f}, "
                  f"Write Rate={val_metrics['write_rate']:.3f}")

            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/accuracy": train_metrics["accuracy"],
                    "train/loss": train_metrics["loss"],
                    "train/write_rate": train_metrics["write_rate"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/write_rate": val_metrics["write_rate"],
                    "val/avg_uncertainty": val_metrics["avg_uncertainty"]
                })

            # Save periodic checkpoint
            if epoch % 2 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

        self.save_checkpoint("final_model_3b.pt")

        if use_wandb:
            wandb.finish()

        print(f"Training complete! Best accuracy: {self.best_accuracy:.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/rtx3060_3b.yaml")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.quick_test:
        # Run a quick test with minimal settings
        trainer = AdaptiveBelief3BTrainer(args.config)
        if args.max_sessions:
            trainer.config["training"]["max_sessions"] = args.max_sessions
        if args.epochs:
            trainer.config["training"]["num_epochs"] = args.epochs
        trainer.train(use_wandb=args.use_wandb)
    else:
        trainer = AdaptiveBelief3BTrainer(args.config)
        if args.max_sessions:
            trainer.config["training"]["max_sessions"] = args.max_sessions
        if args.epochs:
            trainer.config["training"]["num_epochs"] = args.epochs
        trainer.train(use_wandb=args.use_wandb)


if __name__ == "__main__":
    main()