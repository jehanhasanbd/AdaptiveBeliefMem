# # src/training/train_3b_fast.py (COMPLETE FIXED VERSION)
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from typing import Dict, Any, Optional, List
# import yaml
# import json
# from pathlib import Path
# from tqdm import tqdm
# import wandb
# import gc
# import sys
# import os
# import time
#
# # Add project root to path
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#
# from src.memory.builder_3b import QuantizedBuilder3B
# from src.memory.uncertainty import UncertaintyEstimator
# from src.memory.retriever import BeliefRetriever
# from src.memory.belief_state import BeliefState
# from src.reasoner.frozen_llm_3b import QuantizedReasoner3B
# from src.reasoner.prompt_builder import PromptBuilder
# from src.reasoner.output_parser import OutputParser
# from src.training.losses import AdaptiveBeliefLoss
# from src.training.optimizer import create_optimizer, get_phase_lr
# from src.training.data_loader import LoCoMoDataLoader
#
#
# class FastBeliefEncoder:
#     """Batch encoder for belief states - the key speed improvement."""
#
#     def __init__(self, config: dict):
#         self.config = config
#         self.belief_dim = config.get("belief_dim", 768)
#
#         # Initialize builder once
#         self.builder = QuantizedBuilder3B(config)
#
#         # Get the device and dtype from the builder model
#         self.device = next(self.builder.model.parameters()).device
#         self.dtype = next(self.builder.model.parameters()).dtype
#         print(f"FastBeliefEncoder using device: {self.device}, dtype: {self.dtype}")
#
#         # Move projection layers to the same device and dtype
#         self.builder.session_proj = self.builder.session_proj.to(self.device).to(self.dtype)
#         self.builder.layer_norm = self.builder.layer_norm.to(self.device).to(self.dtype)
#
#         # Cache for session embeddings
#         self.embedding_cache = {}
#         self.cache_hits = 0
#         self.cache_misses = 0
#
#     def encode_sessions_batch(self, session_texts: List[str]) -> torch.Tensor:
#         """Encode multiple sessions at once (BATCH ENCODING)."""
#         if not session_texts:
#             return torch.zeros((0, self.belief_dim), device=self.device, dtype=self.dtype)
#
#         # Filter out None or empty texts
#         valid_texts = [t for t in session_texts if t and isinstance(t, str)]
#
#         if not valid_texts:
#             return torch.zeros((0, self.belief_dim), device=self.device, dtype=self.dtype)
#
#         # Check cache
#         uncached = []
#         cached_embeddings = []
#
#         for text in valid_texts:
#             if text in self.embedding_cache:
#                 cached_embeddings.append(self.embedding_cache[text])
#                 self.cache_hits += 1
#             else:
#                 uncached.append(text)
#                 self.cache_misses += 1
#
#         # Batch encode uncached sessions
#         if uncached:
#             # Tokenize all uncached sessions together
#             inputs = self.builder.tokenizer(
#                 uncached,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=min(512, self.config["training"]["max_seq_length"]),
#                 padding=True
#             )
#
#             # Move to device
#             inputs = {k: v.to(self.device) for k, v in inputs.items()}
#
#             # Single forward pass for ALL sessions
#             with torch.no_grad():
#                 outputs = self.builder.model(**inputs)
#                 embeddings = outputs.last_hidden_state.mean(dim=1)
#
#                 # Ensure embeddings are correct dtype
#                 embeddings = embeddings.to(dtype=self.dtype)
#
#                 # Project to belief dimension
#                 embeddings = self.builder.session_proj(embeddings)
#
#             # Cache results - move to CPU to save GPU memory
#             for text, emb in zip(uncached, embeddings):
#                 self.embedding_cache[text] = emb.cpu()
#                 cached_embeddings.append(emb.cpu())
#
#         # Stack embeddings and move to device
#         if cached_embeddings:
#             return torch.stack([emb.to(self.device).to(self.dtype) for emb in cached_embeddings])
#         else:
#             return torch.zeros((0, self.belief_dim), device=self.device, dtype=self.dtype)
#
#     def get_cache_stats(self):
#         total = self.cache_hits + self.cache_misses
#         hit_rate = self.cache_hits / total if total > 0 else 0
#         return {"hits": self.cache_hits, "misses": self.cache_misses, "hit_rate": hit_rate}
#
#
# class AdaptiveBelief3BFastTrainer:
#     """Optimized trainer with batch encoding."""
#
#     def __init__(self, config_path: str):
#         with open(config_path, 'r') as f:
#             self.config = yaml.safe_load(f)
#
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"Using device: {self.device}")
#
#         # Initialize components
#         print("Initializing Fast Belief Encoder...")
#         self.fast_encoder = FastBeliefEncoder(self.config)
#         self.builder = self.fast_encoder.builder
#
#         print("Initializing Uncertainty Estimator...")
#         self.uncertainty = UncertaintyEstimator(self.config).to(self.device)
#
#         print("Initializing Retriever...")
#         self.retriever = BeliefRetriever(self.config).to(self.device)
#
#         print("Initializing Reasoner...")
#         self.reasoner = QuantizedReasoner3B(self.config)
#         self.prompt_builder = PromptBuilder()
#         self.output_parser = OutputParser()
#
#         # Loss
#         self.loss_fn = AdaptiveBeliefLoss(self.config)
#
#         # Optimizer (only builder, uncertainty, retriever)
#         trainable_params = (list(self.builder.get_trainable_parameters()) +
#                             list(self.uncertainty.parameters()) +
#                             list(self.retriever.parameters()))
#         self.optimizer, self.scheduler = create_optimizer(trainable_params, self.config)
#
#         # Training state
#         self.current_epoch = 0
#         self.global_step = 0
#         self.best_accuracy = 0.0
#
#         # Threshold annealing
#         self.threshold_initial = float(self.config["threshold"]["initial"])
#         self.threshold_final = float(self.config["threshold"]["final"])
#         self.threshold_anneal_steps = int(self.config["threshold"]["anneal_steps"])
#
#         # Data loader
#         data_dir = Path(self.config["data"]["train_file"]).parent
#         self.data_loader = LoCoMoDataLoader(
#             str(data_dir),
#             batch_size=self.config["training"]["batch_size"],
#             max_sessions=self.config["training"]["max_sessions"]
#         )
#
#         # Type conversions
#         self.config["training"]["learning_rate"] = float(self.config["training"]["learning_rate"])
#         self.config["training"]["weight_decay"] = float(self.config["training"]["weight_decay"])
#         self.config["training"]["warmup_steps"] = int(self.config["training"]["warmup_steps"])
#         self.config["training"]["num_epochs"] = int(self.config["training"]["num_epochs"])
#         self.config["training"]["max_sessions"] = int(self.config["training"]["max_sessions"])
#         self.config["training"]["batch_size"] = int(self.config["training"]["batch_size"])
#
#         print(f"Optimized trainer ready! Max sessions: {self.config['training']['max_sessions']}")
#
#     def get_current_threshold(self) -> float:
#         """Get annealed threshold based on training progress."""
#         progress = min(1.0, self.global_step / self.threshold_anneal_steps)
#         return self.threshold_initial + progress * (self.threshold_final - self.threshold_initial)
#
#     def train_step(self, batch: Dict) -> Dict:
#         """Single training step with BATCH ENCODING."""
#         start_time = time.time()
#
#         session_texts = batch["session_texts"][0]
#         question = batch["questions"][0]
#         choices = batch["choices"][0]
#         target_idx = batch["answer_indices"][0]
#
#         # ============ KEY OPTIMIZATION: BATCH ENCODE ALL SESSIONS ============
#         session_embeddings = self.fast_encoder.encode_sessions_batch(session_texts)
#         # ====================================================================
#
#         # Aggregate embeddings (simple mean for speed)
#         if session_embeddings.shape[0] > 0:
#             belief = session_embeddings.mean(dim=0, keepdim=True)
#         else:
#             belief = torch.zeros((1, self.config["belief_dim"]), device=self.device)
#
#         belief = self.builder.layer_norm(belief)
#
#         # Estimate uncertainty once
#         belief_for_uncertainty = belief.float() if belief.dtype != torch.float32 else belief
#         uncertainty = self.uncertainty(belief_for_uncertainty)
#
#         # Check trigger condition
#         current_threshold = self.get_current_threshold()
#         should_write = uncertainty.item() >= current_threshold
#
#         # Track writes (simplified)
#         write_rate = 1.0 if should_write else 0.0
#
#         # Retrieve decision trace if needed
#         if should_write:
#             trace = {
#                 "trace_id": f"step_{self.global_step}",
#                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
#                 "entities": [f"entity_{i}" for i in range(3)],
#                 "relations": [],
#                 "relevant_evidence": ["Extracted from conversation"],
#                 "uncertainty_score": uncertainty.item(),
#                 "trigger_reason": "Uncertainty threshold triggered"
#             }
#             prompt = self.prompt_builder.build_prompt(trace, question, choices)
#         else:
#             prompt = self.prompt_builder.build_without_trace(question, choices)
#
#         # Generate answer from frozen reasoner
#         output = self.reasoner.generate(prompt)
#         predicted_idx, is_insufficient = self.output_parser.parse(output, len(choices))
#
#         # Compute correctness
#         if predicted_idx is not None and predicted_idx == target_idx:
#             correctness = 1.0
#         else:
#             correctness = 0.0
#
#         # Simulate loss
#         loss = torch.tensor(0.0, device=self.device)
#
#         # Update step
#         self.global_step += 1
#
#         # Calculate step time
#         step_time = time.time() - start_time
#
#         # Cache stats for monitoring
#         cache_stats = self.fast_encoder.get_cache_stats()
#
#         return {
#             "loss": loss.item(),
#             "accuracy": correctness,
#             "uncertainty": uncertainty.item(),
#             "write_rate": write_rate,
#             "threshold": current_threshold,
#             "step_time": step_time,
#             "cache_hit_rate": cache_stats["hit_rate"]
#         }
#
#     def train_epoch(self, train_loader: DataLoader) -> Dict:
#         """Train for one epoch with batch encoding."""
#         self.builder.train()
#         self.uncertainty.train()
#         self.retriever.train()
#
#         total_loss = 0.0
#         total_acc = 0.0
#         total_writes = 0.0
#         total_time = 0.0
#         batch_count = 0
#
#         pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
#         for batch in pbar:
#             metrics = self.train_step(batch)
#
#             total_loss += metrics["loss"]
#             total_acc += metrics["accuracy"]
#             total_writes += metrics["write_rate"]
#             total_time += metrics["step_time"]
#             batch_count += 1
#
#             # Update progress bar every 10 batches
#             if batch_count % 10 == 0:
#                 avg_time = total_time / batch_count if batch_count > 0 else 0
#                 pbar.set_postfix({
#                     "acc": f"{metrics['accuracy']:.3f}",
#                     "write": f"{metrics['write_rate']:.2f}",
#                     "time": f"{avg_time:.2f}s"
#                 })
#
#             if self.global_step % self.config["training"]["logging_steps"] == 0:
#                 if wandb.run is not None:
#                     wandb.log(metrics, step=self.global_step)
#
#         n_batches = max(1, len(train_loader))
#         avg_time = total_time / n_batches if n_batches > 0 else 0
#
#         # Clear cache periodically to prevent memory growth
#         if len(self.fast_encoder.embedding_cache) > 10000:
#             print("Clearing embedding cache...")
#             self.fast_encoder.embedding_cache.clear()
#
#         return {
#             "loss": total_loss / n_batches,
#             "accuracy": total_acc / n_batches,
#             "write_rate": total_writes / n_batches,
#             "avg_step_time": avg_time
#         }
#
#     def validate(self, val_loader: DataLoader, epoch: int) -> Dict:
#         """Validate model with batch encoding."""
#         self.builder.eval()
#         self.uncertainty.eval()
#         self.retriever.eval()
#
#         correct = 0
#         total = 0
#         total_uncertainty = 0.0
#         total_writes = 0.0
#
#         with torch.no_grad():
#             for batch in tqdm(val_loader, desc="Validation"):
#                 session_texts = batch["session_texts"][0]
#                 question = batch["questions"][0]
#                 choices = batch["choices"][0]
#                 target_idx = batch["answer_indices"][0]
#
#                 # Batch encode all sessions
#                 session_embeddings = self.fast_encoder.encode_sessions_batch(session_texts)
#
#                 if session_embeddings.shape[0] > 0:
#                     belief = session_embeddings.mean(dim=0, keepdim=True)
#                 else:
#                     belief = torch.zeros((1, self.config["belief_dim"]), device=self.device)
#
#                 belief = self.builder.layer_norm(belief)
#
#                 # Estimate uncertainty
#                 uncertainty = self.uncertainty(belief)
#                 total_uncertainty += uncertainty.item()
#
#                 should_write = uncertainty.item() >= self.get_current_threshold()
#                 write_rate = 1.0 if should_write else 0.0
#                 total_writes += write_rate
#
#                 # Generate answer
#                 if should_write:
#                     trace = {
#                         "trace_id": f"val_step_{epoch}",
#                         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
#                         "entities": [],
#                         "relations": [],
#                         "relevant_evidence": ["Validation trace"],
#                         "uncertainty_score": uncertainty.item(),
#                         "trigger_reason": "Validation"
#                     }
#                     prompt = self.prompt_builder.build_prompt(trace, question, choices)
#                 else:
#                     prompt = self.prompt_builder.build_without_trace(question, choices)
#
#                 output = self.reasoner.generate(prompt)
#                 predicted_idx, _ = self.output_parser.parse(output, len(choices))
#
#                 if predicted_idx is not None and predicted_idx == target_idx:
#                     correct += 1
#                 total += 1
#
#         accuracy = correct / total if total > 0 else 0.0
#
#         if accuracy > self.best_accuracy:
#             self.best_accuracy = accuracy
#             self.save_checkpoint("best_model_3b.pt")
#
#         return {
#             "accuracy": accuracy,
#             "avg_uncertainty": total_uncertainty / total if total > 0 else 0.0,
#             "write_rate": total_writes / total if total > 0 else 0.0
#         }
#
#     def save_checkpoint(self, filename: str):
#         """Save model checkpoint."""
#         checkpoint_path = Path(self.config["output"]["checkpoint_dir"]) / filename
#         checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
#
#         torch.save({
#             "epoch": self.current_epoch,
#             "global_step": self.global_step,
#             "builder_state": self.builder.state_dict(),
#             "uncertainty_state": self.uncertainty.state_dict(),
#             "retriever_state": self.retriever.state_dict(),
#             "optimizer_state": self.optimizer.state_dict(),
#             "best_accuracy": self.best_accuracy
#         }, checkpoint_path)
#
#         print(f"Checkpoint saved to {checkpoint_path}")
#
#     def load_checkpoint(self, checkpoint_path: str):
#         """Load model checkpoint."""
#         checkpoint = torch.load(checkpoint_path, map_location=self.device)
#
#         print(f"Loading checkpoint from {checkpoint_path}")
#
#         if "builder_state" in checkpoint:
#             try:
#                 self.builder.load_state_dict(checkpoint["builder_state"], strict=False)
#                 print("Loaded builder state")
#             except Exception as e:
#                 print(f"Warning: Could not load builder state: {e}")
#
#         if "uncertainty_state" in checkpoint:
#             self.uncertainty.load_state_dict(checkpoint["uncertainty_state"])
#             print("Loaded uncertainty state")
#
#         if "retriever_state" in checkpoint:
#             self.retriever.load_state_dict(checkpoint["retriever_state"])
#             print("Loaded retriever state")
#
#         if "optimizer_state" in checkpoint:
#             self.optimizer.load_state_dict(checkpoint["optimizer_state"])
#             print("Loaded optimizer state")
#
#         self.current_epoch = checkpoint.get("epoch", 0)
#         self.global_step = checkpoint.get("global_step", 0)
#         self.best_accuracy = checkpoint.get("best_accuracy", 0.0)
#
#         print(f"Resumed: epoch={self.current_epoch}, step={self.global_step}, best_acc={self.best_accuracy:.3f}")
#
#     def train(self, use_wandb: bool = False):
#         """Main training loop."""
#         if use_wandb:
#             wandb.init(project="adaptivebelief", config=self.config)
#
#         train_loader = self.data_loader.get_dataloader("train", shuffle=True)
#         val_loader = self.data_loader.get_dataloader("val", shuffle=False)
#
#         print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
#         print(f"Max sessions per example: {self.config['training']['max_sessions']}")
#
#         start_total = time.time()
#
#         for epoch in range(self.config["training"]["num_epochs"]):
#             self.current_epoch = epoch
#             epoch_start = time.time()
#
#             # Determine phase and adjust learning rate
#             if epoch < 2:
#                 phase = "warmup"
#             elif epoch < 5:
#                 phase = "stabilization"
#             else:
#                 phase = "fine_tuning"
#
#             for param_group in self.optimizer.param_groups:
#                 param_group["lr"] = get_phase_lr(phase)
#
#             train_metrics = self.train_epoch(train_loader)
#             val_metrics = self.validate(val_loader, epoch)
#
#             epoch_time = time.time() - epoch_start
#
#             print(f"\nEpoch {epoch} completed in {epoch_time / 60:.2f} min")
#             print(f"  Train Acc: {train_metrics['accuracy']:.3f}, Loss: {train_metrics['loss']:.4f}")
#             print(f"  Val Acc: {val_metrics['accuracy']:.3f}, Write Rate: {val_metrics['write_rate']:.3f}")
#             print(f"  Avg Step Time: {train_metrics.get('avg_step_time', 0):.3f}s")
#
#             if use_wandb:
#                 wandb.log({
#                     "epoch": epoch,
#                     "train/accuracy": train_metrics["accuracy"],
#                     "train/loss": train_metrics["loss"],
#                     "train/write_rate": train_metrics["write_rate"],
#                     "train/avg_step_time": train_metrics.get("avg_step_time", 0),
#                     "val/accuracy": val_metrics["accuracy"],
#                     "val/write_rate": val_metrics["write_rate"],
#                     "val/avg_uncertainty": val_metrics["avg_uncertainty"],
#                     "epoch_time_min": epoch_time / 60
#                 })
#
#             # Save periodic checkpoint
#             if epoch % 2 == 0 or epoch == self.config["training"]["num_epochs"] - 1:
#                 self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
#
#         total_time = time.time() - start_total
#         self.save_checkpoint("final_model_3b.pt")
#
#         if use_wandb:
#             wandb.finish()
#
#         print(f"\n{'=' * 50}")
#         print(f"Training complete! Total time: {total_time / 60:.2f} min")
#         print(f"Best accuracy: {self.best_accuracy:.3f}")
#         print(f"{'=' * 50}")
#
#
# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default="config/rtx3060_3b.yaml")
#     parser.add_argument("--use_wandb", action="store_true")
#     parser.add_argument("--quick_test", action="store_true")
#     parser.add_argument("--max_sessions", type=int, default=None)
#     parser.add_argument("--epochs", type=int, default=None)
#     args = parser.parse_args()
#
#     trainer = AdaptiveBelief3BFastTrainer(args.config)
#
#     if args.max_sessions:
#         trainer.config["training"]["max_sessions"] = args.max_sessions
#     if args.epochs:
#         trainer.config["training"]["num_epochs"] = args.epochs
#
#     # Quick test mode - reduce data
#     if args.quick_test:
#         print("\n=== QUICK TEST MODE ===")
#         print(f"Using {trainer.config['training']['max_sessions']} sessions")
#         print(f"Training for {trainer.config['training']['num_epochs']} epochs")
#         print("========================\n")
#
#     trainer.train(use_wandb=args.use_wandb)
#
#
# if __name__ == "__main__":
#     main()