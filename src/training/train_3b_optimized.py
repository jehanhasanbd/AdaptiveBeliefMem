# src/training/train_3b_optimized.py (fixed device handling)
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm
import yaml
from pathlib import Path
import sys
import time


class FastBuilder(nn.Module):
    """Optimized builder - no per-session LLM calls."""

    def __init__(self, config):
        super().__init__()
        self.belief_dim = 768

        # Load model once
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        print("Loading model for builder...")
        self.model = AutoModel.from_pretrained(
            config["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        # Get the device of the loaded model
        self.device = next(self.model.parameters()).device
        print(f"Builder model loaded on: {self.device}")

        # Simple projection - move to same device as model
        self.proj = nn.Linear(self.model.config.hidden_size, self.belief_dim)
        self.proj = self.proj.to(self.device)
        self.proj = self.proj.to(dtype=torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["base_model"], trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode_batch(self, texts):
        """Encode all sessions at once."""
        # Tokenize
        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True,
            max_length=512, padding=True
        )

        # Move to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Ensure embeddings are float16 for projection
            embeddings = embeddings.to(dtype=torch.float16)

        return self.proj(embeddings)

    def forward(self, session_texts):
        """Process all sessions in one forward pass."""
        return self.encode_batch(session_texts)


class OptimizedTrainer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = "cuda"
        print("Loading builder...")
        self.builder = FastBuilder(self.config)

        print("Loading data...")
        with open("data/processed/train.json") as f:
            self.train_data = json.load(f)[:100]  # Limit for testing

        print(f"Loaded {len(self.train_data)} examples")

    def train_epoch(self):
        """Fast training epoch."""
        correct = 0
        total = 0

        for example in tqdm(self.train_data, desc="Training"):
            sessions = example["session_texts"][:3]  # Limit to 3 sessions
            question = example["question"]
            choices = example["choices"]
            target = example["answer_index"]

            # Process all sessions in ONE forward pass
            start_time = time.time()
            with torch.no_grad():
                embeddings = self.builder(sessions)  # [num_sessions, 768]
                belief = embeddings.mean(dim=0)  # Average all sessions

                # Simple rule: if uncertainty > 0.5, use full context
                uncertainty = torch.sigmoid(belief[:1]).item()

                if uncertainty > 0.5:
                    # Use full context (simulate trace)
                    context = " ".join(sessions[:3])
                else:
                    context = sessions[-1]  # Last session only

            # Simulate answer (for speed test)
            latency = time.time() - start_time

            # Mock accuracy (replace with real LLM call)
            is_correct = torch.rand(1).item() > 0.5

            if is_correct:
                correct += 1
            total += 1

            if total % 10 == 0:
                print(f"  Batch {total}, Latency: {latency:.2f}s, Acc: {correct / total:.3f}")

        return correct / total

    def train(self):
        print("Starting fast training...")
        start = time.time()
        acc = self.train_epoch()
        elapsed = time.time() - start
        print(f"\nEpoch completed in {elapsed / 60:.1f} minutes")
        print(f"Accuracy: {acc:.3f}")


if __name__ == "__main__":
    trainer = OptimizedTrainer("config/rtx3060_3b.yaml")
    trainer.train()