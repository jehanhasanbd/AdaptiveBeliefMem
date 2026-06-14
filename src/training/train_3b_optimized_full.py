# src/training/train_3b_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from collections import defaultdict
from tqdm import tqdm
import json
import yaml
from pathlib import Path
import numpy as np
import gc
import sys
import os
import time
import re


class OptimizedDataset(Dataset):
    def __init__(self, data_path, max_sessions=3, max_length=512):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.max_sessions = max_sessions
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sessions = item["session_texts"][:self.max_sessions]
        # Combine sessions with separator
        context = " ".join(sessions)

        return {
            "context": context[:self.max_length],
            "question": item["question"],
            "choices": item["choices"],
            "answer_idx": item["answer_index"],
            "question_type": item.get("question_type", "unknown")
        }


class FastTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = "cuda"
        self.setup_components()

    def setup_components(self):
        """Initialize all components with optimizations."""
        print("Loading models...")

        # 4-bit config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Load base model for training (with LoRA)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        # Apply LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, lora_config)

        # Only train LoRA parameters
        for name, param in self.model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["base_model"],
            trust_remote_code=True,
            padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=2e-4,
            weight_decay=0.01
        )

        # Load data
        data_dir = Path(self.config["data"]["train_file"]).parent
        self.train_dataset = OptimizedDataset(
            data_dir / "train.json",
            max_sessions=3,
            max_length=512
        )
        self.val_dataset = OptimizedDataset(
            data_dir / "val.json",
            max_sessions=3,
            max_length=512
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0)

        # Scheduler
        total_steps = len(self.train_loader) * self.config["training"]["num_epochs"]
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

    def train_step(self, batch):
        """Single training step with proper causal LM training."""
        context = batch["context"][0]
        question = batch["question"][0]
        choices = batch["choices"][0]
        answer_idx = batch["answer_idx"][0]

        # Format choices
        choice_text = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices[:10])])

        # Create prompt and target
        prompt = f"""Context: {context}

Question: {question}

Choices:
{choice_text}

Answer:"""

        correct_answer = chr(65 + answer_idx)
        target_text = f" {correct_answer}"

        # Combine prompt and target for training
        full_text = prompt + target_text

        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=768,
            padding=True
        ).to(self.device)

        # Create labels (shifted for causal LM)
        labels = inputs.input_ids.clone()

        # Forward pass with labels
        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return loss.item()

    def validate(self):
        """Validation step."""
        self.model.eval()
        correct = 0
        total = 0
        type_correct = defaultdict(int)
        type_total = defaultdict(int)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                context = batch["context"][0]
                question = batch["question"][0]
                choices = batch["choices"][0]
                answer_idx = batch["answer_idx"][0]
                qtype = batch["question_type"][0]

                # Format choices
                choice_text = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices[:10])])

                prompt = f"""Context: {context}

Question: {question}

Choices:
{choice_text}

Answer:"""

                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(self.device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

                response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

                # Extract answer
                match = re.search(r'([A-J])', response.strip())
                if match:
                    pred = match.group(1)
                    pred_idx = ord(pred) - ord('A')
                else:
                    pred_idx = -1

                if pred_idx == answer_idx:
                    correct += 1
                    type_correct[qtype] += 1
                type_total[qtype] += 1
                total += 1

        accuracy = correct / total if total > 0 else 0
        type_accuracy = {
            qtype: type_correct[qtype] / type_total[qtype]
            for qtype in type_total
        }

        return accuracy, type_accuracy

    def train(self):
        """Main training loop."""
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")

        best_accuracy = 0

        for epoch in range(self.config["training"]["num_epochs"]):
            self.model.train()
            total_loss = 0
            start_time = time.time()

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            for batch in progress_bar:
                loss = self.train_step(batch)
                total_loss += loss
                progress_bar.set_postfix({"loss": f"{loss:.4f}"})

            avg_loss = total_loss / len(self.train_loader)
            epoch_time = time.time() - start_time

            # Validate
            val_accuracy, type_accuracy = self.validate()

            print(f"\nEpoch {epoch + 1} completed in {epoch_time / 60:.2f} min")
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy * 100:.2f}%")

            # Print per-type accuracy
            print("  Per-type Accuracy:")
            for qtype, acc in sorted(type_accuracy.items()):
                print(f"    {qtype}: {acc * 100:.2f}%")

            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                os.makedirs("outputs/best_lora_model", exist_ok=True)
                self.model.save_pretrained("outputs/best_lora_model")
                print(f"  ✓ Saved best model (accuracy: {best_accuracy * 100:.2f}%)")

            print("-" * 50)

        print(f"\nTraining complete! Best accuracy: {best_accuracy * 100:.2f}%")
        return best_accuracy


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/rtx3060_3b.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    # Update config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config["training"]["num_epochs"] = args.epochs

    # Save temp config
    os.makedirs("config", exist_ok=True)
    temp_config = "config/temp.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    # Train
    trainer = FastTrainer(temp_config)
    trainer.train()

    # Cleanup
    if os.path.exists(temp_config):
        os.remove(temp_config)


if __name__ == "__main__":
    main()