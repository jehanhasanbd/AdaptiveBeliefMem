# scripts/final_evaluation_with_lora.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import re
from tqdm import tqdm
from collections import defaultdict
import numpy as np


class FinalEvaluatorWithLoRA:
    def __init__(self, base_model="Qwen/Qwen2.5-3B-Instruct", lora_path="outputs/best_lora_model"):
        print("Loading base model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        # Load LoRA weights if they exist
        import os
        if os.path.exists(lora_path):
            print(f"Loading LoRA weights from {lora_path}...")
            self.model = PeftModel.from_pretrained(self.base_model, lora_path)
            print("✓ LoRA weights loaded successfully!")
        else:
            print(f"⚠️ No LoRA weights found at {lora_path}, using base model only")
            self.model = self.base_model

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

    def calculate_f1(self, prediction: str, ground_truth: str) -> float:
        """Calculate token-level F1 score."""
        pred_tokens = self.normalize_text(prediction).split()
        truth_tokens = self.normalize_text(ground_truth).split()

        if not pred_tokens and not truth_tokens:
            return 1.0
        if not pred_tokens or not truth_tokens:
            return 0.0

        common = set(pred_tokens) & set(truth_tokens)
        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def calculate_bleu(self, prediction: str, ground_truth: str) -> float:
        """Simple BLEU-like score."""
        pred_tokens = self.normalize_text(prediction).split()
        truth_tokens = self.normalize_text(ground_truth).split()

        if not pred_tokens or not truth_tokens:
            return 0.0

        # 1-gram overlap
        common_1gram = set(pred_tokens) & set(truth_tokens)
        precision_1 = len(common_1gram) / len(pred_tokens) if pred_tokens else 0

        # 2-gram overlap
        pred_2gram = set([' '.join(pred_tokens[i:i + 2]) for i in range(len(pred_tokens) - 1)])
        truth_2gram = set([' '.join(truth_tokens[i:i + 2]) for i in range(len(truth_tokens) - 1)])
        common_2gram = pred_2gram & truth_2gram
        precision_2 = len(common_2gram) / len(pred_2gram) if pred_2gram else 0

        return (precision_1 + precision_2) / 2

    def generate_answer(self, question: str, choices: list) -> tuple:
        """Generate answer using the model with LoRA weights."""
        choice_lines = []
        for i, choice in enumerate(choices[:10]):
            letter = chr(ord('A') + i)
            choice_lines.append(f"{letter}. {choice}")
        choices_text = "\n".join(choice_lines)

        prompt = f"""Question: {question}

Choices:
{choices_text}

Answer with the letter of the correct choice (A, B, C, etc.):

Answer:"""

        messages = [
            {"role": "system",
             "content": "You are a helpful assistant that answers multiple choice questions with just the letter of the correct answer."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.01,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        response = response.strip()

        letter_match = re.search(r'([A-J])', response)
        if letter_match:
            letter = letter_match.group(1)
            idx = ord(letter) - ord('A')
            if 0 <= idx < len(choices):
                return idx, letter, choices[idx]

        return -1, "", ""

    def evaluate(self, test_data, max_samples=None):
        """Evaluate on test data."""
        if max_samples:
            test_data = test_data[:max_samples]

        results = defaultdict(lambda: {
            "correct": 0,
            "total": 0,
            "f1_scores": [],
            "bleu_scores": [],
            "predictions": [],
            "ground_truths": []
        })

        print(f"\nEvaluating {len(test_data)} samples with LoRA model...")

        for example in tqdm(test_data, desc="Evaluating"):
            question = example["question"]
            choices = example["choices"]
            target_idx = example["answer_index"]
            qtype = example.get("question_type", "unknown")
            ground_truth = choices[target_idx] if target_idx >= 0 else ""

            pred_idx, pred_letter, predicted_answer = self.generate_answer(question, choices)

            is_correct = (pred_idx == target_idx)
            f1 = self.calculate_f1(predicted_answer, ground_truth)
            bleu = self.calculate_bleu(predicted_answer, ground_truth)

            results[qtype]["correct"] += 1 if is_correct else 0
            results[qtype]["total"] += 1
            results[qtype]["f1_scores"].append(f1)
            results[qtype]["bleu_scores"].append(bleu)
            results[qtype]["predictions"].append(predicted_answer)
            results[qtype]["ground_truths"].append(ground_truth)

            results["all"]["correct"] += 1 if is_correct else 0
            results["all"]["total"] += 1
            results["all"]["f1_scores"].append(f1)
            results["all"]["bleu_scores"].append(bleu)

        metrics = {}
        for qtype, data in results.items():
            total = data["total"]
            if total > 0:
                metrics[qtype] = {
                    "accuracy": data["correct"] / total,
                    "f1_score": np.mean(data["f1_scores"]),
                    "f1_std": np.std(data["f1_scores"]),
                    "bleu_score": np.mean(data["bleu_scores"]),
                    "bleu_std": np.std(data["bleu_scores"]),
                    "total_examples": total,
                    "correct": data["correct"]
                }

        return metrics

    def print_results(self, metrics):
        """Print formatted results."""
        print("\n" + "=" * 80)
        print("FINAL EVALUATION RESULTS (WITH LoRA)")
        print("=" * 80)

        order = ["all", "single_hop", "multi_hop", "temporal_reasoning", "open_domain", "adversarial"]

        print(f"\n{'Type':<20} {'Accuracy':<12} {'F1 Score':<18} {'BLEU Score':<15} {'Samples':<8}")
        print("-" * 80)

        for qtype in order:
            if qtype in metrics:
                m = metrics[qtype]
                print(
                    f"{qtype:<20} {m['accuracy'] * 100:>6.2f}%     {m['f1_score']:>6.3f} ±{m['f1_std']:>5.3f}   {m['bleu_score']:>6.3f} ±{m['bleu_std']:>5.3f}   {m['total_examples']:>6}")

        print("-" * 80)

        if "all" in metrics:
            print(f"\n{'=' * 40}")
            print("OVERALL PERFORMANCE (WITH LoRA)")
            print(f"{'=' * 40}")
            print(f"Accuracy:  {metrics['all']['accuracy'] * 100:.2f}%")
            print(f"F1 Score:  {metrics['all']['f1_score']:.3f} (±{metrics['all']['f1_std']:.3f})")
            print(f"BLEU Score: {metrics['all']['bleu_score']:.3f} (±{metrics['all']['bleu_std']:.3f})")
            print(f"Total Samples: {metrics['all']['total_examples']}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, default="outputs/best_lora_model",
                        help="Path to LoRA model weights")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (None = all)")
    args = parser.parse_args()

    # Load test data
    print("Loading test data...")
    with open("data/processed/test.json", 'r') as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} test examples")

    # Count question types
    type_counts = defaultdict(int)
    for ex in test_data:
        type_counts[ex.get("question_type", "unknown")] += 1

    print("\nQuestion type distribution:")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count}")

    # Initialize evaluator with LoRA
    evaluator = FinalEvaluatorWithLoRA(lora_path=args.lora_path)

    # Evaluate
    metrics = evaluator.evaluate(test_data, max_samples=args.max_samples)

    # Print results
    evaluator.print_results(metrics)

    # Save results
    output_path = "outputs/results/final_evaluation_with_lora.json"
    import os
    os.makedirs("outputs/results", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()