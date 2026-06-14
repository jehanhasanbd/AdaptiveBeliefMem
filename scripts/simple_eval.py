# scripts/simple_eval.py
import json
import random
from tqdm import tqdm


def evaluate_random_baseline():
    """Simple baseline evaluation without loading models."""

    # Load test data
    with open("data/processed/test.json", 'r') as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} test examples")

    # Random baseline
    random.seed(42)
    correct = 0
    total = 0

    for example in tqdm(test_data[:100], desc="Evaluating"):  # First 100 examples
        choices = example["choices"]
        target = example["answer_index"]

        # Random guess
        pred = random.randint(0, len(choices) - 1)

        if pred == target:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\nRandom Baseline Accuracy: {accuracy:.3f} ({correct}/{total})")

    # Also evaluate "first choice" baseline
    correct = 0
    for example in tqdm(test_data[:100], desc="First choice baseline"):
        pred = 0  # Always pick first choice
        if pred == example["answer_index"]:
            correct += 1

    accuracy = correct / 100
    print(f"First Choice Baseline Accuracy: {accuracy:.3f} ({correct}/100)")

    # Print sample questions
    print("\n=== Sample Questions from Test Set ===")
    for i, example in enumerate(test_data[:3]):
        print(f"\n{i + 1}. Question: {example['question']}")
        print(f"   Choices: {example['choices'][:3]}...")
        print(f"   Answer index: {example['answer_index']}")
        print(f"   Type: {example.get('question_type', 'unknown')}")


if __name__ == "__main__":
    evaluate_random_baseline()