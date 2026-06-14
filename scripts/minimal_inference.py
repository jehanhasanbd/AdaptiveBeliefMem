# scripts/minimal_inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json


class MinimalInference:
    def __init__(self):
        print("Loading model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, question, choices):
        """Generate answer for a question."""
        # Format choices
        choice_text = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])

        prompt = f"""You are a helpful assistant. Answer the following question based on the conversation history.

Question: {question}

Choices:
{choice_text}

Answer with just the letter (A, B, C, etc.) of the correct choice.

Answer:"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the answer letter
        response = response[len(text):].strip()

        return response


def main():
    # Load test data
    with open("data/processed/test.json", 'r') as f:
        test_data = json.load(f)

    # Initialize inference
    infer = MinimalInference()

    # Test on first few examples
    print("Testing on 5 examples...\n")
    correct = 0
    total = 0

    for i, example in enumerate(test_data[:5]):
        question = example["question"]
        choices = example["choices"]
        target = example["answer_index"]

        print(f"Example {i + 1}:")
        print(f"Question: {question[:100]}...")

        # Generate answer
        response = infer.generate(question, choices)

        # Parse response
        response_letter = response.strip()[0] if response else '?'
        pred_idx = ord(response_letter.upper()) - ord('A') if response_letter.isalpha() else -1

        is_correct = (pred_idx == target)
        if is_correct:
            correct += 1
        total += 1

        print(f"Predicted: {response_letter} (index {pred_idx})")
        print(f"Target: {chr(65 + target)} (index {target})")
        print(f"Correct: {is_correct}\n")

    print(f"Accuracy: {correct}/{total} = {correct / total:.3f}")


if __name__ == "__main__":
    main()