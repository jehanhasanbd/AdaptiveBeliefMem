# scripts/test_trained_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json


class TrainedModelTester:
    def __init__(self, base_model="Qwen/Qwen2.5-3B-Instruct", lora_path="outputs/best_lora_model"):
        print("Loading base model...")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        print(f"Loading LoRA weights from {lora_path}...")
        self.model = PeftModel.from_pretrained(self.base_model, lora_path)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict(self, context, question, choices):
        choice_text = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices[:10])])

        prompt = f"""Context: {context}

Question: {question}

Choices:
{choice_text}

Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        import re
        match = re.search(r'([A-J])', response.strip())
        if match:
            return match.group(1)
        return "?"


def main():
    # Load test data
    with open("data/processed/test.json", 'r') as f:
        test_data = json.load(f)

    # Initialize tester
    tester = TrainedModelTester()

    # Test on a few examples
    print("\nTesting trained model:\n")
    for i, example in enumerate(test_data[:10]):
        context = " ".join(example["session_texts"][:3])
        question = example["question"]
        choices = example["choices"]
        true_answer = chr(65 + example["answer_index"])

        prediction = tester.predict(context, question, choices)

        print(f"{i + 1}. Q: {question[:80]}...")
        print(f"   Prediction: {prediction}, True: {true_answer}")
        print(f"   Correct: {prediction == true_answer}\n")


if __name__ == "__main__":
    main()