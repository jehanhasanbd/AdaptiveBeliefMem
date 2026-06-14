# src/reasoner/frozen_llm_3b.py (fixed)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, List, Dict, Any
import gc
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class QuantizedReasoner3B:
    """
    Frozen Qwen2.5-3B-Instruct reasoner with 4-bit quantization.
    Completely frozen - no gradient updates.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config["quantization"]["load_in_4bit"],
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=config["quantization"]["bnb_4bit_use_double_quant"],
        )

        # Load model
        print("Loading Reasoner model (Qwen2.5-3B-Instruct)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["base_model"],
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_new_tokens = 512
        self.temperature = 0.1  # Deterministic for evaluation

        print(f"Reasoner loaded. Device: {self.model.device}")

    def generate(self, prompt: str) -> str:
        """Generate answer from prompt."""
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant that answers questions based on provided information."},
            {"role": "user", "content": prompt}
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config["training"]["max_seq_length"]
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from response
        response = response[len(text):].strip()

        return response

    def clear_cache(self):
        """Clear GPU cache."""
        torch.cuda.empty_cache()
        gc.collect()