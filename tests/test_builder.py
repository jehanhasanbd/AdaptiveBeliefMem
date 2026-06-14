# tests/test_builder.py
import unittest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.builder_3b import QuantizedBuilder3B


class TestBuilder(unittest.TestCase):

    def setUp(self):
        self.config = {
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",  # Use smaller model for testing
            "belief_dim": 768,
            "hidden_dim": 256,
            "quantization": {
                "load_in_4bit": False,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True
            },
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
            },
            "training": {
                "max_seq_length": 512
            }
        }

    def test_initialization(self):
        builder = QuantizedBuilder3B(self.config)
        self.assertIsNotNone(builder)

    def test_forward_shape(self):
        builder = QuantizedBuilder3B(self.config)
        batch_size = 1
        belief = torch.zeros(batch_size, self.config["belief_dim"])

        # Test with a dummy session
        session_text = "This is a test session."
        new_belief, embedding = builder(belief, session_text)

        self.assertEqual(new_belief.shape, (batch_size, self.config["belief_dim"]))
        self.assertEqual(embedding.shape, (batch_size, self.config["belief_dim"]))


if __name__ == "__main__":
    unittest.main()