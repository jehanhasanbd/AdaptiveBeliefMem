# tests/test_retriever.py
import unittest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.retriever import BeliefRetriever


class TestRetriever(unittest.TestCase):

    def setUp(self):
        self.config = {
            "belief_dim": 768,
            "hidden_dim": 256,
            "top_k": {
                "dynamic": True,
                "base_k": 3
            }
        }

    def test_initialization(self):
        retriever = BeliefRetriever(self.config)
        self.assertIsNotNone(retriever)

    def test_hard_topk_selection(self):
        retriever = BeliefRetriever(self.config)
        num_beliefs = 10
        beliefs = torch.randn(num_beliefs, self.config["belief_dim"])
        uncertainties = torch.rand(num_beliefs)

        selected_beliefs, mask = retriever.hard_top_k_selection(beliefs, uncertainties, num_beliefs)

        self.assertEqual(selected_beliefs.shape[0], mask.sum().item())

    def test_trace_generation(self):
        retriever = BeliefRetriever(self.config)
        beliefs = torch.randn(5, self.config["belief_dim"])

        trace = retriever.generate_trace(beliefs, 0, 0.5, "test")

        self.assertIn("trace_id", trace)
        self.assertIn("entities", trace)
        self.assertIn("relations", trace)
        self.assertIn("uncertainty_score", trace)


if __name__ == "__main__":
    unittest.main()