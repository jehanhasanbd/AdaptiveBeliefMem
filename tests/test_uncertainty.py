# tests/test_uncertainty.py
import unittest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.uncertainty import UncertaintyEstimator


class TestUncertainty(unittest.TestCase):

    def setUp(self):
        self.config = {
            "belief_dim": 768,
            "hidden_dim": 256,
            "use_history": False
        }

    def test_initialization(self):
        estimator = UncertaintyEstimator(self.config)
        self.assertIsNotNone(estimator)

    def test_forward_shape(self):
        estimator = UncertaintyEstimator(self.config)
        batch_size = 4
        belief = torch.randn(batch_size, self.config["belief_dim"])

        uncertainty = estimator(belief)

        self.assertEqual(uncertainty.shape, (batch_size, 1))
        self.assertTrue((uncertainty >= 0).all() and (uncertainty <= 1).all())

    def test_uncertainty_range(self):
        estimator = UncertaintyEstimator(self.config)
        belief = torch.randn(1, self.config["belief_dim"])

        u = estimator(belief)
        self.assertGreaterEqual(u.item(), 0)
        self.assertLessEqual(u.item(), 1)


if __name__ == "__main__":
    unittest.main()