# tests/test_hard_topk.py
import unittest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.ste import StraightThroughEstimator, HardTopKSTE


class TestHardTopK(unittest.TestCase):

    def test_ste_forward(self):
        ste = StraightThroughEstimator()
        inputs = torch.randn(10, 5)
        mask = torch.zeros(10, 1)
        mask[:3] = 1

        output = ste.apply(inputs, mask)

        # Check that unselected positions are zeroed
        self.assertTrue((output[3:] == 0).all())
        self.assertTrue((output[:3] == inputs[:3]).all())

    def test_ste_backward(self):
        ste = StraightThroughEstimator()
        inputs = torch.randn(10, 5, requires_grad=True)
        mask = torch.zeros(10, 1)
        mask[:3] = 1

        output = ste.apply(inputs, mask)
        loss = output.sum()
        loss.backward()

        # Gradients should flow through selected positions
        self.assertIsNotNone(inputs.grad)
        self.assertTrue((inputs.grad[3:] == 0).all())

    def test_hard_topk_ste(self):
        k = 3
        selector = HardTopKSTE(k)
        scores = torch.randn(10)
        values = torch.randn(10, 5)

        selected, mask = selector(scores, values)

        self.assertEqual(selected.shape, (10, 5))
        self.assertEqual(mask.sum().item(), k)


if __name__ == "__main__":
    unittest.main()