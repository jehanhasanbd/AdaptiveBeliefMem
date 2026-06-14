# tests/test_integration.py
import unittest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.belief_state import BeliefState


class TestIntegration(unittest.TestCase):

    def test_belief_state_initialization(self):
        belief_state = BeliefState(768, "cpu")
        belief = belief_state.initialize()

        self.assertIsNotNone(belief)
        self.assertEqual(belief.shape, (1, 768))

    def test_belief_update(self):
        belief_state = BeliefState(768, "cpu")
        belief = belief_state.initialize()

        new_belief = torch.randn(1, 768)
        belief_state.update(new_belief, topic="test")

        self.assertEqual(len(belief_state.snapshots), 2)  # Initial + update

    def test_belief_history(self):
        belief_state = BeliefState(768, "cpu")
        belief_state.initialize()

        for i in range(5):
            new_belief = torch.randn(1, 768)
            belief_state.update(new_belief)

        history = belief_state.get_belief_history()
        self.assertEqual(history.shape[0], 6)  # 5 updates + initial

    def test_temporal_snapshots(self):
        belief_state = BeliefState(768, "cpu")
        belief_state.initialize()

        for i in range(10):
            new_belief = torch.randn(1, 768)
            belief_state.update(new_belief, topic=f"topic_{i}")

        snapshots = belief_state.get_temporal_snapshots(3)
        self.assertEqual(len(snapshots), 3)


if __name__ == "__main__":
    unittest.main()