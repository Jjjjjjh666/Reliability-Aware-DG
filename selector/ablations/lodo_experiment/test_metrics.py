import math
import unittest

import torch

from selector.ablations.lodo_experiment.metrics import summarize


class MetricTests(unittest.TestCase):
    def test_hard_bin_and_frequency_weighting(self):
        logits = torch.tensor([[math.log(4), 0.0], [0.0, math.log(4)]])
        labels = torch.tensor([0, 0])
        result = summarize(logits, labels, n_bins=15)
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["hard_ece"], 0.3, places=6)
        self.assertAlmostEqual(result["hard_cwece"], 0.5, places=6)

    def test_accuracy_uses_logits_when_softmax_rounds_to_a_tie(self):
        logits = torch.tensor([[-1e-8, 0.0]], dtype=torch.float32)
        labels = torch.tensor([1])
        probabilities = torch.softmax(logits, dim=1)
        self.assertEqual(probabilities[0, 0].item(), probabilities[0, 1].item())
        result = summarize(logits, labels)
        self.assertEqual(result["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
