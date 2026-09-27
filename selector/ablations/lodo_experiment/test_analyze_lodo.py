import unittest

from selector.ablations.lodo_experiment.analyze_lodo import bootstrap_plan, summarize_paired


class AnalyzeLodoTests(unittest.TestCase):
    def test_fixed_bootstrap_keeps_four_targets_in_each_block(self):
        values = []
        for hparams in range(3):
            for trial in range(3):
                for target in range(4):
                    values.append({
                        "dataset": "OfficeHome", "target": target,
                        "hparams": hparams, "trial": trial,
                        "value": float(hparams + trial),
                    })
        samples = bootstrap_plan("fixed_hparams", ["OfficeHome"], 100, 7)
        mean, low, high = summarize_paired(
            values, "fixed_hparams", ["OfficeHome"], samples,
        )
        self.assertAlmostEqual(mean, 2.0)
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)


if __name__ == "__main__":
    unittest.main()
