import unittest

import numpy as np
import pandas as pd

from selector.rules import make_selection_rules, select_checkpoints


def checkpoint(step, accuracy, nll, cwece, ece=0.1):
    return {
        "dataset": "PACS",
        "test_domain": "art_painting",
        "method_family": "ERM",
        "method": "ERM",
        "seed": 7,
        "hparams_seed": 0,
        "trial_seed": 0,
        "step": step,
        "run_dir": "/tmp/run",
        "results_path": "/tmp/run/results.jsonl",
        "model_path": "",
        "source_in_acc": accuracy,
        "source_out_acc": accuracy,
        "source_out_ece": ece,
        "source_out_cwece": cwece,
        "source_out_nll": nll,
        "target_acc": np.nan,
        "target_ece": np.nan,
        "target_cwece": np.nan,
        "target_nll": np.nan,
        "target_worst_domain": np.nan,
        "target_worst_class": np.nan,
        "source_out_acc_star": 0.90,
    }


class SelectionRulesTest(unittest.TestCase):
    def test_reference_rule_uses_only_accuracy_feasible_checkpoints(self):
        table = pd.DataFrame(
            [
                checkpoint(100, 0.900, 0.80, 0.50),
                checkpoint(200, 0.898, 0.60, 0.40),
                checkpoint(300, 0.895, 0.70, 0.45),
                checkpoint(400, 0.880, 0.10, 0.10),
            ]
        )
        rules = make_selection_rules(
            objective_sets=["NC"],
            distances=["linf"],
            delta_pp=0.5,
        )
        selected = select_checkpoints(table, rules)

        source = selected[selected["selection_rule"] == "source_acc"].iloc[0]
        ac = selected[selected["selection_rule"] == "ac_nc_linf_d0p5pp"].iloc[0]
        self.assertEqual(source["step"], 100)
        self.assertEqual(ac["step"], 200)
        self.assertEqual(ac["feasible_count"], 3)
        self.assertAlmostEqual(ac["source_out_acc_gap_pp"], 0.2)

    def test_constant_objectives_use_accuracy_then_earlier_step_tie_break(self):
        table = pd.DataFrame(
            [
                checkpoint(0, 0.900, 0.5, 0.2),
                checkpoint(100, 0.900, 0.5, 0.2),
                checkpoint(200, 0.898, 0.5, 0.2),
            ]
        )
        for distance in ("l1", "l2", "linf"):
            with self.subTest(distance=distance):
                rules = make_selection_rules(
                    objective_sets=["NC"],
                    distances=[distance],
                    delta_pp=0.5,
                )
                selected = select_checkpoints(table, rules)
                ac = selected[selected["selector_family"] == "ac"].iloc[0]
                self.assertEqual(ac["step"], 0)
                self.assertEqual(ac["selection_score"], 0.0)

    def test_legacy_objective_aliases_are_canonicalized(self):
        table = pd.DataFrame(
            [
                checkpoint(0, 0.900, 0.7, 0.3),
                checkpoint(100, 0.899, 0.6, 0.2),
            ]
        )
        rules = make_selection_rules(
            objective_sets=["A"],
            distances=["linf"],
            delta_pp=0.5,
        )
        selected = select_checkpoints(table, rules)
        ac = selected[selected["selector_family"] == "ac"].iloc[0]
        self.assertEqual(ac["objective_set"], "NC")
        self.assertEqual(ac["selection_rule"], "ac_nc_linf_d0p5pp")


if __name__ == "__main__":
    unittest.main()
