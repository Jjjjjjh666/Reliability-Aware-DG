import json
import tempfile
import unittest
from pathlib import Path

from selector.ablations.lodo_experiment.pipeline import (
    EXPECTED_STEPS, choose_ac, choose_accuracy, cv_series, fold_at_step,
    make_plan, select, source_at_step,
)


class SelectionTests(unittest.TestCase):
    def test_accuracy_tie_uses_earlier_step(self):
        series = [{"step": 100, "accuracy": 0.9}, {"step": 0, "accuracy": 0.9}]
        self.assertEqual(choose_accuracy(series)["step"], 0)

    def test_ac_uses_accuracy_band_then_max_normalized_component(self):
        series = [
            {"step": 0, "accuracy": 0.90, "nll": 1.0, "cwece": 3.0},
            {"step": 100, "accuracy": 0.897, "nll": 2.0, "cwece": 2.0},
            {"step": 200, "accuracy": 0.90, "nll": 3.0, "cwece": 1.0},
            {"step": 300, "accuracy": 0.894, "nll": 0.0, "cwece": 0.0},
        ]
        chosen, count = choose_ac(series)
        self.assertEqual((chosen["step"], count), (100, 3))

    def test_source_score_excludes_real_target(self):
        row = {f"env{i}_out_acc": (0.0 if i == 0 else i / 10) for i in range(4)}
        row["step"] = 100
        row["per_domain_calibration"] = [
            {"domain_index": i, "split": "out", "nll": i,
             "classwise_ece": i / 100} for i in range(4)
        ]
        source = source_at_step(row, target=0)
        self.assertAlmostEqual(source["accuracy"], 0.2)
        self.assertAlmostEqual(source["nll"], 2.0)

    def test_missing_inner_checkpoint_is_error(self):
        with self.assertRaisesRegex(ValueError, "step grid"):
            fold_at_step([{"step": 0, "scores": {"1": {"split": "in"}}}], 1)

    def test_cv_uses_only_pseudo_target_side_of_each_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            for heldout in (1, 2, 3):
                folder = Path(directory) / "OfficeHome" / "ERM" / "h0_t0" / f"ex0_{heldout}"
                folder.mkdir(parents=True)
                (folder / "done").write_text("done")
                (folder / "config.json").write_text(json.dumps({
                    "excluded_envs": [0, heldout], "dataset": "OfficeHome",
                    "algorithm": "ERM", "hparams_seed": 0, "trial_seed": 0,
                    "steps": 5001, "checkpoint_freq": 100,
                }))
                with (folder / "metrics.jsonl").open("w") as destination:
                    for step in EXPECTED_STEPS:
                        destination.write(json.dumps({"step": step, "scores": {
                            "0": {"split": "in", "accuracy": 0.0, "nll": 99.0,
                                  "logged_soft_cwece": 99.0},
                            str(heldout): {"split": "in", "accuracy": heldout / 10,
                                           "nll": heldout, "logged_soft_cwece": heldout / 100},
                        }}) + "\n")
            cv = cv_series(0, "OfficeHome", "ERM", 0, 0, directory, {})
            self.assertEqual(len(cv), 51)
            self.assertAlmostEqual(cv[0]["accuracy"], 0.2)
            self.assertAlmostEqual(cv[0]["nll"], 2.0)

    def test_complete_one_algorithm_selects_both_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = {}
            inner_root = Path(directory) / "inner"
            hard_root = Path(directory) / "hard"
            for h in range(3):
                for trial in range(3):
                    for target in range(4):
                        rows = []
                        for step in EXPECTED_STEPS:
                            accuracy = 0.90 if step == 0 else 0.89 if step == 100 else 0.80
                            row = {
                                "step": step,
                                "args": {"holdout_fraction": 0.2},
                                "hparams": {"batch_size": 32},
                                "per_domain_calibration": [
                                    {"domain_index": i, "split": "out", "nll": 1.0,
                                     "classwise_ece": 0.01} for i in range(4)
                                ] + [{"domain_index": target, "split": "in",
                                      "nll": 2.0, "ece": 0.02, "classwise_ece": 0.03}],
                            }
                            for i in range(4):
                                row[f"env{i}_out_acc"] = accuracy
                                row[f"env{i}_in_acc"] = 0.77
                            rows.append(row)
                        path = Path(directory) / f"full_{target}_{h}_{trial}" / "results.jsonl"
                        runs[("OfficeHome", "ERM", target, h, trial)] = (path, rows)
                        hard_path = hard_root / "OfficeHome" / "ERM" / f"h{h}_t{trial}" / f"target_{target}.jsonl"
                        hard_path.parent.mkdir(parents=True, exist_ok=True)
                        with hard_path.open("w") as destination:
                            for step in EXPECTED_STEPS:
                                destination.write(json.dumps({
                                    "step": step, "metric_profile": "hard_deterministic",
                                    "source_accuracy": 0.95 if step == 200 else 0.8,
                                    "source_nll": 1.0, "source_cwece": 0.01,
                                }) + "\n")
                        hard_path.with_suffix(".done").write_text("done")
                    for left in range(4):
                        for right in range(left + 1, 4):
                            folder = inner_root / "OfficeHome" / "ERM" / f"h{h}_t{trial}" / f"ex{left}_{right}"
                            folder.mkdir(parents=True)
                            (folder / "done").write_text("done")
                            (folder / "config.json").write_text(json.dumps({
                                "excluded_envs": [left, right], "dataset": "OfficeHome",
                                "algorithm": "ERM", "hparams_seed": h, "trial_seed": trial,
                                "steps": 5001, "checkpoint_freq": 100,
                                "hparams": {"batch_size": 32}, "holdout_fraction": 0.2,
                            }))
                            with (folder / "metrics.jsonl").open("w") as destination:
                                for step in EXPECTED_STEPS:
                                    score = {"split": "in", "accuracy": 0.9 if step == 100 else 0.5,
                                             "nll": 1.0, "logged_soft_cwece": 0.01,
                                             "hard_cwece": 0.02}
                                    destination.write(json.dumps({
                                        "step": step,
                                        "scores": {str(left): score, str(right): score},
                                    }) + "\n")
            self.assertEqual(len(make_plan(runs, inner_root, ["OfficeHome"], ["ERM"])), 54)
            self.assertEqual(
                len(make_plan(
                    runs, inner_root, ["OfficeHome"], ["ERM"],
                    hparams_seeds=[0], trial_seeds=[0],
                )),
                6,
            )
            selected = select(runs, inner_root, ["OfficeHome"], ["ERM"])
            self.assertEqual(len(selected), 192)
            fixed = [row for row in selected if row["selection_scope"] == "fixed_hparams"]
            self.assertEqual(len(fixed), 144)
            self.assertEqual({row["selected_step"] for row in fixed if row["selector"] == "LODO-Acc"}, {100})
            self.assertEqual({row["selected_step"] for row in fixed if row["selector"] == "Source-Acc"}, {0})
            hard_selected = select(runs, inner_root, ["OfficeHome"], ["ERM"], "hard", hard_root)
            self.assertEqual(len(hard_selected), 192)
            self.assertEqual({row["selected_step"] for row in hard_selected
                              if row["selector"] == "Source-Acc"}, {200})
            self.assertTrue(all("target_accuracy" not in row for row in hard_selected))
            fixed_only = select(
                runs, inner_root, ["OfficeHome"], ["ERM"],
                hparams_seeds=[0], trial_seeds=[0],
                selection_scopes=("fixed_hparams",),
            )
            self.assertEqual(len(fixed_only), 16)
            self.assertEqual({row["selection_scope"] for row in fixed_only}, {"fixed_hparams"})


if __name__ == "__main__":
    unittest.main()
