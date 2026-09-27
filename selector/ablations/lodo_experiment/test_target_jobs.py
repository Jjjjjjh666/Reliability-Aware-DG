import csv
import tempfile
import unittest
from pathlib import Path

from selector.ablations.lodo_experiment.run_target_jobs import TARGET_FIELDS, combine, shard_path, write_csv


class TargetJobTests(unittest.TestCase):
    def test_combines_shards_in_frozen_selection_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = [
                "selection_scope", "dataset", "algorithm", "target_domain",
                "hparams_seed", "trial_seed", "selector", "selected_step",
                "full_source_run", "metric_profile",
            ]
            rows = []
            for target in (1, 0):
                for selector in ("AC", "LODO-Acc"):
                    rows.append({
                        "selection_scope": "fixed_hparams", "dataset": "OfficeHome",
                        "algorithm": "ERM", "target_domain": str(target),
                        "hparams_seed": "0", "trial_seed": "0", "selector": selector,
                        "selected_step": "100", "full_source_run": f"run-{target}",
                        "metric_profile": "hard_deterministic",
                    })
            by_run = {}
            for row in rows:
                by_run.setdefault(row["full_source_run"], []).append(row)
            for run_rows in by_run.values():
                path = shard_path(root / "shards", run_rows)
                scored = [{**row, **{field: "0.5" for field in TARGET_FIELDS}} for row in run_rows]
                write_csv(path, scored, fields + list(TARGET_FIELDS))
                path.with_suffix(".done").write_text("done\n")
            output = root / "target.csv"
            count, _ = combine(rows, fields, root / "shards", output)
            with output.open(newline="") as source:
                combined = list(csv.DictReader(source))
            self.assertEqual(count, 4)
            self.assertEqual(
                [(row["target_domain"], row["selector"]) for row in combined],
                [(row["target_domain"], row["selector"]) for row in rows],
            )
            self.assertTrue(output.with_suffix(".csv.sha256").exists())


if __name__ == "__main__":
    unittest.main()
