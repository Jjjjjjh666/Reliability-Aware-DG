import json
import tempfile
import unittest
from pathlib import Path

from selector.config import ResultSource
from selector.data import load_checkpoint_table


def record(step, source_acc, target_acc):
    domain_rows = []
    for env_i, name, role, accuracy in [
        (0, "art_painting", "target", target_acc),
        (1, "cartoon", "source", source_acc),
        (2, "photo", "source", source_acc),
        (3, "sketch", "source", source_acc),
    ]:
        domain_rows.extend(
            [
                {
                    "domain_index": env_i,
                    "domain_name": name,
                    "role": role,
                    "split": "in",
                    "accuracy": accuracy,
                    "nll": 0.8,
                    "ece": 0.1,
                    "classwise_ece": 0.2,
                },
                {
                    "domain_index": env_i,
                    "domain_name": name,
                    "role": role,
                    "split": "out",
                    "accuracy": accuracy,
                    "nll": 0.7,
                    "ece": 0.09,
                    "classwise_ece": 0.18,
                },
            ]
        )
    return {
        "step": step,
        "args": {
            "algorithm": "ERM",
            "dataset": "PACS",
            "hparams_seed": 0,
            "trial_seed": 0,
            "seed": 123,
            "test_envs": [0],
        },
        "env0_in_acc": target_acc,
        "env0_out_acc": target_acc,
        "env1_in_acc": source_acc,
        "env1_out_acc": source_acc,
        "env2_in_acc": source_acc,
        "env2_out_acc": source_acc,
        "env3_in_acc": source_acc,
        "env3_out_acc": source_acc,
        "per_domain_calibration": domain_rows,
        "per_domain_class_calibration": [],
    }


class DataLoadingTest(unittest.TestCase):
    def test_loads_a_completed_domainbed_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            with (run_dir / "results.jsonl").open("w") as handle:
                for row in [record(0, 0.90, 0.80), record(100, 0.91, 0.81)]:
                    handle.write(json.dumps(row) + "\n")
            (run_dir / "done").write_text("done")

            table = load_checkpoint_table(
                [ResultSource(Path(tmp))],
                require_done=True,
            )

        self.assertEqual(len(table), 2)
        self.assertEqual(table["dataset"].unique().tolist(), ["PACS"])
        self.assertEqual(table["method"].unique().tolist(), ["ERM"])
        self.assertEqual(table["source_out_acc_star"].unique().tolist(), [0.91])
        self.assertEqual(table["test_domain"].unique().tolist(), ["art_painting"])


if __name__ == "__main__":
    unittest.main()
