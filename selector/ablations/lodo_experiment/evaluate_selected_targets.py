"""Evaluate hard-bin target metrics only after selection is frozen."""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

from domainbed.lib.fast_data_loader import FastDataLoader
from selector.ablations.lodo_experiment.evaluate_full_hard import full_setup, load_step
from selector.ablations.lodo_experiment.metrics import hard_summary
from selector.ablations.lodo_experiment.train_inner import logits_and_labels, split_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ.setdefault("TORCH_HOME", str(Path(args.data_dir).resolve().parent / "torch_cache"))
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    with open(args.selection, newline="") as source:
        selected = list(csv.DictReader(source))
    if not selected or any(row["metric_profile"] != "hard_deterministic" for row in selected):
        raise ValueError("This evaluator requires a frozen hard-profile selection")
    groups = defaultdict(set)
    for row in selected:
        groups[row["full_source_run"]].add(int(row["selected_step"]))

    evaluated = {}
    for run_dir, steps in groups.items():
        _, full_args, _, target, dataset, in_splits, _, model, device = full_setup(
            run_dir, args.data_dir,
        )
        loader = FastDataLoader(in_splits[target], 64, dataset.N_WORKERS)
        target_hash = split_hash(in_splits[target])
        for step in sorted(steps):
            load_step(model, run_dir, step)
            logits, labels = logits_and_labels(model, loader, device)
            score = hard_summary(logits, labels)
            evaluated[(run_dir, step)] = {
                "target_accuracy": score["accuracy"],
                "target_nll": score["nll"],
                "target_ece": score["hard_ece"],
                "target_cwece": score["hard_cwece"],
                "target_split_hash": target_hash,
            }
            print(run_dir, step, flush=True)
    fields = list(selected[0]) + list(next(iter(evaluated.values())))
    with open(args.output, "w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({**row, **evaluated[(row["full_source_run"], int(row["selected_step"]))]})
    print(f"wrote {len(selected)} selected results to {args.output}")


if __name__ == "__main__":
    main()
