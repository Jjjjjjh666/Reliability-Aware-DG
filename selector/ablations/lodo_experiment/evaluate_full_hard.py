"""Recompute hard-bin source validation metrics from saved full-source weights."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from domainbed import algorithms, datasets
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader
from selector.ablations.lodo_experiment.metrics import hard_summary
from selector.ablations.lodo_experiment.pipeline import EXPECTED_STEPS, read_jsonl
from selector.ablations.lodo_experiment.train_inner import logits_and_labels, split_hash


def full_setup(run_dir, data_dir):
    rows = read_jsonl(Path(run_dir) / "results.jsonl")
    if tuple(row["step"] for row in rows) != EXPECTED_STEPS:
        raise ValueError("Full-source checkpoint grid is incomplete")
    args = rows[0]["args"]
    hparams = rows[0]["hparams"]
    target = args["test_envs"]
    if len(target) != 1:
        raise ValueError("Expected one real target domain")
    # Full-source training augments source domains, including their out splits
    # in its logged evaluations. Re-evaluation uses deterministic transforms
    # for every domain while keeping the exact same trial-seed sample indices.
    dataset = getattr(datasets, args["dataset"])(data_dir, list(range(4)), hparams)
    in_splits, out_splits = [], []
    for i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(
            env, int(len(env) * args["holdout_fraction"]),
            misc.seed_hash(args["trial_seed"], i),
        )
        if i in target:
            _, in_ = misc.split_dataset(in_, 0, misc.seed_hash(args["trial_seed"], i))
        in_splits.append(in_)
        out_splits.append(out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = algorithms.get_algorithm_class(args["algorithm"])(
        dataset.input_shape, dataset.num_classes, 3, hparams,
    ).to(device)
    return rows, args, hparams, target[0], dataset, in_splits, out_splits, model, device


def load_step(model, run_dir, step):
    path = Path(run_dir) / f"model_step{step}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_dict"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-run", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, help="Smoke-test prefix; use a separate output path")
    args = parser.parse_args()
    os.environ.setdefault("TORCH_HOME", str(Path(args.data_dir).resolve().parent / "torch_cache"))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if not (Path(args.full_run) / "done").exists():
        raise ValueError("Full-source run is incomplete")
    rows, full_args, hparams, target, dataset, in_splits, out_splits, model, device = full_setup(
        args.full_run, args.data_dir,
    )
    source_domains = [i for i in range(4) if i != target]
    loaders = {
        i: FastDataLoader(out_splits[i], 64, dataset.N_WORKERS)
        for i in source_domains
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as destination:
        for row in rows[:args.max_steps]:
            step = row["step"]
            load_step(model, args.full_run, step)
            scores = {}
            for i in source_domains:
                logits, labels = logits_and_labels(model, loaders[i], device)
                scores[i] = hard_summary(logits, labels)
            record = {
                "step": step,
                "source_accuracy": float(np.mean([scores[i]["accuracy"] for i in source_domains])),
                "source_nll": float(np.mean([scores[i]["nll"] for i in source_domains])),
                "source_ece": float(np.mean([scores[i]["hard_ece"] for i in source_domains])),
                "source_cwece": float(np.mean([scores[i]["hard_cwece"] for i in source_domains])),
                "source_split_hashes": {str(i): split_hash(out_splits[i]) for i in source_domains},
                "metric_profile": "hard_deterministic",
            }
            destination.write(json.dumps(record, sort_keys=True) + "\n")
            destination.flush()
            print(f"step={step}", flush=True)
    output.with_suffix(".done").write_text("done\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
