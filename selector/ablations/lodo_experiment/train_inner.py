"""Train one two-source-domain model and score both excluded domains."""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from domainbed import algorithms, datasets, hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader, InfiniteDataLoader
from selector.ablations.lodo_experiment.metrics import summarize


def split_hash(split):
    indices = np.arange(len(split), dtype=np.int64)
    current = split
    while hasattr(current, "keys") and hasattr(current, "underlying_dataset"):
        indices = np.asarray(current.keys, dtype=np.int64)[indices]
        current = current.underlying_dataset
    return hashlib.sha256(np.sort(indices).tobytes()).hexdigest()


@torch.no_grad()
def logits_and_labels(algorithm, loader, device):
    algorithm.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(algorithm.predict(x.to(device)))
        labels.append(y.to(device))
    algorithm.train()
    return torch.cat(logits), torch.cat(labels)


def run(args):
    excluded = sorted(args.exclude_envs)
    if len(excluded) != 2 or excluded[0] == excluded[1]:
        raise ValueError("Exactly two distinct excluded domains are required")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    if args.hparams_json:
        hparams = json.loads(args.hparams_json)
    elif args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(
            args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed),
        )
    if args.hparams_json and hparams.get("resnet50_augmix"):
        raise ValueError("Full-source AugMix hparams cannot be changed in an inner run")
    if not args.hparams_json:
        hparams["resnet50_augmix"] = False
    torch_home = Path(os.environ.setdefault(
        "TORCH_HOME", str(Path(args.data_dir).resolve().parent / "torch_cache"),
    ))
    if not hparams.get("resnet18") and not hparams.get("vit") and not hparams.get("dinov2"):
        weight = torch_home / "hub" / "checkpoints" / "resnet50-0676ba61.pth"
        if not weight.exists():
            raise FileNotFoundError(f"Missing offline pretrained weights: {weight}")
    seed = args.seed if args.seed is not None else misc.seed_hash(
        args.dataset, args.algorithm, excluded, args.hparams_seed, args.trial_seed,
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = getattr(datasets, args.dataset)(args.data_dir, excluded, hparams)
    if len(dataset) != 4 or excluded[-1] >= len(dataset):
        raise ValueError("This experiment requires four valid domains")

    in_splits = []
    for env_i, env in enumerate(dataset):
        _, in_split = misc.split_dataset(
            env, int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i),
        )
        if env_i in excluded:
            _, in_split = misc.split_dataset(
                in_split, 0, misc.seed_hash(args.trial_seed, env_i),
            )
        in_splits.append(in_split)
    train_loaders = []
    for env_i, split in enumerate(in_splits):
        if env_i in excluded:
            continue
        weights = misc.make_weights_for_balanced_classes(split) if hparams["class_balanced"] else None
        train_loaders.append(InfiniteDataLoader(
            split, weights, hparams["batch_size"], dataset.N_WORKERS,
        ))
    eval_loaders = {
        env_i: FastDataLoader(in_splits[env_i], 64, dataset.N_WORKERS)
        for env_i in excluded
    }
    algorithm = algorithms.get_algorithm_class(args.algorithm)(
        dataset.input_shape, dataset.num_classes, 2, hparams,
    ).to(device)
    domain_names = getattr(dataset, "ENVIRONMENTS", [str(i) for i in range(4)])
    config = {
        "dataset": args.dataset, "algorithm": args.algorithm,
        "excluded_envs": excluded, "hparams_seed": args.hparams_seed,
        "trial_seed": args.trial_seed, "seed": seed,
        "hparams": hparams, "steps": args.steps,
        "checkpoint_freq": args.checkpoint_freq,
        "holdout_fraction": args.holdout_fraction,
        "per_domain_batch_size": hparams["batch_size"],
        "training_domain_count": len(train_loaders),
        "total_batch_size": hparams["batch_size"] * len(train_loaders),
        "samples_per_5001_steps": hparams["batch_size"] * len(train_loaders) * args.steps,
        "split_hashes": {str(i): split_hash(s) for i, s in enumerate(in_splits)},
        "metric_profiles": ["logged_soft", "hard_15_bin_frequency_weighted"],
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    minibatches = zip(*train_loaders)
    started = time.perf_counter()
    train_seconds = 0.0
    evaluation_seconds = 0.0
    with (output / "metrics.jsonl").open("w") as destination:
        for step in range(args.steps):
            tick = time.perf_counter()
            batch = [(x.to(device), y.to(device)) for x, y in next(minibatches)]
            algorithm.update(batch, None)
            train_seconds += time.perf_counter() - tick
            if step % args.checkpoint_freq and step != args.steps - 1:
                continue
            tick = time.perf_counter()
            scores = {}
            for env_i in excluded:
                logits, labels = logits_and_labels(algorithm, eval_loaders[env_i], device)
                scores[str(env_i)] = {
                    "domain": domain_names[env_i], "split": "in",
                    "n_samples": int(labels.numel()),
                    **summarize(
                        logits, labels,
                        n_bins=hparams.get("kde_ece_n_bins", 15),
                        bandwidth=hparams.get("kde_ece_bandwidth", 0.1),
                    ),
                }
            evaluation_seconds += time.perf_counter() - tick
            row = {
                "step": step, "scores": scores,
                "train_seconds": train_seconds,
                "evaluation_seconds": evaluation_seconds,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0,
            }
            destination.write(json.dumps(row, sort_keys=True) + "\n")
            destination.flush()
            print(f"step={step} elapsed={row['elapsed_seconds']:.1f}s", flush=True)
    (output / "done").write_text("done\n")
    print(f"completed in {time.perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--exclude-envs", nargs=2, type=int, required=True)
    parser.add_argument("--hparams-seed", type=int, required=True)
    parser.add_argument("--trial-seed", type=int, required=True)
    parser.add_argument("--hparams-json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=5001)
    parser.add_argument("--checkpoint-freq", type=int, default=100)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
