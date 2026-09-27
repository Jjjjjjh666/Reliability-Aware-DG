"""Evaluate one post-hoc checkpoint-SWAD baseline run.

LossValley follows the official queue rules, while each saved endpoint model
approximates an unavailable dense within-segment averaged model.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from domainbed import datasets
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader, InfiniteDataLoader
from selector.ablations.lodo_experiment.evaluate_full_hard import full_setup
from selector.ablations.lodo_experiment.metrics import hard_summary
from selector.ablations.lodo_experiment.train_inner import logits_and_labels, split_hash
from selector.ablations.checkpoint_swad.selection import loss_valley


@torch.inference_mode()
def loader_nll(model, loader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        y = y.to(device)
        logits = model.predict(x.to(device))
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        total += len(y)
    return total_loss / total


def read_checkpoint(run_dir, step, hparams):
    path = Path(run_dir) / f"model_step{step}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["model_hparams"] != hparams:
        raise ValueError(f"Checkpoint hparams differ from log: {path}")
    return checkpoint["model_dict"]


def load_model_state(model, state):
    # GroupDRO creates q lazily on its first training update. A fresh model
    # therefore has an empty registered buffer, while every saved checkpoint
    # has one value per source domain. q is not used by predict(), but its
    # registered shape must match before strict state loading.
    if hasattr(model, "q") and "q" in state and model.q.shape != state["q"].shape:
        model.q.resize_(state["q"].shape)
    model.load_state_dict(state)


def average_checkpoints(model, run_dir, steps, hparams):
    first = read_checkpoint(run_dir, steps[0], hparams)
    load_model_state(model, first)
    names = set(dict(model.named_parameters()))
    average = {name: first[name].clone() for name in names}
    for count, step in enumerate(steps[1:], start=1):
        state = read_checkpoint(run_dir, step, hparams)
        for name in names:
            average[name].add_((state[name] - average[name]) / (count + 1))
    for name, parameter in model.named_parameters():
        parameter.data.copy_(average[name].to(parameter.device))
    return model


@torch.inference_mode()
def recalibrate_bn(
    model, args, hparams, target, data_dir, device, batches,
    seed_namespace="checkpoint_swad_bn",
):
    """Re-estimate BN statistics from source-domain training examples.

    ``seed_namespace`` makes the sample stream explicit.  The original
    checkpoint-SWAD experiment keeps its historical namespace; paired
    comparisons can supply a method-neutral namespace so every candidate
    receives identical source batches.
    """
    if hparams["freeze_bn"]:
        return 0
    seed = misc.seed_hash(seed_namespace, args["dataset"], args["algorithm"],
                          target, args["hparams_seed"], args["trial_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = getattr(datasets, args["dataset"])(data_dir, [target], hparams)
    loaders = []
    for i, env in enumerate(dataset):
        if i == target:
            continue
        _, in_split = misc.split_dataset(
            env, int(len(env) * args["holdout_fraction"]),
            misc.seed_hash(args["trial_seed"], i),
        )
        weights = misc.make_weights_for_balanced_classes(in_split) if hparams["class_balanced"] else None
        loaders.append(InfiniteDataLoader(in_split, weights, hparams["batch_size"], dataset.N_WORKERS))

    bn_modules = [module for module in model.modules()
                  if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if not bn_modules:
        return 0
    momenta = {module: module.momentum for module in bn_modules}
    previous_mode = model.training
    model.train()
    for module in bn_modules:
        module.reset_running_stats()
        module.momentum = None
    iterator = zip(*loaders)
    for _ in range(batches):
        x = torch.cat([x for x, _ in next(iterator)]).to(device)
        model.predict(x)
    for module, momentum in momenta.items():
        module.momentum = momentum
    model.train(previous_mode)
    return batches


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(full_run, data_dir, output_dir, bn_batches=500):
    started = time.perf_counter()
    full_run = Path(full_run).resolve()
    output = Path(output_dir)
    if not (full_run / "done").exists():
        raise ValueError(f"Full-source run is incomplete: {full_run}")
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint-SWAD requires a CUDA GPU")
    os.environ.setdefault("TORCH_HOME", str(Path(data_dir).resolve().parent / "torch_cache"))
    rows, args, hparams, target, dataset, in_splits, out_splits, model, device = full_setup(
        full_run, data_dir,
    )
    config = {
        "profile": "checkpoint_swad_lossvalley_deterministic_v1",
        "full_run": str(full_run), "dataset": args["dataset"],
        "data_dir": str(Path(data_dir).resolve()),
        "algorithm": args["algorithm"], "target_domain": target,
        "hparams_seed": args["hparams_seed"], "trial_seed": args["trial_seed"],
        "hparams": hparams, "bn_batches": bn_batches,
        "validation": "mean of three deterministic source-out NLLs",
        "source_split_hashes": {str(i): split_hash(out_splits[i])
                                for i in range(4) if i != target},
        "target_split_hash": split_hash(in_splits[target]),
    }
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(f"Output configuration differs: {output}")
    else:
        write_json(config_path, config)
    if (output / "done").exists():
        if not (output / "result.json").exists() or not (output / "averaged_model.pt").exists():
            raise ValueError(f"Incomplete done marker: {output}")
        print(f"already complete: {output}", flush=True)
        return

    source_domains = [i for i in range(4) if i != target]
    validation_path = output / "source_validation.jsonl"
    existing = []
    if validation_path.exists():
        with validation_path.open() as source:
            existing = [json.loads(line) for line in source if line.strip()]
        for index, record in enumerate(existing):
            if index >= len(rows) or record["step"] != rows[index]["step"]:
                raise ValueError(f"Invalid validation resume prefix: {validation_path}")
            if set(record["per_domain_nll"]) != {str(i) for i in source_domains}:
                raise ValueError(f"Missing source domain: {validation_path}")
            if not math.isfinite(record["source_nll"]):
                raise ValueError(f"Nonfinite validation loss: {validation_path}")
    if len(existing) < len(rows):
        loaders = {i: FastDataLoader(out_splits[i], 64, dataset.N_WORKERS)
                   for i in source_domains}
        with validation_path.open("a") as destination:
            for row in rows[len(existing):]:
                step = row["step"]
                tick = time.perf_counter()
                load_model_state(model, read_checkpoint(full_run, step, hparams))
                per_domain = {str(i): loader_nll(model, loaders[i], device)
                              for i in source_domains}
                mean_loss = sum(per_domain.values()) / 3
                if not math.isfinite(mean_loss):
                    raise ValueError(f"Nonfinite source loss at step {step}: {full_run}")
                record = {"step": step, "source_nll": mean_loss,
                          "per_domain_nll": per_domain,
                          "evaluation_seconds": time.perf_counter() - tick}
                destination.write(json.dumps(record, sort_keys=True) + "\n")
                destination.flush()
                existing.append(record)
                print(f"validation step={step} loss={mean_loss:.5f}", flush=True)

    selection = loss_valley([record["source_nll"] for record in existing])
    selection["selected_steps"] = [rows[i]["step"] for i in selection["indices"]]
    selection["converge_step"] = (
        rows[selection["converge_index"]]["step"]
        if selection["converge_index"] is not None else None
    )
    selection["unique_checkpoints"] = len(set(selection["selected_steps"]))
    write_json(output / "selection.json", selection)

    tick = time.perf_counter()
    average_checkpoints(model, full_run, selection["selected_steps"], hparams)
    average_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    actual_bn_batches = recalibrate_bn(model, args, hparams, target, data_dir, device, bn_batches)
    bn_seconds = time.perf_counter() - tick

    tick = time.perf_counter()
    model_path = output / "averaged_model.pt"
    temporary_model = output / "averaged_model.pt.tmp"
    torch.save({"model_dict": model.cpu().state_dict(), "config": config,
                "selection": selection}, temporary_model)
    os.replace(temporary_model, model_path)
    model.to(device)
    save_seconds = time.perf_counter() - tick

    tick = time.perf_counter()
    target_loader = FastDataLoader(in_splits[target], 64, dataset.N_WORKERS)
    target_logits, target_labels = logits_and_labels(model, target_loader, device)
    target_score = hard_summary(target_logits, target_labels)
    if not all(math.isfinite(value) for value in target_score.values()):
        raise ValueError(f"Nonfinite target score: {full_run}")
    result = {
        "profile": config["profile"], "dataset": args["dataset"],
        "algorithm": args["algorithm"], "target_domain": target,
        "hparams_seed": args["hparams_seed"], "trial_seed": args["trial_seed"],
        "full_run": str(full_run), "selected_steps": selection["selected_steps"],
        "converge_step": selection["converge_step"],
        "dead_valley": selection["dead_valley"],
        "fallback_last": selection["fallback_last"],
        "target_split": "in", "target_split_hash": split_hash(in_splits[target]),
        "target_samples": len(in_splits[target]),
        "target_accuracy": target_score["accuracy"],
        "target_nll": target_score["nll"],
        "target_ece": target_score["hard_ece"],
        "target_cwece": target_score["hard_cwece"],
        "validation_seconds": sum(record["evaluation_seconds"] for record in existing),
        "averaging_seconds": average_seconds, "bn_seconds": bn_seconds,
        "bn_batches": actual_bn_batches, "model_save_seconds": save_seconds,
        "target_evaluation_seconds": time.perf_counter() - tick,
        "elapsed_this_attempt_seconds": time.perf_counter() - started,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "model_bytes": model_path.stat().st_size,
    }
    write_json(output / "result.json", result)
    (output / "done").write_text("done\n")
    print(json.dumps(result, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-run", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bn-batches", type=int, default=500)
    args = parser.parse_args()
    if args.bn_batches < 1:
        parser.error("--bn-batches must be positive")
    run(args.full_run, args.data_dir, args.output_dir, args.bn_batches)


if __name__ == "__main__":
    main()
