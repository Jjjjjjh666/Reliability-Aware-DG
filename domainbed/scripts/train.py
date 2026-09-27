# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import shutil
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.calibration import differentiable_calibration_summary
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader


AC_OBJECTIVE_SETS = {
    "ECE": ("source_out_ece",),
    "NLL": ("source_out_nll",),
    "CwECE": ("source_out_cwece",),
    "NC": ("source_out_nll", "source_out_cwece"),
    "NE": ("source_out_nll", "source_out_ece"),
    "NEC": ("source_out_nll", "source_out_ece", "source_out_cwece"),
}
LEGACY_OBJECTIVE_ALIASES = {"A": "NC", "B": "NE", "C": "NEC"}


@torch.no_grad()
def collect_logits_labels(algorithm, loader, device):
    logits_list = []
    labels_list = []
    algorithm.eval()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits_list.append(algorithm.predict(x))
        labels_list.append(y)
    algorithm.train()
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def split_sample_metadata(split):
    """Return stable original-dataset indices and paths for a dataset split."""
    indices = np.arange(len(split), dtype=np.int64)
    current = split
    while hasattr(current, "keys") and hasattr(current, "underlying_dataset"):
        keys = np.asarray(current.keys, dtype=np.int64)
        indices = keys[indices]
        current = current.underlying_dataset

    samples = getattr(current, "samples", None)
    if samples is None:
        paths = np.asarray([""] * len(indices))
    else:
        paths = np.asarray([str(samples[int(i)][0]) for i in indices])
    return indices, paths


@torch.no_grad()
def save_target_predictions(
    algorithm,
    target_split,
    *,
    output_dir,
    step,
    device,
    num_workers,
    class_names,
    dataset_name,
    target_domain_index,
    target_domain_name,
):
    """Save aligned target predictions for optional post-selection diagnostics."""
    loader = torch.utils.data.DataLoader(
        target_split,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
    )
    logits, labels = collect_logits_labels(algorithm, loader, device)
    probabilities = torch.softmax(logits, dim=1)
    confidence, predictions = probabilities.max(dim=1)
    true_label_probability = probabilities.gather(
        1, labels.view(-1, 1)
    ).squeeze(1)

    dataset_indices, sample_paths = split_sample_metadata(target_split)
    if len(dataset_indices) != labels.numel():
        raise RuntimeError("Target prediction/sample metadata length mismatch")

    filename = "target_predictions_step{}.npz".format(int(step))
    np.savez_compressed(
        os.path.join(output_dir, filename),
        schema_version=np.asarray(1, dtype=np.int64),
        dataset=np.asarray(str(dataset_name)),
        target_domain_index=np.asarray(target_domain_index, dtype=np.int64),
        target_domain_name=np.asarray(str(target_domain_name)),
        step=np.asarray(step, dtype=np.int64),
        dataset_index=dataset_indices,
        sample_path=sample_paths,
        y_true=labels.detach().cpu().numpy().astype(np.int64),
        y_pred=predictions.detach().cpu().numpy().astype(np.int64),
        confidence=confidence.detach().cpu().numpy().astype(np.float32),
        true_label_probability=(
            true_label_probability.detach().cpu().numpy().astype(np.float32)
        ),
        probabilities=probabilities.detach().cpu().numpy().astype(np.float32),
        class_names=np.asarray(class_names),
    )
    return filename


def eval_loader_metadata(name, domain_names, test_envs):
    env_token, split = name.split("_", 1)
    env_i = int(env_token.replace("env", ""))
    domain_name = domain_names[env_i] if env_i < len(domain_names) else str(env_i)
    role = "target" if env_i in test_envs else "source"
    return env_i, domain_name, split, role


def class_names_from_dataset(dataset, num_classes):
    stack = [dataset]
    seen = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        classes = getattr(current, "classes", None)
        if classes:
            return [str(name) for name in classes]

        if hasattr(current, "underlying_dataset"):
            stack.append(current.underlying_dataset)
        if hasattr(current, "dataset"):
            stack.append(current.dataset)
        if hasattr(current, "datasets"):
            stack.extend(current.datasets)

    return ["class{}".format(i) for i in range(num_classes)]


def source_out_acc_score(results, test_envs):
    source_accs = []
    test_envs = set(test_envs)
    for key, value in results.items():
        if not (key.startswith("env") and key.endswith("_out_acc")):
            continue
        env_i = int(key[3:].split("_")[0])
        if env_i not in test_envs:
            source_accs.append(value)
    if not source_accs:
        return -float("inf")
    return float(np.mean(source_accs))


def mean_clean(values):
    clean = [float(value) for value in values if value is not None]
    clean = [value for value in clean if np.isfinite(value)]
    if not clean:
        return float("nan")
    return float(np.mean(clean))


def source_out_metric_row(results, test_envs):
    source_out_rows = [
        row for row in results.get("per_domain_calibration", [])
        if row.get("role") == "source" and row.get("split") == "out"
    ]
    return {
        "step": int(results["step"]),
        "source_out_acc": source_out_acc_score(results, test_envs),
        "source_out_ece": mean_clean(row.get("ece") for row in source_out_rows),
        "source_out_cwece": mean_clean(
            row.get("classwise_ece") for row in source_out_rows
        ),
        "source_out_nll": mean_clean(row.get("nll") for row in source_out_rows),
    }


def select_source_out_checkpoint(checkpoint_history):
    selected = sorted(
        checkpoint_history,
        key=lambda row: (-float(row["source_out_acc"]), int(row["step"])),
    )[0].copy()
    selected.update({
        "selection_rule": "source_acc",
        "selection_score": -float(selected["source_out_acc"]),
        "feasible_count": len(checkpoint_history),
        "source_out_acc_star": float(selected["source_out_acc"]),
        "source_out_acc_gap_pp": 0.0,
    })
    return selected


def select_ac_checkpoint(
    checkpoint_history,
    *,
    objective_set,
    distance,
    delta_pp,
):
    objective_set = LEGACY_OBJECTIVE_ALIASES.get(objective_set, objective_set)
    if objective_set not in AC_OBJECTIVE_SETS:
        raise ValueError("Unknown AC objective set: {}".format(objective_set))
    if distance not in {"l1", "l2", "linf"}:
        raise ValueError("Unknown AC distance: {}".format(distance))

    metrics = AC_OBJECTIVE_SETS[objective_set]
    acc_star = max(float(row["source_out_acc"]) for row in checkpoint_history)
    delta_acc = float(delta_pp) / 100.0
    candidates = [
        row.copy() for row in checkpoint_history
        if float(row["source_out_acc"]) >= acc_star - delta_acc
    ]
    if not candidates:
        raise RuntimeError("AC produced an empty candidate set")

    values = np.asarray(
        [[float(row[metric]) for metric in metrics] for row in candidates],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise RuntimeError(
            "AC requires finite source-out reliability metrics"
        )

    mins = values.min(axis=0)
    ranges = values.max(axis=0) - mins
    normalized = (values - mins) / (ranges + 1e-12)
    if distance == "l1":
        scores = np.sum(normalized, axis=1)
    elif distance == "l2":
        scores = np.sqrt(np.sum(normalized ** 2, axis=1))
    else:
        scores = np.max(normalized, axis=1)

    ranked = []
    for candidate_i, score in enumerate(scores):
        row = candidates[int(candidate_i)].copy()
        ranked.append((
            float(score),
            -float(row["source_out_acc"]),
            int(row["step"]),
            row,
        ))
    score, _neg_acc, _step, selected = sorted(ranked, key=lambda item: item[:3])[0]
    delta_token = "{:g}".format(float(delta_pp)).replace(".", "p")
    selected.update({
        "selection_rule": "ac_{}_{}_d{}pp".format(
            objective_set.lower(), distance, delta_token
        ),
        "selection_score": float(score),
        "feasible_count": len(candidates),
        "source_out_acc_star": float(acc_star),
        "source_out_acc_gap_pp": (
            float(acc_star) - float(selected["source_out_acc"])
        ) * 100.0,
        "ac_objective_set": objective_set,
        "ac_distance": distance,
        "ac_delta_pp": float(delta_pp),
        "ac_metrics": list(metrics),
    })
    return selected


def write_best_model_selection(output_dir, selected):
    path = os.path.join(output_dir, "best_model_selection.json")
    with open(path, "w") as f:
        json.dump(selected, f, sort_keys=True, indent=2)


def copy_step_checkpoint_to_best(output_dir, step):
    src = os.path.join(output_dir, "model_step{}.pkl".format(int(step)))
    dst = os.path.join(output_dir, "best_model.pkl")
    if not os.path.exists(src):
        raise RuntimeError("Selected checkpoint is missing: {}".format(src))
    shutil.copyfile(src, dst)


def delete_unselected_step_checkpoints(output_dir, selected_step):
    selected_name = "model_step{}.pkl".format(int(selected_step))
    deleted = 0
    for filename in os.listdir(output_dir):
        if not filename.startswith("model_step") or not filename.endswith(".pkl"):
            continue
        if filename == selected_name:
            continue
        os.remove(os.path.join(output_dir, filename))
        deleted += 1
    return deleted


def preserve_selected_prediction_files(
    output_dir,
    checkpoint_history,
    *,
    objective_set,
    distance,
    delta_pp,
):
    selections = {
        "source_acc": select_source_out_checkpoint(checkpoint_history),
        "ac": select_ac_checkpoint(
            checkpoint_history,
            objective_set=objective_set,
            distance=distance,
            delta_pp=delta_pp,
        ),
    }
    output_files = {
        "source_acc": "target_predictions_source_acc.npz",
        "ac": "target_predictions_ac.npz",
    }

    for name, selected in selections.items():
        src = os.path.join(
            output_dir,
            "target_predictions_step{}.npz".format(int(selected["step"])),
        )
        dst = os.path.join(output_dir, output_files[name])
        if not os.path.exists(src):
            raise RuntimeError("Selected prediction file is missing: {}".format(src))
        shutil.copyfile(src, dst)

    for filename in os.listdir(output_dir):
        if (
            filename.startswith("target_predictions_step")
            and filename.endswith(".npz")
        ):
            os.remove(os.path.join(output_dir, filename))

    manifest = {
        "schema_version": 1,
        "sample_alignment": (
            "dataset_index and sample_path identify the same target in-split "
            "sample in both selected prediction files"
        ),
        "selections": {
            name: {
                "prediction_file": output_files[name],
                "selection": selected,
            }
            for name, selected in selections.items()
        },
    }
    with open(os.path.join(output_dir, "selection_predictions.json"), "w") as f:
        json.dump(manifest, f, sort_keys=True, indent=2)
    return selections


def print_calibration_tables(domain_rows, class_rows):
    print("Per-domain differentiable calibration")
    misc.print_row(
        ["domain", "split", "role", "accuracy", "ece", "classwise_ece", "n"],
        colwidth=16,
    )
    for row in domain_rows:
        misc.print_row(
            [
                row["domain"],
                row["split"],
                row["role"],
                row["accuracy"],
                row["ece"],
                row["classwise_ece"],
                row["n"],
            ],
            colwidth=16,
        )

    print("Per-domain per-class differentiable calibration")
    misc.print_row(
        ["domain", "split", "class", "count", "class_accuracy", "class_ece"],
        colwidth=16,
    )
    for row in class_rows:
        misc.print_row(
            [
                row["domain"],
                row["split"],
                row["class"],
                row["count"],
                "NA" if row["class_accuracy"] is None else row["class_accuracy"],
                row["class_ece"],
            ],
            colwidth=16,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--save_best_source_out_model', action='store_true')
    parser.add_argument('--best_model_selection_rule', type=str, default=None,
        choices=["none", "source_acc", "source_out_acc", "ac", "ac_pareto"],
        help="Rule used to create best_model.pkl. If omitted, legacy \
        --save_best_source_out_model maps to source_acc.")
    parser.add_argument(
        '--ac_objectives', '--ac_pareto_version',
        dest='ac_objectives',
        type=str,
        default="NC",
        choices=sorted(set(AC_OBJECTIVE_SETS) | set(LEGACY_OBJECTIVE_ALIASES)),
        help="Reliability objective set: ECE, NLL, CwECE, NC, NE, or NEC.",
    )
    parser.add_argument(
        '--ac_distance', '--ac_pareto_distance',
        dest='ac_distance',
        type=str,
        default="linf",
        choices=["l1", "l2", "linf"],
    )
    parser.add_argument(
        '--ac_delta_pp', '--ac_pareto_delta_pp',
        dest='ac_delta_pp',
        type=float,
        default=0.5,
    )
    parser.add_argument(
        '--save_selection_predictions',
        action='store_true',
        help=(
            "Save aligned target predictions for the Source-Acc and AC selected "
            "checkpoints. This is evaluation-only and requires one test domain."
        ),
    )
    args = parser.parse_args()

    if args.best_model_selection_rule is None:
        args.best_model_selection_rule = (
            "source_acc" if args.save_best_source_out_model else "none"
        )
    if args.best_model_selection_rule == "source_out_acc":
        args.best_model_selection_rule = "source_acc"
    elif args.best_model_selection_rule == "ac_pareto":
        args.best_model_selection_rule = "ac"
    if args.best_model_selection_rule == "ac" and args.skip_model_save:
        raise ValueError("AC best-model selection requires model saving")
    if args.save_selection_predictions and len(args.test_envs) != 1:
        raise ValueError(
            "--save_selection_predictions requires exactly one test environment"
        )

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    target_prediction_split = None
    if args.save_selection_predictions:
        target_prediction_split = in_splits[args.test_envs[0]][0]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    last_results_keys = None
    best_source_out_acc = -float("inf")
    best_model_checkpoint_history = []
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            domain_names = getattr(
                dataset, "ENVIRONMENTS", [str(i) for i in range(len(dataset))]
            )
            class_names = class_names_from_dataset(dataset, dataset.num_classes)
            calibration_n_bins = hparams.get("kde_ece_n_bins", 15)
            calibration_bandwidth = hparams.get("kde_ece_bandwidth", 0.1)
            per_domain_calibration = []
            per_domain_class_calibration = []

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc
                logits, labels = collect_logits_labels(algorithm, loader, device)
                calibration = differentiable_calibration_summary(
                    logits=logits,
                    labels=labels,
                    num_classes=dataset.num_classes,
                    n_bins=calibration_n_bins,
                    bandwidth=calibration_bandwidth,
                )

                results[name+'_eval_nll'] = calibration["nll"]
                results[name+'_eval_ece'] = calibration["ece"]
                results[name+'_eval_classwise_ece'] = calibration["classwise_ece"]

                env_i, domain_name, split, role = eval_loader_metadata(
                    name, domain_names, args.test_envs
                )
                domain_label = "env{}:{}".format(env_i, domain_name)
                per_domain_calibration.append({
                    "domain_index": env_i,
                    "domain_name": domain_name,
                    "domain": domain_label,
                    "split": split,
                    "role": role,
                    "accuracy": calibration["accuracy"],
                    "nll": calibration["nll"],
                    "ece": calibration["ece"],
                    "classwise_ece": calibration["classwise_ece"],
                    "n": int(labels.numel()),
                })

                for class_i in range(dataset.num_classes):
                    class_name = (
                        class_names[class_i]
                        if class_i < len(class_names)
                        else "class{}".format(class_i)
                    )
                    per_domain_class_calibration.append({
                        "domain_index": env_i,
                        "domain_name": domain_name,
                        "domain": domain_label,
                        "split": split,
                        "role": role,
                        "class_index": class_i,
                        "class_name": class_name,
                        "class": "class{}:{}".format(class_i, class_name),
                        "count": calibration["class_count"][class_i],
                        "class_accuracy": calibration["class_accuracy"][class_i],
                        "class_ece": calibration["class_ece"][class_i],
                    })

            current_source_out_acc = None
            if (
                args.best_model_selection_rule in {"source_acc", "ac"}
                or args.save_selection_predictions
            ):
                current_source_out_acc = source_out_acc_score(results, args.test_envs)
                results['source_out_acc_score'] = current_source_out_acc
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)
            print_calibration_tables(
                per_domain_calibration, per_domain_class_calibration
            )

            results.update({
                'per_domain_calibration': per_domain_calibration,
                'per_domain_class_calibration': per_domain_class_calibration,
                'hparams': hparams,
                'args': vars(args)
            })

            if (
                args.best_model_selection_rule in {"source_acc", "ac"}
                or args.save_selection_predictions
            ):
                best_model_checkpoint_history.append(
                    source_out_metric_row(results, args.test_envs)
                )

            if args.save_selection_predictions:
                target_env_i = args.test_envs[0]
                results['target_prediction_file'] = save_target_predictions(
                    algorithm,
                    target_prediction_split,
                    output_dir=args.output_dir,
                    step=step,
                    device=device,
                    num_workers=dataset.N_WORKERS,
                    class_names=class_names,
                    dataset_name=args.dataset,
                    target_domain_index=target_env_i,
                    target_domain_name=domain_names[target_env_i],
                )

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if (args.save_model_every_checkpoint
                    or args.best_model_selection_rule == "ac"):
                save_checkpoint(f'model_step{step}.pkl')
            if (args.best_model_selection_rule == "source_acc"
                    and current_source_out_acc > best_source_out_acc):
                print('Saving best source-out model...')
                best_source_out_acc = current_source_out_acc
                save_checkpoint('best_model.pkl')

    save_checkpoint('model.pkl')

    if args.best_model_selection_rule == "source_acc" and best_model_checkpoint_history:
        selected_best_model = select_source_out_checkpoint(best_model_checkpoint_history)
        write_best_model_selection(args.output_dir, selected_best_model)
        print("Best model selected by Source-Acc: step {}".format(
            selected_best_model["step"]
        ))
    elif args.best_model_selection_rule == "ac" and best_model_checkpoint_history:
        selected_best_model = select_ac_checkpoint(
            best_model_checkpoint_history,
            objective_set=args.ac_objectives,
            distance=args.ac_distance,
            delta_pp=args.ac_delta_pp,
        )
        copy_step_checkpoint_to_best(args.output_dir, selected_best_model["step"])
        deleted_step_checkpoints = 0
        if not args.save_model_every_checkpoint:
            deleted_step_checkpoints = delete_unselected_step_checkpoints(
                args.output_dir,
                selected_best_model["step"],
            )
        write_best_model_selection(args.output_dir, selected_best_model)
        print("Best model selected by {}: step {}".format(
            selected_best_model["selection_rule"],
            selected_best_model["step"],
        ))
        print("Deleted {} unselected step checkpoints".format(
            deleted_step_checkpoints
        ))

    if args.save_selection_predictions and best_model_checkpoint_history:
        prediction_selections = preserve_selected_prediction_files(
            args.output_dir,
            best_model_checkpoint_history,
            objective_set=args.ac_objectives,
            distance=args.ac_distance,
            delta_pp=args.ac_delta_pp,
        )
        print(
            "Saved selected target predictions: Source-Acc step {}, AC step {}".format(
                prediction_selections["source_acc"]["step"],
                prediction_selections["ac"]["step"],
            )
        )

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
