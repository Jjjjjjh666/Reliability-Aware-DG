"""Plan unique double-holdout runs and select full-source checkpoints.

The selector reads only source-side fields until the selected step is fixed.
Existing full-source logs use soft, squared calibration metrics; results from
this script are therefore marked ``logged_soft`` and are not the manuscript's
hard-bin reliability analysis.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path


DATASETS = ("OfficeHome", "TerraIncognita", "PACS")
ALGORITHMS = ("ERM", "CORAL", "GroupDRO", "IRM", "VREx")
EXPECTED_STEPS = tuple(range(0, 5001, 100))


def read_jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def require_finite(values, context):
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"Nonfinite selection metric: {context}")


def full_runs(root):
    """Index completed four-domain source models, checking the step grid."""
    runs = {}
    for done in Path(root).glob("*/done"):
        path = done.parent / "results.jsonl"
        rows = read_jsonl(path)
        if tuple(row["step"] for row in rows) != EXPECTED_STEPS:
            raise ValueError(f"Incomplete step grid: {path}")
        args = rows[0]["args"]
        target = args["test_envs"]
        if len(target) != 1:
            raise ValueError(f"Expected one real target domain: {path}")
        key = (
            args["dataset"], args["algorithm"], target[0],
            args["hparams_seed"], args["trial_seed"],
        )
        if key in runs:
            raise ValueError(f"Duplicate full-source run: {key}")
        runs[key] = (path, rows)
    return runs


def inner_dir(root, dataset, algorithm, hparams_seed, trial_seed, pair):
    return Path(root) / dataset / algorithm / f"h{hparams_seed}_t{trial_seed}" / f"ex{pair[0]}_{pair[1]}"


def make_plan(runs, inner_root, datasets, algorithms,
              hparams_seeds=range(3), trial_seeds=range(3)):
    jobs = []
    for dataset in datasets:
        for algorithm in algorithms:
            for hparams_seed, trial_seed in itertools.product(hparams_seeds, trial_seeds):
                outer = [runs.get((dataset, algorithm, t, hparams_seed, trial_seed)) for t in range(4)]
                if any(run is None for run in outer):
                    missing = [t for t, run in enumerate(outer) if run is None]
                    raise ValueError(f"Missing full-source runs: {dataset} {algorithm} h={hparams_seed} trial={trial_seed} targets={missing}")
                hparams = outer[0][1][0]["hparams"]
                if any(run[1][0]["hparams"] != hparams for run in outer):
                    raise ValueError(f"Hparams differ across targets: {dataset} {algorithm} h={hparams_seed} trial={trial_seed}")
                holdout = outer[0][1][0]["args"]["holdout_fraction"]
                if any(run[1][0]["args"]["holdout_fraction"] != holdout for run in outer):
                    raise ValueError("Holdout fractions differ across targets")
                for pair in itertools.combinations(range(4), 2):
                    jobs.append({
                        "dataset": dataset, "algorithm": algorithm,
                        "hparams_seed": hparams_seed, "trial_seed": trial_seed,
                        "excluded_envs": pair, "hparams": hparams,
                        "holdout_fraction": holdout,
                        "output_dir": str(inner_dir(
                            inner_root, dataset, algorithm, hparams_seed, trial_seed, pair,
                        )),
                    })
    return jobs


def source_at_step(row, target):
    source_domains = [i for i in range(4) if i != target]
    accuracy = sum(row[f"env{i}_out_acc"] for i in source_domains) / 3
    calibration = {
        item["domain_index"]: item
        for item in row["per_domain_calibration"]
        if item["split"] == "out" and item["domain_index"] in source_domains
    }
    if set(calibration) != set(source_domains):
        raise ValueError("Missing source-out calibration domain")
    score = {
        "step": row["step"], "accuracy": accuracy,
        "nll": sum(calibration[i]["nll"] for i in source_domains) / 3,
        "cwece": sum(calibration[i]["classwise_ece"] for i in source_domains) / 3,
    }
    require_finite((score[name] for name in ("accuracy", "nll", "cwece")), "source")
    return score


def fold_at_step(inner_rows, heldout):
    by_step = {}
    for row in inner_rows:
        score = row["scores"].get(str(heldout))
        if score is None or score.get("split") != "in":
            raise ValueError(f"Missing held-out in-split: {heldout}")
        if row["step"] in by_step:
            raise ValueError("Duplicate inner step")
        by_step[row["step"]] = score
    if tuple(sorted(by_step)) != EXPECTED_STEPS:
        raise ValueError("Inner fold step grid differs from full source")
    return by_step


def cv_series(target, dataset, algorithm, hparams_seed, trial_seed, inner_root, cache,
              metric_profile="logged_soft", expected_hparams=None, expected_holdout=None):
    folds = []
    for heldout in range(4):
        if heldout == target:
            continue
        pair = tuple(sorted((target, heldout)))
        folder = inner_dir(inner_root, dataset, algorithm, hparams_seed, trial_seed, pair)
        if not (folder / "done").exists():
            raise ValueError(f"Missing completed inner fold: {folder}")
        if folder not in cache:
            cache[folder] = read_jsonl(folder / "metrics.jsonl")
        config = json.loads((folder / "config.json").read_text())
        if config["excluded_envs"] != list(pair):
            raise ValueError(f"Inner pair mismatch: {folder}")
        for field, expected in (
            ("dataset", dataset), ("algorithm", algorithm),
            ("hparams_seed", hparams_seed), ("trial_seed", trial_seed),
            ("steps", 5001), ("checkpoint_freq", 100),
        ):
            if config.get(field) != expected:
                raise ValueError(f"Inner {field} mismatch: {folder}")
        if expected_hparams is not None and config.get("hparams") != expected_hparams:
            raise ValueError(f"Inner hparams mismatch: {folder}")
        if expected_holdout is not None and config.get("holdout_fraction") != expected_holdout:
            raise ValueError(f"Inner holdout fraction mismatch: {folder}")
        folds.append(fold_at_step(cache[folder], heldout))
    cwece_key = "hard_cwece" if metric_profile == "hard" else "logged_soft_cwece"
    series = [{
        "step": step,
        "accuracy": sum(f[step]["accuracy"] for f in folds) / 3,
        "nll": sum(f[step]["nll"] for f in folds) / 3,
        "cwece": sum(f[step][cwece_key] for f in folds) / 3,
    } for step in EXPECTED_STEPS]
    for row in series:
        require_finite((row[name] for name in ("accuracy", "nll", "cwece")), "LODO")
    return series


def choose_accuracy(series):
    return min(series, key=lambda row: (-row["accuracy"], row["step"]))


def choose_ac(series, delta_pp=0.5):
    maximum = max(row["accuracy"] for row in series)
    candidates = [row for row in series if row["accuracy"] >= maximum - delta_pp / 100 - 1e-12]
    lo = {name: min(row[name] for row in candidates) for name in ("nll", "cwece")}
    hi = {name: max(row[name] for row in candidates) for name in ("nll", "cwece")}

    def score(row):
        values = [
            (row[name] - lo[name]) / (hi[name] - lo[name]) if hi[name] > lo[name] else 0
            for name in ("nll", "cwece")
        ]
        return max(values), -row["accuracy"], row["step"]

    return min(candidates, key=score), len(candidates)


def target_at_step(row, target):
    matches = [
        item for item in row["per_domain_calibration"]
        if item["domain_index"] == target and item["split"] == "in"
    ]
    if len(matches) != 1:
        raise ValueError("Missing target-in result")
    item = matches[0]
    return {
        "target_accuracy": row[f"env{target}_in_acc"],
        "target_nll": item["nll"],
        "target_ece": item["ece"],
        "target_cwece": item["classwise_ece"],
    }


def hard_source_series(hard_root, dataset, algorithm, target, hparams_seed, trial_seed):
    path = (Path(hard_root) / dataset / algorithm / f"h{hparams_seed}_t{trial_seed}"
            / f"target_{target}.jsonl")
    if not path.with_suffix(".done").exists():
        raise ValueError(f"Hard source evaluation is incomplete: {path}")
    rows = read_jsonl(path)
    if tuple(row["step"] for row in rows) != EXPECTED_STEPS:
        raise ValueError(f"Incomplete hard-metric step grid: {path}")
    if any(row.get("metric_profile") != "hard_deterministic" for row in rows):
        raise ValueError(f"Hard metric profile mismatch: {path}")
    series = [{
        "step": row["step"], "accuracy": row["source_accuracy"],
        "nll": row["source_nll"], "cwece": row["source_cwece"],
    } for row in rows]
    for row in series:
        require_finite((row[name] for name in ("accuracy", "nll", "cwece")), "hard source")
    return series


def select(runs, inner_root, datasets, algorithms, metric_profile="logged_soft", hard_root=None,
           hparams_seeds=range(3), trial_seeds=range(3),
           selection_scopes=("fixed_hparams", "select_hparams")):
    if metric_profile == "hard" and not hard_root:
        raise ValueError("Hard-metric selection requires --hard-root")
    cache = {}
    all_rows = []
    groups = defaultdict(dict)
    for key, (path, raw) in runs.items():
        dataset, algorithm, target, hparams_seed, trial_seed = key
        if (dataset not in datasets or algorithm not in algorithms
                or hparams_seed not in hparams_seeds or trial_seed not in trial_seeds):
            continue
        source = (
            hard_source_series(hard_root, dataset, algorithm, target, hparams_seed, trial_seed)
            if metric_profile == "hard" else
            [source_at_step(row, target) for row in raw]
        )
        cv = cv_series(target, dataset, algorithm, hparams_seed, trial_seed,
                       inner_root, cache, metric_profile,
                       expected_hparams=raw[0]["hparams"],
                       expected_holdout=raw[0]["args"]["holdout_fraction"])
        groups[(dataset, algorithm, target, trial_seed)][hparams_seed] = (path, raw, source, cv)
        choices = {
            "Source-Acc": (choose_accuracy(source), None),
            "AC": choose_ac(source),
            "LODO-Acc": (choose_accuracy(cv), None),
            "AC-LODO": choose_ac(cv),
        }
        if "fixed_hparams" in selection_scopes:
            for selector, (chosen, feasible_count) in choices.items():
                all_rows.append({
                    "selection_scope": "fixed_hparams", "dataset": dataset,
                    "algorithm": algorithm, "target_domain": target,
                    "hparams_seed": hparams_seed, "trial_seed": trial_seed,
                    "selector": selector, "selected_step": chosen["step"],
                    "validation_accuracy": chosen["accuracy"],
                    "feasible_count": feasible_count,
                    "full_source_run": str(path.parent),
                    "metric_profile": ("hard_deterministic" if metric_profile == "hard" else "logged_soft"),
                    **(target_at_step(raw[chosen["step"] // 100], target)
                       if metric_profile == "logged_soft" else {}),
                })
    if "select_hparams" not in selection_scopes:
        return all_rows
    if set(hparams_seeds) != {0, 1, 2}:
        raise ValueError("select_hparams requires hparams seeds 0, 1, and 2")
    for (dataset, algorithm, target, trial_seed), candidates in groups.items():
        if set(candidates) != set(hparams_seeds):
            raise ValueError("Incomplete three-hyperparameter search pool")
        h_source = min(candidates, key=lambda h: (-choose_accuracy(candidates[h][2])["accuracy"], h))
        h_cv = min(candidates, key=lambda h: (-choose_accuracy(candidates[h][3])["accuracy"], h))
        for selector, h in (("Source-Acc", h_source), ("AC", h_source),
                            ("LODO-Acc", h_cv), ("AC-LODO", h_cv)):
            path, raw, source, cv = candidates[h]
            series = source if selector in ("Source-Acc", "AC") else cv
            chosen, feasible_count = (
                (choose_accuracy(series), None)
                if selector in ("Source-Acc", "LODO-Acc") else choose_ac(series)
            )
            all_rows.append({
                "selection_scope": "select_hparams", "dataset": dataset,
                "algorithm": algorithm, "target_domain": target,
                "hparams_seed": h, "trial_seed": trial_seed,
                "selector": selector, "selected_step": chosen["step"],
                "validation_accuracy": chosen["accuracy"],
                "feasible_count": feasible_count,
                "full_source_run": str(path.parent),
                "metric_profile": ("hard_deterministic" if metric_profile == "hard" else "logged_soft"),
                **(target_at_step(raw[chosen["step"] // 100], target)
                   if metric_profile == "logged_soft" else {}),
            })
    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "select"))
    parser.add_argument("--full-root", required=True)
    parser.add_argument("--inner-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS[:2]))
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--hparams-seeds", nargs="+", type=int, choices=range(3), default=list(range(3)))
    parser.add_argument("--trial-seeds", nargs="+", type=int, choices=range(3), default=list(range(3)))
    parser.add_argument(
        "--selection-scopes", nargs="+",
        choices=("fixed_hparams", "select_hparams"),
        default=["fixed_hparams", "select_hparams"],
    )
    parser.add_argument("--metric-profile", choices=("logged_soft", "hard"), default="logged_soft")
    parser.add_argument("--hard-root")
    parser.add_argument("--protocol-output", help="Write a frozen protocol next to a plan manifest")
    args = parser.parse_args()
    runs = full_runs(args.full_root)
    if args.command == "plan":
        jobs = make_plan(
            runs, args.inner_root, args.datasets, args.algorithms,
            args.hparams_seeds, args.trial_seeds,
        )
        with open(args.output, "w") as destination:
            for job in jobs:
                destination.write(json.dumps(job, sort_keys=True) + "\n")
        if args.protocol_output:
            manifest_bytes = Path(args.output).read_bytes()
            code_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).parent.glob("*.py"))
            }
            relevant_runs = []
            for key, (path, rows) in sorted(runs.items()):
                dataset, algorithm, target, hparams_seed, trial_seed = key
                if (dataset in args.datasets and algorithm in args.algorithms
                        and hparams_seed in args.hparams_seeds
                        and trial_seed in args.trial_seeds):
                    relevant_runs.append({
                        "dataset": dataset, "algorithm": algorithm,
                        "target_domain": target, "hparams_seed": hparams_seed,
                        "trial_seed": trial_seed, "run_id": path.parent.name,
                        "training_seed": rows[0]["args"]["seed"],
                    })
            protocol = {
                "schema_version": 1,
                "full_root": str(Path(args.full_root).resolve()),
                "inner_root": str(Path(args.inner_root).resolve()),
                "datasets": args.datasets, "algorithms": args.algorithms,
                "hparams_seeds": args.hparams_seeds,
                "trial_seeds": args.trial_seeds,
                "excluded_domain_pairs": list(itertools.combinations(range(4), 2)),
                "steps": list(EXPECTED_STEPS), "holdout_fraction": 0.2,
                "inner_seed_rule": "seed_hash(dataset,algorithm,sorted_excluded_envs,hparams_seed,trial_seed)",
                "metric_profile": "hard_15_bin_true_class_frequency_weighted",
                "selectors": ["Source-Acc", "AC", "LODO-Acc", "AC-LODO"],
                "ac_delta_pp": 0.5, "ac_distance": "linf",
                "job_count": len(jobs),
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "code_sha256": code_hashes,
                "full_source_runs": relevant_runs,
                "split_hashes": "recorded and checked in each completed inner config",
            }
            Path(args.protocol_output).write_text(
                json.dumps(protocol, indent=2, sort_keys=True) + "\n"
            )
        print(f"planned {len(jobs)} unique inner jobs")
    else:
        rows = select(runs, args.inner_root, args.datasets, args.algorithms,
                      args.metric_profile, args.hard_root,
                      args.hparams_seeds, args.trial_seeds, args.selection_scopes)
        with open(args.output, "w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"selected {len(rows)} checkpoints; metric_profile={args.metric_profile}")


if __name__ == "__main__":
    main()
