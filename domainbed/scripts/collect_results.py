# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections


import argparse
import csv
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys

import numpy as np
import tqdm

from domainbed.lib import misc, reporting
from domainbed.lib.query import Q
import warnings

try:
    from domainbed import datasets
    from domainbed import algorithms
    from domainbed import model_selection
    _DOMAINBED_TABLE_IMPORT_ERROR = None
except ModuleNotFoundError as e:
    datasets = None
    algorithms = None
    model_selection = None
    _DOMAINBED_TABLE_IMPORT_ERROR = e

def remove_key(d,key):
    new_d = d.copy()
    new_d.pop(key)
    return new_d

def recursive_freeze(obj):
    if isinstance(obj, dict):
        return frozenset((key, recursive_freeze(val)) for key, val in obj.items())
    elif isinstance(obj, list):
        return tuple(recursive_freeze(item) for item in obj)
    elif isinstance(obj, set):
        return frozenset(recursive_freeze(item) for item in obj)
    elif isinstance(obj, tuple):
        return tuple(recursive_freeze(item) for item in obj)
    else:
        return obj

def merge_records(records):
    merged_records = []
    args_set = set()  # Store unique args dictionaries

    # Group records by unique 'args' dictionaries
    for record in records:
        args = record['args'].copy()
        args.pop('holdout_fraction', None)  # Remove 'holdout_fraction' from comparison
        args_key = recursive_freeze(args)
        args_set.add(args_key)

    # Merge records with the same 'args' except for 'holdout_fraction'
    for args_key in args_set:
        args_dict = dict(args_key)
        filtered_records = [record for record in records if dict(recursive_freeze(remove_key(record['args'],'holdout_fraction'))) == args_dict]
        merged_record = {}
        for record in filtered_records:
            merged_record.update(record)
        merged_records.append(merged_record)
    return Q(merged_records)

def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "X"
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    if latex:
        return mean, err, "{:.1f} $\\pm$ {:.1f}".format(mean, err)
    else:
        return mean, err, "{:.1f} +/- {:.1f}".format(mean, err)

def print_table(table, header_text, row_labels, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")

def print_results_tables(records, selection_method, latex):
    """Given all records, print a results table for each dataset."""
    grouped_records = reporting.get_grouped_records(records)

    if selection_method == model_selection.IIDAutoLRAccuracySelectionMethod:
        for r in grouped_records:
            r['records'] = merge_records(r['records'])

    grouped_records = grouped_records.map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)


    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))
        test_envs = range(datasets.num_environments(dataset))

        table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            means = []
            for j, test_env in enumerate(test_envs):
                trial_accs = (grouped_records
                    .filter_equals(
                        "dataset, algorithm, test_env",
                        (dataset, algorithm, test_env)
                    ).select("sweep_acc"))
                mean, err, table[i][j] = format_mean(trial_accs, latex)
                means.append(mean)
            if None in means:
                table[i][-1] = "X"
            else:
                table[i][-1] = "{:.1f}".format(sum(means) / len(means))

        col_labels = [
            "Algorithm",
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ]
        header_text = (f"Dataset: {dataset}, "
            f"model selection method: {selection_method.name}")
        print_table(table, header_text, alg_names, list(col_labels),
            colwidth=20, latex=latex)

    # Print an "averages" table
    if latex:
        print()
        print("\\subsubsection{Averages}")

    table = [[None for _ in [*dataset_names, "Avg"]] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        means = []
        for j, dataset in enumerate(dataset_names):
            trial_averages = (grouped_records
                .filter_equals("algorithm, dataset", (algorithm, dataset))
                .group("trial_seed")
                .map(lambda trial_seed, group:
                    group.select("sweep_acc").mean()
                )
            )
            mean, err, table[i][j] = format_mean(trial_averages, latex)
            means.append(mean)
        if None in means:
            table[i][-1] = "X"
        else:
            table[i][-1] = "{:.1f}".format(sum(means) / len(means))

    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {selection_method.name}"
    print_table(table, header_text, alg_names, col_labels, colwidth=25,
        latex=latex)

def _clean_float(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value

def _mean(values):
    values = [_clean_float(v) for v in values]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)

def _write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _env_index_from_key(key):
    try:
        return int(key[3:key.index("_")])
    except Exception:
        return None

def _row_env_index(row):
    if "domain_index" in row:
        return row.get("domain_index")
    domain = row.get("domain")
    if isinstance(domain, str) and domain.startswith("env") and ":" in domain:
        try:
            return int(domain[3:domain.index(":")])
        except Exception:
            return None
    return row.get("domain")

def _result_files(input_dir, recursive=True):
    if os.path.isfile(input_dir):
        return [input_dir] if os.path.basename(input_dir) == "results.jsonl" else []
    if recursive:
        paths = []
        for root, _, files in os.walk(input_dir):
            if "results.jsonl" in files:
                paths.append(os.path.join(root, "results.jsonl"))
        return sorted(paths)
    paths = []
    direct = os.path.join(input_dir, "results.jsonl")
    if os.path.exists(direct):
        paths.append(direct)
    for subdir in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, subdir, "results.jsonl")
        if os.path.exists(path):
            paths.append(path)
    return paths

def _read_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def _test_envs(record):
    test_envs = record.get("args", {}).get("test_envs", [])
    if not isinstance(test_envs, list):
        test_envs = [test_envs]
    return test_envs

def _iid_val_acc(record, selection_split="out"):
    test_envs = set(_test_envs(record))
    source_keys = [
        k for k in record
        if k.startswith("env") and k.endswith("_{}_acc".format(selection_split))
        and _env_index_from_key(k) not in test_envs
    ]
    score = _mean(record[k] for k in source_keys)
    return -1.0 if score is None else score

def _best_record(records, selection_split="out"):
    test_records = [r for r in records if len(_test_envs(r)) == 1]
    if not test_records:
        test_records = records
    if not test_records:
        return None
    return max(test_records, key=lambda r: _iid_val_acc(r, selection_split))

def _cal_rows(record, role=None, split=None, env=None):
    rows = []
    for row in record.get("per_domain_calibration", []):
        if role is not None and row.get("role") != role:
            continue
        if split is not None and row.get("split") != split:
            continue
        if env is not None and _row_env_index(row) != env:
            continue
        rows.append(row)
    return rows

def _class_rows(record, role=None, split=None, env=None):
    rows = []
    for row in record.get("per_domain_class_calibration", []):
        if role is not None and row.get("role") != role:
            continue
        if split is not None and row.get("split") != split:
            continue
        if env is not None and _row_env_index(row) != env:
            continue
        rows.append(row)
    return rows

def _domain_name(record, env):
    for row in record.get("per_domain_calibration", []):
        if _row_env_index(row) == env:
            return row.get("domain_name") or row.get("domain") or str(env)
    return str(env)

def _metric(rows, key):
    return _mean(row.get(key) for row in rows)

def _source_envs(record):
    test_envs = set(_test_envs(record))
    envs = set()
    for key in record:
        if key.startswith("env") and (key.endswith("_in_acc") or key.endswith("_out_acc")):
            env = _env_index_from_key(key)
            if env is not None and env not in test_envs:
                envs.add(env)
    return sorted(envs)

def _method_label(record, result_path):
    args = record.get("args", {})
    hparams = record.get("hparams", {}) or {}
    algorithm = args.get("algorithm", "")
    parts = []
    sweep = os.path.basename(os.path.dirname(os.path.dirname(result_path)))
    if sweep and sweep != "outputs":
        parts.append(sweep)
    if algorithm:
        parts.append(algorithm)
    tags = []
    if hparams.get("lambda_cal_mean") not in (None, 0, 0.0):
        tags.append("mean={}".format(hparams.get("lambda_cal_mean")))
    if hparams.get("lambda_cal_var") not in (None, 0, 0.0):
        tags.append("var={}".format(hparams.get("lambda_cal_var")))
    if hparams.get("cal_var_type"):
        tags.append("var_type={}".format(hparams.get("cal_var_type")))
    if algorithm in ("VREx", "CwECEREx") and hparams.get("vrex_lambda") is not None:
        tags.append("vrex={}".format(hparams.get("vrex_lambda")))
    if algorithm in ("ERMCwECE", "CwECEREx") and hparams.get("cal_warmup_iters") is not None:
        tags.append("warmup={}".format(hparams.get("cal_warmup_iters")))
    if tags:
        parts.append("({})".format(",".join(tags)))
    return " ".join(parts) if parts else sweep

CHECKPOINT_FIELDS = [
    "dataset", "test_domain", "method", "seed", "step",
    "source_in_acc", "source_out_acc", "source_out_ece",
    "source_out_cwece", "source_out_nll", "target_acc", "target_ece",
    "target_cwece", "target_nll", "target_worst_domain",
    "target_worst_class", "source_path", "run_id", "algorithm",
    "hparams_seed", "trial_seed", "test_env", "target_split"
]

def _checkpoint_row(record, result_path, input_dir, target_split="in"):
    args = record.get("args", {})
    test_envs = _test_envs(record)
    test_env = test_envs[0] if test_envs else None
    source_envs = _source_envs(record)
    source_in_acc = _mean(record.get("env{}_in_acc".format(e)) for e in source_envs)
    source_out_acc = _mean(record.get("env{}_out_acc".format(e)) for e in source_envs)
    source_out_rows = _cal_rows(record, role="source", split="out")
    target_rows = _cal_rows(record, role="target", split=target_split, env=test_env)
    if not target_rows:
        target_rows = _cal_rows(record, role="target", split=target_split)
    target_class_rows = _class_rows(record, role="target", split=target_split, env=test_env)
    if not target_class_rows:
        target_class_rows = _class_rows(record, role="target", split=target_split)
    target_acc = None
    if test_env is not None:
        target_acc = record.get("env{}_{}_acc".format(test_env, target_split))
    if target_acc is None:
        target_acc = _metric(target_rows, "accuracy")
    target_class_acc = [
        _clean_float(r.get("class_accuracy")) for r in target_class_rows
    ]
    target_class_acc = [v for v in target_class_acc if v is not None]
    try:
        source_path = os.path.relpath(result_path, input_dir)
    except ValueError:
        source_path = result_path
    return {
        "dataset": args.get("dataset", ""),
        "test_domain": _domain_name(record, test_env) if test_env is not None else "",
        "method": _method_label(record, result_path),
        "seed": args.get("seed", ""),
        "step": record.get("step", ""),
        "source_in_acc": source_in_acc,
        "source_out_acc": source_out_acc,
        "source_out_ece": _metric(source_out_rows, "ece"),
        "source_out_cwece": _metric(source_out_rows, "classwise_ece"),
        "source_out_nll": _metric(source_out_rows, "nll"),
        "target_acc": target_acc,
        "target_ece": _metric(target_rows, "ece"),
        "target_cwece": _metric(target_rows, "classwise_ece"),
        "target_nll": _metric(target_rows, "nll"),
        "target_worst_domain": target_acc,
        "target_worst_class": min(target_class_acc) if target_class_acc else None,
        "source_path": source_path,
        "run_id": os.path.basename(os.path.dirname(result_path)),
        "algorithm": args.get("algorithm", ""),
        "hparams_seed": args.get("hparams_seed", ""),
        "trial_seed": args.get("trial_seed", ""),
        "test_env": test_env if test_env is not None else "",
        "target_split": target_split,
    }

def export_checkpoint_level(input_dir, output_path, target_split="in", recursive=True):
    rows = []
    for path in _result_files(input_dir, recursive=recursive):
        for record in _read_jsonl(path):
            rows.append(_checkpoint_row(record, path, input_dir, target_split))
    rows.sort(key=lambda r: (
        r["dataset"], r["method"], str(r["test_env"]), str(r["seed"]),
        int(r["step"]) if str(r["step"]).isdigit() else -1, r["source_path"]
    ))
    _write_csv(output_path, rows, CHECKPOINT_FIELDS)
    return rows

def _selected_runs(input_dir, selection_split="out", recursive=True):
    runs = []
    for path in _result_files(input_dir, recursive=recursive):
        record = _best_record(_read_jsonl(path), selection_split)
        if record is not None:
            runs.append((os.path.basename(os.path.dirname(path)), path, record))
    return runs

def _selected_metric_row(run_id, path, record, input_dir, selection_split, target_split):
    row = _checkpoint_row(record, path, input_dir, target_split)
    row["selection_split"] = selection_split
    row["selection_score"] = _iid_val_acc(record, selection_split)
    return row

def _build_calibration_tables(runs, split, selection_split, input_dir):
    domain_rows = []
    class_rows = []
    summary_rows = []
    for run_id, path, record in runs:
        test_envs = _test_envs(record)
        test_env = test_envs[0] if test_envs else None
        step = record.get("step")
        selection_score = _iid_val_acc(record, selection_split)
        for row in record.get("per_domain_calibration", []):
            if split != "all" and row.get("split") != split:
                continue
            domain_rows.append({
                "run_id": run_id,
                "step": step,
                "selection_split": selection_split,
                "selection_score": selection_score,
                "target_env": test_env,
                "domain": row.get("domain"),
                "domain_name": row.get("domain_name"),
                "role": row.get("role"),
                "split": row.get("split"),
                "n": row.get("n"),
                "accuracy": row.get("accuracy"),
                "nll": row.get("nll"),
                "ece": row.get("ece"),
                "classwise_ece": row.get("classwise_ece"),
            })
        for row in record.get("per_domain_class_calibration", []):
            if split != "all" and row.get("split") != split:
                continue
            class_rows.append({
                "run_id": run_id,
                "step": step,
                "selection_split": selection_split,
                "selection_score": selection_score,
                "target_env": test_env,
                "domain": row.get("domain"),
                "domain_name": row.get("domain_name"),
                "role": row.get("role"),
                "split": row.get("split"),
                "class": row.get("class"),
                "class_name": row.get("class_name"),
                "count": row.get("count"),
                "class_accuracy": row.get("class_accuracy"),
                "class_ece": row.get("class_ece"),
            })
        if split != "all":
            summary_rows.append(_selected_metric_row(
                run_id, path, record, input_dir, selection_split, split
            ))
    return domain_rows, class_rows, summary_rows

def export_selected_calibration(input_dir, prefix, split, selection_split, recursive=True):
    runs = _selected_runs(input_dir, selection_split, recursive)
    domain_rows, class_rows, summary_rows = _build_calibration_tables(
        runs, split, selection_split, input_dir
    )
    table1_path = os.path.join(input_dir, "table1_domain_metrics_{}_split.csv".format(prefix))
    table2_path = os.path.join(input_dir, "table2_domain_class_metrics_{}_split.csv".format(prefix))
    summary_path = os.path.join(input_dir, "selected_metrics_{}_split.csv".format(prefix))
    _write_csv(table1_path, domain_rows)
    _write_csv(table2_path, class_rows)
    _write_csv(summary_path, summary_rows, CHECKPOINT_FIELDS + ["selection_split", "selection_score"])
    return {
        "runs": len(runs),
        "table1": table1_path,
        "table1_rows": len(domain_rows),
        "table2": table2_path,
        "table2_rows": len(class_rows),
        "summary": summary_path,
        "summary_rows": len(summary_rows),
    }

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--auto_lr", action="store_true")
    parser.add_argument("--skip_standard_tables", action="store_true")
    parser.add_argument("--export_checkpoint_level", action="store_true")
    parser.add_argument("--checkpoint_level_path", type=str, default=None)
    parser.add_argument("--export_calibration_tables", action="store_true")
    parser.add_argument("--calibration_split", type=str, default="in",
        choices=["in", "out", "uda", "all"])
    parser.add_argument("--selection_split", type=str, default="out",
        choices=["in", "out"])
    parser.add_argument("--metrics_prefix", type=str, default=None)
    parser.add_argument("--no_recursive_metrics", action="store_true")
    args = parser.parse_args()

    results_file = "results.tex" if args.latex else "results.txt"

    sys.stdout = misc.Tee(os.path.join(args.input_dir, results_file), "w")

    records = reporting.load_records(args.input_dir)

    if args.latex and not args.skip_standard_tables:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\section{Full DomainBed results}")
        print("% Total records:", len(records))
    else:
        print("Total records:", len(records))

    if not args.skip_standard_tables and len(records):
        if _DOMAINBED_TABLE_IMPORT_ERROR is not None:
            raise _DOMAINBED_TABLE_IMPORT_ERROR
        if args.auto_lr:
            SELECTION_METHODS = [model_selection.IIDAutoLRAccuracySelectionMethod]
        else:
            SELECTION_METHODS = [
                model_selection.IIDAccuracySelectionMethod,
                model_selection.LeaveOneOutSelectionMethod,
                model_selection.OracleSelectionMethod,
            ]

        for selection_method in SELECTION_METHODS:
            if args.latex:
                print()
                print("\\subsection{{Model selection: {}}}".format(
                    selection_method.name))
            print_results_tables(records, selection_method, args.latex)

    recursive_metrics = not args.no_recursive_metrics
    if args.export_checkpoint_level:
        checkpoint_path = args.checkpoint_level_path or os.path.join(
            args.input_dir, "checkpoint_level_all.csv"
        )
        rows = export_checkpoint_level(
            args.input_dir, checkpoint_path, "in", recursive_metrics
        )
        print("checkpoint_level =", checkpoint_path, "rows=", len(rows))

    if args.export_calibration_tables:
        prefix = args.metrics_prefix or "{}_selection_{}".format(
            args.selection_split, args.calibration_split
        )
        info = export_selected_calibration(
            args.input_dir, prefix, args.calibration_split,
            args.selection_split, recursive_metrics
        )
        print("selected_runs =", info["runs"])
        print("table1 =", info["table1"], "rows=", info["table1_rows"])
        print("table2 =", info["table2"], "rows=", info["table2_rows"])
        print("summary =", info["summary"], "rows=", info["summary_rows"])

    if args.latex and not args.skip_standard_tables:
        print("\\end{document}")
