import argparse
import csv
import json
import os


def _iid_val_acc(record, selection_split="out"):
    """Compute training-domain validation accuracy.

    DomainBed's IIDAccuracySelectionMethod uses ``selection_split="out"``:
    select the checkpoint with the highest mean source-domain holdout
    accuracy (env*_out_acc). ``selection_split="in"`` is kept only for
    diagnostic/backward-compatibility comparisons with older reports.
    """
    args = record.get("args", {})
    test_envs = args.get("test_envs", [0])
    test_envs = set(test_envs if isinstance(test_envs, list) else [test_envs])
    source_keys = [
        k for k in record
        if k.startswith("env") and k.endswith("_{}_acc".format(selection_split))
        and int(k[3:k.index("_")]) not in test_envs
    ]
    if not source_keys:
        return -1.0
    return sum(float(record[k]) for k in source_keys) / len(source_keys)


def read_best_iid_jsonl(path, selection_split="out"):
    """Read all checkpoints and return the one with best training-domain val acc."""
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return None
    records = [json.loads(l) for l in lines]
    return max(records, key=lambda r: _iid_val_acc(r, selection_split))


def load_completed_runs(input_dir, selection_split="out"):
    runs = []
    results_path = os.path.join(input_dir, "results.jsonl")
    done_path = os.path.join(input_dir, "done")
    if os.path.exists(results_path) and os.path.exists(done_path):
        record = read_best_iid_jsonl(results_path, selection_split)
        if record is not None:
            runs.append((os.path.basename(os.path.abspath(input_dir)), record))
        return runs

    for run_id in sorted(os.listdir(input_dir)):
        run_dir = os.path.join(input_dir, run_id)
        if not os.path.isdir(run_dir):
            continue
        results_path = os.path.join(run_dir, "results.jsonl")
        done_path = os.path.join(run_dir, "done")
        if os.path.exists(results_path) and os.path.exists(done_path):
            record = read_best_iid_jsonl(results_path, selection_split)
            if record is not None:
                runs.append((run_id, record))
    return runs


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def build_tables(runs, split, selection_split):
    domain_rows = []
    class_rows = []

    for run_id, record in runs:
        test_env = record["args"]["test_envs"][0]
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

    role_order = {"target": 0, "source": 1}
    domain_rows.sort(
        key=lambda r: (
            r["target_env"],
            role_order.get(r["role"], 9),
            r["split"],
            r["domain_name"],
        )
    )
    class_rows.sort(
        key=lambda r: (
            r["target_env"],
            role_order.get(r["role"], 9),
            r["split"],
            r["domain_name"],
            r["class_name"],
        )
    )
    return domain_rows, class_rows


def _valid_float_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        values.append(float(value))
    return values


def build_summary(domain_rows, class_rows):
    target_domain_rows = [r for r in domain_rows if r["role"] == "target"]
    target_class_rows = [r for r in class_rows if r["role"] == "target"]
    target_domain_acc = _valid_float_values(target_domain_rows, "accuracy")
    target_domain_nll = _valid_float_values(target_domain_rows, "nll")
    target_domain_ece = _valid_float_values(target_domain_rows, "ece")
    target_domain_cwece = _valid_float_values(target_domain_rows, "classwise_ece")
    target_class_acc = _valid_float_values(target_class_rows, "class_accuracy")
    return [{
        "target_domain_accuracy": mean(target_domain_acc),
        "target_domain_nll": mean(target_domain_nll),
        "target_domain_ece": mean(target_domain_ece),
        "target_domain_classwise_ece": mean(target_domain_cwece),
        "worst_target_domain_accuracy": min(target_domain_acc, default=None),
        "worst_target_class_accuracy": min(target_class_acc, default=None),
        "num_target_domains": len(target_domain_acc),
        "num_target_classes": len(target_class_acc),
    }]


def main():
    parser = argparse.ArgumentParser(
        description="Export calibration observation tables from a DomainBed sweep."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="out",
                        choices=["in", "out", "uda", "all"])
    parser.add_argument("--selection_split", type=str, default="out",
                        choices=["in", "out"],
                        help=(
                            "Source-domain split used for checkpoint selection. "
                            "'out' matches DomainBed IID model selection; 'in' "
                            "is for reproducing older diagnostic reports."
                        ))
    parser.add_argument("--prefix", type=str, default=None)
    args = parser.parse_args()

    runs = load_completed_runs(args.input_dir, args.selection_split)
    domain_rows, class_rows = build_tables(runs, args.split, args.selection_split)
    summary_rows = build_summary(domain_rows, class_rows)
    prefix = args.prefix or args.split

    table1_path = os.path.join(
        args.input_dir, "table1_domain_metrics_{}_split.csv".format(prefix)
    )
    table2_path = os.path.join(
        args.input_dir, "table2_domain_class_metrics_{}_split.csv".format(prefix)
    )
    summary_path = os.path.join(
        args.input_dir, "summary_calibration_{}_split.csv".format(prefix)
    )

    write_csv(table1_path, domain_rows)
    write_csv(table2_path, class_rows)
    write_csv(summary_path, summary_rows)

    print("runs_done =", len(runs))
    print("selection_split =", args.selection_split)
    print("eval_split =", args.split)
    print("table1 =", table1_path, "rows=", len(domain_rows))
    print("table2 =", table2_path, "rows=", len(class_rows))
    print("summary =", summary_path, "rows=", len(summary_rows))


if __name__ == "__main__":
    main()
