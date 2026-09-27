"""Validate and summarize the frozen ERM LODO experiment."""

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


SELECTORS = ("Source-Acc", "AC", "LODO-Acc", "AC-LODO")
METRICS = (
    ("target_accuracy", "Accuracy", 100.0),
    ("target_nll", "NLL", 1.0),
    ("target_ece", "ECE ×100", 100.0),
    ("target_cwece", "CwECE ×100", 100.0),
)
COMPARISONS = (
    ("AC - LODO-Acc", "AC", "LODO-Acc"),
    ("AC - Source-Acc", "AC", "Source-Acc"),
    ("AC-LODO - LODO-Acc", "AC-LODO", "LODO-Acc"),
)


def read_csv(path):
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        return list(reader), reader.fieldnames


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0])
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_target(selection_rows, selection_fields, target_rows, target_fields, inner_root):
    if len(selection_rows) != 384 or len(target_rows) != len(selection_rows):
        raise ValueError("Expected 384 frozen selection and target rows")
    if target_fields[:len(selection_fields)] != selection_fields:
        raise ValueError("Target output does not preserve frozen selection fields")
    target_metric_fields = target_fields[len(selection_fields):]
    expected_metrics = [
        "target_accuracy", "target_nll", "target_ece", "target_cwece",
        "target_split_hash",
    ]
    if target_metric_fields != expected_metrics:
        raise ValueError("Unexpected target metric fields")
    for selected, evaluated in zip(selection_rows, target_rows):
        if any(evaluated[field] != selected[field] for field in selection_fields):
            raise ValueError("Target output changed a frozen selection row")
    hash_groups = defaultdict(set)
    for row in target_rows:
        scope_key = (
            row["selection_scope"], row["dataset"], row["algorithm"],
            row["target_domain"], row["trial_seed"],
        )
        if row["selection_scope"] == "fixed_hparams":
            scope_key += (row["hparams_seed"],)
        hash_groups[scope_key].add(row["target_split_hash"])
        target = int(row["target_domain"])
        other = 0 if target != 0 else 1
        pair = sorted((target, other))
        config_path = (
            Path(inner_root) / row["dataset"] / row["algorithm"]
            / f"h{row['hparams_seed']}_t{row['trial_seed']}"
            / f"ex{pair[0]}_{pair[1]}" / "config.json"
        )
        config = json.loads(config_path.read_text())
        if config["split_hashes"][str(target)] != row["target_split_hash"]:
            raise ValueError(f"Target split hash differs from inner protocol: {config_path}")
    if any(len(hashes) != 1 for hashes in hash_groups.values()):
        raise ValueError("Selectors used different target samples")


def flatten_inner(inner_root):
    rows = []
    for done in sorted(Path(inner_root).glob("*/ERM/h*_t*/ex?_?/done")):
        folder = done.parent
        config = json.loads((folder / "config.json").read_text())
        pair = config["excluded_envs"]
        metrics = [
            json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if [row["step"] for row in metrics] != list(range(0, 5001, 100)):
            raise ValueError(f"Incomplete inner step grid: {folder}")
        for target, heldout in ((pair[0], pair[1]), (pair[1], pair[0])):
            for metric in metrics:
                score = metric["scores"][str(heldout)]
                rows.append({
                    "dataset": config["dataset"], "algorithm": config["algorithm"],
                    "target_domain": target, "heldout_source_domain": heldout,
                    "hparams_seed": config["hparams_seed"],
                    "trial_seed": config["trial_seed"], "model_id": str(folder),
                    "step": metric["step"], "eval_split": score["split"],
                    "n_samples": score["n_samples"], "accuracy": score["accuracy"],
                    "nll": score["nll"], "ece": score["hard_ece"],
                    "cwece": score["hard_cwece"],
                    "split_hash": config["split_hashes"][str(heldout)],
                })
    if len(rows) != 108 * 2 * 51:
        raise ValueError(f"Expected 11016 inner metric rows, found {len(rows)}")
    return rows


def task_key(row, scope):
    key = (row["dataset"], row["algorithm"], row["target_domain"], row["trial_seed"])
    return key + ((row["hparams_seed"],) if scope == "fixed_hparams" else ())


def paired_values(target_rows, scope, comparison, metric):
    _, left, right = comparison
    tasks = defaultdict(dict)
    for row in target_rows:
        if row["selection_scope"] == scope:
            tasks[task_key(row, scope)][row["selector"]] = row
    values = []
    for key, selectors in tasks.items():
        if set(selectors) != set(SELECTORS):
            raise ValueError(f"Incomplete selector task: {key}")
        values.append({
            "dataset": key[0], "target": int(key[2]),
            "trial": int(key[3]),
            "hparams": int(key[4]) if scope == "fixed_hparams" else None,
            "value": float(selectors[left][metric]) - float(selectors[right][metric]),
        })
    expected = 72 if scope == "fixed_hparams" else 24
    if len(values) != expected:
        raise ValueError(f"Expected {expected} paired tasks, found {len(values)}")
    return values


def bootstrap_plan(scope, datasets, repeats, seed):
    count = 9 if scope == "fixed_hparams" else 3
    rng = np.random.default_rng(seed)
    return {dataset: rng.integers(0, count, size=(repeats, count)) for dataset in datasets}


def summarize_paired(values, scope, datasets, samples):
    dataset_bootstraps = []
    dataset_means = []
    for dataset in datasets:
        subset = [row for row in values if row["dataset"] == dataset]
        blocks = defaultdict(list)
        for row in subset:
            block = (row["hparams"], row["trial"]) if scope == "fixed_hparams" else row["trial"]
            blocks[block].append(row["value"])
        ordered = sorted(blocks)
        expected_blocks = 9 if scope == "fixed_hparams" else 3
        if len(ordered) != expected_blocks or any(len(blocks[key]) != 4 for key in ordered):
            raise ValueError(f"Incomplete bootstrap blocks: {scope} {dataset}")
        block_means = np.asarray([np.mean(blocks[key]) for key in ordered])
        dataset_means.append(float(block_means.mean()))
        dataset_bootstraps.append(block_means[samples[dataset]].mean(axis=1))
    bootstraps = np.mean(dataset_bootstraps, axis=0)
    return (
        float(np.mean(dataset_means)),
        float(np.quantile(bootstraps, 0.025)),
        float(np.quantile(bootstraps, 0.975)),
    )


def paired_tables(target_rows, repeats, seed):
    datasets = sorted({row["dataset"] for row in target_rows})
    summaries, trials = [], []
    for scope_index, scope in enumerate(("fixed_hparams", "select_hparams")):
        plans = {}
        for group_index, group in enumerate([[d] for d in datasets] + [datasets]):
            plans[tuple(group)] = bootstrap_plan(
                scope, group, repeats, seed + scope_index * 100 + group_index,
            )
        for comparison in COMPARISONS:
            for metric, label, scale in METRICS:
                values = paired_values(target_rows, scope, comparison, metric)
                for group in [[d] for d in datasets] + [datasets]:
                    mean, low, high = summarize_paired(
                        values, scope, group, plans[tuple(group)],
                    )
                    summaries.append({
                        "evaluation_scope": scope,
                        "dataset_group": group[0] if len(group) == 1 else "combined_equal_dataset",
                        "comparison": comparison[0], "metric": label,
                        "mean_difference": mean * scale,
                        "ci_low": low * scale, "ci_high": high * scale,
                        "resampling_unit": (
                            "(hparams_seed,trial_seed) block with 4 targets"
                            if scope == "fixed_hparams" else
                            "trial_seed block with 4 targets; descriptive only"
                        ),
                        "n_tasks": len([row for row in values if row["dataset"] in group]),
                        "bootstrap_repeats": repeats,
                    })
                if scope == "select_hparams":
                    for dataset in datasets:
                        for trial in range(3):
                            subset = [
                                row["value"] for row in values
                                if row["dataset"] == dataset and row["trial"] == trial
                            ]
                            trials.append({
                                "dataset": dataset, "trial_seed": trial,
                                "comparison": comparison[0], "metric": label,
                                "mean_difference_across_targets": float(np.mean(subset)) * scale,
                                "n_targets": len(subset),
                            })
    return summaries, trials


def aggregate_table(target_rows):
    rows = []
    for scope in ("fixed_hparams", "select_hparams"):
        for dataset in sorted({row["dataset"] for row in target_rows}):
            for selector in SELECTORS:
                subset = [
                    row for row in target_rows
                    if row["selection_scope"] == scope and row["dataset"] == dataset
                    and row["selector"] == selector
                ]
                record = {
                    "selection_scope": scope, "dataset": dataset,
                    "selector": selector, "n_tasks": len(subset),
                }
                for metric, label, scale in METRICS:
                    record[label] = float(np.mean([float(row[metric]) for row in subset])) * scale
                rows.append(record)
    return rows


def diagnostics(target_rows):
    rows = []
    for scope in ("fixed_hparams", "select_hparams"):
        tasks = defaultdict(dict)
        for row in target_rows:
            if row["selection_scope"] == scope:
                tasks[task_key(row, scope)][row["selector"]] = row
        for dataset in sorted({key[0] for key in tasks}):
            subset = [selectors for key, selectors in tasks.items() if key[0] == dataset]
            for _, left, right in COMPARISONS:
                rows.append({
                    "selection_scope": scope, "dataset": dataset,
                    "selector_pair": f"{left} vs {right}",
                    "different_step_fraction": np.mean([
                        selectors[left]["selected_step"] != selectors[right]["selected_step"]
                        or selectors[left]["hparams_seed"] != selectors[right]["hparams_seed"]
                        for selectors in subset
                    ]),
                    "n_tasks": len(subset),
                })
    return rows


def compute_costs(inner_root, hard_root, target_shard_root):
    rows = []
    for done in sorted(Path(inner_root).glob("*/ERM/h*_t*/ex?_?/done")):
        folder = done.parent
        metrics = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        final = metrics[-1]
        rows.append({
            "stage": "inner_training", "model_id": str(folder),
            "training_gpu_hours": final["train_seconds"] / 3600,
            "validation_seconds": final["evaluation_seconds"],
            "gpu_process_hours": final["elapsed_seconds"] / 3600,
            "peak_memory_gb": final["peak_memory_gb"],
            "storage_bytes": sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()),
            "timing_source": "recorded per run",
        })
    for done in sorted(Path(hard_root).glob("*/ERM/h*_t*/target_*.done")):
        path = done.with_suffix(".jsonl")
        rows.append({
            "stage": "hard_source_evaluation", "model_id": str(path),
            "training_gpu_hours": "", "validation_seconds": "",
            "gpu_process_hours": "", "peak_memory_gb": "",
            "storage_bytes": path.stat().st_size,
            "timing_source": "per-run timing not recorded",
        })
    for done in sorted(Path(target_shard_root).glob("*/ERM/h*_t*/target_*.done")):
        path = done.with_suffix(".csv")
        rows.append({
            "stage": "target_evaluation", "model_id": str(path),
            "training_gpu_hours": "", "validation_seconds": "",
            "gpu_process_hours": "", "peak_memory_gb": "",
            "storage_bytes": path.stat().st_size,
            "timing_source": "per-run timing not recorded",
        })
    return rows


def fmt(value):
    return f"{float(value):.3f}"


def write_report(path, aggregate, paired, trials, diagnostics_rows, selection_hash, target_hash,
                 inner_gpu_hours):
    lines = [
        "# ERM-108 LODO / AC checkpoint 选择实验报告", "",
        "## 实验范围与完整性", "",
        "- 数据集：OfficeHome、TerraIncognita。", "- 算法：ERM。",
        "- 内层训练：108/108；hard-source 轨迹：72/72；选择结果：384；目标结果：384。",
        "- fixed-hparams 与 select-hparams 分开汇总；所有目标评估均发生在选择文件冻结之后。",
        f"- 冻结选择 SHA-256：`{selection_hash}`。",
        f"- 目标结果 SHA-256：`{target_hash}`。",
        f"- 内层训练记录的 GPU-process 时间：{inner_gpu_hours:.2f} 小时。", "",
    ]
    for scope, title in (
        ("fixed_hparams", "固定超参数结果"),
        ("select_hparams", "三超参数搜索结果"),
    ):
        lines += [f"## {title}", "", "| Dataset | Selector | Acc ↑ | NLL ↓ | ECE ×100 ↓ | CwECE ×100 ↓ | n |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in aggregate:
            if row["selection_scope"] == scope:
                lines.append(
                    f"| {row['dataset']} | {row['selector']} | {fmt(row['Accuracy'])} | "
                    f"{fmt(row['NLL'])} | {fmt(row['ECE ×100'])} | {fmt(row['CwECE ×100'])} | "
                    f"{row['n_tasks']} |"
                )
        lines += ["", "配对差值采用“前者减后者”；Accuracy 为正表示前者更高，误差指标为负表示前者更低。", "",
                  "| Dataset | Comparison | Metric | Mean | 95% CI | n |",
                  "| --- | --- | --- | ---: | ---: | ---: |"]
        for row in paired:
            if row["evaluation_scope"] == scope and row["dataset_group"] != "combined_equal_dataset":
                lines.append(
                    f"| {row['dataset_group']} | {row['comparison']} | {row['metric']} | "
                    f"{fmt(row['mean_difference'])} | [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | "
                    f"{row['n_tasks']} |"
                )
        lines.append("")
    lines += ["## 两数据集等权合并的主要比较", "",
              "| Scope | Comparison | Metric | Mean | 95% CI |",
              "| --- | --- | --- | ---: | ---: |"]
    for row in paired:
        if row["dataset_group"] == "combined_equal_dataset":
            lines.append(
                f"| {row['evaluation_scope']} | {row['comparison']} | {row['metric']} | "
                f"{fmt(row['mean_difference'])} | [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] |"
            )

    def combined(scope, comparison, metric):
        matches = [
            row for row in paired
            if row["evaluation_scope"] == scope
            and row["dataset_group"] == "combined_equal_dataset"
            and row["comparison"] == comparison and row["metric"] == metric
        ]
        if len(matches) != 1:
            raise ValueError("Missing combined comparison")
        return matches[0]

    fixed_ac_acc = combined("fixed_hparams", "AC - LODO-Acc", "Accuracy")
    fixed_ac_nll = combined("fixed_hparams", "AC - LODO-Acc", "NLL")
    fixed_ac_ece = combined("fixed_hparams", "AC - LODO-Acc", "ECE ×100")
    fixed_hybrid_acc = combined("fixed_hparams", "AC-LODO - LODO-Acc", "Accuracy")
    fixed_hybrid_ece = combined("fixed_hparams", "AC-LODO - LODO-Acc", "ECE ×100")
    search_hybrid_acc = combined("select_hparams", "AC-LODO - LODO-Acc", "Accuracy")
    search_hybrid_ece = combined("select_hparams", "AC-LODO - LODO-Acc", "ECE ×100")
    lines += ["", "## 主要结果解读", "",
              (f"- 固定超参数下，AC 相对 LODO-Acc 的 Accuracy 差为 "
               f"{fmt(fixed_ac_acc['mean_difference'])} pp "
               f"[{fmt(fixed_ac_acc['ci_low'])}, {fmt(fixed_ac_acc['ci_high'])}]，区间跨过 0；"
               f"NLL 差为 {fmt(fixed_ac_nll['mean_difference'])} "
               f"[{fmt(fixed_ac_nll['ci_low'])}, {fmt(fixed_ac_nll['ci_high'])}]，"
               f"ECE 差为 {fmt(fixed_ac_ece['mean_difference'])} "
               f"[{fmt(fixed_ac_ece['ci_low'])}, {fmt(fixed_ac_ece['ci_high'])}]。"
               "后两项均为正且区间未跨 0，因此本实验不支持 AC 比 LODO-Acc 更可靠。"),
              (f"- 固定超参数下，AC-LODO 相对 LODO-Acc 的 Accuracy 差为 "
               f"{fmt(fixed_hybrid_acc['mean_difference'])} pp "
               f"[{fmt(fixed_hybrid_acc['ci_low'])}, {fmt(fixed_hybrid_acc['ci_high'])}]，"
               f"ECE 差为 {fmt(fixed_hybrid_ece['mean_difference'])} "
               f"[{fmt(fixed_hybrid_ece['ci_low'])}, {fmt(fixed_hybrid_ece['ci_high'])}]；"
               "两项区间均跨过 0，未得到稳定增益证据。"),
              (f"- 三超参数搜索下，AC-LODO 相对 LODO-Acc 的 Accuracy 差为 "
               f"{fmt(search_hybrid_acc['mean_difference'])} pp "
               f"[{fmt(search_hybrid_acc['ci_low'])}, {fmt(search_hybrid_acc['ci_high'])}]，"
               f"ECE 差为 {fmt(search_hybrid_ece['mean_difference'])} "
               f"[{fmt(search_hybrid_ece['ci_low'])}, {fmt(search_hybrid_ece['ci_high'])}]。"
               "ECE 方向有利于 AC-LODO，但该 scope 只有 3 个 trial，区间仅作描述。"), ""]
    lines += ["## 诊断", ""]
    for row in diagnostics_rows:
        lines.append(
            f"- {row['selection_scope']} / {row['dataset']} / {row['selector_pair']}："
            f"选点不一致比例 {100 * float(row['different_step_fraction']):.1f}%（n={row['n_tasks']}）。"
        )
    lines += ["", "## 结论边界", "",
              "结果仅覆盖 OfficeHome、TerraIncognita 上的 ERM 和当前 3×3 seed 搜索池，不能外推到其余四种算法或 PACS。",
              "置信区间是给定数据集与配置下的条件不确定性；三超参数结果只有 3 个 trial，其区间仅作描述。",
              "是否满足准确率非劣需要预先给定目标域非劣界限，本报告不把源域 0.5 pp 可行阈值解释为目标域非劣界限。", ""]
    Path(path).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-root", required=True)
    parser.add_argument("--hard-root", required=True)
    parser.add_argument("--target-shard-root", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Nonempty analysis output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selection_rows, selection_fields = read_csv(args.selection)
    target_rows, target_fields = read_csv(args.target)
    validate_target(selection_rows, selection_fields, target_rows, target_fields, args.inner_root)

    shutil.copyfile(args.selection, output / "selected_checkpoints.csv")
    shutil.copyfile(args.target, output / "target_evaluation.csv")
    inner_rows = flatten_inner(args.inner_root)
    write_csv(output / "inner_fold_metrics.csv", inner_rows)
    paired, trials = paired_tables(target_rows, args.bootstrap_repeats, args.bootstrap_seed)
    write_csv(output / "paired_comparisons.csv", paired)
    write_csv(output / "select_hparams_trial_differences.csv", trials)
    aggregate = aggregate_table(target_rows)
    write_csv(output / "aggregate_results.csv", aggregate)
    diagnostic_rows = diagnostics(target_rows)
    write_csv(output / "selection_diagnostics.csv", diagnostic_rows)
    costs = compute_costs(args.inner_root, args.hard_root, args.target_shard_root)
    write_csv(output / "compute_costs.csv", costs)

    selection_hash = hashlib.sha256(Path(args.selection).read_bytes()).hexdigest()
    target_hash = hashlib.sha256(Path(args.target).read_bytes()).hexdigest()
    inner_gpu_hours = sum(float(row["gpu_process_hours"]) for row in costs
                          if row["stage"] == "inner_training")
    write_report(
        output / "lodo_report_zh.md", aggregate, paired, trials, diagnostic_rows,
        selection_hash, target_hash, inner_gpu_hours,
    )
    (output / "artifact_hashes.sha256").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(output.iterdir()) if path.is_file()
    ))
    print(f"wrote analysis to {output}")


if __name__ == "__main__":
    main()
