"""Audit and summarize the 360-run checkpoint-SWAD experiment.

The two single-checkpoint baselines are recomputed from the saved per-sample
prediction files so all three methods use the same hard-bin metrics.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


DATASETS = ("OfficeHome", "TerraIncognita")
ALGORITHMS = ("ERM", "CORAL", "GroupDRO", "IRM", "VREx")
APPROACHES = ("source_acc", "ac_nc", "checkpoint_swad")
METRICS = ("accuracy", "nll", "ece", "cwece")
EXPECTED_KEYS = {
    (dataset, algorithm, target, hparams_seed, trial_seed)
    for dataset in DATASETS
    for algorithm in ALGORITHMS
    for target in range(4)
    for hparams_seed in range(3)
    for trial_seed in range(3)
}


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows and not fields:
        raise ValueError(f"Cannot infer fields for empty table: {path}")
    fields = fields or list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def hard_ece(confidence, correct, n_bins=15):
    bins = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
    count = np.bincount(bins, minlength=n_bins)
    confidence_sum = np.bincount(bins, weights=confidence, minlength=n_bins)
    correct_sum = np.bincount(bins, weights=correct, minlength=n_bins)
    if count.sum() != len(confidence):
        raise ValueError("ECE bin counts do not match the number of samples")
    return float(np.abs(correct_sum - confidence_sum).sum() / len(confidence))


def prediction_metrics(path):
    with np.load(path) as stored:
        probabilities = stored["probabilities"].astype(np.float64)
        labels = stored["y_true"].astype(np.int64)
        dataset_indices = stored["dataset_index"].copy()
        sample_paths = stored["sample_path"].copy()
        step = int(stored["step"])
    if probabilities.ndim != 2 or len(labels) != len(probabilities):
        raise ValueError(f"Malformed predictions: {path}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Nonfinite probabilities: {path}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-5):
        raise ValueError(f"Probabilities do not sum to one: {path}")
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(labels)), predicted]
    cwece = 0.0
    for klass in range(probabilities.shape[1]):
        frequency = np.mean(labels == klass)
        if frequency:
            cwece += frequency * hard_ece(
                probabilities[:, klass], labels == klass
            )
    metrics = {
        "accuracy": float(np.mean(predicted == labels)),
        "ece": hard_ece(confidence, predicted == labels),
        "cwece": float(cwece),
    }
    return metrics, step, dataset_indices, sample_paths


def run_metadata(run_dir):
    with (run_dir / "results.jsonl").open() as source:
        args = json.loads(source.readline())["args"]
    if len(args["test_envs"]) != 1:
        raise ValueError(f"Expected one target domain: {run_dir}")
    return (
        args["dataset"], args["algorithm"], int(args["test_envs"][0]),
        int(args["hparams_seed"]), int(args["trial_seed"]),
    )


def collect(full_root, swad_root):
    full_runs = {}
    for path in Path(full_root).glob("*/done"):
        run_dir = path.parent
        key = run_metadata(run_dir)
        if key in full_runs:
            raise ValueError(f"Duplicate full-source key: {key}")
        full_runs[key] = run_dir
    if set(full_runs) != EXPECTED_KEYS:
        missing = sorted(EXPECTED_KEYS - set(full_runs))
        extra = sorted(set(full_runs) - EXPECTED_KEYS)
        raise ValueError(f"Full-source coverage mismatch; missing={missing[:5]} extra={extra[:5]}")

    rows = []
    selections = []
    alignment_checks = 0
    for key in sorted(EXPECTED_KEYS):
        dataset, algorithm, target, hparams_seed, trial_seed = key
        run_dir = full_runs[key]
        logged_rows = [
            json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()
            if line.strip()
        ]
        logged_by_step = {int(row["step"]): row for row in logged_rows}
        if len(logged_by_step) != 51:
            raise ValueError(f"Expected 51 logged checkpoints: {run_dir}")
        selection = json.loads((run_dir / "selection_predictions.json").read_text())
        baseline_alignment = None
        for approach, filename, selection_name in (
            ("source_acc", "target_predictions_source_acc.npz", "source_acc"),
            ("ac_nc", "target_predictions_ac_nc.npz", "ac_nc"),
        ):
            metrics, step, indices, paths = prediction_metrics(run_dir / filename)
            declared = selection["selections"][selection_name]["selection"]
            if step != int(declared["step"]):
                raise ValueError(f"Prediction/selection step mismatch: {run_dir} {approach}")
            logged = logged_by_step[step]
            logged_accuracy = float(logged[f"env{target}_in_acc"])
            if abs(metrics["accuracy"] - logged_accuracy) > 1e-12:
                raise ValueError(f"Prediction/log accuracy mismatch: {run_dir} {approach}")
            # Use the cross-entropy recorded directly from logits. Reconstructing
            # NLL from saved float32 probabilities is not safe for extreme IRM
            # errors because true-class probabilities can underflow to zero.
            metrics["nll"] = float(logged[f"env{target}_in_eval_nll"])
            if baseline_alignment is None:
                baseline_alignment = (indices, paths)
            else:
                if not np.array_equal(indices, baseline_alignment[0]) or not np.array_equal(paths, baseline_alignment[1]):
                    raise ValueError(f"Baseline target samples differ: {run_dir}")
                alignment_checks += 1
            rows.append({
                "dataset": dataset, "algorithm": algorithm,
                "target_domain": target, "hparams_seed": hparams_seed,
                "trial_seed": trial_seed, "run_id": run_dir.name,
                "approach": approach, "selected_step": step,
                "selected_count": 1, "unique_selected_count": 1,
                "target_samples": len(indices), **metrics,
            })

        swad_dir = Path(swad_root) / run_dir.name
        if not (swad_dir / "done").exists():
            raise ValueError(f"Missing checkpoint-SWAD done marker: {swad_dir}")
        result = json.loads((swad_dir / "result.json").read_text())
        result_key = (
            result["dataset"], result["algorithm"], int(result["target_domain"]),
            int(result["hparams_seed"]), int(result["trial_seed"]),
        )
        if result_key != key:
            raise ValueError(f"checkpoint-SWAD metadata mismatch: {swad_dir}")
        if int(result["target_samples"]) != len(baseline_alignment[0]):
            raise ValueError(f"Target sample count mismatch: {swad_dir}")
        steps = [int(value) for value in result["selected_steps"]]
        metrics = {
            "accuracy": float(result["target_accuracy"]),
            "nll": float(result["target_nll"]),
            "ece": float(result["target_ece"]),
            "cwece": float(result["target_cwece"]),
        }
        rows.append({
            "dataset": dataset, "algorithm": algorithm,
            "target_domain": target, "hparams_seed": hparams_seed,
            "trial_seed": trial_seed, "run_id": run_dir.name,
            "approach": "checkpoint_swad", "selected_step": "",
            "selected_count": len(steps),
            "unique_selected_count": len(set(steps)),
            "target_samples": int(result["target_samples"]), **metrics,
        })
        selections.append({
            "dataset": dataset, "algorithm": algorithm,
            "target_domain": target, "hparams_seed": hparams_seed,
            "trial_seed": trial_seed, "run_id": run_dir.name,
            "converge_step": result["converge_step"],
            "first_selected_step": min(steps), "last_selected_step": max(steps),
            "selected_count": len(steps), "unique_selected_count": len(set(steps)),
            "dead_valley": int(bool(result["dead_valley"])),
            "fallback_last": int(bool(result["fallback_last"])),
            "validation_seconds": float(result["validation_seconds"]),
            "averaging_seconds": float(result["averaging_seconds"]),
            "bn_seconds": float(result["bn_seconds"]),
            "target_evaluation_seconds": float(result["target_evaluation_seconds"]),
            "model_save_seconds": float(result["model_save_seconds"]),
            "peak_gpu_memory_gb": float(result["peak_gpu_memory_gb"]),
            "model_bytes": int(result["model_bytes"]),
            "config_mtime_epoch": (swad_dir / "config.json").stat().st_mtime,
            "result_mtime_epoch": (swad_dir / "result.json").stat().st_mtime,
        })

    if len(rows) != 1080 or len(selections) != 360 or alignment_checks != 360:
        raise ValueError("Unexpected result or alignment count")
    for row in rows:
        if not all(math.isfinite(row[metric]) for metric in METRICS):
            raise ValueError(f"Nonfinite metric: {row}")
    return rows, selections


def mean(values):
    if not isinstance(values, np.ndarray):
        values = list(values)
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def group_summary(rows, group_fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        record = dict(zip(group_fields, key))
        record["n_tasks"] = len(group)
        for metric in METRICS:
            record[metric] = mean(row[metric] for row in group)
        output.append(record)
    return output


def paired_index(rows):
    index = {}
    for row in rows:
        task = (
            row["dataset"], row["algorithm"], row["target_domain"],
            row["hparams_seed"], row["trial_seed"],
        )
        key = task + (row["approach"],)
        if key in index:
            raise ValueError(f"Duplicate metric row: {key}")
        index[key] = row
    return index


def paired_bootstrap(rows, baseline, filters, bootstrap_samples, seed):
    index = paired_index(rows)
    tasks = sorted({key[:-1] for key in index if key[-1] == "checkpoint_swad"})
    tasks = [task for task in tasks if all(task[{"dataset": 0, "algorithm": 1}[field]] == value
                                           for field, value in filters.items())]
    strata = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        dataset, algorithm, target, hparams_seed, trial_seed = task
        strata[(dataset, algorithm)][(hparams_seed, trial_seed)].append(task)
    for blocks in strata.values():
        if set(blocks) != {(h, t) for h in range(3) for t in range(3)}:
            raise ValueError("A bootstrap stratum does not contain all nine seed blocks")
        if any(len(block) != 4 for block in blocks.values()):
            raise ValueError("A bootstrap seed block does not contain all four targets")

    rng = np.random.default_rng(seed)
    output = []
    for metric in METRICS:
        scale = 100.0 if metric in {"accuracy", "ece", "cwece"} else 1.0
        delta_by_stratum = {}
        for stratum, blocks in strata.items():
            delta_by_stratum[stratum] = np.asarray([
                mean(
                    index[task + ("checkpoint_swad",)][metric]
                    - index[task + (baseline,)][metric]
                    for task in blocks[block]
                ) * scale
                for block in sorted(blocks)
            ])
        observed = mean(value.mean() for value in delta_by_stratum.values())
        samples = np.empty(bootstrap_samples, dtype=np.float64)
        for iteration in range(bootstrap_samples):
            samples[iteration] = mean(
                values[rng.integers(0, len(values), len(values))].mean()
                for values in delta_by_stratum.values()
            )
        raw_deltas = [
            (index[task + ("checkpoint_swad",)][metric]
             - index[task + (baseline,)][metric]) * scale
            for task in tasks
        ]
        if metric == "accuracy":
            wins = mean(value > 0 for value in raw_deltas)
        else:
            wins = mean(value < 0 for value in raw_deltas)
        output.append({
            "dataset": filters.get("dataset", ""),
            "algorithm": filters.get("algorithm", ""),
            "comparison": f"checkpoint_swad-minus-{baseline}",
            "metric": metric, "n_tasks": len(tasks), "mean_delta": observed,
            "ci_low": float(np.quantile(samples, 0.025)),
            "ci_high": float(np.quantile(samples, 0.975)),
            "win_rate": wins,
            "resampling_unit": "(hparams_seed,trial_seed) block with 4 target domains",
            "bootstrap_samples": bootstrap_samples,
        })
    return output


def comparison_tables(rows, bootstrap_samples=10000):
    output = []
    scopes = [{}]
    scopes.extend({"dataset": dataset} for dataset in DATASETS)
    scopes.extend(
        {"dataset": dataset, "algorithm": algorithm}
        for dataset in DATASETS for algorithm in ALGORITHMS
    )
    for scope_index, filters in enumerate(scopes):
        for baseline_index, baseline in enumerate(("source_acc", "ac_nc")):
            output.extend(paired_bootstrap(
                rows, baseline, filters, bootstrap_samples,
                seed=20260921 + 100 * scope_index + baseline_index,
            ))
    return output


def selection_summary(selections):
    groups = defaultdict(list)
    for row in selections:
        groups[(row["dataset"], row["algorithm"])].append(row)
    output = []
    for (dataset, algorithm), group in sorted(groups.items()):
        output.append({
            "dataset": dataset, "algorithm": algorithm, "n_runs": len(group),
            "mean_first_selected_step": mean(row["first_selected_step"] for row in group),
            "mean_last_selected_step": mean(row["last_selected_step"] for row in group),
            "mean_selected_count": mean(row["selected_count"] for row in group),
            "mean_unique_selected_count": mean(row["unique_selected_count"] for row in group),
            "dead_valley_rate": mean(row["dead_valley"] for row in group),
            "fallback_last_rate": mean(row["fallback_last"] for row in group),
        })
    return output


def true_worst_summary(rows):
    by_configuration = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"], row["algorithm"], row["hparams_seed"],
            row["trial_seed"], row["approach"],
        )
        by_configuration[key].append(row)
    summarized = []
    for key, group in by_configuration.items():
        if len(group) != 4 or {row["target_domain"] for row in group} != set(range(4)):
            raise ValueError(f"Incomplete target set for true-worst summary: {key}")
        summarized.append({
            "dataset": key[0], "algorithm": key[1], "approach": key[4],
            "true_worst_accuracy": min(row["accuracy"] for row in group),
        })
    groups = defaultdict(list)
    for row in summarized:
        groups[(row["dataset"], row["algorithm"], row["approach"])].append(row)
    return [
        {"dataset": key[0], "algorithm": key[1], "approach": key[2],
         "n_configurations": len(group),
         "true_worst_accuracy": mean(row["true_worst_accuracy"] for row in group)}
        for key, group in sorted(groups.items())
    ]


def compute_summary(selections):
    phases = (
        "validation_seconds", "averaging_seconds", "bn_seconds",
        "target_evaluation_seconds", "model_save_seconds",
    )
    output = {f"total_{phase}": sum(row[phase] for row in selections) for phase in phases}
    output["total_phase_seconds"] = sum(output[f"total_{phase}"] for phase in phases)
    output["total_phase_hours"] = output["total_phase_seconds"] / 3600
    output["mean_phase_minutes_per_run"] = output["total_phase_seconds"] / len(selections) / 60
    output["mean_peak_gpu_memory_gb"] = mean(row["peak_gpu_memory_gb"] for row in selections)
    output["max_peak_gpu_memory_gb"] = max(row["peak_gpu_memory_gb"] for row in selections)
    output["total_model_bytes"] = sum(row["model_bytes"] for row in selections)
    output["total_model_gib"] = output["total_model_bytes"] / 1024**3
    output["observed_file_span_hours"] = (
        max(row["result_mtime_epoch"] for row in selections)
        - min(row["config_mtime_epoch"] for row in selections)
    ) / 3600
    output["n_runs"] = len(selections)
    return output


def lookup_comparison(comparisons, baseline, metric, dataset=None, algorithm=None):
    dataset = dataset or ""
    algorithm = algorithm or ""
    for row in comparisons:
        if (row["dataset"] == dataset and row["algorithm"] == algorithm
                and row["comparison"] == f"checkpoint_swad-minus-{baseline}"
                and row["metric"] == metric):
            return row
    raise KeyError((baseline, metric, dataset, algorithm))


def fmt_metric(metric, value):
    if metric in {"accuracy", "ece", "cwece"}:
        return f"{100 * value:.2f}"
    return f"{value:.3f}"


def fmt_delta(row):
    return f"{row['mean_delta']:+.2f} [{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]"


def report_markdown(rows, selections, summaries, comparisons, selection_stats,
                    worst_stats, compute):
    summary_index = {
        (row["dataset"], row["algorithm"], row["approach"]): row
        for row in summaries
    }
    dataset_summaries = group_summary(rows, ("dataset", "approach"))
    dataset_index = {(row["dataset"], row["approach"]): row for row in dataset_summaries}
    overall = group_summary(rows, ("approach",))
    overall_index = {row["approach"]: row for row in overall}
    selection_index = {(row["dataset"], row["algorithm"]): row for row in selection_stats}
    worst_index = {
        (row["dataset"], row["algorithm"], row["approach"]): row
        for row in worst_stats
    }
    jointly_clear = []
    for dataset in DATASETS:
        for algorithm in ALGORITHMS:
            clear = True
            for baseline in ("source_acc", "ac_nc"):
                accuracy = lookup_comparison(comparisons, baseline, "accuracy", dataset, algorithm)
                nll = lookup_comparison(comparisons, baseline, "nll", dataset, algorithm)
                clear &= accuracy["ci_low"] > 0 and nll["ci_high"] < 0
            if clear:
                jointly_clear.append(f"{dataset}/{algorithm}")
    worst_improved = sum(
        worst_index[(dataset, algorithm, "checkpoint_swad")]["true_worst_accuracy"]
        > max(
            worst_index[(dataset, algorithm, "source_acc")]["true_worst_accuracy"],
            worst_index[(dataset, algorithm, "ac_nc")]["true_worst_accuracy"],
        )
        for dataset in DATASETS for algorithm in ALGORITHMS
    )

    lines = [
        "# Checkpoint-SWAD（含源域 BN 重估）基线实验汇总报告",
        "",
        "## 结论摘要",
        "",
    ]
    for baseline, label in (("source_acc", "Source-Acc"), ("ac_nc", "AC-NC")):
        accuracy = lookup_comparison(comparisons, baseline, "accuracy")
        nll = lookup_comparison(comparisons, baseline, "nll")
        lines.append(
            f"- 相对 {label}，checkpoint-SWAD 的总体 Accuracy 差为 "
            f"**{fmt_delta(accuracy)} 个百分点**，NLL 差为 "
            f"**{nll['mean_delta']:+.3f} [{nll['ci_low']:+.3f}, {nll['ci_high']:+.3f}]**。"
        )
    lines.extend([
        "- 上述差值均为 `checkpoint-SWAD − 对照`；Accuracy 为正表示 checkpoint-SWAD 更好，NLL/ECE/CwECE 为负表示 checkpoint-SWAD 更好。",
        "- 总体 Accuracy、ECE、CwECE 相对两个对照的 95% 区间均不跨 0；总体 NLL 均值更低，但区间跨 0，因此只能报告下降趋势。",
        "- TerraIncognita 上相对两个对照的 Accuracy、NLL、ECE、CwECE 区间均指向改善。OfficeHome 的均值同向，但 Accuracy/NLL 区间跨 0，算法间差异更大。",
        f"- 同时对两个对照取得明确 Accuracy 提升和 NLL 降低的组合有 **{len(jointly_clear)}/10**：{', '.join(jointly_clear)}。checkpoint-SWAD 的 true worst-domain Accuracy 高于两个对照的组合有 **{worst_improved}/10**。",
        "- 本实验完成的是稀疏 checkpoint 近似 SWAD。它复用每 100 step 的模型端点，不能表述为官方逐更新段平均 SWAD 的精确复现。",
        "- **BN 协议说明：本项目 360 条轨迹的 `freeze_bn=false`，因此每个平均模型都额外使用 500 批源域训练样本重估 BN。当前结果同时包含权重平均和 BN 重估的作用，不能直接等同于冻结 BN 统计的论文 DG 主配置，也不能据此把全部增益归因于权重平均。**",
        "",
        "## 实验范围与完整性",
        "",
        "- 数据集：OfficeHome、TerraIncognita。",
        "- 算法：ERM、CORAL、GroupDRO、IRM、VREx。",
        "- 设计：4 个目标域 × 3 个 hparams seeds × 3 个 trial seeds，共 **360** 条全源域任务。",
        "- 三种方法均在同一任务、同一目标 `in` 样本上评估，共 **1,080** 条方法级结果。",
        "- Source-Acc 与 AC-NC 的 Accuracy、hard ECE、hard CwECE 从本轮逐样本预测计算；NLL 使用同一评估中直接从 logits 计算的日志值，避免 float32 概率下溢。没有拼接旧表中的 soft ECE/CwECE。",
        "- 完整性检查通过：360 个 SWAD `done/result.json`、360 对基线预测文件、任务键、目标样本数、两种基线样本索引与路径均一致；所有指标有限。",
        "",
        "## 数据集总体结果",
        "",
        "指标为 180 个任务的等权平均；Accuracy、ECE、CwECE 的单位均为 `%`。",
        "",
        "| 数据集 | 方法 | Accuracy ↑ | NLL ↓ | ECE ↓ | CwECE ↓ |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    approach_labels = {
        "source_acc": "Source-Acc", "ac_nc": "AC-NC",
        "checkpoint_swad": "checkpoint-SWAD+BN-recal"
    }
    for dataset in DATASETS:
        for approach in APPROACHES:
            row = dataset_index[(dataset, approach)]
            lines.append(
                f"| {dataset} | {approach_labels[approach]} | {fmt_metric('accuracy', row['accuracy'])} | "
                f"{fmt_metric('nll', row['nll'])} | {fmt_metric('ece', row['ece'])} | "
                f"{fmt_metric('cwece', row['cwece'])} |"
            )
    lines.extend([
        "",
        "## 分算法结果与配对差值",
        "",
        "每行含 36 个任务。`ΔAcc` 为百分点；`ΔNLL` 为原始 NLL 差；括号内为按 9 个 seed 块重采样的 95% bootstrap 区间。",
        "",
        "| 数据集 | 算法 | SWAD Acc | ΔAcc vs Source-Acc | ΔAcc vs AC-NC | SWAD NLL | ΔNLL vs Source-Acc | ΔNLL vs AC-NC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for dataset in DATASETS:
        for algorithm in ALGORITHMS:
            swad = summary_index[(dataset, algorithm, "checkpoint_swad")]
            acc_source = lookup_comparison(comparisons, "source_acc", "accuracy", dataset, algorithm)
            acc_ac = lookup_comparison(comparisons, "ac_nc", "accuracy", dataset, algorithm)
            nll_source = lookup_comparison(comparisons, "source_acc", "nll", dataset, algorithm)
            nll_ac = lookup_comparison(comparisons, "ac_nc", "nll", dataset, algorithm)
            lines.append(
                f"| {dataset} | {algorithm} | {fmt_metric('accuracy', swad['accuracy'])} | "
                f"{fmt_delta(acc_source)} | {fmt_delta(acc_ac)} | {swad['nll']:.3f} | "
                f"{nll_source['mean_delta']:+.3f} [{nll_source['ci_low']:+.3f}, {nll_source['ci_high']:+.3f}] | "
                f"{nll_ac['mean_delta']:+.3f} [{nll_ac['ci_low']:+.3f}, {nll_ac['ci_high']:+.3f}] |"
            )

    lines.extend([
        "",
        "## 校准指标的总体配对比较",
        "",
        "ECE/CwECE 差值单位为百分点，负值表示 checkpoint-SWAD 更低。",
        "",
        "| 对照 | ΔECE [95% CI] | ΔCwECE [95% CI] | Accuracy 胜率 | NLL 胜率 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for baseline, label in (("source_acc", "Source-Acc"), ("ac_nc", "AC-NC")):
        ece = lookup_comparison(comparisons, baseline, "ece")
        cwece = lookup_comparison(comparisons, baseline, "cwece")
        accuracy = lookup_comparison(comparisons, baseline, "accuracy")
        nll = lookup_comparison(comparisons, baseline, "nll")
        lines.append(
            f"| {label} | {fmt_delta(ece)} | {fmt_delta(cwece)} | "
            f"{100 * accuracy['win_rate']:.1f}% | {100 * nll['win_rate']:.1f}% |"
        )

    lines.extend([
        "",
        "## True worst-domain Accuracy",
        "",
        "对每个 `dataset × algorithm × hparams_seed × trial_seed` 先在 4 个目标域取最小 Accuracy，再对 9 组配置求平均。单位为 `%`。",
        "",
        "| 数据集 | 算法 | Source-Acc | AC-NC | checkpoint-SWAD+BN-recal |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for dataset in DATASETS:
        for algorithm in ALGORITHMS:
            values = [
                100 * worst_index[(dataset, algorithm, approach)]["true_worst_accuracy"]
                for approach in APPROACHES
            ]
            lines.append(
                f"| {dataset} | {algorithm} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} |"
            )

    lines.extend([
        "",
        "## LossValley 选择区间诊断",
        "",
        "`selected count` 保留官方队列规则产生的重复端点；`unique count` 为实际不同 checkpoint 数。",
        "",
        "| 数据集 | 算法 | 首个 step | 末个 step | selected count | unique count | dead valley | fallback-last |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for dataset in DATASETS:
        for algorithm in ALGORITHMS:
            row = selection_index[(dataset, algorithm)]
            lines.append(
                f"| {dataset} | {algorithm} | {row['mean_first_selected_step']:.0f} | "
                f"{row['mean_last_selected_step']:.0f} | {row['mean_selected_count']:.1f} | "
                f"{row['mean_unique_selected_count']:.1f} | {100 * row['dead_valley_rate']:.1f}% | "
                f"{100 * row['fallback_last_rate']:.1f}% |"
            )

    phase_hours = {
        phase: compute[f"total_{phase}_seconds"] / 3600
        for phase in ("validation", "averaging", "bn", "target_evaluation", "model_save")
    }
    lines.extend([
        "",
        "## 计算成本",
        "",
        f"- 360 条任务各阶段累计 **{compute['total_phase_hours']:.1f} GPU-process hours**，平均 **{compute['mean_phase_minutes_per_run']:.1f} 分钟/任务**。这是各任务阶段时间之和，不是 8 并发下的墙钟时间。",
        f"- 从首个 `config.json` 到最后一个 `result.json` 的文件时间跨度为 **{compute['observed_file_span_hours']:.1f} 小时**，包含首次失败、修复和续跑间隔，可作为本次实际日历耗时记录。",
        f"- 阶段累计：source validation {phase_hours['validation']:.1f} h；权重平均 {phase_hours['averaging']:.1f} h；BN 重校准 {phase_hours['bn']:.1f} h；目标评估 {phase_hours['target_evaluation']:.1f} h；模型保存 {phase_hours['model_save']:.1f} h。",
        f"- 峰值显存：均值 {compute['mean_peak_gpu_memory_gb']:.2f} GiB，最大 {compute['max_peak_gpu_memory_gb']:.2f} GiB。",
        f"- 360 个平均模型共 {compute['total_model_gib']:.1f} GiB。Source-Acc 与 AC-NC 的预测已经随全源域实验保存，本次汇总没有新增训练成本。",
        "",
        "## 与原实验计划的关系",
        "",
        "- 对齐部分：OfficeHome/TerraIncognita、5 种算法、4×3×3 的固定超参数任务、目标 `in` 子集、统一 hard-bin 指标、同任务配对和 seed-block bootstrap 均与计划一致。",
        "- checkpoint-SWAD 的 LossValley 只使用三个源域的验证 NLL；目标域没有参与选点、平均或 BN 更新，因此不存在目标泄漏。",
        "- 本次 360 条轨迹全部为 `freeze_bn=false`，并在平均后使用 500 批源域训练样本重估 BN。这对应非冻结 BN 的条件分支；若对照采用冻结 BN 统计的论文 DG 配置，当前结果应视为 `checkpoint-SWAD+BN-recal` 变体。现有输出没有 no-recalibration 反事实，不能分离 BN 重估与参数平均的贡献。",
        "- 该结果是计划之外的新增轨迹平均基线。它不能回答 `AC − LODO-Acc` 或 `AC-LODO − LODO-Acc`，也不能替代 1,080 条内层训练的 LODO 主实验。",
        "- 由于保存间隔为 100 step，每个端点代表官方 SWAD 中不可恢复的稠密训练段；报告结论应使用“checkpoint-SWAD 近似”。",
        "",
        "## 复现文件",
        "",
        "- `run_metrics.csv`：360 任务 × 3 方法的统一目标指标。",
        "- `summary_by_dataset_algorithm.csv`：按数据集、算法和方法汇总。",
        "- `paired_comparisons.csv`：总体、数据集和数据集×算法层面的配对差值与区间。",
        "- `selection_diagnostics.csv`：每条 SWAD 的选择区间与资源字段。",
        "- `selection_summary.csv`：选择区间分组汇总。",
        "- `true_worst_domain.csv`：四目标域内先取最差后的汇总。",
        "- `compute_costs.json`：累计阶段时间、显存和模型存储。",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-root", required=True)
    parser.add_argument("--swad-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, selections = collect(args.full_root, args.swad_root)
    summaries = group_summary(rows, ("dataset", "algorithm", "approach"))
    comparisons = comparison_tables(rows, args.bootstrap_samples)
    selection_stats = selection_summary(selections)
    worst_stats = true_worst_summary(rows)
    compute = compute_summary(selections)

    write_csv(output / "run_metrics.csv", rows)
    write_csv(output / "summary_by_dataset_algorithm.csv", summaries)
    write_csv(output / "paired_comparisons.csv", comparisons)
    write_csv(output / "selection_diagnostics.csv", selections)
    write_csv(output / "selection_summary.csv", selection_stats)
    write_csv(output / "true_worst_domain.csv", worst_stats)
    (output / "compute_costs.json").write_text(json.dumps(compute, indent=2, sort_keys=True) + "\n")
    report = report_markdown(
        rows, selections, summaries, comparisons, selection_stats, worst_stats, compute
    )
    (output / "checkpoint_swad_report_zh.md").write_text(report)
    print(f"wrote report and tables to {output}")


if __name__ == "__main__":
    main()
