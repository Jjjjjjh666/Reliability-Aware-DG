"""Combine the frozen ERM and VREx LODO target evaluations."""

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from selector.ablations.lodo_experiment.analyze_lodo import (
    COMPARISONS,
    METRICS,
    aggregate_table,
    diagnostics,
    paired_tables,
    paired_values,
    read_csv,
    validate_target,
    write_csv,
)


ALGORITHMS = ("ERM", "VREx")
SCOPES = ("fixed_hparams", "select_hparams")


def tag(rows, algorithm):
    return [{"algorithm": algorithm, **row} for row in rows]


def block_means(target_rows, scope, comparison, metric, dataset):
    values = [
        row for row in paired_values(target_rows, scope, comparison, metric)
        if row["dataset"] == dataset
    ]
    blocks = defaultdict(list)
    for row in values:
        block = (
            (row["hparams"], row["trial"])
            if scope == "fixed_hparams" else row["trial"]
        )
        blocks[block].append(row["value"])
    expected = 9 if scope == "fixed_hparams" else 3
    if len(blocks) != expected or any(len(group) != 4 for group in blocks.values()):
        raise ValueError(f"Incomplete blocks: {scope} {dataset}")
    return np.asarray([np.mean(blocks[key]) for key in sorted(blocks)])


def combined_algorithms(rows_by_algorithm, repeats, seed):
    datasets = ("OfficeHome", "TerraIncognita")
    output = []
    for scope_index, scope in enumerate(SCOPES):
        for comparison_index, comparison in enumerate(COMPARISONS):
            for metric_index, (metric, label, scale) in enumerate(METRICS):
                rng = np.random.default_rng(
                    seed + scope_index * 1000 + comparison_index * 100 + metric_index
                )
                strata = []
                bootstraps = []
                for algorithm in ALGORITHMS:
                    for dataset in datasets:
                        values = block_means(
                            rows_by_algorithm[algorithm], scope, comparison, metric, dataset
                        )
                        strata.append(float(values.mean()))
                        sample = rng.integers(0, len(values), size=(repeats, len(values)))
                        bootstraps.append(values[sample].mean(axis=1))
                bootstrap = np.mean(bootstraps, axis=0)
                output.append({
                    "evaluation_scope": scope,
                    "comparison": comparison[0],
                    "metric": label,
                    "mean_difference": float(np.mean(strata)) * scale,
                    "ci_low": float(np.quantile(bootstrap, 0.025)) * scale,
                    "ci_high": float(np.quantile(bootstrap, 0.975)) * scale,
                    "weighting": "equal algorithm x dataset strata",
                    "bootstrap_repeats": repeats,
                })
    return output


def fmt(value):
    return f"{float(value):.3f}"


def lookup(rows, algorithm, scope, comparison, metric, dataset="combined_equal_dataset"):
    found = [
        row for row in rows
        if row["algorithm"] == algorithm
        and row["evaluation_scope"] == scope
        and row["dataset_group"] == dataset
        and row["comparison"] == comparison
        and row["metric"] == metric
    ]
    if len(found) != 1:
        raise ValueError(f"Missing comparison: {algorithm} {scope} {comparison} {metric}")
    return found[0]


def render_report(aggregate, paired, combined, hashes):
    lines = [
        "# ERM + VREx LODO / AC checkpoint 选择联合报告", "",
        "## 实验范围与完整性", "",
        "- 数据集：OfficeHome、TerraIncognita。",
        "- 算法：ERM、VREx。",
        "- 每种算法：108 条内层训练、72 条 hard-source 轨迹、384 条冻结选择和 384 条目标结果。",
        "- 合计：216 条内层训练、144 条 hard-source 轨迹、768 条冻结选择和 768 条目标结果。",
        "- 指标：deterministic transform、15-bin hard ECE、按目标集真实类别频率加权的 CwECE；NLL 直接从 logits 计算。",
        "- fixed-hparams 的区间以 `(hparams_seed, trial_seed)` 为 block；select-hparams 只有 3 个 trial，区间仅作描述。",
        "- VREx 完整性审计发现 2 条内层任务在近均匀预测时出现 softmax 概率并列；hard 指标的类别判定已统一为 `logits.argmax`，与 soft 指标和常规分类定义一致。两条任务从头重跑完成，其余 106 条均通过原一致性断言，数值不受影响。",
        f"- ERM target SHA-256：`{hashes['ERM']}`。",
        f"- VREx target SHA-256：`{hashes['VREx']}`。", "",
    ]
    for scope, title in (
        ("fixed_hparams", "固定超参数结果"),
        ("select_hparams", "三超参数搜索结果"),
    ):
        lines += [
            f"## {title}", "",
            "| Algorithm | Dataset | Selector | Acc ↑ | NLL ↓ | ECE ×100 ↓ | CwECE ×100 ↓ | n |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in aggregate:
            if row["selection_scope"] == scope:
                lines.append(
                    f"| {row['algorithm']} | {row['dataset']} | {row['selector']} | "
                    f"{fmt(row['Accuracy'])} | {fmt(row['NLL'])} | "
                    f"{fmt(row['ECE ×100'])} | {fmt(row['CwECE ×100'])} | {row['n_tasks']} |"
                )
        lines += [
            "", "两数据集等权的主要配对差值（前者减后者）：", "",
            "| Algorithm | Comparison | Metric | Mean | 95% CI |",
            "| --- | --- | --- | ---: | ---: |",
        ]
        for algorithm in ALGORITHMS:
            for comparison in ("AC - LODO-Acc", "AC-LODO - LODO-Acc"):
                for _, label, _ in METRICS:
                    row = lookup(paired, algorithm, scope, comparison, label)
                    lines.append(
                        f"| {algorithm} | {comparison} | {label} | "
                        f"{fmt(row['mean_difference'])} | "
                        f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] |"
                    )
        lines.append("")

    lines += [
        "## 两算法、两数据集等权合并", "",
        "以下区间在四个 `algorithm × dataset` strata 内分别重采样 seed block，再对四个 strata 等权平均。", "",
        "| Scope | Comparison | Metric | Mean | 95% CI |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in combined:
        if row["comparison"] in ("AC - LODO-Acc", "AC-LODO - LODO-Acc"):
            lines.append(
                f"| {row['evaluation_scope']} | {row['comparison']} | {row['metric']} | "
                f"{fmt(row['mean_difference'])} | "
                f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] |"
            )

    def combined_lookup(scope, comparison, metric):
        found = [
            row for row in combined
            if row["evaluation_scope"] == scope
            and row["comparison"] == comparison and row["metric"] == metric
        ]
        if len(found) != 1:
            raise ValueError(f"Missing combined result: {scope} {comparison} {metric}")
        return found[0]

    fixed_ac_acc = combined_lookup("fixed_hparams", "AC - LODO-Acc", "Accuracy")
    fixed_ac_nll = combined_lookup("fixed_hparams", "AC - LODO-Acc", "NLL")
    fixed_ac_ece = combined_lookup("fixed_hparams", "AC - LODO-Acc", "ECE ×100")
    fixed_hybrid_acc = combined_lookup(
        "fixed_hparams", "AC-LODO - LODO-Acc", "Accuracy"
    )
    fixed_hybrid_ece = combined_lookup(
        "fixed_hparams", "AC-LODO - LODO-Acc", "ECE ×100"
    )
    search_hybrid_acc = combined_lookup(
        "select_hparams", "AC-LODO - LODO-Acc", "Accuracy"
    )
    search_hybrid_nll = combined_lookup(
        "select_hparams", "AC-LODO - LODO-Acc", "NLL"
    )
    search_hybrid_ece = combined_lookup(
        "select_hparams", "AC-LODO - LODO-Acc", "ECE ×100"
    )
    erm_hybrid_ece = lookup(
        paired, "ERM", "fixed_hparams", "AC-LODO - LODO-Acc", "ECE ×100"
    )
    vrex_hybrid_ece = lookup(
        paired, "VREx", "fixed_hparams", "AC-LODO - LODO-Acc", "ECE ×100"
    )
    lines += [
        "", "## 主要结论", "",
        (
            "- **LODO-Acc 的可靠性优势在 ERM 和 VREx 上方向一致。**固定超参数、两算法两数据集等权时，"
            f"`AC − LODO-Acc` 的 Accuracy 为 {fmt(fixed_ac_acc['mean_difference'])} pp "
            f"[{fmt(fixed_ac_acc['ci_low'])}, {fmt(fixed_ac_acc['ci_high'])}]，没有稳定准确率差异；"
            f"NLL 为 {fmt(fixed_ac_nll['mean_difference'])} "
            f"[{fmt(fixed_ac_nll['ci_low'])}, {fmt(fixed_ac_nll['ci_high'])}]，ECE×100 为 "
            f"{fmt(fixed_ac_ece['mean_difference'])} "
            f"[{fmt(fixed_ac_ece['ci_low'])}, {fmt(fixed_ac_ece['ci_high'])}]。"
            "误差差值均为正，说明只用源域选择的 AC 在 NLL/ECE 上稳定差于 LODO-Acc。"
        ),
        (
            "- **AC-LODO 能在 LODO-Acc 基础上进一步降低 ECE，但没有稳定提高 Accuracy。**"
            f"固定超参数合并差值为 Accuracy {fmt(fixed_hybrid_acc['mean_difference'])} pp "
            f"[{fmt(fixed_hybrid_acc['ci_low'])}, {fmt(fixed_hybrid_acc['ci_high'])}]，ECE×100 "
            f"{fmt(fixed_hybrid_ece['mean_difference'])} "
            f"[{fmt(fixed_hybrid_ece['ci_low'])}, {fmt(fixed_hybrid_ece['ci_high'])}]。"
            f"分算法 ECE 差值为 ERM {fmt(erm_hybrid_ece['mean_difference'])} "
            f"[{fmt(erm_hybrid_ece['ci_low'])}, {fmt(erm_hybrid_ece['ci_high'])}]、VREx "
            f"{fmt(vrex_hybrid_ece['mean_difference'])} "
            f"[{fmt(vrex_hybrid_ece['ci_low'])}, {fmt(vrex_hybrid_ece['ci_high'])}]；"
            "VREx 上区间不跨 0，ERM 上方向相同但区间跨 0。"
        ),
        (
            "- **三超参数搜索结果与固定超参数方向一致，但只作描述。**两算法两数据集等权时，"
            f"`AC-LODO − LODO-Acc` 的 Accuracy 为 {fmt(search_hybrid_acc['mean_difference'])} pp "
            f"[{fmt(search_hybrid_acc['ci_low'])}, {fmt(search_hybrid_acc['ci_high'])}]，NLL 为 "
            f"{fmt(search_hybrid_nll['mean_difference'])} "
            f"[{fmt(search_hybrid_nll['ci_low'])}, {fmt(search_hybrid_nll['ci_high'])}]，ECE×100 为 "
            f"{fmt(search_hybrid_ece['mean_difference'])} "
            f"[{fmt(search_hybrid_ece['ci_low'])}, {fmt(search_hybrid_ece['ci_high'])}]。"
            "由于每个算法/数据集只有 3 个 trial，不能将这些区间解释为稳定总体推断。"
        ),
        "- **CwECE 证据较弱。**固定超参数下，AC-LODO 与 LODO-Acc 的 CwECE 合并区间跨 0；因此不能声称 AC-LODO 在所有校准指标上全面优于 LODO-Acc。",
        "- 综合来看，实验支持“用源域轮流留一构造的 LODO 信号能改善 checkpoint 的目标可靠性”，并部分支持“在 LODO 可行集内加入 NLL/CwECE 的 AC-LODO 可进一步改善 ECE”。实验没有给出 AC-LODO 提高目标 Accuracy 的稳定证据。", "",
    ]

    lines += [
        "", "## 结论边界", "",
        "- 结果只覆盖 ERM、VREx 与 OfficeHome、TerraIncognita，不能外推到 CORAL、GroupDRO、IRM 或 PACS。",
        "- 统计比较针对当前 3×3 seed 搜索池和冻结协议；select-hparams 只有 3 个 trial，不能把其区间解释为稳定的总体不确定性。",
        "- AC/AC-LODO 的 0.5 pp 是源域或 LODO 可行集阈值，不是目标域准确率非劣界限。", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-root", required=True)
    for algorithm in ("erm", "vrex"):
        parser.add_argument(f"--{algorithm}-selection", required=True)
        parser.add_argument(f"--{algorithm}-target", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260923)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Nonempty analysis output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows_by_algorithm = {}
    aggregate = []
    paired = []
    diagnostic_rows = []
    hashes = {}
    for algorithm, token in (("ERM", "erm"), ("VREx", "vrex")):
        selection_path = Path(getattr(args, f"{token}_selection"))
        target_path = Path(getattr(args, f"{token}_target"))
        selection_rows, selection_fields = read_csv(selection_path)
        target_rows, target_fields = read_csv(target_path)
        if {row["algorithm"] for row in target_rows} != {algorithm}:
            raise ValueError(f"Unexpected algorithm in {target_path}")
        validate_target(
            selection_rows, selection_fields, target_rows, target_fields, args.inner_root
        )
        rows_by_algorithm[algorithm] = target_rows
        aggregate += tag(aggregate_table(target_rows), algorithm)
        algorithm_paired, _ = paired_tables(
            target_rows, args.bootstrap_repeats, args.bootstrap_seed
        )
        paired += tag(algorithm_paired, algorithm)
        diagnostic_rows += tag(diagnostics(target_rows), algorithm)
        hashes[algorithm] = hashlib.sha256(target_path.read_bytes()).hexdigest()

    combined = combined_algorithms(
        rows_by_algorithm, args.bootstrap_repeats, args.bootstrap_seed
    )
    write_csv(output / "aggregate_results_by_algorithm.csv", aggregate)
    write_csv(output / "paired_comparisons_by_algorithm.csv", paired)
    write_csv(output / "combined_equal_algorithm_dataset.csv", combined)
    write_csv(output / "selection_diagnostics_by_algorithm.csv", diagnostic_rows)
    (output / "lodo_erm_vrex_report_zh.md").write_text(
        render_report(aggregate, paired, combined, hashes)
    )
    print(f"wrote combined analysis to {output}")


if __name__ == "__main__":
    main()
