from __future__ import annotations

import pandas as pd

from selector.metrics import summarize_selected


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    text_df = df.copy()
    for column in text_df.columns:
        if pd.api.types.is_float_dtype(text_df[column]):
            text_df[column] = text_df[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
        else:
            text_df[column] = text_df[column].fillna("").astype(str)
    headers = list(text_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in text_df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[column for column in columns if column in df.columns]].copy()


def build_report(
    checkpoints: pd.DataFrame,
    selected: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    oracle_regret: pd.DataFrame,
) -> str:
    coverage = (
        checkpoints.groupby(["dataset", "method_family"], sort=False)
        .agg(
            checkpoint_rows=("step", "size"),
            trajectories=("seed", "nunique"),
        )
        .reset_index()
    )
    selection_summary = summarize_selected(
        selected,
        ["dataset", "method_family", "selection_rule"],
    )
    selection_summary = existing_columns(
        selection_summary,
        [
            "dataset",
            "method_family",
            "selection_rule",
            "n_runs",
            "mean_selected_step",
            "source_out_acc_gap_pp",
            "target_acc",
            "target_ece",
            "target_cwece",
            "target_nll",
        ],
    )
    paired_overall = existing_columns(
        pairwise[pairwise["scope"] == "overall"] if not pairwise.empty else pairwise,
        [
            "selection_rule",
            "n_pairs",
            "source_out_acc_delta_pp",
            "target_acc_delta_pp",
            "target_ece_delta",
            "target_cwece_delta",
            "target_nll_delta",
        ],
    )
    oracle_summary = pd.DataFrame()
    if not oracle_regret.empty:
        oracle_summary = (
            oracle_regret.groupby(
                ["selection_rule", "objective_set", "distance"],
                sort=False,
            )
            .agg(
                n_runs=("step", "size"),
                acc_regret=("acc_regret", "mean"),
                ece_regret=("ece_regret", "mean"),
                cwece_regret=("cwece_regret", "mean"),
                nll_regret=("nll_regret", "mean"),
            )
            .reset_index()
        )

    lines = [
        "# Checkpoint selection report",
        "",
        "Selection uses source-domain validation statistics only. Target metrics in "
        "this report are read after selection and are used only for evaluation.",
        "",
        "## Coverage",
        "",
        markdown_table(coverage),
        "",
        "## Selected checkpoints",
        "",
        markdown_table(selection_summary),
        "",
        "## Paired changes versus Source-Acc",
        "",
        "Positive accuracy changes and negative reliability-error changes favor AC.",
        "",
        markdown_table(paired_overall),
        "",
        "## Target-oracle regret (diagnostic only)",
        "",
        markdown_table(oracle_summary),
        "",
        "## Dataset and method summary",
        "",
        markdown_table(summary),
        "",
    ]
    return "\n".join(lines)
