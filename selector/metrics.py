from __future__ import annotations

import pandas as pd

from selector.config import RUN_KEYS, TRUE_WORST_KEYS


def attach_true_worst_domain(selected: pd.DataFrame) -> pd.DataFrame:
    selected = selected.drop(columns=["true_worst_domain"], errors="ignore")
    true_worst = (
        selected.groupby(TRUE_WORST_KEYS, sort=False)["target_acc"]
        .min()
        .rename("true_worst_domain")
        .reset_index()
    )
    return selected.merge(
        true_worst,
        on=TRUE_WORST_KEYS,
        how="left",
        validate="many_to_one",
    )


def summarize_selected(selected: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    with_worst = attach_true_worst_domain(selected)
    rows = []
    for key, group in with_worst.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["selector_family"] = group["selector_family"].iloc[0]
        row["objective_set"] = group["objective_set"].iloc[0]
        row["distance"] = group["distance"].iloc[0]
        row["reliability_metrics"] = group["reliability_metrics"].iloc[0]
        row["n_runs"] = len(group)
        row["mean_selected_step"] = group["step"].mean()
        row["source_out_acc_gap_pp"] = group["source_out_acc_gap_pp"].mean()
        row["source_out_acc"] = group["source_out_acc"].mean()
        row["target_acc"] = group["target_acc"].mean()
        row["target_ece"] = group["target_ece"].mean()
        row["target_cwece"] = group["target_cwece"].mean()
        row["target_nll"] = group["target_nll"].mean()
        row["true_worst_domain"] = group.drop_duplicates(TRUE_WORST_KEYS)[
            "true_worst_domain"
        ].mean()
        row["target_worst_class"] = group["target_worst_class"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def add_oracle_regrets(
    selected: pd.DataFrame,
    checkpoints: pd.DataFrame,
) -> pd.DataFrame:
    oracle = (
        checkpoints.groupby(RUN_KEYS, sort=False)
        .agg(
            oracle_target_acc=("target_acc", "max"),
            oracle_target_ece=("target_ece", "min"),
            oracle_target_cwece=("target_cwece", "min"),
            oracle_target_nll=("target_nll", "min"),
        )
        .reset_index()
    )
    out = selected.merge(oracle, on=RUN_KEYS, how="left", validate="many_to_one")
    out["acc_regret"] = out["oracle_target_acc"] - out["target_acc"]
    out["ece_regret"] = out["target_ece"] - out["oracle_target_ece"]
    out["cwece_regret"] = out["target_cwece"] - out["oracle_target_cwece"]
    out["nll_regret"] = out["target_nll"] - out["oracle_target_nll"]
    return out


def build_pairwise_vs_out_acc(selected: pd.DataFrame) -> pd.DataFrame:
    selected = attach_true_worst_domain(selected)
    baseline = selected[selected["selection_rule"] == "source_acc"].copy()
    other = selected[selected["selection_rule"] != "source_acc"].copy()
    if baseline.empty or other.empty:
        return pd.DataFrame()

    base_cols = RUN_KEYS + [
        "step",
        "source_out_acc",
        "target_acc",
        "target_ece",
        "target_cwece",
        "target_nll",
        "true_worst_domain",
        "target_worst_class",
    ]
    baseline = baseline[base_cols].rename(
        columns={column: f"out_acc_{column}" for column in base_cols if column not in RUN_KEYS}
    )
    paired = other.merge(baseline, on=RUN_KEYS, how="inner", validate="many_to_one")

    paired["selected_same_step"] = paired["step"] == paired["out_acc_step"]
    paired["step_delta"] = paired["step"] - paired["out_acc_step"]
    paired["source_out_acc_delta_pp"] = (
        paired["source_out_acc"] - paired["out_acc_source_out_acc"]
    ) * 100.0
    paired["target_acc_delta_pp"] = (
        paired["target_acc"] - paired["out_acc_target_acc"]
    ) * 100.0
    paired["target_ece_delta"] = paired["target_ece"] - paired["out_acc_target_ece"]
    paired["target_cwece_delta"] = paired["target_cwece"] - paired["out_acc_target_cwece"]
    paired["target_nll_delta"] = paired["target_nll"] - paired["out_acc_target_nll"]
    paired["true_worst_domain_delta_pp"] = (
        paired["true_worst_domain"] - paired["out_acc_true_worst_domain"]
    ) * 100.0
    paired["target_worst_class_delta_pp"] = (
        paired["target_worst_class"] - paired["out_acc_target_worst_class"]
    ) * 100.0

    def aggregate(group_cols: list[str], scope: str) -> pd.DataFrame:
        rows = []
        for key, group in paired.groupby(group_cols, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            row = dict(zip(group_cols, key))
            row["scope"] = scope
            row["n_pairs"] = len(group)
            row["selector_family"] = group["selector_family"].iloc[0]
            row["objective_set"] = group["objective_set"].iloc[0]
            row["distance"] = group["distance"].iloc[0]
            row["reliability_metrics"] = group["reliability_metrics"].iloc[0]
            row["mean_step_delta"] = group["step_delta"].mean()
            row["same_step_rate"] = group["selected_same_step"].mean()
            row["source_out_acc_delta_pp"] = group["source_out_acc_delta_pp"].mean()
            row["target_acc_delta_pp"] = group["target_acc_delta_pp"].mean()
            row["target_acc_win_rate"] = (group["target_acc_delta_pp"] > 0).mean()
            row["target_ece_delta"] = group["target_ece_delta"].mean()
            row["target_ece_improve_rate"] = (group["target_ece_delta"] < 0).mean()
            row["target_cwece_delta"] = group["target_cwece_delta"].mean()
            row["target_cwece_improve_rate"] = (group["target_cwece_delta"] < 0).mean()
            row["target_nll_delta"] = group["target_nll_delta"].mean()
            row["target_nll_improve_rate"] = (group["target_nll_delta"] < 0).mean()
            row["true_worst_domain_delta_pp"] = group["true_worst_domain_delta_pp"].mean()
            row["target_worst_class_delta_pp"] = group["target_worst_class_delta_pp"].mean()
            rows.append(row)
        return pd.DataFrame(rows)

    tables = [
        aggregate(["dataset", "method_family", "selection_rule"], "dataset_method"),
        aggregate(["method_family", "selection_rule"], "method"),
        aggregate(["dataset", "selection_rule"], "dataset"),
        aggregate(["selection_rule"], "overall"),
    ]
    out = pd.concat(tables, ignore_index=True, sort=False)
    for column in ["dataset", "method_family"]:
        if column not in out.columns:
            out[column] = "ALL"
        out[column] = out[column].fillna("ALL")
    return out
