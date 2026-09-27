from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from selector.config import (
    AC_OBJECTIVE_SETS,
    DEFAULT_DELTA_PP,
    EPS,
    LEGACY_OBJECTIVE_ALIASES,
    RUN_KEYS,
    SELECTED_COLUMNS,
)


@dataclass(frozen=True)
class SelectionRule:
    name: str
    kind: str
    selector_family: str
    objective_set: str
    distance: str
    metrics: tuple[str, ...]
    delta_pp: float | None = None


def canonical_objective_set(name: str) -> str:
    return LEGACY_OBJECTIVE_ALIASES.get(name, name)


def delta_token(delta_pp: float) -> str:
    text = f"{delta_pp:g}".replace(".", "p")
    return f"d{text}pp"


def make_ac_rules(
    *,
    objective_sets: Iterable[str],
    distances: Iterable[str],
    delta_pp: float = DEFAULT_DELTA_PP,
) -> list[SelectionRule]:
    rules = []
    token = delta_token(delta_pp)
    for raw_name in objective_sets:
        objective_set = canonical_objective_set(raw_name)
        if objective_set not in AC_OBJECTIVE_SETS:
            raise ValueError(f"unknown AC objective set: {raw_name}")
        metrics = tuple(AC_OBJECTIVE_SETS[objective_set])
        for distance in distances:
            if distance not in {"l1", "l2", "linf"}:
                raise ValueError(f"unsupported distance: {distance}")
            rules.append(
                SelectionRule(
                    name=f"ac_{objective_set.lower()}_{distance}_{token}",
                    kind="ac",
                    selector_family="ac",
                    objective_set=objective_set,
                    distance=distance,
                    metrics=metrics,
                    delta_pp=delta_pp,
                )
            )
    return rules


def make_selection_rules(
    *,
    objective_sets: Iterable[str],
    distances: Iterable[str],
    delta_pp: float = DEFAULT_DELTA_PP,
) -> list[SelectionRule]:
    return [
        SelectionRule(
            name="source_acc",
            kind="source_acc",
            selector_family="baseline",
            objective_set="accuracy",
            distance="none",
            metrics=("source_out_acc",),
        ),
        *make_ac_rules(
            objective_sets=objective_sets,
            distances=distances,
            delta_pp=delta_pp,
        ),
    ]


def accuracy_candidates(group: pd.DataFrame, delta_pp: float) -> pd.DataFrame:
    delta_acc = delta_pp / 100.0
    acc_star = float(group["source_out_acc"].max())
    candidates = group[group["source_out_acc"] >= acc_star - delta_acc].copy()
    if candidates.empty:
        key = group[RUN_KEYS].iloc[0].to_dict()
        raise ValueError(f"empty candidate set for {key}")
    return candidates


def normalize_metrics(
    candidates: pd.DataFrame,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    values = candidates[list(metrics)].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"non-finite reliability metric in {list(metrics)}")
    mins = values.min(axis=0)
    ranges = values.max(axis=0) - mins
    normalized = (values - mins) / (ranges + EPS)
    normalized.columns = [f"norm__{metric}" for metric in metrics]
    return pd.concat(
        [candidates.reset_index(drop=True), normalized.reset_index(drop=True)],
        axis=1,
    )


def row_with_metadata(
    row: pd.Series,
    rule: SelectionRule,
    *,
    selection_score: float,
    feasible_count: int,
) -> dict:
    out = {
        "dataset": row["dataset"],
        "test_domain": row["test_domain"],
        "method_family": row["method_family"],
        "method": row["method"],
        "seed": row["seed"],
        "hparams_seed": row["hparams_seed"],
        "trial_seed": row["trial_seed"],
        "step": row["step"],
        "run_dir": row["run_dir"],
        "results_path": row["results_path"],
        "model_path": row["model_path"],
        "selector_family": rule.selector_family,
        "objective_set": rule.objective_set,
        "distance": rule.distance,
        "reliability_metrics": ",".join(rule.metrics),
        "selection_rule": rule.name,
        "selection_score": float(selection_score),
        "delta_pp": np.nan if rule.delta_pp is None else float(rule.delta_pp),
        "feasible_count": int(feasible_count),
        "source_out_acc_star": float(row["source_out_acc_star"]),
        "source_out_acc_gap_pp": (
            float(row["source_out_acc_star"]) - float(row["source_out_acc"])
        )
        * 100.0,
        "source_in_acc": row.get("source_in_acc", np.nan),
        "source_out_acc": row["source_out_acc"],
        "source_out_ece": row["source_out_ece"],
        "source_out_cwece": row["source_out_cwece"],
        "source_out_nll": row["source_out_nll"],
        "target_acc": row.get("target_acc", np.nan),
        "target_ece": row.get("target_ece", np.nan),
        "target_cwece": row.get("target_cwece", np.nan),
        "target_nll": row.get("target_nll", np.nan),
        "target_worst_domain": row.get("target_worst_domain", np.nan),
        "target_worst_class": row.get("target_worst_class", np.nan),
    }
    return out


def select_source_acc(group: pd.DataFrame, rule: SelectionRule) -> dict:
    selected = group.sort_values(
        ["source_out_acc", "step"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    return row_with_metadata(
        selected,
        rule,
        selection_score=-float(selected["source_out_acc"]),
        feasible_count=len(group),
    )


def select_ac(group: pd.DataFrame, rule: SelectionRule) -> dict:
    candidates = accuracy_candidates(group, float(rule.delta_pp))
    work = normalize_metrics(candidates, rule.metrics)
    norm_cols = [f"norm__{metric}" for metric in rule.metrics]
    values = work[norm_cols].to_numpy(dtype=float)

    if rule.distance == "l1":
        scores = np.sum(values, axis=1)
    elif rule.distance == "l2":
        scores = np.sqrt(np.sum(values**2, axis=1))
    elif rule.distance == "linf":
        scores = np.max(values, axis=1)
    else:
        raise ValueError(f"unsupported distance: {rule.distance}")

    work["selection_score"] = scores
    selected = work.sort_values(
        ["selection_score", "source_out_acc", "step"],
        ascending=[True, False, True],
        kind="mergesort",
    ).iloc[0]
    return row_with_metadata(
        selected,
        rule,
        selection_score=float(selected["selection_score"]),
        feasible_count=len(candidates),
    )


def select_with_rule(group: pd.DataFrame, rule: SelectionRule) -> dict:
    if rule.kind == "source_acc":
        return select_source_acc(group, rule)
    if rule.kind == "ac":
        return select_ac(group, rule)
    raise ValueError(f"unknown selection rule kind: {rule.kind}")


def select_checkpoints(
    checkpoints: pd.DataFrame,
    rules: list[SelectionRule],
) -> pd.DataFrame:
    rows = []
    for _key, group in checkpoints.groupby(RUN_KEYS, sort=False):
        group = group.sort_values("step", kind="mergesort").copy()
        for rule in rules:
            rows.append(select_with_rule(group, rule))

    selected = pd.DataFrame(rows)
    for column in SELECTED_COLUMNS:
        if column not in selected.columns:
            selected[column] = np.nan
    selected = selected[SELECTED_COLUMNS].sort_values(
        [
            "dataset",
            "method_family",
            "method",
            "seed",
            "test_domain",
            "selector_family",
            "selection_rule",
        ],
        kind="mergesort",
    )
    return selected.reset_index(drop=True)
