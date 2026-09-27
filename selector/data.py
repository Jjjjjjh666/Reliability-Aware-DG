from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from selector.config import (
    CHECKPOINT_COLUMNS,
    REQUIRED_CHECKPOINT_COLUMNS,
    RUN_KEYS,
    ResultSource,
)


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def find_results_files(root: Path, *, require_done: bool = False) -> list[Path]:
    if (root / "results.jsonl").exists():
        result_files = [root / "results.jsonl"]
    else:
        result_files = sorted(root.glob("*/results.jsonl"))
    if require_done:
        result_files = [
            path for path in result_files if (path.parent / "done").exists()
        ]
    return result_files


def mean(values: Iterable[float | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return np.nan
    return float(sum(clean) / len(clean))


def clean_float(value) -> float:
    if value is None:
        return np.nan
    return float(value)


def prefer_metric(primary: float, fallback: float) -> float:
    if np.isnan(primary):
        return fallback
    return primary


def rows_matching(rows: list[dict], role: str, split: str) -> list[dict]:
    return [
        row
        for row in rows
        if row.get("role") == role and row.get("split") == split
    ]


def metric_mean(rows: list[dict], key: str) -> float:
    return mean(row.get(key) for row in rows)


def target_domain_name(record: dict, target_rows: list[dict]) -> str:
    if target_rows:
        return str(target_rows[0].get("domain_name") or target_rows[0].get("domain"))
    test_envs = record.get("args", {}).get("test_envs") or []
    if test_envs:
        return f"env{test_envs[0]}"
    return ""


def model_path_for(run_dir: Path, step: int | None) -> str:
    if step is None:
        return ""
    candidates = [
        run_dir / f"model_step{step}.pkl",
        run_dir / f"model_step{step:05d}.pkl",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def test_env_index(record: dict) -> int | None:
    test_envs = record.get("args", {}).get("test_envs") or []
    if not test_envs:
        return None
    return int(test_envs[0])


def source_env_indices(record: dict) -> list[int]:
    test_env = test_env_index(record)
    envs = []
    for i in range(100):
        if f"env{i}_out_acc" not in record:
            break
        if i != test_env:
            envs.append(i)
    return envs


def env_acc(record: dict, env: int | None, split: str) -> float:
    if env is None:
        return np.nan
    return clean_float(record.get(f"env{env}_{split}_acc"))


def env_acc_mean(record: dict, envs: Iterable[int], split: str) -> float:
    return mean(record.get(f"env{env}_{split}_acc") for env in envs)


def record_to_checkpoint_row(
    record: dict,
    *,
    source_label: str | None,
    results_path: Path,
) -> dict:
    args = record.get("args", {})
    algorithm = str(args.get("algorithm", "unknown"))
    method = source_label or algorithm
    domain_rows = record.get("per_domain_calibration") or []
    class_rows = record.get("per_domain_class_calibration") or []

    source_in = rows_matching(domain_rows, "source", "in")
    source_out = rows_matching(domain_rows, "source", "out")
    target_in = rows_matching(domain_rows, "target", "in")
    target_class_in = rows_matching(class_rows, "target", "in")

    target_accs = [row.get("accuracy") for row in target_in if row.get("accuracy") is not None]
    target_class_accs = [
        row.get("class_accuracy")
        for row in target_class_in
        if row.get("class_accuracy") is not None
    ]
    step = record.get("step")
    step = int(step) if step is not None else None
    test_env = test_env_index(record)
    source_envs = source_env_indices(record)
    source_in_acc = prefer_metric(
        env_acc_mean(record, source_envs, "in"),
        metric_mean(source_in, "accuracy"),
    )
    source_out_acc = prefer_metric(
        env_acc_mean(record, source_envs, "out"),
        metric_mean(source_out, "accuracy"),
    )
    target_acc = prefer_metric(
        env_acc(record, test_env, "in"),
        metric_mean(target_in, "accuracy"),
    )

    run_dir = results_path.parent
    return {
        "dataset": args.get("dataset", ""),
        "test_domain": target_domain_name(record, target_in),
        "method_family": method,
        "method": method,
        "seed": args.get("seed", ""),
        "hparams_seed": args.get("hparams_seed", ""),
        "trial_seed": args.get("trial_seed", ""),
        "step": step,
        "run_dir": str(run_dir),
        "results_path": str(results_path),
        "model_path": model_path_for(run_dir, step),
        "source_in_acc": source_in_acc,
        "source_out_acc": source_out_acc,
        "source_out_ece": metric_mean(source_out, "ece"),
        "source_out_cwece": metric_mean(source_out, "classwise_ece"),
        "source_out_nll": metric_mean(source_out, "nll"),
        "target_acc": target_acc,
        "target_ece": metric_mean(target_in, "ece"),
        "target_cwece": metric_mean(target_in, "classwise_ece"),
        "target_nll": metric_mean(target_in, "nll"),
        "target_worst_domain": prefer_metric(
            target_acc,
            min(target_accs) if target_accs else np.nan,
        ),
        "target_worst_class": min(target_class_accs) if target_class_accs else np.nan,
    }


def validate_checkpoint_table(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_CHECKPOINT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"checkpoint table is missing required columns: {missing}")
    null_required = [
        column
        for column in REQUIRED_CHECKPOINT_COLUMNS
        if df[column].isna().any()
    ]
    if null_required:
        raise ValueError(f"checkpoint table has null required columns: {null_required}")


def load_checkpoint_table(
    sources: list[ResultSource],
    *,
    datasets: set[str] | None = None,
    require_done: bool = False,
) -> pd.DataFrame:
    rows = []
    for source in sources:
        if not source.root.exists():
            raise FileNotFoundError(source.root)
        result_files = find_results_files(source.root, require_done=require_done)
        if not result_files:
            raise FileNotFoundError(f"no results.jsonl files under {source.root}")
        for results_path in result_files:
            for record in iter_jsonl(results_path):
                dataset = record.get("args", {}).get("dataset")
                if datasets is not None and dataset not in datasets:
                    continue
                rows.append(
                    record_to_checkpoint_row(
                        record,
                        source_label=source.label,
                        results_path=results_path,
                    )
                )

    checkpoints = pd.DataFrame(rows)
    if checkpoints.empty:
        raise RuntimeError("no checkpoint rows loaded")

    validate_checkpoint_table(checkpoints)
    checkpoints["seed"] = checkpoints["seed"].astype(int)
    checkpoints["hparams_seed"] = checkpoints["hparams_seed"].astype(int)
    checkpoints["trial_seed"] = checkpoints["trial_seed"].astype(int)
    checkpoints["step"] = checkpoints["step"].astype(int)
    duplicated = checkpoints.duplicated(RUN_KEYS + ["step"], keep=False)
    if duplicated.any():
        example = checkpoints.loc[duplicated, RUN_KEYS + ["step"]].iloc[0].to_dict()
        raise ValueError(f"duplicate checkpoint key: {example}")

    checkpoints["source_out_acc_star"] = checkpoints.groupby(RUN_KEYS, sort=False)[
        "source_out_acc"
    ].transform("max")
    checkpoints = checkpoints.sort_values(
        ["dataset", "method_family", "method", "seed", "test_domain", "step"],
        kind="mergesort",
    ).reset_index(drop=True)

    for column in CHECKPOINT_COLUMNS:
        if column not in checkpoints.columns:
            checkpoints[column] = np.nan
    return checkpoints[CHECKPOINT_COLUMNS]
