from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selector.config import (
    DEFAULT_DELTA_PP,
    DEFAULT_DISTANCES,
    DEFAULT_OBJECTIVE_SETS,
    OUT_DIR,
    ResultSource,
)
from selector.data import load_checkpoint_table
from selector.metrics import (
    add_oracle_regrets,
    attach_true_worst_domain,
    build_pairwise_vs_out_acc,
    summarize_selected,
)
from selector.reporting import build_report
from selector.rules import make_selection_rules, select_checkpoints


def csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def labeled_path(value: str) -> ResultSource:
    if "=" in value:
        label, path = value.split("=", 1)
        return ResultSource(Path(path).expanduser(), label=label.strip() or None)
    return ResultSource(Path(value).expanduser())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select checkpoints from DomainBed results.jsonl trajectories."
    )
    parser.add_argument(
        "--input-root",
        type=labeled_path,
        action="append",
        required=True,
        help=(
            "Sweep root containing run directories with results.jsonl. Repeat for "
            "multiple roots. Use LABEL=PATH to distinguish variants of the same "
            "training algorithm."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=csv_arg,
        default=None,
        help="Optional comma-separated dataset filter.",
    )
    parser.add_argument(
        "--require-done",
        action="store_true",
        help="Only load run directories containing a done marker.",
    )
    parser.add_argument(
        "--objective-sets",
        "--ac-versions",
        dest="objective_sets",
        type=csv_arg,
        default=DEFAULT_OBJECTIVE_SETS,
        help="Comma-separated AC objective sets: ECE,NLL,CwECE,NC,NE,NEC.",
    )
    parser.add_argument(
        "--distances",
        type=csv_arg,
        default=DEFAULT_DISTANCES,
        help="Comma-separated aggregation distances: l1,l2,linf.",
    )
    parser.add_argument(
        "--delta-pp",
        type=float,
        default=DEFAULT_DELTA_PP,
        help="Source-validation accuracy tolerance in percentage points.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for selections and evaluation tables.",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Also write a Markdown summary.",
    )
    return parser.parse_args()


def write_outputs(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rules = make_selection_rules(
        objective_sets=args.objective_sets,
        distances=args.distances,
        delta_pp=args.delta_pp,
    )
    checkpoints = load_checkpoint_table(
        args.input_root,
        datasets=set(args.datasets) if args.datasets else None,
        require_done=args.require_done,
    )
    selected = attach_true_worst_domain(select_checkpoints(checkpoints, rules))
    summary = summarize_selected(
        selected,
        ["dataset", "method_family", "selection_rule"],
    )
    pairwise = build_pairwise_vs_out_acc(selected)
    oracle_regret = add_oracle_regrets(selected, checkpoints)

    checkpoints.to_csv(args.out_dir / "checkpoint_metrics.csv", index=False)
    selected.to_csv(args.out_dir / "selected_checkpoints.csv", index=False)
    summary.to_csv(args.out_dir / "summary_by_dataset_method.csv", index=False)
    pairwise.to_csv(args.out_dir / "pairwise_vs_source_acc.csv", index=False)
    oracle_regret.to_csv(args.out_dir / "oracle_regret.csv", index=False)

    if args.write_report:
        report = build_report(
            checkpoints,
            selected,
            summary,
            pairwise,
            oracle_regret,
        )
        (args.out_dir / "report.md").write_text(report, encoding="utf-8")

    print(f"Rules: {', '.join(rule.name for rule in rules)}")
    print(f"Loaded checkpoint rows: {len(checkpoints)}")
    print(f"Selected rows: {len(selected)}")
    print(f"Wrote outputs to {args.out_dir}")


def main() -> None:
    write_outputs(parse_args())


if __name__ == "__main__":
    main()
