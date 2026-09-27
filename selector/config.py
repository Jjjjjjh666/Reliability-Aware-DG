from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "selection"

RUN_KEYS = ["dataset", "test_domain", "method", "seed"]
TRUE_WORST_KEYS = ["dataset", "method", "selection_rule"]
EPS = 1e-12

DEFAULT_DELTA_PP = 0.5

# Names follow the final paper: N = NLL, E = ECE, and C = CwECE.
AC_OBJECTIVE_SETS = {
    "ECE": ["source_out_ece"],
    "NLL": ["source_out_nll"],
    "CwECE": ["source_out_cwece"],
    "NC": ["source_out_nll", "source_out_cwece"],
    "NE": ["source_out_nll", "source_out_ece"],
    "NEC": ["source_out_nll", "source_out_ece", "source_out_cwece"],
}
LEGACY_OBJECTIVE_ALIASES = {"A": "NC", "B": "NE", "C": "NEC"}
DEFAULT_OBJECTIVE_SETS = ["NC"]
DEFAULT_DISTANCES = ["linf"]

CHECKPOINT_COLUMNS = [
    "dataset",
    "test_domain",
    "method_family",
    "method",
    "seed",
    "hparams_seed",
    "trial_seed",
    "step",
    "run_dir",
    "results_path",
    "model_path",
    "source_in_acc",
    "source_out_acc",
    "source_out_ece",
    "source_out_cwece",
    "source_out_nll",
    "target_acc",
    "target_ece",
    "target_cwece",
    "target_nll",
    "target_worst_domain",
    "target_worst_class",
    "source_out_acc_star",
]

# Target metrics are evaluation-only. They are deliberately not required for
# source-only checkpoint selection.
REQUIRED_CHECKPOINT_COLUMNS = [
    "dataset",
    "test_domain",
    "method_family",
    "method",
    "seed",
    "hparams_seed",
    "trial_seed",
    "step",
    "source_out_acc",
    "source_out_ece",
    "source_out_cwece",
    "source_out_nll",
]

SELECTED_COLUMNS = [
    "dataset",
    "test_domain",
    "method_family",
    "method",
    "seed",
    "hparams_seed",
    "trial_seed",
    "step",
    "run_dir",
    "results_path",
    "model_path",
    "selector_family",
    "objective_set",
    "distance",
    "reliability_metrics",
    "selection_rule",
    "selection_score",
    "delta_pp",
    "feasible_count",
    "source_out_acc_star",
    "source_out_acc_gap_pp",
    "source_in_acc",
    "source_out_acc",
    "source_out_ece",
    "source_out_cwece",
    "source_out_nll",
    "target_acc",
    "target_ece",
    "target_cwece",
    "target_nll",
    "target_worst_domain",
    "true_worst_domain",
    "target_worst_class",
]


@dataclass(frozen=True)
class ResultSource:
    root: Path
    label: str | None = None
