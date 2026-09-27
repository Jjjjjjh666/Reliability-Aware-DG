"""Metrics for the held-out domains in the LODO experiment."""

import numpy as np
import torch
import torch.nn.functional as F

from domainbed.lib.calibration import differentiable_calibration_summary


def _hard_ece(confidence, correct, n_bins):
    bins = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
    count = np.bincount(bins, minlength=n_bins)
    conf_sum = np.bincount(bins, weights=confidence, minlength=n_bins)
    correct_sum = np.bincount(bins, weights=correct, minlength=n_bins)
    return float(np.abs(correct_sum - conf_sum).sum() / len(confidence))


def hard_summary(logits, labels, n_bins=15):
    """15-bin ECE and true-class-frequency weighted one-vs-rest CwECE."""
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        y = labels.detach().cpu().numpy()
        pred = logits.argmax(dim=1).detach().cpu().numpy()
    confidence = probs[np.arange(len(y)), pred]
    ece = _hard_ece(confidence, pred == y, n_bins)
    cwece = 0.0
    for klass in range(probs.shape[1]):
        frequency = np.mean(y == klass)
        if frequency:
            cwece += frequency * _hard_ece(probs[:, klass], y == klass, n_bins)
    return {
        "accuracy": float(np.mean(pred == y)),
        "nll": float(F.cross_entropy(logits, labels).item()),
        "hard_ece": ece,
        "hard_cwece": float(cwece),
    }


def summarize(logits, labels, n_bins=15, bandwidth=0.1):
    """Return both logged soft metrics and manuscript hard-bin metrics."""
    soft = differentiable_calibration_summary(
        logits, labels, num_classes=logits.shape[1],
        n_bins=n_bins, bandwidth=bandwidth,
    )
    hard = hard_summary(logits, labels, n_bins=n_bins)
    if abs(soft["accuracy"] - hard["accuracy"]) > 1e-5:
        raise ValueError("Soft and hard metric accuracy disagree")
    return {
        **hard,
        "logged_soft_ece": soft["ece"],
        "logged_soft_cwece": soft["classwise_ece"],
    }
