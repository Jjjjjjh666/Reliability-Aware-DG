import torch
import torch.nn.functional as F


def _as_float_list(values):
    """Convert a tensor to plain Python floats for JSON serialization."""
    return [float(v) for v in values.detach().cpu().tolist()]


def _ece_from_confidence(confidence, correctness, n_bins=15):
    """Expected Calibration Error for confidence/correctness pairs."""
    if confidence.numel() == 0:
        return 0.0

    # Uniform bins in [0, 1]
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=confidence.device)
    ece = torch.zeros(1, device=confidence.device, dtype=confidence.dtype)
    n = confidence.numel()

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence >= lower) & (confidence < upper)

        if in_bin.any():
            conf_bin = confidence[in_bin].mean()
            acc_bin = correctness[in_bin].float().mean()
            weight = in_bin.float().mean()
            ece += torch.abs(conf_bin - acc_bin) * weight

    return ece.item()


def expected_calibration_error(logits, labels, n_bins=15):
    """
    Top-label ECE:
      - confidence: max softmax probability
      - correctness: whether argmax prediction is correct
    """
    probs = torch.softmax(logits, dim=1)
    confidence, preds = probs.max(dim=1)
    correctness = preds.eq(labels)
    return _ece_from_confidence(confidence, correctness, n_bins=n_bins)


def domain_confidence_accuracy_residual(logits, labels):
    """
    Domain-level metrics used in stage-1 motivation experiment:
      - accuracy: mean [argmax(logits)==label]
      - avg_confidence: mean max softmax probability
      - residual: |avg_confidence - accuracy|
    """
    probs = torch.softmax(logits, dim=1)
    confidence, preds = probs.max(dim=1)
    accuracy = preds.eq(labels).float().mean().item()
    avg_confidence = confidence.mean().item()
    residual = abs(avg_confidence - accuracy)
    return {
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "residual": residual,
    }


def classwise_confidence_accuracy_residual(logits, labels, num_classes):
    """
    Compute class-wise confidence, accuracy and residual.

    For class c:
      - confidence: mean p(y=c|x) over samples with true label c
      - accuracy:   mean [argmax(logits)==c] over samples with true label c
      - residual:   |confidence - accuracy|
    """
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)

    class_confidences = []
    class_accuracies = []
    class_residuals = []
    class_counts = []

    for c in range(num_classes):
        class_mask = labels.eq(c)
        count_c = int(class_mask.sum().item())
        class_counts.append(count_c)

        if count_c == 0:
            class_confidences.append(float("nan"))
            class_accuracies.append(float("nan"))
            class_residuals.append(float("nan"))
            continue

        conf_c = probs[class_mask, c].mean().item()
        acc_c = preds[class_mask].eq(labels[class_mask]).float().mean().item()
        class_confidences.append(conf_c)
        class_accuracies.append(acc_c)
        class_residuals.append(abs(conf_c - acc_c))

    return {
        "class_confidence": class_confidences,
        "class_accuracy": class_accuracies,
        "class_residual": class_residuals,
        "class_count": class_counts,
    }


def classwise_ece_ovr(logits, labels, num_classes, n_bins=15):
    """
    One-vs-rest class-wise ECE:
      ECE_c uses p(y=c|x) and binary target [label==c].
    """
    probs = torch.softmax(logits, dim=1)
    ece_per_class = []

    for c in range(num_classes):
        confidence_c = probs[:, c]
        target_c = labels.eq(c)
        ece_c = _ece_from_confidence(confidence_c, target_c, n_bins=n_bins)
        ece_per_class.append(ece_c)

    return ece_per_class


def differentiable_ece(logits, labels, n_bins=15, bandwidth=0.1, eps=1e-8):
    """
    Differentiable top-label ECE with Gaussian soft binning.

    This follows the KDE-style loss in ece_renderable.md:
      - confidence is max softmax probability
      - bin membership is a normalized Gaussian kernel
      - calibration gap uses squared error
    """
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).to(confidences.dtype)

    centers = torch.linspace(
        0.0, 1.0, steps=n_bins, device=logits.device, dtype=logits.dtype
    )
    scaled_distances = (confidences[:, None] - centers[None, :]) / bandwidth
    weights = torch.exp(-0.5 * scaled_distances.pow(2))
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)

    bin_weights = weights.sum(dim=0)
    safe_bin_weights = bin_weights.clamp_min(eps)
    bin_accuracies = (weights * accuracies[:, None]).sum(dim=0) / safe_bin_weights
    bin_confidences = (
        weights * confidences[:, None]
    ).sum(dim=0) / safe_bin_weights
    bin_masses = bin_weights / bin_weights.sum().clamp_min(eps)

    return (bin_masses * (bin_accuracies - bin_confidences).pow(2)).sum()


def classwise_differentiable_ece(
    logits,
    labels,
    n_bins=15,
    bandwidth=0.1,
    class_weight="uniform",
    eps=1e-8,
    return_per_class=False,
):
    """
    Class-wise differentiable ECE with Gaussian soft binning.

    For each class k, weights are:
      p_ik * exp(-((p_ik - mu_m)^2) / (2 * bandwidth^2))
    and each class uses squared calibration gaps across soft bins.
    """
    probs = torch.softmax(logits, dim=1)
    num_samples, num_classes = probs.shape
    labels_onehot = torch.zeros_like(probs)
    labels_onehot.scatter_(1, labels.view(-1, 1), 1.0)

    centers = torch.linspace(
        0.0, 1.0, steps=n_bins, device=logits.device, dtype=logits.dtype
    ).view(1, 1, n_bins)

    class_probs = probs.unsqueeze(-1)
    kernel = torch.exp(-((class_probs - centers).pow(2)) / (2 * bandwidth**2))
    weights = class_probs * kernel

    weight_sum = weights.sum(dim=0)
    safe_weight_sum = weight_sum.clamp_min(eps)
    correctness = labels_onehot.unsqueeze(-1)

    bin_accuracies = (weights * correctness).sum(dim=0) / safe_weight_sum
    bin_confidences = (weights * class_probs).sum(dim=0) / safe_weight_sum
    bin_masses = weight_sum / weight_sum.sum(dim=1, keepdim=True).clamp_min(eps)

    ece_per_class = (
        bin_masses * (bin_accuracies - bin_confidences).pow(2)
    ).sum(dim=1)

    if class_weight == "uniform":
        loss = ece_per_class.mean()
    elif class_weight == "frequency":
        class_counts = labels_onehot.sum(dim=0)
        class_weights = class_counts / max(num_samples, 1)
        loss = (ece_per_class * class_weights).sum()
    else:
        raise ValueError("Unknown class_weight: {}".format(class_weight))

    if return_per_class:
        return loss, ece_per_class
    return loss


def domain_class_differentiable_ece(
    logits,
    labels,
    env_ids,
    num_envs,
    num_classes,
    n_bins=15,
    bandwidth=0.1,
    eps=1e-8,
    power=2,
):

    probs = torch.softmax(logits, dim=1)
    centers = torch.linspace(
        0.0, 1.0, steps=n_bins, device=logits.device, dtype=logits.dtype
    ).view(1, 1, n_bins)
    calibration = torch.zeros(
        num_envs, num_classes, device=logits.device, dtype=logits.dtype
    )

    for env_idx in range(num_envs):
        env_mask = env_ids.eq(env_idx)
        if not env_mask.any():
            continue

        probs_env = probs[env_mask]
        labels_env = labels[env_mask]
        labels_onehot = torch.zeros(
            probs_env.shape[0],
            num_classes,
            device=logits.device,
            dtype=logits.dtype,
        )
        labels_onehot.scatter_(1, labels_env.view(-1, 1), 1.0)

        class_probs = probs_env.unsqueeze(-1)
        kernel = torch.exp(
            -((class_probs - centers).pow(2)) / (2 * bandwidth**2)
        )
        weights = kernel / kernel.sum(dim=2, keepdim=True).clamp_min(eps)

        weight_sum = weights.sum(dim=0)
        safe_weight_sum = weight_sum.clamp_min(eps)
        bin_frequency = (
            weights * labels_onehot.unsqueeze(-1)
        ).sum(dim=0) / safe_weight_sum
        bin_confidence = (weights * class_probs).sum(dim=0) / safe_weight_sum
        bin_mass = weight_sum / weight_sum.sum(dim=1, keepdim=True).clamp_min(eps)

        gap = bin_frequency - bin_confidence
        if power == 2:
            gap_loss = gap.pow(2)
        elif power == 1:
            gap_loss = torch.sqrt(gap.pow(2) + eps)
        else:
            raise ValueError("Unknown power: {}".format(power))

        calibration[env_idx] = (bin_mass * gap_loss).sum(dim=1)

    return calibration


def differentiable_calibration_summary(
    logits,
    labels,
    num_classes,
    n_bins=15,
    bandwidth=0.1,
    class_weight="uniform",
):
    """Aggregate differentiable calibration metrics into JSON-safe values."""
    predictions = logits.argmax(dim=1)
    overall_acc = predictions.eq(labels).float().mean().item()
    nll = F.cross_entropy(logits, labels, reduction="mean")
    overall_ece = differentiable_ece(
        logits, labels, n_bins=n_bins, bandwidth=bandwidth
    )
    classwise_ece, ece_per_class = classwise_differentiable_ece(
        logits,
        labels,
        n_bins=n_bins,
        bandwidth=bandwidth,
        class_weight=class_weight,
        return_per_class=True,
    )

    class_accuracies = []
    class_counts = []
    for class_idx in range(num_classes):
        class_mask = labels.eq(class_idx)
        class_count = int(class_mask.sum().item())
        class_counts.append(class_count)
        if class_count == 0:
            class_accuracies.append(None)
        else:
            class_acc = predictions[class_mask].eq(labels[class_mask]).float().mean()
            class_accuracies.append(float(class_acc.item()))

    return {
        "accuracy": float(overall_acc),
        "nll": float(nll.detach().cpu().item()),
        "ece": float(overall_ece.detach().cpu().item()),
        "classwise_ece": float(classwise_ece.detach().cpu().item()),
        "class_accuracy": class_accuracies,
        "class_ece": _as_float_list(ece_per_class),
        "class_count": class_counts,
    }


def calibration_summary(logits, labels, num_classes, n_bins=15):
    """Aggregate useful calibration metrics in one dict."""
    overall_acc = logits.argmax(dim=1).eq(labels).float().mean().item()
    overall_ece = expected_calibration_error(logits, labels, n_bins=n_bins)
    classwise = classwise_confidence_accuracy_residual(
        logits, labels, num_classes=num_classes
    )
    classwise_ece = classwise_ece_ovr(
        logits, labels, num_classes=num_classes, n_bins=n_bins
    )

    return {
        "accuracy": overall_acc,
        "ece": overall_ece,
        "classwise_confidence": classwise["class_confidence"],
        "classwise_accuracy": classwise["class_accuracy"],
        "classwise_residual": classwise["class_residual"],
        "classwise_ece_ovr": classwise_ece,
        "class_count": classwise["class_count"],
    }
