import argparse
import json
import os

import torch

from domainbed import algorithms
from domainbed import datasets
from domainbed.lib import misc
from domainbed.lib.calibration import calibration_summary
from domainbed.lib.calibration import domain_confidence_accuracy_residual
from domainbed.lib.fast_data_loader import FastDataLoader


@torch.no_grad()
def collect_logits_labels(algorithm, loader, device):
    logits_list = []
    labels_list = []
    algorithm.eval()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = algorithm.predict(x)
        logits_list.append(logits)
        labels_list.append(y)
    algorithm.train()
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def build_eval_envs(dataset, eval_split, holdout_fraction, trial_seed):
    if eval_split == "full":
        return [env for env in dataset]

    envs = []
    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(
            env,
            int(len(env) * holdout_fraction),
            misc.seed_hash(trial_seed, env_i),
        )
        if eval_split == "in":
            envs.append(in_)
        else:
            envs.append(out)
    return envs


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CMNIST stage-1 metrics and calibration details."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--algorithm", type=str, default=None)
    parser.add_argument("--test_envs", type=int, nargs="+", default=None)
    parser.add_argument("--trial_seed", type=int, default=None)
    parser.add_argument(
        "--eval_split",
        type=str,
        default="full",
        choices=["full", "in", "out"],
        help="Evaluate on full env, in-split or out-split.",
    )
    parser.add_argument("--holdout_fraction", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_bins", type=int, default=15)
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="JSON output path. Default: model dir + calibration_metrics.json",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.model_path, map_location="cpu")
    train_args = checkpoint.get("args", {})

    dataset_name = args.dataset or train_args.get("dataset")
    if dataset_name is None:
        raise ValueError("Dataset is missing. Pass --dataset or use a train checkpoint.")

    algorithm_name = args.algorithm or train_args.get("algorithm")
    if algorithm_name is None:
        raise ValueError("Algorithm is missing. Pass --algorithm or use a train checkpoint.")

    data_dir = args.data_dir or train_args.get("data_dir")
    if data_dir is None:
        raise ValueError("Data dir is missing. Pass --data_dir or use a train checkpoint.")

    test_envs = args.test_envs if args.test_envs is not None else train_args.get("test_envs", [0])
    trial_seed = args.trial_seed if args.trial_seed is not None else train_args.get("trial_seed", 0)
    holdout_fraction = (
        args.holdout_fraction
        if args.holdout_fraction is not None
        else train_args.get("holdout_fraction", 0.2)
    )

    hparams = checkpoint["model_hparams"]
    dataset = vars(datasets)[dataset_name](data_dir, test_envs, hparams)

    algorithm_class = algorithms.get_algorithm_class(algorithm_name)
    algorithm = algorithm_class(
        checkpoint["model_input_shape"],
        checkpoint["model_num_classes"],
        checkpoint["model_num_domains"],
        hparams,
    )
    algorithm.load_state_dict(checkpoint["model_dict"], strict=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    algorithm.to(device)

    eval_envs = build_eval_envs(
        dataset=dataset,
        eval_split=args.eval_split,
        holdout_fraction=holdout_fraction,
        trial_seed=trial_seed,
    )

    env_names = getattr(dataset, "ENVIRONMENTS", [str(i) for i in range(len(eval_envs))])
    domain_results = []
    detailed_results = []

    for env_i, env_dataset in enumerate(eval_envs):
        loader = FastDataLoader(
            dataset=env_dataset, batch_size=args.batch_size, num_workers=dataset.N_WORKERS
        )
        logits, labels = collect_logits_labels(algorithm, loader, device)
        basic_metrics = domain_confidence_accuracy_residual(logits=logits, labels=labels)
        basic_metrics.update(
            {
                "env_index": env_i,
                "env_name": env_names[env_i] if env_i < len(env_names) else str(env_i),
                "split": args.eval_split,
                "is_test_env": env_i in test_envs,
                "num_samples": int(labels.numel()),
            }
        )
        domain_results.append(basic_metrics)

        detailed_metrics = calibration_summary(
            logits=logits, labels=labels, num_classes=dataset.num_classes, n_bins=args.n_bins
        )
        detailed_metrics.update(
            {
                "env_index": env_i,
                "env_name": env_names[env_i] if env_i < len(env_names) else str(env_i),
                "split": args.eval_split,
                "is_test_env": env_i in test_envs,
                "num_samples": int(labels.numel()),
            }
        )
        detailed_results.append(detailed_metrics)

    source_domain_results = [r for r in domain_results if not r["is_test_env"]]
    source_residuals = [r["residual"] for r in source_domain_results]
    source_residual_variance = (
        float(torch.tensor(source_residuals).var(unbiased=False).item())
        if len(source_residuals) >= 2
        else 0.0
    )

    if args.output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(args.model_path)), "calibration_metrics.json"
        )
    else:
        output_path = args.output_path

    with open(output_path, "w") as f:
        json.dump(
            {
                "domain_results": domain_results,
                "source_residual_variance": source_residual_variance,
                "detailed_results": detailed_results,
            },
            f,
            indent=2,
        )

    print(f"Saved calibration metrics to: {output_path}")
    print("=" * 78)
    print("Per-domain overall metrics")
    print("-" * 78)
    for r in domain_results:
        role = "target" if r["is_test_env"] else "source"
        print(
            f"[env{r['env_index']}:{r['env_name']}][{role}] "
            f"acc={r['accuracy']:.4f} conf={r['avg_confidence']:.4f} "
            f"residual={r['residual']:.4f} n={r['num_samples']}"
        )
    print(f"source_residual_variance={source_residual_variance:.6f}")

    print("=" * 78)
    print("Per-domain class-wise metrics (confidence / accuracy / residual)")
    print("-" * 78)
    num_classes = len(detailed_results[0]["classwise_confidence"]) if detailed_results else 0
    header = (
        f"{'env':>18} {'role':>6} "
        + " ".join(
            f"c{c}:{'conf':>6} {'acc':>6} {'|c-a|':>6}" for c in range(num_classes)
        )
    )
    print(header)
    for r in detailed_results:
        role = "target" if r["is_test_env"] else "source"
        tag = f"env{r['env_index']}:{r['env_name']}"
        row = f"{tag:>18} {role:>6} "
        for c in range(num_classes):
            conf_c = r["classwise_confidence"][c]
            acc_c = r["classwise_accuracy"][c]
            res_c = r["classwise_residual"][c]
            row += f"     {conf_c:6.4f} {acc_c:6.4f} {res_c:6.4f}"
        print(row)

    print("-" * 78)
    print("Class-wise residual variance across source domains")
    src_detailed = [r for r in detailed_results if not r["is_test_env"]]
    for c in range(num_classes):
        vals = [r["classwise_residual"][c] for r in src_detailed]
        var_c = (
            float(torch.tensor(vals).var(unbiased=False).item())
            if len(vals) >= 2
            else 0.0
        )
        print(f"  class {c}: residuals={['%.4f' % v for v in vals]}  var={var_c:.6f}")


if __name__ == "__main__":
    main()
