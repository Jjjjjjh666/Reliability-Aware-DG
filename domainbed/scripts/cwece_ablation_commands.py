import argparse
import json


def sweep_command(
    output_dir,
    data_dir,
    algorithm,
    hparams,
    command_launcher,
    n_trials,
    steps=None,
):
    command = [
        "python -m domainbed.scripts.sweep launch",
        "--data_dir={}".format(data_dir),
        "--output_dir={}".format(output_dir),
        "--command_launcher={}".format(command_launcher),
        "--datasets PACS",
        "--algorithms {}".format(algorithm),
        "--n_hparams 1",
        "--n_trials {}".format(n_trials),
        "--single_test_envs",
        "--hparams '{}'".format(json.dumps(hparams, separators=(",", ":"))),
        "--skip_confirmation",
    ]
    if steps is not None:
        command.insert(-1, "--steps {}".format(steps))
    return " \\\n  ".join(command)


def main():
    parser = argparse.ArgumentParser(
        description="Print PACS sweep commands for CwECE-REx ablations."
    )
    parser.add_argument("--data_dir", type=str, default="./domainbed/data")
    parser.add_argument("--output_root", type=str, default="./outputs")
    parser.add_argument("--command_launcher", type=str, default="multi_gpu")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--n_trials", type=int, default=3)
    args = parser.parse_args()

    lambda_var_values = [0.1, 0.3, 1.0, 3.0, 10.0]
    lambda_mean = 0.01
    common_calibration = {
        "resnet50_augmix": False,
        "kde_ece_n_bins": 15,
        "kde_ece_bandwidth": 0.05,
        "cal_warmup_iters": 500,
        "cal_nvar_eps": 1e-8,
    }

    def lambda_tag(value):
        return str(value).replace(".", "p")

    variants = []
    variants.append((
        "sweep_pacs_erm_control",
        "ERMCwECE",
        dict(
            common_calibration,
            lambda_cal_mean=0.0,
            lambda_cal_var=0.0,
            cal_var_type="plain",
        ),
    ))
    variants.append((
        "sweep_pacs_erm_mean001",
        "ERMCwECE",
        dict(
            common_calibration,
            lambda_cal_mean=lambda_mean,
            lambda_cal_var=0.0,
            cal_var_type="plain",
        ),
    ))

    for cal_var_type in ["plain", "nvar", "weighted"]:
        for lambda_var in lambda_var_values:
            variants.append((
                "sweep_pacs_erm_var_{}_lambda{}".format(
                    cal_var_type, lambda_tag(lambda_var)),
                "ERMCwECE",
                dict(
                    common_calibration,
                    lambda_cal_mean=0.0,
                    lambda_cal_var=lambda_var,
                    cal_var_type=cal_var_type,
                ),
            ))

    for lambda_var in lambda_var_values:
        variants.append((
            "sweep_pacs_erm_mean001_var_plain_lambda{}".format(
                lambda_tag(lambda_var)),
            "ERMCwECE",
            dict(
                common_calibration,
                lambda_cal_mean=lambda_mean,
                lambda_cal_var=lambda_var,
                cal_var_type="plain",
            ),
        ))

    common_vrex = dict(
        common_calibration,
        vrex_lambda=10.0,
        vrex_penalty_anneal_iters=500,
    )
    variants.append((
        "sweep_pacs_vrex_control",
        "CwECEREx",
        dict(
            common_vrex,
            lambda_cal_mean=0.0,
            lambda_cal_var=0.0,
            cal_var_type="plain",
        ),
    ))

    for cal_var_type in ["plain", "nvar", "weighted"]:
        for lambda_var in lambda_var_values:
            variants.append((
                "sweep_pacs_vrex_var_{}_lambda{}".format(
                    cal_var_type, lambda_tag(lambda_var)),
                "CwECEREx",
                dict(
                    common_vrex,
                    lambda_cal_mean=0.0,
                    lambda_cal_var=lambda_var,
                    cal_var_type=cal_var_type,
                ),
            ))

    for lambda_var in lambda_var_values:
        variants.append((
            "sweep_pacs_vrex_mean001_var_plain_lambda{}".format(
                lambda_tag(lambda_var)),
            "CwECEREx",
            dict(
                common_vrex,
                lambda_cal_mean=lambda_mean,
                lambda_cal_var=lambda_var,
                cal_var_type="plain",
            ),
        ))

    for name, algorithm, hparams in variants:
        output_dir = "{}/{}".format(args.output_root.rstrip("/"), name)
        print("# {}".format(name))
        print(sweep_command(
            output_dir=output_dir,
            data_dir=args.data_dir,
            algorithm=algorithm,
            hparams=hparams,
            command_launcher=args.command_launcher,
            n_trials=args.n_trials,
            steps=args.steps,
        ))
        print()


if __name__ == "__main__":
    main()
