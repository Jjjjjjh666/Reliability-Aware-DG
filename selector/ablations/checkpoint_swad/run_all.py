"""Run the fixed 360-run OfficeHome/TerraIncognita checkpoint-SWAD baseline."""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue

from selector.ablations.lodo_experiment.pipeline import ALGORITHMS, EXPECTED_STEPS


DATASETS = ("OfficeHome", "TerraIncognita")
PROFILE = "checkpoint_swad_lossvalley_deterministic_v1"


def expected_keys():
    return set(itertools.product(DATASETS, ALGORITHMS, range(4), range(3), range(3)))


def index_completed_runs(root):
    """Read only the first log row; run_one validates the full step grid."""
    runs = {}
    for done in Path(root).glob("*/done"):
        path = done.parent / "results.jsonl"
        with path.open() as source:
            args = json.loads(source.readline())["args"]
        targets = args["test_envs"]
        if len(targets) != 1:
            raise ValueError(f"Expected one target domain: {path}")
        key = (args["dataset"], args["algorithm"], targets[0],
               args["hparams_seed"], args["trial_seed"])
        if key in runs:
            raise ValueError(f"Duplicate full-source run: {key}")
        runs[key] = (path, None)
    return runs


def collect(runs, output_root):
    records = []
    for key, (path, _) in sorted(runs.items()):
        folder = output_root / path.parent.name
        if not (folder / "done").exists():
            raise ValueError(f"Missing completed checkpoint-SWAD result: {folder}")
        record = json.loads((folder / "result.json").read_text())
        if record["profile"] != PROFILE or (
            record["dataset"], record["algorithm"], record["target_domain"],
            record["hparams_seed"], record["trial_seed"],
        ) != key:
            raise ValueError(f"Result metadata mismatch: {folder}")
        records.append(record)
    fields = list(records[0])
    temporary = output_root / "results.csv.tmp"
    with temporary.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                             for key, value in record.items()})
    os.replace(temporary, output_root / "results.csv")
    print(f"wrote {len(records)} rows to {output_root / 'results.csv'}", flush=True)


def launch(path, output, data_dir, gpu):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "done").exists():
        return "already_done", str(output)
    command = [
        sys.executable, "-m", "selector.ablations.checkpoint_swad.run_one",
        "--full-run", str(path.parent), "--data-dir", data_dir,
        "--output-dir", str(output),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("TORCH_HOME", str(Path(data_dir).resolve().parent / "torch_cache"))
    with (output / "run.log").open("a") as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Run failed ({result.returncode}); inspect {output / 'run.log'}")
    return "completed", str(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.jobs_per_gpu < 1 or not args.gpus:
        parser.error("At least one GPU and one job per GPU are required")

    runs = index_completed_runs(args.full_root)
    expected = expected_keys()
    missing = expected - set(runs)
    extra = set(runs) - expected
    print(f"full-source runs: {len(runs)}/360; missing={len(missing)} extra={len(extra)}")
    if missing:
        print("missing examples:", sorted(missing)[:5])
    if extra:
        print("unexpected examples:", sorted(extra)[:5])
    missing_weights = []
    for key, (path, _) in runs.items():
        if key not in expected:
            continue
        present = {file.name for file in path.parent.iterdir()}
        missing_weights.extend(
            str(path.parent / f"model_step{step}.pkl")
            for step in EXPECTED_STEPS
            if f"model_step{step}.pkl" not in present
        )
    print(f"missing checkpoint files: {len(missing_weights)}")
    if missing_weights:
        print("missing examples:", missing_weights[:5])
    if args.dry_run:
        return
    if missing or extra or missing_weights:
        raise SystemExit("Wait for 360 complete runs with all 51 checkpoint files")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    queue = Queue()
    for key, (path, _) in sorted(runs.items()):
        output = output_root / path.parent.name
        if not (output / "done").exists():
            queue.put((path, output))
    print(f"pending={queue.qsize()} workers={len(args.gpus) * args.jobs_per_gpu}", flush=True)

    def worker(gpu):
        failures = 0
        while True:
            try:
                path, output = queue.get_nowait()
            except Empty:
                return failures
            try:
                status, location = launch(path, output, args.data_dir, gpu)
            except Exception as error:
                status, location = "FAILED", str(error)
                failures += 1
            finally:
                queue.task_done()
            print(status, location, flush=True)

    slots = [gpu for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        failures = sum(future.result() for future in
                       [pool.submit(worker, gpu) for gpu in slots])
    if failures:
        raise SystemExit(f"{failures} runs failed; retry after checking their run.log files")
    collect(runs, output_root)


if __name__ == "__main__":
    main()
