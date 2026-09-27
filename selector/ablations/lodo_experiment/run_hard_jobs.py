"""Evaluate all full-source checkpoints with the manuscript hard metrics."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue

from selector.ablations.lodo_experiment.pipeline import ALGORITHMS, DATASETS, full_runs


def output_path(hard_root, key):
    dataset, algorithm, target, hparams_seed, trial_seed = key
    return (Path(hard_root) / dataset / algorithm /
            f"h{hparams_seed}_t{trial_seed}" / f"target_{target}.jsonl")


def launch(key, run_dir, output, data_dir, gpu):
    if output.with_suffix(".done").exists():
        return "already_done", str(output)
    if output.exists():
        raise RuntimeError(f"Incomplete hard-metric evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "selector.ablations.lodo_experiment.evaluate_full_hard",
        "--full-run", str(run_dir), "--data-dir", data_dir,
        "--output", str(output),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("TORCH_HOME", str(Path(data_dir).resolve().parent / "torch_cache"))
    with output.with_suffix(".log").open("w") as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Hard-metric evaluation failed ({result.returncode}): {output}")
    return "completed", str(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-root", required=True)
    parser.add_argument("--hard-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS[:2]))
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs-per-gpu", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    runs = full_runs(args.full_root)
    jobs = [
        (key, path.parent, output_path(args.hard_root, key))
        for key, (path, _) in runs.items()
        if key[0] in args.datasets and key[1] in args.algorithms
    ]
    pending = [job for job in jobs if not job[2].with_suffix(".done").exists()]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"full_source={len(jobs)} pending={len(pending)}")
    if args.dry_run:
        for key, path, output in pending[:5]:
            print(key, path, output)
        return
    queue = Queue()
    for job in pending:
        queue.put(job)

    def worker(gpu):
        failures = 0
        while True:
            try:
                key, path, output = queue.get_nowait()
            except Empty:
                return failures
            try:
                status, result = launch(key, path, output, args.data_dir, gpu)
            except Exception as error:
                status, result = "FAILED", str(error)
            finally:
                queue.task_done()
            print(status, result, flush=True)
            failures += status == "FAILED"

    slots = [gpu for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in slots]
        failures = sum(future.result() for future in as_completed(futures))
    if failures:
        raise SystemExit(f"{failures} hard-metric evaluations failed")


if __name__ == "__main__":
    main()
