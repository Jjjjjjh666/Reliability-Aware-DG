"""Resume a planned inner sweep without touching incomplete or outer runs."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue


def validate_completed(job, output):
    if not (output / "done").exists():
        return False
    config_path = output / "config.json"
    metrics_path = output / "metrics.jsonl"
    if not config_path.exists() or not metrics_path.exists():
        raise RuntimeError(f"Completed inner run is missing files: {output}")
    config = json.loads(config_path.read_text())
    expected = {
        "dataset": job["dataset"], "algorithm": job["algorithm"],
        "hparams_seed": job["hparams_seed"], "trial_seed": job["trial_seed"],
        "excluded_envs": sorted(job["excluded_envs"]),
        "hparams": job["hparams"], "holdout_fraction": job["holdout_fraction"],
        "steps": 5001, "checkpoint_freq": 100,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise RuntimeError(f"Completed inner {field} mismatch: {output}")
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if [row.get("step") for row in rows] != list(range(0, 5001, 100)):
        raise RuntimeError(f"Completed inner step grid mismatch: {output}")
    return True


def launch(job, gpu, data_dir):
    output = Path(job["output_dir"])
    if validate_completed(job, output):
        return "already_done", str(output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Incomplete inner run requires inspection: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial.{uuid.uuid4().hex}")
    command = [
        sys.executable, "-m", "selector.ablations.lodo_experiment.train_inner",
        "--data-dir", data_dir,
        "--dataset", job["dataset"],
        "--algorithm", job["algorithm"],
        "--exclude-envs", *(str(x) for x in job["excluded_envs"]),
        "--hparams-seed", str(job["hparams_seed"]),
        "--trial-seed", str(job["trial_seed"]),
        "--hparams-json", json.dumps(job["hparams"]),
        "--holdout-fraction", str(job["holdout_fraction"]),
        "--output-dir", str(temporary),
        "--steps", "5001",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("TORCH_HOME", str(Path(data_dir).resolve().parent / "torch_cache"))
    with output.with_suffix(".log").open("w") as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(
            f"Inner training failed ({result.returncode}): {output}; partial={temporary}"
        )
    if output.exists():
        raise RuntimeError(f"Canonical output appeared while job was running: {output}")
    os.replace(temporary, output)
    validate_completed(job, output)
    return "completed", str(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest_bytes = Path(args.manifest).read_bytes()
    jobs = [json.loads(line) for line in manifest_bytes.splitlines() if line.strip()]
    pending = []
    for job in jobs:
        if not validate_completed(job, Path(job["output_dir"])):
            pending.append(job)
    if args.limit is not None:
        pending = pending[:args.limit]
    print(
        f"manifest={len(jobs)} pending={len(pending)} "
        f"slots={len(args.gpus) * args.jobs_per_gpu} "
        f"sha256={hashlib.sha256(manifest_bytes).hexdigest()}"
    )
    if args.dry_run:
        for job in pending[:5]:
            print(job["dataset"], job["algorithm"], job["excluded_envs"], job["output_dir"])
        return
    slots = [gpu for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
    queue = Queue()
    for job in pending:
        queue.put(job)

    def worker(gpu):
        failures = 0
        while True:
            try:
                job = queue.get_nowait()
            except Empty:
                return failures
            try:
                status, path = launch(job, gpu, args.data_dir)
            except Exception as error:
                status, path = "FAILED", str(error)
            finally:
                queue.task_done()
            print(status, path, flush=True)
            failures += status == "FAILED"

    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in slots]
        failures = sum(future.result() for future in as_completed(futures))
    if failures:
        raise SystemExit(f"{failures} inner jobs failed; inspect their logs")


if __name__ == "__main__":
    main()
