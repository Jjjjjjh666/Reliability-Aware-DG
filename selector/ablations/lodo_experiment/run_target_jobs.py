"""Evaluate a frozen target-selection CSV in resumable per-run shards."""

import argparse
import csv
import hashlib
import os
import subprocess
import sys
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue


TARGET_FIELDS = (
    "target_accuracy", "target_nll", "target_ece", "target_cwece",
    "target_split_hash",
)


def read_csv(path):
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        return list(reader), reader.fieldnames


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row, selection_fields):
    return tuple(row[field] for field in selection_fields)


def shard_path(root, rows):
    values = {
        (row["dataset"], row["algorithm"], row["hparams_seed"],
         row["trial_seed"], row["target_domain"])
        for row in rows
    }
    if len(values) != 1:
        raise ValueError("A target shard must describe exactly one full-source run")
    dataset, algorithm, hparams_seed, trial_seed, target = values.pop()
    return (Path(root) / dataset / algorithm / f"h{hparams_seed}_t{trial_seed}"
            / f"target_{target}.csv")


def validate_shard(path, expected_rows, selection_fields):
    path = Path(path)
    if not path.with_suffix(".done").exists():
        return False
    rows, fields = read_csv(path)
    if fields != selection_fields + list(TARGET_FIELDS):
        raise RuntimeError(f"Target shard fields differ: {path}")
    expected = {row_key(row, selection_fields) for row in expected_rows}
    actual = {row_key(row, selection_fields) for row in rows}
    if len(rows) != len(expected_rows) or actual != expected:
        raise RuntimeError(f"Target shard rows differ: {path}")
    return True


def launch(rows, selection_fields, shard_root, data_dir, gpu):
    output = shard_path(shard_root, rows)
    if validate_shard(output, rows, selection_fields):
        return "already_done", str(output)
    if output.exists():
        raise RuntimeError(f"Incomplete target shard requires inspection: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shard_selection = output.with_suffix(".selection.csv")
    write_csv(shard_selection, rows, selection_fields)
    temporary = output.with_name(f".{output.stem}.partial.{uuid.uuid4().hex}.csv")
    command = [
        sys.executable, "-m", "selector.ablations.lodo_experiment.evaluate_selected_targets",
        "--selection", str(shard_selection),
        "--data-dir", data_dir,
        "--output", str(temporary),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("TORCH_HOME", str(Path(data_dir).resolve().parent / "torch_cache"))
    with output.with_suffix(".log").open("w") as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(
            f"Target evaluation failed ({result.returncode}): {output}; partial={temporary}"
        )
    os.replace(temporary, output)
    output.with_suffix(".done").write_text("done\n")
    validate_shard(output, rows, selection_fields)
    return "completed", str(output)


def combine(selection_rows, selection_fields, shard_root, output):
    evaluated = {}
    grouped = defaultdict(list)
    for row in selection_rows:
        grouped[row["full_source_run"]].append(row)
    for rows in grouped.values():
        path = shard_path(shard_root, rows)
        if not validate_shard(path, rows, selection_fields):
            raise RuntimeError(f"Incomplete target shard: {path}")
        shard_rows, _ = read_csv(path)
        for row in shard_rows:
            key = row_key(row, selection_fields)
            if key in evaluated:
                raise RuntimeError("Duplicate target result")
            evaluated[key] = {field: row[field] for field in TARGET_FIELDS}
    expected = {row_key(row, selection_fields) for row in selection_rows}
    if set(evaluated) != expected:
        raise RuntimeError("Combined target results differ from frozen selection")
    final_rows = [
        {**row, **evaluated[row_key(row, selection_fields)]}
        for row in selection_rows
    ]
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(f".{output.name}.partial.{uuid.uuid4().hex}")
    write_csv(temporary, final_rows, selection_fields + list(TARGET_FIELDS))
    os.replace(temporary, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output}\n")
    return len(final_rows), digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected, selection_fields = read_csv(args.selection)
    if not selected or any(row["metric_profile"] != "hard_deterministic" for row in selected):
        raise ValueError("Target evaluation requires a frozen hard-profile selection")
    if any(field in selection_fields for field in TARGET_FIELDS):
        raise ValueError("Frozen selection already contains target fields")
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["full_source_run"]].append(row)
    jobs = [rows for _, rows in sorted(grouped.items())]
    pending = [
        rows for rows in jobs
        if not validate_shard(shard_path(args.shard_root, rows), rows, selection_fields)
    ]
    slots = [gpu for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
    print(f"selected_rows={len(selected)} runs={len(jobs)} pending={len(pending)} slots={len(slots)}")
    if args.dry_run:
        for rows in pending[:5]:
            print(rows[0]["full_source_run"], shard_path(args.shard_root, rows))
        return

    queue = Queue()
    for rows in pending:
        queue.put(rows)

    def worker(gpu):
        failures = 0
        while True:
            try:
                rows = queue.get_nowait()
            except Empty:
                return failures
            try:
                status, result = launch(rows, selection_fields, args.shard_root, args.data_dir, gpu)
            except Exception as error:
                status, result = "FAILED", str(error)
            finally:
                queue.task_done()
            print(status, result, flush=True)
            failures += status == "FAILED"

    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in slots]
        failures = sum(future.result() for future in as_completed(futures))
    if failures:
        raise SystemExit(f"{failures} target shards failed")
    count, digest = combine(selected, selection_fields, args.shard_root, args.output)
    print(f"wrote {count} target rows; sha256={digest}")


if __name__ == "__main__":
    main()
