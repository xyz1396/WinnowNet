#!/usr/bin/env python3
"""Regenerate CNN feature pickles, then retrain WinnowNet checkpoints.

The default job list matches the existing top-level ``data/*.pt`` CNN/T-Net
checkpoints and runs four training processes at a time.
"""

import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_GENERATOR = REPO_ROOT / "script" / "SpectraFeatures.py"
TRAINER = REPO_ROOT / "script" / "WinnowNet_CNN.py"


@dataclass(frozen=True)
class FeatureJob:
    name: str
    input_paths: str
    log_path: str
    max_peaks: int
    blank_label: bool = False
    config: str = "script/SIP.cfg"
    jobs: int = 15
    threads: int = 3
    window: int = 10
    tolerance: int = 10
    feature_type: str = "cnn"


@dataclass(frozen=True)
class TrainingJob:
    name: str
    target: str
    decoy: str
    model_path: str
    model_arch: str = "pure_cnn"
    epochs: int = 80
    learning_rate: str = "3e-4"
    class_weight: str = "none"
    train_batch_size: int = 512
    eval_batch_size: int = 4096
    exclude_protein_prefix: str = "Con_"
    target_exclude: str = ""
    decoy_exclude: str = ""

    @property
    def log_path(self):
        return str(Path(self.model_path).with_suffix(".log"))


FEATURE_JOBS = [
    FeatureJob(
        name="cnn_features_all",
        input_paths="data/pct1,data/pct2,data/pct5,data/pct25,data/pct50,data/pct99,data/spike_control,data/spike_pct02,data/spike_pct05,data/spike_pct50,data/mouse_fecal_12C_glucose,data/mouse_fecal_12C_inulin,data/mouse_fecal_13C_glucose,data/mouse_fecal_13C_inulin",
        log_path="data/cnn_feature_generation.log",
        max_peaks=256,
    ),
    FeatureJob(
        name="cnn_features_b",
        input_paths="data/pct1b,data/pct2b,data/pct5b,data/spike_control_b,data/spike_pct02b,data/spike_pct05b",
        log_path="data/cnn_feature_generation_b.log",
        max_peaks=128,
        blank_label=True,
    ),
]


JOBS = [
    TrainingJob(
        name="pure_cnn",
        target="data/pct2b",
        decoy="data/pct1b,data/spike_control_b",
        model_path="data/pure_cnn.pt",
    ),
    TrainingJob(
        name="pure_cnn_pct5b",
        target="data/pct5b",
        decoy="data/pct1b,data/spike_control_b",
        model_path="data/pure_cnn_pct5b.pt",
    ),
    TrainingJob(
        name="pure_cnn_pct5",
        target="data/pct5",
        decoy="data/pct1,data/spike_control",
        model_path="data/pure_cnn_pct5.pt",
    ),
    TrainingJob(
        name="pure_cnn_pct50",
        target="data/pct50",
        decoy="data/pct1,data/spike_control",
        model_path="data/pure_cnn_pct50.pt",
    ),
    TrainingJob(
        name="pure_cnn_pct50_5",
        target="data/pct50",
        target_exclude="5",
        decoy="data/pct1,data/spike_control",
        decoy_exclude="5,5",
        model_path="data/pure_cnn_pct50_5.pt",
    ),
    TrainingJob(
        name="pure_cnn_all",
        target="data/pct2,data/pct5,data/pct25,data/pct50,data/pct99",
        target_exclude="0,0,5,5,5",
        decoy="data/pct1,data/spike_control,data/mouse_fecal_12C_glucose,data/mouse_fecal_12C_inulin",
        decoy_exclude="5,5,5,5",
        model_path="data/pure_cnn_all.pt",
    ),
    TrainingJob(
        name="tnet_all",
        target="data/pct2,data/pct5,data/pct25,data/pct50,data/pct99",
        target_exclude="0,0,5,5,5",
        decoy="data/pct1,data/spike_control,data/mouse_fecal_12C_glucose,data/mouse_fecal_12C_inulin",
        decoy_exclude="5,5,5,5",
        model_path="data/tnet_all.pt",
        model_arch="tnet",
        train_batch_size=1024,
    ),
]


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _resolve_csv_paths(value):
    resolved = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        path = Path(item)
        if not path.is_absolute():
            path = REPO_ROOT / path
        resolved.append(str(path))
    if not resolved:
        raise ValueError("path list must not be empty")
    return ",".join(resolved)


def _resolve_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _build_command(args, job):
    command = [
        args.python,
        str(TRAINER),
        "--target",
        _resolve_csv_paths(job.target),
        "--decoy",
        _resolve_csv_paths(job.decoy),
        "-m",
        str(_resolve_path(job.model_path)),
        "--model-arch",
        job.model_arch,
        "--epochs",
        str(job.epochs),
        "--learning-rate",
        job.learning_rate,
        "--class-weight",
        job.class_weight,
        "--train-batch-size",
        str(job.train_batch_size),
        "--eval-batch-size",
        str(job.eval_batch_size),
        "--exclude-protein-prefix",
        job.exclude_protein_prefix,
    ]
    if job.target_exclude:
        command.extend(["--target-exclude", job.target_exclude])
    if job.decoy_exclude:
        command.extend(["--decoy-exclude", job.decoy_exclude])
    return command


def _build_feature_command(args, job):
    command = [
        args.python,
        str(FEATURE_GENERATOR),
        "-c",
        str(_resolve_path(job.config)),
        "-i",
        _resolve_csv_paths(job.input_paths),
        "-j",
        str(job.jobs),
        "-t",
        str(job.threads),
        "-w",
        str(job.window),
        "-d",
        str(job.tolerance),
        "--max-peaks",
        str(job.max_peaks),
        "-f",
        job.feature_type,
    ]
    if job.blank_label:
        command.append("-b")
    return command


def _write_command_file(command_path, command):
    command_path.write_text(
        " ".join(shlex.quote(part) for part in command) + "\n",
        encoding="utf-8",
    )


def _run_feature_job(args, job):
    log_path = _resolve_path(job.log_path)
    command_path = log_path.with_suffix(".command.sh")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _build_feature_command(args, job)
    _write_command_file(command_path, command)

    if args.dry_run:
        return {"status": "dry_run", "returncode": "", "log_path": log_path}

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("# " + " ".join(shlex.quote(part) for part in command) + "\n")
        log_handle.flush()
        process = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )

    return {
        "status": "ok" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "log_path": log_path,
    }


def _run_job(args, job):
    model_path = _resolve_path(job.model_path)
    log_path = _resolve_path(job.log_path)
    command_path = log_path.with_suffix(".command.sh")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _build_command(args, job)
    _write_command_file(command_path, command)

    if args.dry_run:
        return job, {"status": "dry_run", "returncode": "", "log_path": log_path}

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("# " + " ".join(shlex.quote(part) for part in command) + "\n")
        log_handle.flush()
        process = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )

    status = "ok" if process.returncode == 0 else "failed"
    return job, {
        "status": status,
        "returncode": process.returncode,
        "model_path": model_path,
        "log_path": log_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain the data/*.pt WinnowNet CNN/T-Net models from note.md in parallel."
    )
    parser.add_argument("--jobs", type=_positive_int, default=4, help="Parallel training processes.")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated job names to run. Default: all existing data/*.pt jobs from note.md.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child trainers.")
    parser.add_argument("--cuda-visible-devices", default="", help="Optional CUDA_VISIBLE_DEVICES value.")
    parser.add_argument(
        "--skip-feature-generation",
        action="store_true",
        help="Skip regenerating CNN feature pickles before training.",
    )
    parser.add_argument("--features-only", action="store_true", help="Regenerate feature pickles and stop.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running training.")
    return parser.parse_args()


def _select_jobs(only):
    if not only:
        return JOBS
    wanted = {item.strip() for item in only.split(",") if item.strip()}
    known = {job.name for job in JOBS}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError("unknown job(s): " + ", ".join(unknown) + "; known jobs: " + ", ".join(sorted(known)))
    return [job for job in JOBS if job.name in wanted]


def main():
    args = parse_args()
    jobs = _select_jobs(args.only)
    feature_jobs = [] if args.skip_feature_generation else FEATURE_JOBS
    print(f"feature_jobs={len(feature_jobs)} training_jobs={len(jobs)} parallel_jobs={args.jobs}")

    if args.dry_run:
        for feature_job in feature_jobs:
            print(
                f"[{feature_job.name}] "
                + " ".join(shlex.quote(part) for part in _build_feature_command(args, feature_job))
            )
        if args.features_only:
            return
        for job in jobs:
            print(f"[{job.name}] " + " ".join(shlex.quote(part) for part in _build_command(args, job)))
        return

    for completed_count, feature_job in enumerate(feature_jobs, start=1):
        result = _run_feature_job(args, feature_job)
        print(
            f"[features {completed_count}/{len(feature_jobs)}] {feature_job.name} "
            f"status={result['status']} returncode={result['returncode']} "
            f"log={result['log_path']}"
        )
        if result["status"] != "ok":
            print(f"feature_generation_failed={feature_job.name}", file=sys.stderr)
            raise SystemExit(1)

    if args.features_only:
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_job = {executor.submit(_run_job, args, job): job for job in jobs}
        for completed_count, future in enumerate(as_completed(future_to_job), start=1):
            job, result = future.result()
            results.append((job, result))
            print(
                f"[{completed_count}/{len(jobs)}] {job.name} "
                f"status={result['status']} returncode={result['returncode']} "
                f"log={result['log_path']}"
            )

    failed = [job.name for job, result in results if result["status"] != "ok"]
    if failed:
        print("failed_jobs=" + ",".join(failed), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
