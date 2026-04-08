#!/usr/bin/env python3
"""Parallel grid search launcher for WinnowNet pure CNN training.

Each hyperparameter combination runs in its own directory so the trainer's
relative checkpoints/ directory cannot collide across parallel jobs.
"""

import argparse
import csv
import itertools
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER = REPO_ROOT / "script" / "WinnowNet_CNN.py"
DEFAULT_OUTDIR = REPO_ROOT / "data" / "grid_search_pure_cnn"

EPOCH_RE = re.compile(r"^Epoch\s+(?P<epoch>\d+),")
CHECKPOINT_RE = re.compile(r"Checkpoint: .*?\(Epoch (?P<epoch>\d+)\)")
TEST_RE = re.compile(
    r"Test accuracy:\s+(?P<test_acc>[-+0-9.eE]+)%,\s+F1-Score:\s+(?P<test_f1>[-+0-9.eE]+)%"
)


@dataclass(frozen=True)
class Trial:
    trial_id: str
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    epochs: int
    class_weight: str
    run_dir: Path
    model_path: Path
    log_path: Path


def _parse_csv_values(value, cast, flag_name):
    items = []
    for raw_item in str(value).split(","):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        try:
            items.append(cast(raw_item))
        except ValueError as exc:
            raise ValueError(f"{flag_name} contains invalid value {raw_item!r}.") from exc
    if not items:
        raise ValueError(f"{flag_name} must contain at least one value.")
    return items


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise ValueError
    return parsed


def _positive_float(value):
    parsed = float(value)
    if parsed <= 0:
        raise ValueError
    return parsed


def _normalize_input_paths(value):
    """Convert comma-separated relative paths/globs to repo-root absolute paths."""
    normalized = []
    for raw_item in str(value).split(","):
        item = raw_item.strip()
        if not item:
            continue
        path = Path(item)
        if not path.is_absolute():
            path = REPO_ROOT / path
        normalized.append(str(path))
    if not normalized:
        raise ValueError("input path list must not be empty.")
    return ",".join(normalized)


def _format_lr(value):
    text = f"{value:.0e}" if value < 0.001 else f"{value:g}"
    return text.replace("+", "")


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _make_trial(outdir, learning_rate, train_batch_size, eval_batch_size, epochs, class_weight):
    trial_id = (
        f"pure_cnn_lr{_safe_name(_format_lr(learning_rate))}"
        f"_train{train_batch_size}_eval{eval_batch_size}"
        f"_ep{epochs}_cw{_safe_name(class_weight)}"
    )
    run_dir = outdir / trial_id
    return Trial(
        trial_id=trial_id,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        epochs=epochs,
        class_weight=class_weight,
        run_dir=run_dir,
        model_path=run_dir / "model.pt",
        log_path=run_dir / "train.log",
    )


def _build_command(args, trial):
    command = [
        args.python,
        str(TRAINER),
        "--target",
        args.target,
        "--decoy",
        args.decoy,
        "-m",
        str(trial.model_path),
        "--model-arch",
        args.model_arch,
        "--epochs",
        str(trial.epochs),
        "--learning-rate",
        f"{trial.learning_rate:g}",
        "--class-weight",
        trial.class_weight,
        "--train-batch-size",
        str(trial.train_batch_size),
        "--eval-batch-size",
        str(trial.eval_batch_size),
    ]
    if args.exclude_protein_prefix:
        command.extend(["--exclude-protein-prefix", args.exclude_protein_prefix])
    if args.pretrained_model:
        command.extend(["-p", str(Path(args.pretrained_model).resolve())])
    return command


def _parse_log(log_path):
    best = None
    test_by_epoch = {}
    pending_test_epoch = None
    if not log_path.exists():
        return {}

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                parsed = {
                    "best_epoch": int(epoch_match.group("epoch")),
                    "train_loss": _extract_float(line, r"Train_loss:\s*([-+0-9.eE]+)"),
                    "train_acc": _extract_float(line, r"Train_acc\s+([-+0-9.eE]+)%"),
                    "val_loss": _extract_float(line, r"Val_loss:\s*([-+0-9.eE]+)"),
                    "val_acc": _extract_float(line, r"Val_acc\s+([-+0-9.eE]+)%"),
                    "best_threshold": _extract_float(line, r"BestThreshold\s+([-+0-9.eE]+)"),
                    "best_targets": _extract_int(line, r"BestTargets@FDR<=1%\s+(-?\d+)"),
                    "best_val_fdr": _extract_float(line, r"BestValFDR\s+([-+0-9.eE]+)%"),
                }
                if best is None or _is_better_epoch(parsed, best):
                    best = parsed
                continue

            checkpoint_match = CHECKPOINT_RE.search(line)
            if checkpoint_match:
                pending_test_epoch = int(checkpoint_match.group("epoch"))
                continue

            test_match = TEST_RE.search(line)
            if test_match and pending_test_epoch is not None:
                test_by_epoch[pending_test_epoch] = {
                    "test_acc": float(test_match.group("test_acc")),
                    "test_f1": float(test_match.group("test_f1")),
                }
                pending_test_epoch = None

    if best is None:
        return {}
    best.update(test_by_epoch.get(best["best_epoch"], {}))
    return best


def _extract_float(text, pattern):
    match = re.search(pattern, text)
    if not match:
        return float("nan")
    return float(match.group(1))


def _extract_int(text, pattern):
    match = re.search(pattern, text)
    if not match:
        return -1
    return int(match.group(1))


def _is_better_epoch(candidate, current):
    """Prefer more targets at <=1% FDR, then lower FDR, then higher val accuracy."""
    return (
        candidate["best_targets"],
        -candidate["best_val_fdr"],
        candidate["val_acc"],
        -candidate["val_loss"],
    ) > (
        current["best_targets"],
        -current["best_val_fdr"],
        current["val_acc"],
        -current["val_loss"],
    )


def _run_trial(args, trial):
    trial.run_dir.mkdir(parents=True, exist_ok=True)
    command = _build_command(args, trial)
    command_path = trial.run_dir / "command.sh"
    command_path.write_text(
        " ".join(shlex.quote(part) for part in command) + "\n",
        encoding="utf-8",
    )
    if args.skip_existing and trial.log_path.exists() and "done" in trial.log_path.read_text(
        encoding="utf-8", errors="replace"
    ):
        result = {"status": "skipped", "returncode": 0}
        result.update(_parse_log(trial.log_path))
        return trial, result

    if args.dry_run:
        return trial, {"status": "dry_run", "returncode": ""}

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    with trial.log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.run(
            command,
            cwd=str(trial.run_dir),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )

    result = {
        "status": "ok" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
    }
    result.update(_parse_log(trial.log_path))
    return trial, result


def _write_summary(summary_path, rows):
    fieldnames = [
        "trial_id",
        "status",
        "returncode",
        "learning_rate",
        "train_batch_size",
        "eval_batch_size",
        "epochs",
        "class_weight",
        "best_epoch",
        "best_targets",
        "best_val_fdr",
        "best_threshold",
        "val_acc",
        "val_loss",
        "test_acc",
        "test_f1",
        "model_path",
        "log_path",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _result_sort_key(row):
    try:
        best_targets = int(row.get("best_targets", -10**12))
    except (TypeError, ValueError):
        best_targets = -10**12
    try:
        best_val_fdr = float(row.get("best_val_fdr", 100.0))
    except (TypeError, ValueError):
        best_val_fdr = 100.0
    try:
        val_acc = float(row.get("val_acc", -1.0))
    except (TypeError, ValueError):
        val_acc = -1.0
    return best_targets, -best_val_fdr, val_acc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run parallel grid search jobs for script/WinnowNet_CNN.py pure_cnn."
    )
    parser.add_argument("--target", default="data/pct2")
    parser.add_argument("--decoy", default="data/pct1,data/spike_control")
    parser.add_argument("--exclude-protein-prefix", default="Con_")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--summary", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-arch", default="pure_cnn", choices=["pure_cnn", "tnet"])
    parser.add_argument("--learning-rates", default="1e-4,2e-4,3e-4")
    parser.add_argument("--train-batch-sizes", default="128,512,1024,2048")
    parser.add_argument("--eval-batch-sizes", default=None)
    parser.add_argument("--epochs-list", default="30")
    parser.add_argument("--class-weights", default="none")
    parser.add_argument("--jobs", type=_positive_int, default=2)
    parser.add_argument("--max-runs", type=_positive_int, default=None)
    parser.add_argument("--pretrained-model", default="")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.target = _normalize_input_paths(args.target)
    args.decoy = _normalize_input_paths(args.decoy)
    args.outdir = Path(args.outdir).resolve()
    args.summary = Path(args.summary).resolve() if args.summary else args.outdir / "summary.tsv"
    args.learning_rates = _parse_csv_values(args.learning_rates, _positive_float, "--learning-rates")
    args.train_batch_sizes = _parse_csv_values(args.train_batch_sizes, _positive_int, "--train-batch-sizes")
    if args.eval_batch_sizes:
        args.eval_batch_sizes = _parse_csv_values(args.eval_batch_sizes, _positive_int, "--eval-batch-sizes")
    else:
        args.eval_batch_sizes = None
    args.epochs_values = _parse_csv_values(args.epochs_list, _positive_int, "--epochs-list")
    args.class_weights = _parse_csv_values(args.class_weights, str, "--class-weights")
    for class_weight in args.class_weights:
        if class_weight not in {"none", "balanced"}:
            raise ValueError("--class-weights values must be none or balanced.")
    return args


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    trials = []
    for lr, train_bs, epochs, class_weight in itertools.product(
        args.learning_rates,
        args.train_batch_sizes,
        args.epochs_values,
        args.class_weights,
    ):
        eval_batch_sizes = args.eval_batch_sizes or [train_bs]
        for eval_bs in eval_batch_sizes:
            trials.append(_make_trial(args.outdir, lr, train_bs, eval_bs, epochs, class_weight))
    if args.max_runs is not None:
        trials = trials[: args.max_runs]

    print(f"grid_search_trials={len(trials)} jobs={args.jobs} outdir={args.outdir}")
    if args.dry_run:
        for trial in trials:
            print(" ".join(shlex.quote(part) for part in _build_command(args, trial)))

    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_trial = {executor.submit(_run_trial, args, trial): trial for trial in trials}
        for future in as_completed(future_to_trial):
            trial, result = future.result()
            row = {
                "trial_id": trial.trial_id,
                "learning_rate": trial.learning_rate,
                "train_batch_size": trial.train_batch_size,
                "eval_batch_size": trial.eval_batch_size,
                "epochs": trial.epochs,
                "class_weight": trial.class_weight,
                "model_path": trial.model_path,
                "log_path": trial.log_path,
                **result,
            }
            rows.append(row)
            best_targets = row.get("best_targets", "")
            print(
                f"[{len(rows)}/{len(trials)}] {trial.trial_id} "
                f"status={row['status']} best_targets={best_targets} log={trial.log_path}"
            )
            _write_summary(args.summary, sorted(rows, key=_result_sort_key, reverse=True))

    rows = sorted(rows, key=_result_sort_key, reverse=True)
    _write_summary(args.summary, rows)
    print(f"summary={args.summary}")
    if rows:
        best = rows[0]
        print(
            "best="
            f"{best['trial_id']} "
            f"best_targets={best.get('best_targets', '')} "
            f"best_epoch={best.get('best_epoch', '')} "
            f"best_val_fdr={best.get('best_val_fdr', '')} "
            f"model={best['model_path']}"
        )


if __name__ == "__main__":
    main()
