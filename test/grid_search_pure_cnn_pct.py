#!/usr/bin/env python3
"""Parallel grid search launcher for WinnowNet pure_cnn_pct training."""

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
DEFAULT_OUTDIR = REPO_ROOT / "data" / "grid_search_pure_cnn_pct"

EPOCH_RE = re.compile(r"^Epoch\s+(?P<epoch>\d+),")


@dataclass(frozen=True)
class Trial:
    trial_id: str
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    epochs: int
    class_weight: str
    pct_loss_weight: float
    run_dir: Path
    model_path: Path
    log_path: Path


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return parsed


def _nonnegative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _csv_values(value, cast, flag_name):
    values = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(cast(item))
        except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
            raise argparse.ArgumentTypeError(f"{flag_name} contains invalid value {item!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{flag_name} must contain at least one value")
    return values


def _csv_count(value):
    return len([item for item in str(value).split(",") if item.strip()])


def _normalize_paths(value):
    paths = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        path = Path(item)
        if not path.is_absolute():
            path = REPO_ROOT / path
        paths.append(str(path))
    if not paths:
        raise argparse.ArgumentTypeError("path list must not be empty")
    return ",".join(paths)


def _format_float(value):
    text = f"{value:.0e}" if 0 < abs(value) < 0.001 else f"{value:g}"
    return text.replace("+", "")


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _extract_float(text, pattern):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else float("nan")


def _extract_int(text, pattern):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else -1


def _make_trial(args, learning_rate, train_batch_size, eval_batch_size, epochs, class_weight, pct_loss_weight):
    trial_id = (
        f"pure_cnn_pct_lr{_safe_name(_format_float(learning_rate))}"
        f"_train{train_batch_size}_eval{eval_batch_size}"
        f"_ep{epochs}_cw{_safe_name(class_weight)}"
        f"_pctloss{_safe_name(_format_float(pct_loss_weight))}"
    )
    run_dir = args.outdir / trial_id
    return Trial(
        trial_id=trial_id,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        epochs=epochs,
        class_weight=class_weight,
        pct_loss_weight=pct_loss_weight,
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
        "--target-pct",
        args.target_pct,
        "--decoy-pct",
        args.decoy_pct,
        "-m",
        str(trial.model_path),
        "--model-arch",
        "pure_cnn_pct",
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
        "--pct-loss-weight",
        f"{trial.pct_loss_weight:g}",
    ]
    if args.exclude_protein_prefix:
        command.extend(["--exclude-protein-prefix", args.exclude_protein_prefix])
    if args.target_exclude:
        command.extend(["--target-exclude", args.target_exclude])
    if args.decoy_exclude:
        command.extend(["--decoy-exclude", args.decoy_exclude])
    if args.pretrained_model:
        command.extend(["-p", str(Path(args.pretrained_model).resolve())])
    return command


def _parse_log(log_path):
    best = None
    if not log_path.exists():
        return {}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not EPOCH_RE.search(line):
                continue
            row = {
                "best_epoch": int(EPOCH_RE.search(line).group("epoch")),
                "train_loss": _extract_float(line, r"Train_loss:\s*([-+0-9.eE]+)"),
                "train_acc": _extract_float(line, r"Train_acc\s+([-+0-9.eE]+)%"),
                "val_loss": _extract_float(line, r"Val_loss:\s*([-+0-9.eE]+)"),
                "val_acc": _extract_float(line, r"Val_acc\s+([-+0-9.eE]+)%"),
                "best_threshold": _extract_float(line, r"BestThreshold\s+([-+0-9.eE]+)"),
                "best_targets": _extract_int(line, r"BestTargets@FDR<=1%\s+(-?\d+)"),
                "best_val_fdr": _extract_float(line, r"BestValFDR\s+([-+0-9.eE]+)%"),
                "best_mean_pct_recall": _extract_float(line, r"BestMeanPctRecall\s+([-+0-9.eE]+)%"),
                "best_min_pct_recall": _extract_float(line, r"BestMinPctRecall\s+([-+0-9.eE]+)%"),
            }
            if best is None or _result_sort_key(row) > _result_sort_key(best):
                best = row
    return best or {}


def _run_trial(args, trial):
    trial.run_dir.mkdir(parents=True, exist_ok=True)
    command = _build_command(args, trial)
    command_text = " ".join(shlex.quote(part) for part in command)
    (trial.run_dir / "command.sh").write_text(command_text + "\n", encoding="utf-8")

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
        log_handle.write("# " + command_text + "\n")
        log_handle.flush()
        process = subprocess.run(
            command,
            cwd=str(trial.run_dir),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )

    result = {"status": "ok" if process.returncode == 0 else "failed", "returncode": process.returncode}
    result.update(_parse_log(trial.log_path))
    return trial, result


def _result_sort_key(row):
    try:
        best_mean_pct_recall = float(row.get("best_mean_pct_recall", float("nan")))
    except (TypeError, ValueError):
        best_mean_pct_recall = float("nan")
    try:
        best_min_pct_recall = float(row.get("best_min_pct_recall", float("nan")))
    except (TypeError, ValueError):
        best_min_pct_recall = float("nan")
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
    if best_mean_pct_recall != best_mean_pct_recall:
        best_mean_pct_recall = float("-inf")
    if best_min_pct_recall != best_min_pct_recall:
        best_min_pct_recall = float("-inf")
    return best_mean_pct_recall, best_min_pct_recall, best_targets, -best_val_fdr, val_acc


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
        "pct_loss_weight",
        "best_epoch",
        "best_targets",
        "best_val_fdr",
        "best_mean_pct_recall",
        "best_min_pct_recall",
        "best_threshold",
        "val_acc",
        "val_loss",
        "train_acc",
        "train_loss",
        "model_path",
        "log_path",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pure_cnn_pct grid search with three parallel jobs by default."
    )
    parser.add_argument("--target", default="data/pct2,data/pct5,data/pct25,data/pct50,data/pct99")
    parser.add_argument(
        "--decoy",
        default="data/pct1,data/spike_control,data/mouse_fecal_12C_glucose,data/mouse_fecal_12C_inulin",
    )
    parser.add_argument("--target-pct", default="2,5,25,50,99")
    parser.add_argument("--decoy-pct", default="1,1,1,1")
    parser.add_argument("--target-exclude", default="0,0,5,5,5")
    parser.add_argument("--decoy-exclude", default="5,5,5,5")
    parser.add_argument("--exclude-protein-prefix", default="Con_")
    parser.add_argument("--learning-rates", default="1e-4,2e-4,3e-4")
    parser.add_argument("--train-batch-sizes", default="512,1024,2048")
    parser.add_argument("--eval-batch-sizes", default="4096")
    parser.add_argument("--epochs-list", default="80")
    parser.add_argument("--class-weights", default="none")
    parser.add_argument("--pct-loss-weights", default="0.1,0.5,1.0")
    parser.add_argument("--jobs", type=_positive_int, default=5)
    parser.add_argument("--max-runs", type=_positive_int, default=None)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--summary", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pretrained-model", default="")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if _csv_count(args.target) != _csv_count(args.target_pct):
        raise ValueError("--target-pct must provide one value per --target input item.")
    if _csv_count(args.decoy) != _csv_count(args.decoy_pct):
        raise ValueError("--decoy-pct must provide one value per --decoy input item.")
    if args.target_exclude and _csv_count(args.target) != _csv_count(args.target_exclude):
        raise ValueError("--target-exclude must provide one value per --target input item.")
    if args.decoy_exclude and _csv_count(args.decoy) != _csv_count(args.decoy_exclude):
        raise ValueError("--decoy-exclude must provide one value per --decoy input item.")

    args.target = _normalize_paths(args.target)
    args.decoy = _normalize_paths(args.decoy)
    args.outdir = Path(args.outdir).resolve()
    args.summary = Path(args.summary).resolve() if args.summary else args.outdir / "summary.tsv"
    args.learning_rates = _csv_values(args.learning_rates, _positive_float, "--learning-rates")
    args.train_batch_sizes = _csv_values(args.train_batch_sizes, _positive_int, "--train-batch-sizes")
    args.eval_batch_sizes = _csv_values(args.eval_batch_sizes, _positive_int, "--eval-batch-sizes") if args.eval_batch_sizes else None
    args.epochs_values = _csv_values(args.epochs_list, _positive_int, "--epochs-list")
    args.class_weights = _csv_values(args.class_weights, str, "--class-weights")
    args.pct_loss_weights = _csv_values(args.pct_loss_weights, _nonnegative_float, "--pct-loss-weights")
    for class_weight in args.class_weights:
        if class_weight not in {"none", "balanced"}:
            raise ValueError("--class-weights values must be none or balanced.")
    return args


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    trials = []
    for lr, train_bs, epochs, class_weight, pct_loss_weight in itertools.product(
        args.learning_rates,
        args.train_batch_sizes,
        args.epochs_values,
        args.class_weights,
        args.pct_loss_weights,
    ):
        for eval_bs in args.eval_batch_sizes or [train_bs]:
            trials.append(_make_trial(args, lr, train_bs, eval_bs, epochs, class_weight, pct_loss_weight))
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
                "pct_loss_weight": trial.pct_loss_weight,
                "model_path": trial.model_path,
                "log_path": trial.log_path,
                **result,
            }
            rows.append(row)
            rows = sorted(rows, key=_result_sort_key, reverse=True)
            _write_summary(args.summary, rows)
            print(
                f"[{len(rows)}/{len(trials)}] {trial.trial_id} "
                f"status={row['status']} "
                f"best_mean_pct_recall={row.get('best_mean_pct_recall', '')} "
                f"best_targets={row.get('best_targets', '')} log={trial.log_path}"
            )

    rows = sorted(rows, key=_result_sort_key, reverse=True)
    _write_summary(args.summary, rows)
    print(f"summary={args.summary}")
    if rows:
        best = rows[0]
        print(
            "best="
            f"{best['trial_id']} "
            f"best_mean_pct_recall={best.get('best_mean_pct_recall', '')} "
            f"best_min_pct_recall={best.get('best_min_pct_recall', '')} "
            f"best_targets={best.get('best_targets', '')} "
            f"best_epoch={best.get('best_epoch', '')} "
            f"best_val_fdr={best.get('best_val_fdr', '')} "
            f"model={best['model_path']}"
        )


if __name__ == "__main__":
    main()
