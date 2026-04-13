#!/usr/bin/env python3
"""Run CNN prediction jobs for spike-in directories and summarize E. coli ratios."""

import argparse
import csv
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from check_ecoli_ratio import (
    DEFAULT_EXCLUDE_PROTEIN_PREFIXES,
    DEFAULT_QVALUE_THRESHOLD,
    SUMMARY_COLUMNS,
    _format_value,
    _infer_spike_in_13c_pct,
    _parse_prefixes,
    summarize_tsv,
)


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
PREDICTION_SCRIPT = REPO_ROOT / "script" / "Prediction_CNN.py"
SUMMARY_ID_COLUMNS = ["Input", "Model"]
OUTPUT_COLUMNS = SUMMARY_ID_COLUMNS + SUMMARY_COLUMNS


@dataclass(frozen=True)
class PredictionJob:
    input_path: str
    model_path: str


DEFAULT_JOBS = [
    PredictionJob("data/spike_pct02", "data/pure_cnn_all.pt"),
    PredictionJob("data/spike_pct02", "data/pure_cnn_all_pct.pt"),
    PredictionJob("data/spike_pct02", "data/tnet_all.pt"),
    PredictionJob("data/spike_pct02b", "data/pure_cnn.pt"),
    PredictionJob("data/spike_pct05", "data/pure_cnn_all.pt"),
    PredictionJob("data/spike_pct05", "data/pure_cnn_all_pct.pt"),
    PredictionJob("data/spike_pct05", "data/tnet_all.pt"),
    PredictionJob("data/spike_pct05", "data/pure_cnn_pct5.pt"),
    PredictionJob("data/spike_pct05b", "data/pure_cnn_pct5b.pt"),
    PredictionJob("data/spike_pct50", "data/pure_cnn_all.pt"),
    PredictionJob("data/spike_pct50", "data/pure_cnn_all_pct.pt"),
    PredictionJob("data/spike_pct50", "data/tnet_all.pt"),
    PredictionJob("data/spike_pct50", "data/pure_cnn_pct50.pt"),
    PredictionJob("data/spike_pct50", "data/pure_cnn_pct50_5.pt"),
]
DEFAULT_PARALLEL_JOBS = 8


def _checkpoint_label(model_path):
    return Path(model_path).stem


def _safe_stem(value):
    text = str(value).strip().replace("\\", "_").replace("/", "_")
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)


def _output_path_for_job(output_dir, job):
    input_label = _safe_stem(job.input_path)
    model_label = _safe_stem(_checkpoint_label(job.model_path))
    return output_dir / f"{input_label}__{model_label}.tsv"


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _run_prediction_job(job, output_path, python_executable, batch_size, decision_threshold, known_pct):
    command = [
        python_executable,
        str(PREDICTION_SCRIPT),
        "-i",
        job.input_path,
        "-m",
        job.model_path,
        "-o",
        str(output_path),
        "--batch-size",
        str(batch_size),
    ]
    if known_pct is not None:
        command.extend(["--known-pct", str(known_pct)])
    if decision_threshold is not None:
        command.extend(["--decision-threshold", str(decision_threshold)])
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def _summarize_prediction_output(job, output_path, qvalue_threshold, include_all_rows, exclude_protein_prefixes):
    summary = summarize_tsv(
        str(output_path),
        qvalue_threshold=qvalue_threshold,
        include_all_rows=include_all_rows,
        exclude_protein_prefixes=exclude_protein_prefixes,
    )
    summary["Input"] = job.input_path
    summary["Model"] = _checkpoint_label(job.model_path)
    summary["Spike-in 13C (%)"] = _infer_spike_in_13c_pct(job.input_path)
    return summary


def _run_and_summarize_job(
    job,
    output_dir,
    python_executable,
    batch_size,
    decision_threshold,
    qvalue_threshold,
    include_all_rows,
    exclude_protein_prefixes,
):
    output_path = _output_path_for_job(output_dir, job)
    known_pct = _infer_spike_in_13c_pct(job.input_path)
    print(
        f"Predicting {job.input_path} with {_checkpoint_label(job.model_path)} -> {output_path}",
        file=sys.stderr,
    )
    _run_prediction_job(
        job,
        output_path,
        python_executable,
        batch_size,
        decision_threshold,
        known_pct,
    )
    return _summarize_prediction_output(
        job,
        output_path,
        qvalue_threshold,
        include_all_rows,
        exclude_protein_prefixes,
    )


def write_summary(rows, output_path=None):
    if output_path:
        with open(output_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: _format_value(row.get(column, "")) for column in OUTPUT_COLUMNS})
        return

    writer = csv.DictWriter(sys.stdout, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _format_value(row.get(column, "")) for column in OUTPUT_COLUMNS})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Predict the configured spike-in pickle directories with the requested CNN models, "
            "then summarize E. coli ratios in the same style as test/check_ecoli_ratio.py."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output summary TSV path. Defaults to stdout.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run script/Prediction_CNN.py. Default: current interpreter.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=1024,
        help="Prediction batch size passed to script/Prediction_CNN.py. Default: 1024.",
    )
    parser.add_argument(
        "--jobs",
        type=_positive_int,
        default=DEFAULT_PARALLEL_JOBS,
        help=f"Number of prediction jobs to run in parallel. Default: {DEFAULT_PARALLEL_JOBS}.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help="Optional override passed to script/Prediction_CNN.py for all models.",
    )
    parser.add_argument(
        "--q-value",
        type=float,
        default=DEFAULT_QVALUE_THRESHOLD,
        help="Optionally also require predicted target rows to have q-value <= this threshold.",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Summarize all predicted target rows, ignoring any --q-value threshold.",
    )
    parser.add_argument(
        "--exclude-protein-prefix",
        default=",".join(DEFAULT_EXCLUDE_PROTEIN_PREFIXES),
        help=(
            "Drop PSMs when all proteins start with one of these comma-separated prefixes. "
            "Default: Decoy_,Con_."
        ),
    )
    parser.add_argument(
        "--keep-predictions-dir",
        default="",
        help="Optional directory to keep the combined prediction TSVs. Defaults to a temporary directory.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    exclude_protein_prefixes = _parse_prefixes(args.exclude_protein_prefix)

    if args.keep_predictions_dir:
        output_dir = Path(args.keep_predictions_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_context = None
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="predict_check_ecoli_ratio_")
        output_dir = Path(temp_context.name)

    try:
        summaries = [None] * len(DEFAULT_JOBS)
        with ThreadPoolExecutor(max_workers=min(args.jobs, len(DEFAULT_JOBS))) as executor:
            future_to_index = {
                executor.submit(
                    _run_and_summarize_job,
                    job,
                    output_dir,
                    args.python,
                    args.batch_size,
                    args.decision_threshold,
                    args.q_value,
                    args.all_rows,
                    exclude_protein_prefixes,
                ): idx
                for idx, job in enumerate(DEFAULT_JOBS)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                summaries[idx] = future.result()

        write_summary(summaries, args.output)
    finally:
        if temp_context is not None:
            temp_context.cleanup()


if __name__ == "__main__":
    main()
