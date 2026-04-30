#!/usr/bin/env python3
"""Filter DIA spike-in PSMs with Mokapot using metadata-defined controls."""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# Mokapot 0.10 still references np.float_, which was removed in NumPy 2.
if not hasattr(np, "float_"):
    np.float_ = np.float64

import mokapot

from check_ecoli_ratio import (
    DEFAULT_EXCLUDE_PROTEIN_PREFIXES,
    SUMMARY_COLUMNS,
    _format_value,
    _parse_prefixes,
    summarize_tsv,
)
from filter_dia_spike_in_percolator import (
    CONTROL_GROUP,
    DEFAULT_METADATA,
    DEFAULT_MS2_THRESHOLD,
    DEFAULT_PSM_DIR,
    FEATURE_COLUMNS,
    PIN_COLUMNS,
    FilterStats,
    _display_path,
    _format_pct_for_summary,
    _group_records,
    _iter_filtered_rows,
    _load_metadata,
    _pct_label,
    _to_float,
    _validate_compatible_headers,
)


DEFAULT_OUTPUT_DIR = DEFAULT_PSM_DIR / "mokapot"
DEFAULT_SUMMARY = DEFAULT_PSM_DIR / "mokapot_summary.tsv"
DEFAULT_Q_VALUE = 0.01
DEFAULT_TEST_FDR = 0.01
DEFAULT_RETRY_TEST_FDR = 0.2
DEFAULT_TRAIN_FDR = 0.01
DEFAULT_RETRY_TRAIN_FDR = 0.2
DEFAULT_FOLDS = 3
DEFAULT_MAX_WORKERS = 1
DEFAULT_SEED = 1
MODEL_LABEL = "mokapot"
OUTPUT_COLUMNS = ["Input", "Model"] + SUMMARY_COLUMNS
MOKAPOT_SCORE_COLUMNS = {
    "score": "mokapot score",
    "q-value": "mokapot q-value",
    "posterior_error_prob": "mokapot PEP",
}


def _rows_to_dataframe(records, label, ms2_threshold, exclude_prefixes):
    rows = []
    stats = FilterStats()
    target = label == 1
    for record in records:
        record_stats = FilterStats()
        for row in _iter_filtered_rows(record.path, label, ms2_threshold, exclude_prefixes, record_stats):
            item = {column: row.get(column, "") for column in PIN_COLUMNS}
            item["target"] = target
            rows.append(item)
        stats.add(record_stats)

    if not rows:
        role = "target" if target else "control"
        raise ValueError(f"No {role} rows survived filtering.")

    frame = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS + ["ScanNr"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame, stats


def _make_dataset(frame, seed):
    return mokapot.LinearPsmDataset(
        frame,
        target_column="target",
        spectrum_columns=["PSMId"],
        peptide_column="Peptide",
        protein_column="Proteins",
        feature_columns=FEATURE_COLUMNS,
        scan_column="ScanNr",
        copy_data=False,
        rng=seed,
    )


def _make_model(train_fdr, seed):
    return mokapot.PercolatorModel(train_fdr=train_fdr, rng=seed, n_jobs=1)


def _run_mokapot(dataset, train_fdr, retry_train_fdr, test_fdr, retry_test_fdr, folds, max_workers, seed):
    attempts = [(train_fdr, test_fdr)]
    if retry_test_fdr > test_fdr:
        attempts.append((train_fdr, retry_test_fdr))
    if retry_train_fdr > train_fdr:
        attempts.append((retry_train_fdr, retry_test_fdr if retry_test_fdr > test_fdr else test_fdr))

    last_error = None
    for attempt_idx, (attempt_train_fdr, attempt_test_fdr) in enumerate(attempts, start=1):
        model = _make_model(attempt_train_fdr, seed)
        try:
            if attempt_idx > 1:
                print(
                    f"Warning: retrying Mokapot with train_fdr={attempt_train_fdr:g}, "
                    f"test_fdr={attempt_test_fdr:g}. Final accepted rows still require "
                    f"q-value <= {test_fdr:g}.",
                    file=sys.stderr,
                )
            return mokapot.brew(
                dataset,
                model=model,
                test_fdr=attempt_test_fdr,
                folds=folds,
                max_workers=max_workers,
                rng=seed,
            )
        except RuntimeError as exc:
            last_error = exc
            message = str(exc)
        retryable = (
            "No target PSMs were below the 'eval_fdr'" in message
            or "no target PSMs could be found below 'test_fdr'" in message
            or "No PSMs found below the 'eval_fdr'" in message
        )
        if not retryable:
            raise
    raise last_error


def _write_mokapot_psms(confidence, output_path):
    psms = confidence.psms.copy()
    psms.to_csv(output_path, sep="\t", index=False)
    return psms


def _read_mokapot_accepted(psms, q_value_threshold):
    accepted = {}
    required = ["PSMId", "target", *MOKAPOT_SCORE_COLUMNS.values()]
    missing = [column for column in required if column not in psms.columns]
    if missing:
        raise ValueError(f"Mokapot PSM table is missing columns: {', '.join(missing)}")

    for row in psms.to_dict("records"):
        if not bool(row.get("target")):
            continue
        q_value = _to_float(row.get("mokapot q-value", ""))
        if q_value is None or q_value > q_value_threshold:
            continue
        accepted[str(row["PSMId"])] = {
            "score": row.get("mokapot score", ""),
            "q-value": row.get("mokapot q-value", ""),
            "posterior_error_prob": row.get("mokapot PEP", ""),
        }
    return accepted


def _write_accepted_targets(
    output_path,
    target_records,
    output_header,
    accepted,
    ms2_threshold,
    exclude_prefixes,
):
    written = 0
    seen_accepted = set()
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_header, delimiter="\t")
        writer.writeheader()
        for record in target_records:
            for row in _iter_filtered_rows(record.path, 1, ms2_threshold, exclude_prefixes):
                psm_id = row.get("PSMId", "")
                mokapot_scores = accepted.get(psm_id)
                if mokapot_scores is None:
                    continue
                row.update(mokapot_scores)
                row["Label"] = "1"
                writer.writerow({column: row.get(column, "") for column in output_header})
                seen_accepted.add(psm_id)
                written += 1

    missing = sorted(set(accepted) - seen_accepted)
    if missing:
        print(
            f"Warning: {len(missing)} accepted Mokapot PSMs were not found in target inputs for {output_path}",
            file=sys.stderr,
        )
    return written


def _write_empty_accepted_targets(output_path, output_header):
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_header, delimiter="\t")
        writer.writeheader()


def _write_empty_mokapot_psms(output_path):
    fieldnames = [
        "PSMId",
        "Label",
        "ScanNr",
        "Peptide",
        "target",
        "mokapot score",
        "mokapot q-value",
        "mokapot PEP",
        "Proteins",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()


def _summarize_output(output_path, pct_percent, exclude_prefixes):
    summary = summarize_tsv(str(output_path), exclude_protein_prefixes=exclude_prefixes)
    summary["Input"] = _display_path(output_path)
    summary["Model"] = MODEL_LABEL
    summary["Spike-in 13C (%)"] = _format_pct_for_summary(pct_percent)
    return summary


def _write_summary(rows, summary_path):
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_value(row.get(column, "")) for column in OUTPUT_COLUMNS})


def _positive_float(value):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Filter DIA spike-in PSMs with metadata-pooled Mokapot target/control jobs."
    )
    parser.add_argument("--psm-dir", default=str(DEFAULT_PSM_DIR), help="Directory with *_filtered_psms.tsv files.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Spike-in metadata workbook.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for Mokapot outputs.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Output summary TSV path.")
    parser.add_argument(
        "--ms2-threshold",
        type=float,
        default=DEFAULT_MS2_THRESHOLD,
        help=f"Keep PSMs with MS2IsotopicAbundances greater than this value. Default: {DEFAULT_MS2_THRESHOLD}.",
    )
    parser.add_argument(
        "--q-value",
        type=float,
        default=DEFAULT_Q_VALUE,
        help=f"Accept target PSMs with Mokapot q-value at or below this value. Default: {DEFAULT_Q_VALUE}.",
    )
    parser.add_argument(
        "--test-fdr",
        type=_positive_float,
        default=DEFAULT_TEST_FDR,
        help=f"Mokapot test_fdr value for initial calibration. Default: {DEFAULT_TEST_FDR}.",
    )
    parser.add_argument(
        "--retry-test-fdr",
        type=_positive_float,
        default=DEFAULT_RETRY_TEST_FDR,
        help=(
            "Fallback Mokapot test_fdr if 1%% calibration has no confident targets. "
            f"Accepted rows still require --q-value. Default: {DEFAULT_RETRY_TEST_FDR}."
        ),
    )
    parser.add_argument(
        "--train-fdr",
        type=_positive_float,
        default=DEFAULT_TRAIN_FDR,
        help=f"Mokapot/PercolatorModel train_fdr value. Default: {DEFAULT_TRAIN_FDR}.",
    )
    parser.add_argument(
        "--retry-train-fdr",
        type=_positive_float,
        default=DEFAULT_RETRY_TRAIN_FDR,
        help=(
            "Fallback Mokapot train_fdr if the initial direction has no confident targets. "
            f"Accepted rows still require --q-value. Default: {DEFAULT_RETRY_TRAIN_FDR}."
        ),
    )
    parser.add_argument(
        "--folds",
        type=_positive_int,
        default=DEFAULT_FOLDS,
        help=f"Cross-validation folds for Mokapot. Default: {DEFAULT_FOLDS}.",
    )
    parser.add_argument(
        "--max-workers",
        type=_positive_int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Mokapot worker processes. Default: {DEFAULT_MAX_WORKERS}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for Mokapot. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--exclude-protein-prefix",
        default=",".join(DEFAULT_EXCLUDE_PROTEIN_PREFIXES),
        help="Drop rows when all proteins start with one of these comma-separated prefixes. Default: Decoy_,Con_.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    psm_dir = Path(args.psm_dir).resolve()
    metadata_path = Path(args.metadata).resolve()
    output_dir = Path(args.output_dir).resolve()
    summary_path = Path(args.summary).resolve()
    exclude_prefixes = _parse_prefixes(args.exclude_protein_prefix)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    records = _load_metadata(metadata_path, psm_dir)
    output_header = _validate_compatible_headers(records)
    targets_by_pct, control_records = _group_records(records)

    print(f"Loaded {len(records)} metadata samples.", file=sys.stderr)
    print(
        "Control samples: " + ", ".join(record.sample for record in control_records),
        file=sys.stderr,
    )

    summaries = []
    for pct_percent, target_records in targets_by_pct.items():
        label = _pct_label(pct_percent)
        mokapot_psms_path = output_dir / f"{label}.mokapot.psms.tsv"
        accepted_output_path = output_dir / f"{label}_mokapot_filtered_psms.tsv"

        print(
            f"[{label}] target samples: " + ", ".join(record.sample for record in target_records),
            file=sys.stderr,
        )
        target_frame, target_stats = _rows_to_dataframe(
            target_records, 1, args.ms2_threshold, exclude_prefixes
        )
        control_frame, control_stats = _rows_to_dataframe(
            control_records, -1, args.ms2_threshold, exclude_prefixes
        )
        print(
            f"[{label}] dataset rows target={target_stats.rows_kept} "
            f"control={control_stats.rows_kept}; "
            f"removed target label={target_stats.removed_label} "
            f"prefix={target_stats.removed_prefix} "
            f"ms2={target_stats.removed_ms2} "
            f"invalid_feature={target_stats.removed_invalid_feature}",
            file=sys.stderr,
        )

        dataset = _make_dataset(pd.concat([target_frame, control_frame], ignore_index=True), args.seed)
        try:
            confidence, _models = _run_mokapot(
                dataset,
                args.train_fdr,
                args.retry_train_fdr,
                args.test_fdr,
                args.retry_test_fdr,
                args.folds,
                args.max_workers,
                args.seed,
            )
            mokapot_psms = _write_mokapot_psms(confidence, mokapot_psms_path)
            accepted = _read_mokapot_accepted(mokapot_psms, args.q_value)
            written = _write_accepted_targets(
                accepted_output_path,
                target_records,
                output_header,
                accepted,
                args.ms2_threshold,
                exclude_prefixes,
            )
        except RuntimeError as exc:
            print(
                f"Warning: Mokapot failed for {label}: {exc}. Writing empty accepted target output.",
                file=sys.stderr,
            )
            _write_empty_mokapot_psms(mokapot_psms_path)
            _write_empty_accepted_targets(accepted_output_path, output_header)
            written = 0
        print(
            f"[{label}] accepted target PSMs={written} -> {accepted_output_path}",
            file=sys.stderr,
        )
        summaries.append(_summarize_output(accepted_output_path, pct_percent, exclude_prefixes))

    _write_summary(summaries, summary_path)
    print(f"summary={summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
