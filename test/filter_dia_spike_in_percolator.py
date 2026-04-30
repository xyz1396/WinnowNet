#!/usr/bin/env python3
"""Filter DIA spike-in PSMs with Percolator using metadata-defined controls."""

import argparse
import csv
import math
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from check_ecoli_ratio import (
    DEFAULT_EXCLUDE_PROTEIN_PREFIXES,
    SUMMARY_COLUMNS,
    _format_value,
    _parse_prefixes,
    summarize_tsv,
)


DEFAULT_PSM_DIR = REPO_ROOT / "data" / "spike_dia"
DEFAULT_METADATA = (
    DEFAULT_PSM_DIR / "dia astral spike in meta data IDeA Proteomics Sample List 20251006.xlsx"
)
DEFAULT_OUTPUT_DIR = DEFAULT_PSM_DIR / "percolator"
DEFAULT_SUMMARY = DEFAULT_PSM_DIR / "percolator_summary.tsv"
DEFAULT_MS2_THRESHOLD = 1.5
DEFAULT_Q_VALUE = 0.01
DEFAULT_TEST_FDR = 0.01
DEFAULT_TRAIN_FDR = 0.01
DEFAULT_PERCOLATOR = "percolator"
MODEL_LABEL = "percolator"

SAMPLE_COLUMN = "sample number"
PCT_COLUMN = "13C %"
GROUP_COLUMN = "sample group/treatment/condition"
CONTROL_GROUP = "control"

FEATURE_COLUMNS = [
    "ranks",
    "parentCharges",
    "massErrors",
    "isotopicMassWindowShifts",
    "mzShiftFromisolationWindowCenters",
    "peptideLengths",
    "missCleavageSiteNumbers",
    "PTMnumbers",
    "isotopicPeakNumbers",
    "MS1IsotopicAbundances",
    "MS2IsotopicAbundances",
    "isotopicAbundanceDiffs",
    "WDPscores",
    "XcorrScores",
    "MVHscores",
    "diffScores",
    "log10_precursorIntensities",
]
PIN_ID_COLUMNS = ["PSMId", "Label", "ScanNr"]
PIN_TRAILING_COLUMNS = ["Peptide", "Proteins"]
PIN_COLUMNS = PIN_ID_COLUMNS + FEATURE_COLUMNS + PIN_TRAILING_COLUMNS
PERCOLATOR_SCORE_COLUMNS = {
    "score": "score",
    "q-value": "q-value",
    "posterior_error_prob": "posterior_error_prob",
}
OUTPUT_COLUMNS = ["Input", "Model"] + SUMMARY_COLUMNS


@dataclass
class SampleRecord:
    sample: str
    path: Path
    pct_fraction: float
    group: str

    @property
    def is_control(self):
        return self.group.strip().lower() == CONTROL_GROUP

    @property
    def pct_percent(self):
        return self.pct_fraction * 100.0


@dataclass
class FilterStats:
    rows_seen: int = 0
    rows_kept: int = 0
    removed_label: int = 0
    removed_prefix: int = 0
    removed_ms2: int = 0
    removed_invalid_feature: int = 0

    def add(self, other):
        self.rows_seen += other.rows_seen
        self.rows_kept += other.rows_kept
        self.removed_label += other.removed_label
        self.removed_prefix += other.removed_prefix
        self.removed_ms2 += other.removed_ms2
        self.removed_invalid_feature += other.removed_invalid_feature


def _to_float(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _split_proteins(value):
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    return [item.strip() for item in text.split(",") if item.strip()]


def _all_proteins_match_prefixes(proteins, prefixes):
    if not prefixes or not proteins:
        return False
    return all(any(protein.startswith(prefix) for prefix in prefixes) for protein in proteins)


def _is_original_decoy_label(row):
    return str(row.get("Label", "")).strip() in {"-1", "-1.0"}


def _pct_label(pct_percent):
    value = float(pct_percent)
    if value.is_integer():
        return f"pct{int(value):02d}"
    return f"pct{value:g}".replace(".", "p")


def _format_pct_for_summary(pct_percent):
    value = float(pct_percent)
    if value.is_integer():
        return int(value)
    return value


def _display_path(path):
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sample_from_file(path):
    match = re.match(r"^(P\d+)_", path.name)
    if match:
        return match.group(1)
    return None


def _require_columns(fieldnames, path, columns):
    missing = [column for column in columns if column not in fieldnames]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _iter_filtered_rows(path, new_label, ms2_threshold, exclude_prefixes, stats=None):
    if stats is None:
        stats = FilterStats()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a TSV header.")
        _require_columns(reader.fieldnames, path, PIN_COLUMNS)
        for row in reader:
            stats.rows_seen += 1

            if _is_original_decoy_label(row):
                stats.removed_label += 1
                continue

            proteins = _split_proteins(row.get("Proteins", ""))
            if _all_proteins_match_prefixes(proteins, exclude_prefixes):
                stats.removed_prefix += 1
                continue

            ms2_abundance = _to_float(row.get("MS2IsotopicAbundances", ""))
            if ms2_abundance is None or ms2_abundance <= ms2_threshold:
                stats.removed_ms2 += 1
                continue

            invalid_feature = False
            for feature in FEATURE_COLUMNS:
                if _to_float(row.get(feature, "")) is None:
                    invalid_feature = True
                    break
            if invalid_feature:
                stats.removed_invalid_feature += 1
                continue

            row = dict(row)
            row["Label"] = str(new_label)
            stats.rows_kept += 1
            yield row


def _read_tsv_header(path):
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty.") from exc


def _load_metadata(metadata_path, psm_dir):
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata workbook not found: {metadata_path}")
    if not psm_dir.exists():
        raise FileNotFoundError(f"PSM directory not found: {psm_dir}")

    df = pd.read_excel(metadata_path, engine="openpyxl")
    df.columns = [str(column).strip() for column in df.columns]
    for column in [SAMPLE_COLUMN, PCT_COLUMN, GROUP_COLUMN]:
        if column not in df.columns:
            raise ValueError(f"Metadata workbook is missing required column: {column}")

    files_by_sample = defaultdict(list)
    for path in sorted(psm_dir.glob("*_filtered_psms.tsv")):
        sample = _sample_from_file(path)
        if sample is not None:
            files_by_sample[sample].append(path)

    metadata_samples = set()
    records = []
    for _, row in df.iterrows():
        sample = str(row.get(SAMPLE_COLUMN, "")).strip()
        if not sample or sample.lower() == "nan":
            continue
        pct_fraction = _to_float(row.get(PCT_COLUMN, ""))
        if pct_fraction is None:
            raise ValueError(f"Metadata sample {sample} has invalid {PCT_COLUMN!r}: {row.get(PCT_COLUMN)!r}")
        group = str(row.get(GROUP_COLUMN, "")).strip()
        metadata_samples.add(sample)

        matches = files_by_sample.get(sample, [])
        if len(matches) != 1:
            rendered = ", ".join(str(path) for path in matches) or "none"
            raise ValueError(f"Expected exactly one filtered PSM file for {sample}; found {rendered}")
        records.append(SampleRecord(sample=sample, path=matches[0], pct_fraction=pct_fraction, group=group))

    if not records:
        raise ValueError(f"No metadata rows were loaded from {metadata_path}")

    unmapped = []
    for path in sorted(psm_dir.glob("*_filtered_psms.tsv")):
        sample = _sample_from_file(path)
        if sample is None or sample not in metadata_samples:
            unmapped.append(path)
    if unmapped:
        print(
            "Warning: skipping filtered PSM files not listed in metadata: "
            + ", ".join(str(path) for path in unmapped),
            file=sys.stderr,
        )

    return records


def _validate_compatible_headers(records):
    expected = None
    expected_path = None
    for record in records:
        header = _read_tsv_header(record.path)
        _require_columns(header, record.path, PIN_COLUMNS)
        for column in PERCOLATOR_SCORE_COLUMNS:
            if column not in header:
                raise ValueError(f"{record.path} is missing output score column: {column}")
        if expected is None:
            expected = header
            expected_path = record.path
        elif header != expected:
            raise ValueError(f"{record.path} header does not match {expected_path}")
    return expected


def _group_records(records):
    controls = [record for record in records if record.is_control]
    targets_by_pct = defaultdict(list)
    for record in records:
        if record.is_control:
            continue
        targets_by_pct[round(record.pct_percent, 10)].append(record)

    if not controls:
        raise ValueError("No control metadata rows were found.")
    if not targets_by_pct:
        raise ValueError("No target metadata rows were found.")

    return dict(sorted(targets_by_pct.items())), controls


def _write_pin(pin_path, target_records, control_records, ms2_threshold, exclude_prefixes):
    stats = {"target": FilterStats(), "control": FilterStats()}
    with pin_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PIN_COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()

        for record in target_records:
            record_stats = FilterStats()
            for row in _iter_filtered_rows(
                record.path, 1, ms2_threshold, exclude_prefixes, record_stats
            ):
                writer.writerow({column: row.get(column, "") for column in PIN_COLUMNS})
            stats["target"].add(record_stats)

        for record in control_records:
            record_stats = FilterStats()
            for row in _iter_filtered_rows(
                record.path, -1, ms2_threshold, exclude_prefixes, record_stats
            ):
                writer.writerow({column: row.get(column, "") for column in PIN_COLUMNS})
            stats["control"].add(record_stats)

    if stats["target"].rows_kept == 0:
        raise ValueError(f"No target rows survived filtering for {pin_path}")
    if stats["control"].rows_kept == 0:
        raise ValueError(f"No control rows survived filtering for {pin_path}")
    return stats


def _run_percolator(percolator, pin_path, percolator_psms_path, log_path, test_fdr, train_fdr):
    command = [
        percolator,
        "--only-psms",
        "--search-input",
        "separate",
        "--results-psms",
        str(percolator_psms_path),
        "--testFDR",
        str(test_fdr),
        "--trainFDR",
        str(train_fdr),
        str(pin_path),
    ]
    with log_path.open("w") as log_handle:
        log_handle.write("Command: " + " ".join(command) + "\n\n")
        log_handle.flush()
        subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _read_percolator_accepted(psms_path, q_value_threshold):
    accepted = {}
    with psms_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{psms_path} does not contain a TSV header.")
        _require_columns(
            reader.fieldnames,
            psms_path,
            ["PSMId", "score", "q-value", "posterior_error_prob"],
        )
        for row in reader:
            q_value = _to_float(row.get("q-value", ""))
            if q_value is None or q_value > q_value_threshold:
                continue
            accepted[row["PSMId"]] = {
                "score": row.get("score", ""),
                "q-value": row.get("q-value", ""),
                "posterior_error_prob": row.get("posterior_error_prob", ""),
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
                percolator_scores = accepted.get(psm_id)
                if percolator_scores is None:
                    continue
                row.update(percolator_scores)
                row["Label"] = "1"
                writer.writerow({column: row.get(column, "") for column in output_header})
                seen_accepted.add(psm_id)
                written += 1

    missing = sorted(set(accepted) - seen_accepted)
    if missing:
        print(
            f"Warning: {len(missing)} accepted Percolator PSMs were not found in target inputs for {output_path}",
            file=sys.stderr,
        )
    return written


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Filter DIA spike-in PSMs with metadata-pooled Percolator target/control jobs."
    )
    parser.add_argument("--psm-dir", default=str(DEFAULT_PSM_DIR), help="Directory with *_filtered_psms.tsv files.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Spike-in metadata workbook.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for Percolator outputs.")
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
        help=f"Accept target PSMs with Percolator q-value at or below this value. Default: {DEFAULT_Q_VALUE}.",
    )
    parser.add_argument(
        "--test-fdr",
        type=_positive_float,
        default=DEFAULT_TEST_FDR,
        help=f"Percolator --testFDR value. Default: {DEFAULT_TEST_FDR}.",
    )
    parser.add_argument(
        "--train-fdr",
        type=_positive_float,
        default=DEFAULT_TRAIN_FDR,
        help=f"Percolator --trainFDR value. Default: {DEFAULT_TRAIN_FDR}.",
    )
    parser.add_argument("--percolator", default=DEFAULT_PERCOLATOR, help="Percolator executable.")
    parser.add_argument(
        "--exclude-protein-prefix",
        default=",".join(DEFAULT_EXCLUDE_PROTEIN_PREFIXES),
        help="Drop rows when all proteins start with one of these comma-separated prefixes. Default: Decoy_,Con_.",
    )
    parser.add_argument(
        "--keep-pin",
        action="store_true",
        help="Keep generated Percolator PIN files in --output-dir.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    psm_dir = Path(args.psm_dir).resolve()
    metadata_path = Path(args.metadata).resolve()
    output_dir = Path(args.output_dir).resolve()
    summary_path = Path(args.summary).resolve()
    exclude_prefixes = _parse_prefixes(args.exclude_protein_prefix)

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

    temp_context = None
    pin_dir = output_dir
    if not args.keep_pin:
        temp_context = tempfile.TemporaryDirectory(prefix="percolator_pin_", dir=str(output_dir))
        pin_dir = Path(temp_context.name)

    summaries = []
    try:
        for pct_percent, target_records in targets_by_pct.items():
            label = _pct_label(pct_percent)
            pin_path = pin_dir / f"{label}.pin"
            percolator_psms_path = output_dir / f"{label}.percolator.psms.tsv"
            log_path = output_dir / f"{label}.percolator.log"
            accepted_output_path = output_dir / f"{label}_percolator_filtered_psms.tsv"

            print(
                f"[{label}] target samples: "
                + ", ".join(record.sample for record in target_records),
                file=sys.stderr,
            )
            stats = _write_pin(
                pin_path,
                target_records,
                control_records,
                args.ms2_threshold,
                exclude_prefixes,
            )
            print(
                f"[{label}] PIN rows target={stats['target'].rows_kept} "
                f"control={stats['control'].rows_kept}; "
                f"removed target label={stats['target'].removed_label} "
                f"prefix={stats['target'].removed_prefix} "
                f"ms2={stats['target'].removed_ms2} "
                f"invalid_feature={stats['target'].removed_invalid_feature}",
                file=sys.stderr,
            )

            _run_percolator(
                args.percolator,
                pin_path,
                percolator_psms_path,
                log_path,
                args.test_fdr,
                args.train_fdr,
            )
            accepted = _read_percolator_accepted(percolator_psms_path, args.q_value)
            written = _write_accepted_targets(
                accepted_output_path,
                target_records,
                output_header,
                accepted,
                args.ms2_threshold,
                exclude_prefixes,
            )
            print(
                f"[{label}] accepted target PSMs={written} -> {accepted_output_path}",
                file=sys.stderr,
            )
            summaries.append(_summarize_output(accepted_output_path, pct_percent, exclude_prefixes))

        _write_summary(summaries, summary_path)
        print(f"summary={summary_path}", file=sys.stderr)
    finally:
        if temp_context is not None:
            temp_context.cleanup()


if __name__ == "__main__":
    main()
