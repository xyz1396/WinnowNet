#!/usr/bin/env python3
"""Predict DIA spike-in samples with WinnowNet percent thresholds."""

import argparse
import csv
import gc
import importlib
import os
import re
import sys
import traceback
from collections import defaultdict
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SCRIPT_DIR = REPO_ROOT / "script"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_ecoli_ratio import (
    DEFAULT_EXCLUDE_PROTEIN_PREFIXES,
    SUMMARY_COLUMNS,
    _format_value,
    summarize_tsv,
)


DEFAULT_PSM_DIR = REPO_ROOT / "data" / "spike_dia"
DEFAULT_METADATA = (
    DEFAULT_PSM_DIR / "dia astral spike in meta data IDeA Proteomics Sample List 20251006.xlsx"
)
DEFAULT_MODEL = REPO_ROOT / "data" / "pure_cnn_all_pct.pt"
DEFAULT_CONFIG = REPO_ROOT / "script" / "SIP.cfg"
DEFAULT_OUTPUT_DIR = DEFAULT_PSM_DIR / "winnownet"
DEFAULT_SUMMARY = DEFAULT_PSM_DIR / "winnownet_summary.tsv"
DEFAULT_PCT_THRESHOLDS = "2,5,50"
DEFAULT_JOBS = 15
DEFAULT_CHUNK_BATCH_SIZE = None
DEFAULT_TARGET_SCORE_THRESHOLDS = {
    # Post-score cutoffs chosen to keep as many target PSMs as possible while
    # reaching the requested E. coli PSM ratio targets:
    # pct02 >= 90%, pct05 >= 93%.
    2.0: 0.9997105002,
    5.0: 0.9990711212,
}
MODEL_LABEL = "winnownet"
CONTROL_GROUP = "control"
EXPECTED_CONTROL_SAMPLES = ("P19", "P20", "P21")
EXPECTED_TARGET_SAMPLES = {
    2.0: ("P1", "P2", "P3", "P4", "P5", "P6"),
    5.0: ("P7", "P8", "P9", "P10", "P11", "P12"),
    50.0: ("P13", "P14", "P15", "P16", "P17", "P18"),
}
SAMPLE_COLUMN = "sample number"
GROUP_COLUMN = "sample group/treatment/condition"
CARBON_COLUMN = "13C %"
OUTPUT_COLUMNS = ["Input", "Model"] + SUMMARY_COLUMNS


def _load_winnownet_module():
    return importlib.import_module("winnownet")


@dataclass(frozen=True)
class SampleRecord:
    sample: str
    path: Path
    group: str
    carbon_pct: float

    @property
    def is_control(self):
        return self.group.strip().lower() == CONTROL_GROUP


def _repo_relative(path):
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sample_from_file(path):
    match = re.match(r"^(P\d+)_", path.name)
    return match.group(1) if match else None


def _sample_sort_key(sample):
    match = re.match(r"^P(\d+)$", str(sample))
    if match:
        return int(match.group(1))
    return str(sample)


def _chunk_stem_from_psm_id(psm_id):
    match = re.match(r"^(.+)\.FT2\.\d+\.\d+(?:\.\d+)?$", str(psm_id).strip())
    if match:
        return match.group(1)
    raise ValueError(f"Could not infer FT chunk stem from PSMId: {psm_id!r}")


def _pct_label(value):
    pct = float(value)
    if pct.is_integer():
        return f"pct{int(pct):02d}"
    return f"pct{pct:g}".replace(".", "p")


def _pct_summary_value(value):
    pct = float(value)
    return int(pct) if pct.is_integer() else pct


def _parse_pct_thresholds(value):
    pcts = []
    for item in str(value or "").split(","):
        item = item.strip()
        if item:
            pcts.append(float(item))
    if not pcts:
        raise argparse.ArgumentTypeError("at least one pct threshold is required")
    return pcts


def _normalize_pct_key(value):
    pct = float(value)
    return str(int(pct)) if pct.is_integer() else f"{pct:g}"


def _metadata_fraction_to_pct(value):
    pct = float(value)
    if pct <= 1.0:
        pct *= 100.0
    return pct


def _load_metadata_groups(metadata_path, psm_dir):
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata workbook not found: {metadata_path}")
    df = pd.read_excel(metadata_path, engine="openpyxl")
    df.columns = [str(column).strip() for column in df.columns]
    for column in (SAMPLE_COLUMN, GROUP_COLUMN, CARBON_COLUMN):
        if column not in df.columns:
            raise ValueError(f"Metadata workbook is missing required column: {column}")

    files_by_sample = defaultdict(list)
    for path in sorted(psm_dir.glob("*_filtered_psms.tsv")):
        sample = _sample_from_file(path)
        if sample:
            files_by_sample[sample].append(path)

    target_groups = defaultdict(list)
    controls = []
    for _, row in df.iterrows():
        sample = str(row.get(SAMPLE_COLUMN, "")).strip()
        if not sample or sample.lower() == "nan":
            continue
        group = str(row.get(GROUP_COLUMN, "")).strip()
        matches = files_by_sample.get(sample, [])
        if len(matches) != 1:
            rendered = ", ".join(str(path) for path in matches) or "none"
            raise ValueError(f"Expected exactly one filtered PSM file for {sample}; found {rendered}")

        carbon_pct = _metadata_fraction_to_pct(row.get(CARBON_COLUMN))
        record = SampleRecord(sample, matches[0], group, carbon_pct)
        if record.is_control:
            controls.append(record)
        else:
            target_groups[carbon_pct].append(record)

    if not controls:
        raise ValueError("No control samples found in metadata.")
    if not target_groups:
        raise ValueError("No target 13C sample groups found in metadata.")

    target_groups = {
        pct: sorted(records, key=lambda record: _sample_sort_key(record.sample))
        for pct, records in sorted(target_groups.items())
    }
    controls = sorted(controls, key=lambda record: _sample_sort_key(record.sample))
    return target_groups, controls


def _load_pct_thresholds(model_path, requested_pcts):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    if metadata.get("model_arch") != "pure_cnn_pct":
        raise ValueError(f"{model_path} model_arch={metadata.get('model_arch')!r}; expected 'pure_cnn_pct'.")
    raw = metadata.get("pct_decision_thresholds") or {}
    normalized = {_normalize_pct_key(key): value for key, value in raw.items()}
    thresholds = {}
    missing = []
    for pct in requested_pcts:
        key = _normalize_pct_key(pct)
        entry = normalized.get(key)
        if entry is None:
            missing.append(key)
        elif isinstance(entry, dict):
            thresholds[pct] = float(entry["threshold"])
        else:
            thresholds[pct] = float(entry)
    if missing:
        raise ValueError(f"{model_path} is missing pct_decision_thresholds for: {', '.join(missing)}")
    return thresholds


def _target_threshold_for_pct(pct, checkpoint_thresholds):
    return DEFAULT_TARGET_SCORE_THRESHOLDS.get(float(pct), checkpoint_thresholds[pct])


def _safe_symlink(src, dest):
    src = Path(src).resolve()
    dest = Path(dest)
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() and Path(os.readlink(dest)).resolve() == src:
            return
        raise FileExistsError(f"Refusing to replace existing nonmatching path: {dest}")
    os.symlink(src, dest)


def _prepare_chunks(sample_records, output_dir, group_label):
    chunk_dir = output_dir / "chunk_inputs" / group_label
    chunk_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    row_count_by_source = {}

    for record in sample_records:
        with record.path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or "PSMId" not in reader.fieldnames:
                raise ValueError(f"{record.path} is missing a PSMId TSV header.")
            fieldnames = list(reader.fieldnames)
            writers = {}
            handles = {}
            counts = defaultdict(int)
            source_rows = 0
            try:
                for row in reader:
                    source_rows += 1
                    chunk_stem = _chunk_stem_from_psm_id(row["PSMId"])
                    src_base = record.path.with_name(chunk_stem)
                    ft1 = Path(str(src_base) + ".FT1")
                    ft2 = Path(str(src_base) + ".FT2")
                    missing = [str(path) for path in (ft1, ft2) if not path.exists()]
                    if missing:
                        raise FileNotFoundError(
                            f"{record.path} PSMId={row['PSMId']} references missing raw files: "
                            + ", ".join(missing)
                        )

                    out_base = chunk_dir / chunk_stem
                    out_tsv = Path(str(out_base) + "_filtered_psms.tsv")
                    if out_tsv not in writers:
                        out_handle = out_tsv.open("w", newline="")
                        handles[out_tsv] = out_handle
                        writer = csv.DictWriter(out_handle, fieldnames=fieldnames, delimiter="\t")
                        writer.writeheader()
                        writers[out_tsv] = writer
                        _safe_symlink(ft1, Path(str(out_base) + ".FT1"))
                        _safe_symlink(ft2, Path(str(out_base) + ".FT2"))
                        generated.append(out_tsv)
                    writers[out_tsv].writerow({column: row.get(column, "") for column in fieldnames})
                    counts[out_tsv] += 1
            finally:
                for out_handle in handles.values():
                    out_handle.close()

        if sum(counts.values()) != source_rows:
            raise ValueError(f"Chunk row count mismatch for {record.path}")
        row_count_by_source[record.sample] = source_rows

    return sorted(generated), row_count_by_source


def _task_from_chunk(ww, tsv_path):
    base = ww._default_base_from_tsv(tsv_path)
    return ww.RawTask(tsv=tsv_path, ft1=Path(str(base) + ".FT1"), ft2=Path(str(base) + ".FT2"), kind="target")


def _batched(items, size):
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def _diagnose_worker_failure(ww, args, tasks):
    print("Diagnosing worker failure by running chunks one at a time...", file=sys.stderr)
    for index, task in enumerate(tasks, start=1):
        print(f"[diagnose {index}/{len(tasks)}] chunk={task.tsv}", file=sys.stderr)
        try:
            ww._feature_task(task, args)
        except Exception as exc:
            print(f"[diagnose failed] chunk={task.tsv}", file=sys.stderr)
            print(f"[diagnose failed] exception={type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"WinnowNet feature extraction failed for chunk {task.tsv}: {exc}") from exc

    print(
        "No individual chunk failed during diagnostics. The worker probably died outside "
        "Python exception handling, commonly from memory/resource pressure in parallel mode.",
        file=sys.stderr,
    )


def _score_chunks(ww, args, chunk_files):
    tasks = [_task_from_chunk(ww, path) for path in chunk_files]
    scoring_files = len(tasks)
    jobs, threads_per_job, cpu_cores, total_cores = ww._resolve_parallelism(
        args.jobs,
        args.cores,
        args.threads_per_job,
        scoring_files,
    )
    args.threads_per_job = threads_per_job
    device = ww._resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(total_cores)

    model, _metadata = ww._load_model(args.model, device)
    print(f"Scoring chunks={len(tasks)} device={device} jobs={jobs} cores_per_job={threads_per_job}", file=sys.stderr)
    try:
        return ww._score_tasks(tasks, model, device, args.batch_size, jobs, args)
    except BrokenProcessPool as exc:
        print(
            f"Worker failure: a feature extraction process died abruptly with jobs={jobs} "
            f"and cores_per_job={threads_per_job}. Exception={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("Failed batch chunks:", file=sys.stderr)
        for task in tasks:
            print(f"  {task.tsv}", file=sys.stderr)
        raise


def _score_chunk_batches(ww, args, chunk_files):
    batch_size = args.chunk_batch_size or args.jobs or len(chunk_files)
    batch_size = max(1, min(int(batch_size), len(chunk_files)))
    for start, batch in _batched(chunk_files, batch_size):
        batch_number = start // batch_size + 1
        total_batches = (len(chunk_files) + batch_size - 1) // batch_size
        print(
            f"Scoring chunk batch {batch_number}/{total_batches}: "
            f"files={len(batch)} range={start + 1}-{start + len(batch)}",
            file=sys.stderr,
        )
        yield _score_chunks(ww, args, batch)


def _write_threshold_output(ww, scored, threshold, output_path):
    rows, metas = ww._rescored_rows(scored, threshold)
    scored_rows = _scored_rows(rows)
    accepted_count = _accepted_count(scored_rows, threshold)
    ww._write_output(output_path, scored_rows, metas)
    print(
        f"Wrote scored PSMs={len(scored_rows)} accepted={accepted_count} -> {output_path}",
        file=sys.stderr,
    )


def _scored_rows(rows):
    return [row for row in rows if row.get("score") is not None]


def _accepted_count(rows, threshold):
    return sum(1 for row in rows if float(row["score"]) >= threshold)


def _write_threshold_outputs_batched(ww, args, chunk_files, threshold_specs):
    outputs = {
        label: {"pct": pct, "threshold": threshold, "path": output_path, "rows": [], "metas": []}
        for label, pct, threshold, output_path in threshold_specs
    }

    for scored in _score_chunk_batches(ww, args, chunk_files):
        for label, spec in outputs.items():
            rows, metas = ww._rescored_rows(scored, spec["threshold"])
            scored_rows = _scored_rows(rows)
            accepted_count = _accepted_count(scored_rows, spec["threshold"])
            spec["rows"].extend(scored_rows)
            spec["metas"].extend(metas)
            print(
                f"[{label}] scored batch PSMs={len(scored_rows)} accepted={accepted_count} "
                f"total={len(spec['rows'])}",
                file=sys.stderr,
            )
        del scored
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summaries = []
    for label, spec in outputs.items():
        ww._write_output(spec["path"], spec["rows"], spec["metas"])
        accepted_count = _accepted_count(spec["rows"], spec["threshold"])
        print(
            f"[{label}] wrote scored PSMs={len(spec['rows'])} accepted={accepted_count} "
            f"-> {spec['path']}",
            file=sys.stderr,
        )
        summaries.append(_summarize_output(spec["path"], spec["pct"]))
    return summaries


def _summarize_output(output_path, pct):
    # summarize_tsv only uses rows whose output Label is 1, so rejected scored
    # rows retained in the full output do not contribute to E. coli ratios.
    summary = summarize_tsv(str(output_path), exclude_protein_prefixes=DEFAULT_EXCLUDE_PROTEIN_PREFIXES)
    summary["Input"] = _repo_relative(output_path)
    summary["Model"] = MODEL_LABEL
    summary["Spike-in 13C (%)"] = _pct_summary_value(pct)
    return summary


def _write_summary(rows, output_path):
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_value(row.get(column, "")) for column in OUTPUT_COLUMNS})


def _validate_metadata_groups(target_groups, controls):
    for pct, expected_samples in EXPECTED_TARGET_SAMPLES.items():
        observed = tuple(record.sample for record in target_groups.get(pct, []))
        if observed != expected_samples:
            expected = ", ".join(expected_samples)
            rendered = ", ".join(observed) or "none"
            raise ValueError(f"Expected {pct:g}% target samples {expected}; found {rendered}.")

    control_names = tuple(record.sample for record in controls)
    if control_names != EXPECTED_CONTROL_SAMPLES:
        expected = ", ".join(EXPECTED_CONTROL_SAMPLES)
        observed = ", ".join(control_names) or "none"
        raise ValueError(f"Expected metadata controls {expected}; found {observed}.")


def _score_sample_group(ww, args, records, group_label):
    chunk_files, row_counts = _prepare_chunks(records, args.output_dir, group_label)
    sample_names = ",".join(record.sample for record in records)
    print(
        f"Prepared {group_label} chunks={len(chunk_files)} samples={sample_names} rows={row_counts}",
        file=sys.stderr,
    )
    return chunk_files


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Predict DIA spike-ins and controls with WinnowNet pct thresholds.")
    parser.add_argument("--psm-dir", default=str(DEFAULT_PSM_DIR))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--pct-thresholds", default=DEFAULT_PCT_THRESHOLDS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--jobs", type=_positive_int, default=DEFAULT_JOBS)
    parser.add_argument("--cores", type=_positive_int, default=None)
    parser.add_argument("--threads-per-job", type=_positive_int, default=3)
    parser.add_argument(
        "--chunk-batch-size",
        type=_positive_int,
        default=DEFAULT_CHUNK_BATCH_SIZE,
        help=(
            "Number of chunk TSVs to score per fresh worker pool. "
            "Default: same as --jobs, so each worker handles at most one chunk before memory is released."
        ),
    )
    parser.add_argument("--batch-size", type=_positive_int, default=1024)
    parser.add_argument("--ms1-window", type=_positive_float, default=10.0)
    parser.add_argument("--ppm", type=_positive_float, default=10.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.psm_dir = Path(args.psm_dir).resolve()
    args.metadata = Path(args.metadata).resolve()
    args.model = Path(args.model).resolve()
    args.config = Path(args.config).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    args.summary = Path(args.summary).resolve()

    ww = _load_winnownet_module()
    args.max_peaks = ww.DEFAULT_MAX_PEAKS
    args.target_exclude_protein_prefixes = ww._parse_protein_prefixes(",".join(DEFAULT_EXCLUDE_PROTEIN_PREFIXES))

    control_pct_bins = _parse_pct_thresholds(args.pct_thresholds)
    target_pcts = sorted(EXPECTED_TARGET_SAMPLES)
    required_pcts = sorted(set(target_pcts + control_pct_bins))
    thresholds = _load_pct_thresholds(args.model, required_pcts)
    target_groups, controls = _load_metadata_groups(args.metadata, args.psm_dir)
    _validate_metadata_groups(target_groups, controls)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summaries = []

    for pct in target_pcts:
        label = _pct_label(pct)
        checkpoint_threshold = thresholds[pct]
        threshold = _target_threshold_for_pct(pct, thresholds)
        if threshold == checkpoint_threshold:
            print(f"[{label}] target checkpoint threshold={threshold:.12g}", file=sys.stderr)
        else:
            print(
                f"[{label}] target adjusted threshold={threshold:.12g} "
                f"(checkpoint={checkpoint_threshold:.12g})",
                file=sys.stderr,
            )
        chunk_files = _score_sample_group(ww, args, target_groups[pct], label)
        output_path = args.output_dir / f"{label}_winnownet_filtered_psms.tsv"
        summaries.extend(
            _write_threshold_outputs_batched(
                ww,
                args,
                chunk_files,
                [(label, pct, threshold, output_path)],
            )
        )

    control_chunk_files = _score_sample_group(ww, args, controls, "control")
    control_threshold_specs = []
    for pct in control_pct_bins:
        threshold = thresholds[pct]
        label = _pct_label(pct)
        output_path = args.output_dir / f"{label}_winnownet_control_filtered_psms.tsv"
        print(f"[{label}] control checkpoint threshold={threshold:.12g}", file=sys.stderr)
        control_threshold_specs.append((f"{label}_control", pct, threshold, output_path))
    summaries.extend(_write_threshold_outputs_batched(ww, args, control_chunk_files, control_threshold_specs))

    _write_summary(summaries, args.summary)
    print(f"summary={args.summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
