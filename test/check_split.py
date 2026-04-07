#!/usr/bin/env python3
"""Export the ATT trainer split assignment as a TSV manifest."""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "script"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


DEFAULT_OUTPUT_NAME = "att_split_manifest.tsv"
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
LABEL_NAME = {1: "target", 0: "decoy"}
TSV_COLUMNS = [
    "split",
    "label",
    "psm_id",
    "spectrum_id",
    "peptide",
    "source_file",
]


def _split_csv_args(values):
    items = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                items.append(item)
    return items


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Write a TSV manifest of ATT train/val/test PSM split assignments."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="",
        help="Input directory or pickle path(s) for embedded-label mode, comma-separated if needed.",
    )
    parser.add_argument(
        "-target",
        "--target",
        action="append",
        default=[],
        help="Target pickle path(s) or directories, repeat or comma-separate values.",
    )
    parser.add_argument(
        "-decoy",
        "--decoy",
        action="append",
        default=[],
        help="Decoy pickle path(s) or directories, repeat or comma-separate values.",
    )
    parser.add_argument(
        "--exclude-protein-prefix",
        action="append",
        default=[],
        help="Drop PSMs whose proteins all match one of these prefixes, repeat or comma-separate values.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output TSV path (default: {DEFAULT_OUTPUT_NAME}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Grouped split RNG seed (default: 10).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio passed to the ATT grouped split (default: 0.1).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test ratio passed to the ATT grouped split (default: 0.1).",
    )
    return parser


def _validate_args(args):
    if args.val_ratio < 0.0 or args.val_ratio >= 1.0:
        raise ValueError("--val-ratio must be in [0, 1).")
    if args.test_ratio < 0.0 or args.test_ratio >= 1.0:
        raise ValueError("--test-ratio must be in [0, 1).")
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("--val-ratio + --test-ratio must be less than 1.")

    has_target = len(args.target) > 0
    has_decoy = len(args.decoy) > 0
    if has_target != has_decoy:
        raise ValueError("Provide both --target and --decoy together.")
    if not args.input and not has_target:
        raise ValueError("Use -i for embedded-label mode or provide both --target and --decoy.")


def _load_att_helpers():
    try:
        import WinnowNet_Att as att
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "unknown module")
        raise SystemExit(
            "Failed to import script/WinnowNet_Att.py because dependency "
            f"{missing_name!r} is unavailable. Run this checker in the ATT training environment."
        ) from exc
    return att


def _resolve_inputs(att, args):
    exclude_prefixes = _split_csv_args(args.exclude_protein_prefix)
    target_inputs = _split_csv_args(args.target)
    decoy_inputs = _split_csv_args(args.decoy)
    target_pickles, decoy_pickles, split_mode = att._resolve_training_inputs(
        args.input,
        target_inputs,
        decoy_inputs,
    )
    return target_pickles, decoy_pickles, split_mode, exclude_prefixes


def _load_records(att, args):
    target_pickles, decoy_pickles, split_mode, exclude_prefixes = _resolve_inputs(att, args)
    manifest_records = []

    if split_mode:
        if len(target_pickles) == 0 or len(decoy_pickles) == 0:
            raise ValueError("Missing target/decoy pickle inputs.")

        _, _, group_pos = att._load_feature_records(
            target_pickles,
            force_label=1,
            dataset_name="target",
            exclude_protein_prefixes=exclude_prefixes,
            record_sink=manifest_records,
        )
        _, _, group_neg = att._load_feature_records(
            decoy_pickles,
            force_label=0,
            dataset_name="decoy",
            exclude_protein_prefixes=exclude_prefixes,
            record_sink=manifest_records,
        )
        groups = group_pos + group_neg
    else:
        if len(target_pickles) == 0:
            raise ValueError(f"No feature pickles found under {args.input}.")
        _, _, groups = att._load_feature_records(
            target_pickles,
            dataset_name="embedded",
            exclude_protein_prefixes=exclude_prefixes,
            record_sink=manifest_records,
        )

    if len(groups) != len(manifest_records):
        raise ValueError(
            "Internal mismatch while collecting kept ATT records: "
            f"groups={len(groups)} records={len(manifest_records)}."
        )
    return manifest_records, groups


def _assign_splits(att, records, groups, args):
    indices = list(range(len(records)))
    placeholder_y = [[0, 0] for _ in indices]
    (train_idx, _), (val_idx, _), (test_idx, _) = att.split_grouped(
        indices,
        placeholder_y,
        groups,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    split_by_index = {}
    for split_name, split_indices in (
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ):
        for idx in split_indices:
            split_by_index[idx] = split_name

    if len(split_by_index) != len(records):
        raise ValueError(
            "Not all kept PSMs were assigned to a split: "
            f"assigned={len(split_by_index)} total={len(records)}."
        )

    for idx, record in enumerate(records):
        record["split"] = split_by_index[idx]


def _validate_manifest(records):
    spectrum_splits = defaultdict(set)
    for record in records:
        split_name = record.get("split", "")
        if split_name not in SPLIT_ORDER:
            raise ValueError(f"Unexpected split value in manifest: {split_name!r}")
        spectrum_splits[record["spectrum_id"]].add(split_name)

    leaked_spectra = [
        spectrum_id
        for spectrum_id, splits in spectrum_splits.items()
        if len(splits) > 1
    ]
    if leaked_spectra:
        raise ValueError(
            "Spectrum grouping leak detected across splits for "
            f"{len(leaked_spectra)} spectrum IDs."
        )


def _write_manifest(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_records = sorted(
        records,
        key=lambda record: (
            SPLIT_ORDER[record["split"]],
            record["source_file"],
            record["spectrum_id"],
            record["psm_id"],
        ),
    )
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TSV_COLUMNS, delimiter="\t")
        writer.writeheader()
        for record in sorted_records:
            writer.writerow(
                {
                    "split": record["split"],
                    "label": LABEL_NAME[int(record["label"])],
                    "psm_id": record["psm_id"],
                    "spectrum_id": record["spectrum_id"],
                    "peptide": record["peptide"],
                    "source_file": record["source_file"],
                }
            )


def _print_summary(records, output_path):
    split_counts = Counter()
    label_counts = Counter()
    for record in records:
        split_counts[record["split"]] += 1
        label_counts[(record["split"], int(record["label"]))] += 1

    print(f"total_kept_psms={len(records)}")
    for split_name in ("train", "val", "test"):
        print(
            f"{split_name}_psms={split_counts[split_name]} "
            f"{split_name}_targets={label_counts[(split_name, 1)]} "
            f"{split_name}_decoys={label_counts[(split_name, 0)]}"
        )
    print(f"output_tsv={output_path}")


def main():
    parser = _build_parser()
    args = parser.parse_args()
    try:
        _validate_args(args)
        att = _load_att_helpers()
        records, groups = _load_records(att, args)
        _assign_splits(att, records, groups, args)
        _validate_manifest(records)
        output_path = Path(args.output).resolve()
        _write_manifest(records, output_path)
        _print_summary(records, output_path)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
