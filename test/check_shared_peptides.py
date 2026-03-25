import argparse
import re
from collections import Counter
from pathlib import Path

from pkl_utils import get_entry_row_map, load_feature_pickle


PEPTIDE_CANDIDATE_COLUMNS = [
    "Peptide",
    "IdentifiedPeptide",
    "OriginalPeptide",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check peptide overlap between target and decoy feature pickles."
    )
    parser.add_argument("--target", required=True, help="Target feature pickle path.")
    parser.add_argument("--decoy", required=True, help="Decoy feature pickle path.")
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Show the top N shared peptides by combined PSM count.",
    )
    return parser.parse_args()


def canonicalize_peptide(value):
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"\[(.*)\]", text)
    if match:
        return match.group(1).strip()
    return text


def load_peptide_counts(path):
    meta, entries = load_feature_pickle(path)
    counts = Counter()
    raw_counts = Counter()
    missing = 0

    for psm_id, entry in entries.items():
        if not isinstance(entry, dict):
            missing += 1
            continue
        row_map = get_entry_row_map(meta, entry.get("psm_id", psm_id), entry)
        peptide_value = ""
        for column in PEPTIDE_CANDIDATE_COLUMNS:
            peptide_value = str(row_map.get(column, "")).strip()
            if peptide_value:
                break
        if not peptide_value:
            missing += 1
            continue
        canonical = canonicalize_peptide(peptide_value)
        if not canonical:
            missing += 1
            continue
        counts[canonical] += 1
        raw_counts[peptide_value] += 1

    return counts, raw_counts, missing, len(entries)


def main():
    args = parse_args()
    target_path = Path(args.target).resolve()
    decoy_path = Path(args.decoy).resolve()

    target_counts, target_raw_counts, target_missing, target_total = load_peptide_counts(target_path)
    decoy_counts, decoy_raw_counts, decoy_missing, decoy_total = load_peptide_counts(decoy_path)

    shared = set(target_counts) & set(decoy_counts)
    shared_target_psms = sum(target_counts[peptide] for peptide in shared)
    shared_decoy_psms = sum(decoy_counts[peptide] for peptide in shared)

    print(f"target_file={target_path}")
    print(f"decoy_file={decoy_path}")
    print(f"target_total_psms={target_total} target_missing_peptide={target_missing}")
    print(f"decoy_total_psms={decoy_total} decoy_missing_peptide={decoy_missing}")
    print(f"target_unique_peptides={len(target_counts)}")
    print(f"decoy_unique_peptides={len(decoy_counts)}")
    print(f"shared_unique_peptides={len(shared)}")
    print(
        f"target_psms_with_shared_peptide={shared_target_psms} "
        f"({(100.0 * shared_target_psms / max(1, sum(target_counts.values()))):.2f}%)"
    )
    print(
        f"decoy_psms_with_shared_peptide={shared_decoy_psms} "
        f"({(100.0 * shared_decoy_psms / max(1, sum(decoy_counts.values()))):.2f}%)"
    )

    ranked = sorted(
        shared,
        key=lambda peptide: (
            target_counts[peptide] + decoy_counts[peptide],
            target_counts[peptide],
            decoy_counts[peptide],
            peptide,
        ),
        reverse=True,
    )

    print(f"top_shared_peptides={min(args.top, len(ranked))}")
    for peptide in ranked[: args.top]:
        print(
            f"{peptide}\ttarget_psms={target_counts[peptide]}\tdecoy_psms={decoy_counts[peptide]}"
        )


if __name__ == "__main__":
    main()
