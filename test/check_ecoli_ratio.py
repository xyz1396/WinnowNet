import argparse
import csv
import glob
import math
import os
import re
import sys


ECOLI_TOKEN = "ECOLI"
DEFAULT_EXCLUDE_PROTEIN_PREFIXES = ("Decoy_", "Con_")
DEFAULT_QVALUE_THRESHOLD = None
MS2_ABUNDANCE_COLUMN = "MS2IsotopicAbundances"


SUMMARY_COLUMNS = [
    "Spike-in 13C (%)",
    "E. coli PSMs",
    "E. coli peptides",
    "E. coli proteins",
    "E. coli PSM ratio",
    "E. coli peptide ratio",
    "Peptide 13C (%) median",
    "Peptide 13C (%) MAD",
]


def _split_path_arg(value):
    paths = []
    for item in str(value or "").split(","):
        item = item.strip()
        if item:
            paths.append(item)
    return paths


def _parse_prefixes(value):
    if value is None:
        return []
    return [item for item in _split_path_arg(value)]


def _expand_inputs(values):
    paths = []
    seen = set()
    for value in values:
        for item in _split_path_arg(value):
            if os.path.isdir(item):
                matches = sorted(glob.glob(os.path.join(item, "*.tsv")))
            elif any(ch in item for ch in ["*", "?", "["]):
                matches = sorted(glob.glob(item))
            else:
                matches = [item]
            for match in matches:
                if match.endswith(".tsv") and match not in seen:
                    seen.add(match)
                    paths.append(match)
    return paths


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


def _is_target(row):
    label = str(row.get("Label", "")).strip()
    return label in {"1", "1.0"}


def _is_used_row(row, qvalue_threshold, include_all_rows):
    if not _is_target(row):
        return False
    if include_all_rows or qvalue_threshold is None:
        return True
    qvalue = _to_float(row.get("q-value", row.get("qvalue", "")))
    if qvalue is None:
        return True
    return qvalue <= qvalue_threshold


def _has_ecoli_protein(row, exclude_protein_prefixes):
    for protein in _split_proteins(row.get("Proteins", "")):
        if _protein_matches_prefixes(protein, exclude_protein_prefixes):
            continue
        if ECOLI_TOKEN in protein.upper():
            return True
    return False


def _split_proteins(value):
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    return [item.strip() for item in text.split(",") if item.strip()]


def _protein_matches_prefixes(protein, prefixes):
    return any(protein.startswith(prefix) for prefix in prefixes)


def _proteins_all_match_prefixes(proteins, prefixes):
    if not prefixes or not proteins:
        return False
    return all(_protein_matches_prefixes(protein, prefixes) for protein in proteins)


def _canonical_peptide(value):
    text = str(value or "").strip()
    match = re.search(r"\[(.*)\]", text)
    if match:
        return match.group(1).strip()
    return text


def _median(values):
    if not values:
        return ""
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _mad(values):
    if not values:
        return ""
    center = _median(values)
    deviations = [abs(value - center) for value in values]
    return _median(deviations)


def _format_value(value):
    if value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.10g}"
    return str(value)


def _ratio(numerator, denominator):
    if denominator == 0:
        return ""
    return numerator / float(denominator)


def _infer_spike_in_13c_pct(path):
    match = re.search(r"pct[_-]?(\d+(?:\.\d+)?)", path, flags=re.IGNORECASE)
    if match is None:
        return ""
    value_text = match.group(1)
    if "." in value_text:
        return _to_float(value_text) or ""
    return float(int(value_text))


def summarize_tsv(
    path,
    qvalue_threshold=DEFAULT_QVALUE_THRESHOLD,
    include_all_rows=False,
    exclude_protein_prefixes=DEFAULT_EXCLUDE_PROTEIN_PREFIXES,
):
    rows_used = 0
    ecoli_psms = 0
    peptides = set()
    ecoli_peptides = set()
    ecoli_proteins = set()
    ms2_abundance_values = []

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            row_proteins = _split_proteins(row.get("Proteins", ""))
            if _proteins_all_match_prefixes(row_proteins, exclude_protein_prefixes):
                continue
            if not _is_used_row(row, qvalue_threshold, include_all_rows):
                continue

            rows_used += 1
            is_ecoli = _has_ecoli_protein(row, exclude_protein_prefixes)
            if is_ecoli:
                ecoli_psms += 1

            peptide = _canonical_peptide(row.get("Peptide", ""))
            if peptide:
                peptides.add(peptide)
                if is_ecoli:
                    ecoli_peptides.add(peptide)

            for protein in row_proteins:
                if _protein_matches_prefixes(protein, exclude_protein_prefixes):
                    continue
                if ECOLI_TOKEN in protein.upper():
                    ecoli_proteins.add(protein)

            ms2_abundance = _to_float(row.get(MS2_ABUNDANCE_COLUMN, ""))
            if ms2_abundance is not None:
                ms2_abundance_values.append(ms2_abundance)

    return {
        "Spike-in 13C (%)": _infer_spike_in_13c_pct(path),
        "E. coli PSMs": ecoli_psms,
        "E. coli peptides": len(ecoli_peptides),
        "E. coli proteins": len(ecoli_proteins),
        "E. coli PSM ratio": _ratio(ecoli_psms, rows_used),
        "E. coli peptide ratio": _ratio(len(ecoli_peptides), len(peptides)),
        "Peptide 13C (%) median": _median(ms2_abundance_values),
        "Peptide 13C (%) MAD": _mad(ms2_abundance_values),
    }


def write_summary(rows, output_path=None):
    if output_path:
        with open(output_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: _format_value(row.get(column, "")) for column in SUMMARY_COLUMNS})
        return

    writer = csv.DictWriter(sys.stdout, fieldnames=SUMMARY_COLUMNS, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _format_value(row.get(column, "")) for column in SUMMARY_COLUMNS})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Summarize E. coli ratios and 13C enrichment from WinnowNet predicted SIP TSV files."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Predicted TSV files, directories, globs, or comma-separated lists.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output summary TSV path. Defaults to stdout.",
    )
    parser.add_argument(
        "--q-value",
        type=float,
        default=DEFAULT_QVALUE_THRESHOLD,
        help="Optionally also require Label=1 rows to have q-value <= this threshold.",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Summarize all Label=1 rows, ignoring any --q-value threshold.",
    )
    parser.add_argument(
        "--exclude-protein-prefix",
        default=",".join(DEFAULT_EXCLUDE_PROTEIN_PREFIXES),
        help=(
            "Drop PSMs when all proteins start with one of these comma-separated prefixes. "
            "Default: Decoy_,Con_."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_files = _expand_inputs(args.inputs)
    if not input_files:
        raise ValueError("No input TSV files found.")
    exclude_protein_prefixes = _parse_prefixes(args.exclude_protein_prefix)

    summaries = [
        summarize_tsv(
            path,
            qvalue_threshold=args.q_value,
            include_all_rows=args.all_rows,
            exclude_protein_prefixes=exclude_protein_prefixes,
        )
        for path in input_files
    ]
    write_summary(summaries, args.output)


if __name__ == "__main__":
    main()
