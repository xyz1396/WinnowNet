import glob
import os
import pickle
import re


PKL_META_KEY = "__meta__"
PKL_SCHEMA_VERSION = 2


def normalize_long_flag_aliases(argv, aliases):
    normalized = []
    for token in argv:
        normalized.append(aliases.get(token, token))
    return normalized


def _split_path_arg(value):
    if value is None:
        return []
    parts = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            parts.append(item)
    return parts


def expand_pickle_inputs(values):
    paths = []
    seen = set()
    for value in values:
        for item in _split_path_arg(value):
            matches = []
            if os.path.isdir(item):
                matches = sorted(glob.glob(os.path.join(item, "*.pkl")))
            elif any(ch in item for ch in ["*", "?", "["]):
                matches = sorted(glob.glob(item))
            else:
                matches = [item]
            for match in matches:
                if match.endswith(".pkl") and match not in seen:
                    seen.add(match)
                    paths.append(match)
    return paths


def load_pickle_data(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def split_feature_pickle(data):
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at top level, got {type(data).__name__}.")
    meta = {}
    entries = data
    if PKL_META_KEY in data and isinstance(data[PKL_META_KEY], dict):
        meta = data[PKL_META_KEY]
        entries = {k: v for k, v in data.items() if k != PKL_META_KEY}
    return meta, entries


def load_feature_pickle(path):
    data = load_pickle_data(path)
    return split_feature_pickle(data)


def is_rich_entry(entry):
    return isinstance(entry, dict) and (
        "model_input" in entry or "row_values" in entry or "label" in entry
    )


def get_entry_model_input(entry):
    if is_rich_entry(entry):
        return entry.get("model_input")
    return entry


def get_entry_label(entry):
    if is_rich_entry(entry):
        return entry.get("label")
    return None


def get_entry_label_confidence(entry):
    if is_rich_entry(entry):
        return entry.get("label_confidence")
    return None


def get_entry_label_raw(entry):
    if is_rich_entry(entry):
        return entry.get("label_raw")
    return None


def get_entry_row_index(entry, default_idx=0):
    if is_rich_entry(entry):
        return int(entry.get("row_index", default_idx))
    return default_idx


def get_entry_row_map(meta, psm_id, entry):
    if not is_rich_entry(entry):
        raise ValueError("The pickle entry does not contain stored row metadata.")

    columns = list(meta.get("columns", []))
    row_values = list(entry.get("row_values", []))
    row_map = {}
    for idx, column in enumerate(columns):
        row_map[column] = row_values[idx] if idx < len(row_values) else ""

    if not row_map:
        row_map["PSMId"] = psm_id
    elif "PSMId" not in row_map and "SpecId" not in row_map:
        row_map["PSMId"] = psm_id
    return row_map


def parse_prefix_filters(value):
    if value is None:
        return []
    prefixes = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            prefixes.append(item)
    return prefixes


def _extract_protein_names(text):
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    if cleaned.startswith("{") and cleaned.endswith("}"):
        cleaned = cleaned[1:-1]
    proteins = []
    for item in cleaned.split(","):
        item = item.strip()
        if item:
            proteins.append(item)
    return proteins


def proteins_all_match_prefixes(row_map, prefixes):
    if not prefixes:
        return False
    protein_text = ""
    for column in ["Proteins", "Proteinname", "ProteinNames"]:
        protein_text = str(row_map.get(column, "")).strip()
        if protein_text:
            break
    proteins = _extract_protein_names(protein_text)
    if not proteins:
        return False
    return all(any(protein.startswith(prefix) for prefix in prefixes) for protein in proteins)


def canonicalize_peptide_sequence(value):
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"\[(.*)\]", text)
    if match:
        return match.group(1).strip()
    return text


def _base_spectrum_id(value):
    text = str(value or "").strip()
    if not text:
        return ""
    parts = text.rsplit(".", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return text


def get_entry_group_key(meta, psm_id, entry):
    if not is_rich_entry(entry):
        return str(psm_id)

    row_map = get_entry_row_map(meta, psm_id, entry)

    peptide_value = ""
    for column in ["Peptide", "IdentifiedPeptide", "OriginalPeptide"]:
        peptide_value = str(row_map.get(column, "")).strip()
        if peptide_value:
            break

    peptide_core = canonicalize_peptide_sequence(peptide_value)
    if peptide_core:
        return peptide_core
    return str(psm_id)


def get_entry_spectrum_group_key(meta, psm_id, entry):
    base_id = _base_spectrum_id(psm_id)
    if base_id:
        return base_id
    if not is_rich_entry(entry):
        return str(psm_id)

    row_map = get_entry_row_map(meta, psm_id, entry)
    for column in ["PSMId", "SpecId"]:
        base_id = _base_spectrum_id(row_map.get(column, ""))
        if base_id:
            return base_id

    scan_value = ""
    for column in ["ScanNr", "Scan", "scan"]:
        scan_value = str(row_map.get(column, "")).strip()
        if scan_value:
            break

    source_file = os.path.basename(str(meta.get("source_file", "")).strip())
    if source_file and scan_value:
        return f"{source_file}::{scan_value}"
    if scan_value:
        return f"scan::{scan_value}"
    return str(psm_id)


def choose_output_column(columns, preferred_names, default_name):
    lower_to_actual = {column.lower(): column for column in columns}
    for name in preferred_names:
        actual = lower_to_actual.get(name.lower())
        if actual is not None:
            return actual
    return default_name


def format_label_value(label, raw_template=None):
    if label is None:
        return ""
    return "1" if int(label) == 1 else "-1"
