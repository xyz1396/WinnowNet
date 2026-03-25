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


def canonicalize_peptide_sequence(value):
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"\[(.*)\]", text)
    if match:
        return match.group(1).strip()
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
