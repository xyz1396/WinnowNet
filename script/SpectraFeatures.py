import csv
import getopt
import os
import pickle
import subprocess
import sys
import time
from multiprocessing import get_all_start_methods, get_context

import numpy as np
from sklearn.preprocessing import StandardScaler
from pkl_utils import PKL_META_KEY, PKL_SCHEMA_VERSION, normalize_long_flag_aliases

DEFAULT_MAX_PEAKS = 300
pairmaxlength = DEFAULT_MAX_PEAKS
diffDa = 0.01

AA_dict = {
    "G": 57.021464,
    "R": 156.101111,
    "V": 99.068414,
    "P": 97.052764,
    "S": 87.032028,
    "U": 150.95363,
    "L": 113.084064,
    "M": 131.040485,
    "Q": 128.058578,
    "N": 114.042927,
    "Y": 163.063329,
    "E": 129.042593,
    "C": 103.009185 + 57.0214637236,   #fixed modification
    "F": 147.068414,
    "I": 113.084064,
    "A": 71.037114,
    "T": 101.047679,
    "W": 186.079313,
    "H": 137.058912,
    "D": 115.026943,
    "K": 128.094963,
    "~": 15.99491
}

PROTON_MASS = 1.00727646688
H = 1.007825
O = 15.9949
N_TERMINUS = H
C_TERMINUS = O + H

_WORKER_EXP_MS1 = None
_WORKER_EXP_MS2 = None
_WORKER_THEORY = None
_WORKER_FEATURE = None
_WORKER_SCAN_MAP = None
_WORKER_MODE = "att"
_WORKER_FRAGMENT_TOP_N = 3
_WORKER_MS1_WINDOW_MZ = 10.0

class peptide:

    def __init__(self):
        self.identified_pep = ""
        self.qvalue = 0
        self.qn=0
        self.qnl=0
        self.num_missed_cleavages = 0
        self.mono_mass = 0
        self.mass_error = 0
        self.isotopic_mass_window_shift = 0
        self.theory_mass=0
        self.peplen=0
        self.PSMId=''

class scan:

    def __init__(self):
        self.fidx = 0
        self.charge = 0
        self.scan_number = 0
        self.ms1_scan = ""
        self.pep_list = []

    def add_pep(self, pep):
        self.pep_list.append(pep)


    def get_features(self):
        pep_sorted_list = sorted(self.pep_list, key=lambda pep: pep.qvalue)
        if len(pep_sorted_list) == 0:
            return
        if len(pep_sorted_list) == 1:
            pep_sorted_list[0].qn = 1
            pep_sorted_list[0].qnl = 1
        else:
            lth_qvalue = pep_sorted_list[-1].qvalue
            for i in range(len(pep_sorted_list) - 1):
                if pep_sorted_list[i + 1].qvalue > 0:
                    pep_sorted_list[i].qn = (
                        pep_sorted_list[i + 1].qvalue - pep_sorted_list[i].qvalue
                    ) / pep_sorted_list[i + 1].qvalue
                else:
                    pep_sorted_list[i].qn = 1
                if lth_qvalue > 0:
                    pep_sorted_list[i].qnl = (lth_qvalue - pep_sorted_list[i].qvalue) / lth_qvalue
                else:
                    pep_sorted_list[i].qnl = 1
            pep_sorted_list[-1].qn = 1
            pep_sorted_list[-1].qnl = 1


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_peak_pairs(tokens, start_idx=0):
    peaks = []
    for i in range(start_idx, len(tokens) - 1, 2):
        try:
            mz = float(tokens[i])
            intensity = float(tokens[i + 1])
        except ValueError:
            continue
        peaks.append([mz, intensity])
    return peaks


def _normalize_sip_abundance_args(argv):
    normalized = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in ("-b", "--sip-abundance"):
            next_token = argv[i + 1] if i + 1 < len(argv) else None
            if next_token is None or len(next_token) == 0 or next_token.startswith("-"):
                normalized.append("--sip-abundance=config")
            else:
                normalized.append("--sip-abundance=" + next_token)
                i += 1
        else:
            normalized.append(token)
        i += 1
    return normalized


def _most_abundant_peak_mz(peaks):
    if len(peaks) == 0:
        return 0.0
    return max(peaks, key=lambda x: x[1])[0]


def _parse_psm_id(psm_id):
    psm_id = psm_id.strip()
    if "." in psm_id:
        parts = psm_id.split(".")
        if len(parts) >= 3 and parts[-2].isdigit():
            scan_no = parts[-2]
            file_prefix = ".".join(parts[:-2])
            return file_prefix, scan_no
    parts = psm_id.split("_")
    if len(parts) >= 3 and parts[-3].isdigit():
        scan_no = parts[-3]
        file_prefix = "_".join(parts[:-3])
        return file_prefix, scan_no
    return "", ""


def _clean_peptide(peptide_str):
    peptide_str = peptide_str.strip()
    if "[" in peptide_str and "]" in peptide_str:
        return peptide_str.split("[", 1)[1].split("]", 1)[0]
    if "." in peptide_str:
        parts = peptide_str.split(".")
        if len(parts) >= 3:
            return parts[1]
    return peptide_str


def _build_field_lookup(fieldnames):
    lookup = {}
    for field in fieldnames:
        if field is None:
            continue
        lookup[str(field).strip().lower()] = field
    return lookup


def _resolve_field_name(field_lookup, aliases):
    for alias in aliases:
        actual = field_lookup.get(alias.lower())
        if actual is not None:
            return actual
    return None


def _get_field_value(row, field_lookup, aliases, default=""):
    field_name = _resolve_field_name(field_lookup, aliases)
    if field_name is None:
        return default
    return row.get(field_name, default)


def _parse_label_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    if lower in {"1", "+1", "true", "t", "target"}:
        return 1
    if lower in {"0", "-1", "false", "f", "decoy"}:
        return 0
    return None


def _compute_label_confidence(label, qvalue_raw):
    if label is None:
        return None
    if str(qvalue_raw).strip() == "":
        return 1.0 if label == 1 else 0.0
    qvalue = max(0.0, min(1.0, _to_float(qvalue_raw, 0.0)))
    if label == 1:
        return max(0.0, min(1.0, 1.0 - qvalue))
    return 0.0


def FTtoDict(file_path, reduce_peak_charge_to_one=False):
    msdict = {}
    scan_id = None
    charge = 0
    parent_scan = ""
    isolation_center_mz = 0.0
    precursor_candidates = []
    peaks = []

    def _flush():
        if scan_id is None:
            return
        sorted_peaks = sorted(peaks, key=lambda x: x[0])
        msdict[scan_id] = {
            "isolation_center_mz__charge": f"{isolation_center_mz}_{charge}",
            "peaks": sorted_peaks,
            "parent_scan": parent_scan,
            "isolation_center_mz": isolation_center_mz,
            "precursor_candidates": precursor_candidates.copy(),
        }

    with open(file_path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            first_char = line[0]
            if first_char.isalpha():
                tokens = line.split()
                if not tokens:
                    continue
                if tokens[0] == "S":
                    _flush()
                    scan_id = str(int(tokens[1]))
                    isolation_center_mz = _to_float(tokens[2]) if len(tokens) > 2 else 0.0
                    charge = 0
                    parent_scan = ""
                    precursor_candidates = []
                    peaks = []
                elif tokens[0] == "Z" and scan_id is not None:
                    # FT2 Z-line format:
                    # Z <selected_z> <selected_z * isolation_center_mz> [<cand_z1> <cand_mz1> ...]
                    selected_charge = int(_to_float(tokens[1], 0))
                    if selected_charge > 0:
                        charge = selected_charge
                    candidate_pairs = []
                    seen = set()
                    if len(tokens) > 3:
                        remain = tokens[3:]
                        pair_count = len(remain) // 2
                        for i in range(pair_count):
                            cand_charge = int(_to_float(remain[2 * i], 0))
                            cand_mz = _to_float(remain[2 * i + 1], 0.0)
                            cand_key = (cand_charge, cand_mz)
                            if cand_charge > 0 and cand_mz > 0 and cand_key not in seen:
                                seen.add(cand_key)
                                candidate_pairs.append((cand_charge, cand_mz))
                    precursor_candidates = [{"charge": z, "mz": mz} for z, mz in candidate_pairs]
                elif (
                    tokens[0] == "D"
                    and scan_id is not None
                    and len(tokens) >= 3
                    and tokens[1] == "ParentScanNumber"
                ):
                    parent_scan = tokens[2]
                continue
            vals = line.split()
            if len(vals) >= 2:
                mz = _to_float(vals[0])
                intensity = _to_float(vals[1])
                peak_charge = int(_to_float(vals[5], 0)) if len(vals) >= 6 else 0
                if reduce_peak_charge_to_one and peak_charge > 1:
                    mz = peak_charge * (mz - PROTON_MASS) + PROTON_MASS
                peaks.append([mz, intensity])
    _flush()
    return msdict


def theoryToDict(file_path):
    theory_dic = {}
    with open(file_path) as fh:
        raw_lines = [line.strip() for line in fh]

    lines = [line for line in raw_lines if line and not line.startswith("#")]

    def _parse_float_list(line):
        values = []
        for token in line.split():
            try:
                values.append(float(token))
            except ValueError:
                continue
        return values

    def _build_peaks(mz_vals, intensity_vals):
        n = min(len(mz_vals), len(intensity_vals))
        peaks = [[mz_vals[i], intensity_vals[i]] for i in range(n)]
        return sorted(peaks)

    def _parse_int_list(line):
        values = []
        for token in line.split():
            try:
                values.append(int(float(token)))
            except ValueError:
                continue
        return values

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith(">"):
            i += 1
            continue

        if i + 6 >= len(lines):
            break

        psm_id = line[1:].strip()
        precursor_mz_vals = _parse_float_list(lines[i + 1])
        precursor_int_vals = _parse_float_list(lines[i + 2])
        fragment_mz_vals = _parse_float_list(lines[i + 3])
        fragment_int_vals = _parse_float_list(lines[i + 4])
        fragment_kinds = lines[i + 5].split()
        fragment_positions = _parse_int_list(lines[i + 6])

        precursor_scan = _build_peaks(precursor_mz_vals, precursor_int_vals)
        fragment_scan = _build_peaks(fragment_mz_vals, fragment_int_vals)
        all_scan = sorted(precursor_scan + fragment_scan)

        theory_dic[psm_id] = {
            "all": all_scan,
            "precursor": precursor_scan,
            "fragment": fragment_scan,
            "precursor_abundant_mz": _most_abundant_peak_mz(precursor_scan),
            "fragment_kinds": fragment_kinds,
            "fragment_positions": fragment_positions,
        }
        i += 7

    return theory_dic


def read_tsv(tsv_file, psm_dict, ms2_dict):
    psm_scan_map = {}
    row_records = []
    record_key_counts = {}
    row_stats = {
        "total_rows": 0,
        "rows_with_psm_id": 0,
        "target_label_rows": 0,
        "decoy_label_rows": 0,
        "unlabeled_rows": 0,
        "usable_peptide_rows": 0,
        "skipped_missing_peptide": 0,
    }
    with open(tsv_file, newline="") as fh:
        rows = [row for row in csv.reader(fh, delimiter="\t") if len(row) > 0]

    if len(rows) == 0:
        table_meta = {
            "schema_version": PKL_SCHEMA_VERSION,
            "source_file": os.path.abspath(tsv_file),
            "source_kind": os.path.splitext(tsv_file)[1].lower().lstrip("."),
            "has_header": False,
            "columns": [],
        }
        return psm_scan_map, table_meta, row_records, row_stats

    first_row = rows[0]
    first_row_lookup = {str(value).strip().lower() for value in first_row}
    use_header = "psmid" in first_row_lookup or "specid" in first_row_lookup
    if use_header:
        columns = list(first_row)
        raw_rows = rows[1:]
    else:
        width = max(len(row) for row in rows)
        columns = [f"col{i + 1}" for i in range(width)]
        raw_rows = rows

    field_lookup = _build_field_lookup(columns)
    table_meta = {
        "schema_version": PKL_SCHEMA_VERSION,
        "source_file": os.path.abspath(tsv_file),
        "source_kind": os.path.splitext(tsv_file)[1].lower().lstrip("."),
        "has_header": use_header,
        "columns": columns,
        "id_column": _resolve_field_name(field_lookup, ["PSMId", "SpecId"]),
        "score_column": _resolve_field_name(field_lookup, ["score"]),
        "qvalue_column": _resolve_field_name(field_lookup, ["q-value", "qvalue"]),
        "label_column": _resolve_field_name(field_lookup, ["Label"]),
    }

    for row_index, raw_values in enumerate(raw_rows):
        row_stats["total_rows"] += 1
        values = list(raw_values) + [""] * max(0, len(columns) - len(raw_values))
        row = {columns[i]: values[i] for i in range(len(columns))}

        if use_header:
            idx = (_get_field_value(row, field_lookup, ["PSMId", "SpecId"]) or "").strip()
            qvalue_raw = _get_field_value(row, field_lookup, ["q-value", "qvalue"], "")
            peptide_raw = _get_field_value(
                row,
                field_lookup,
                ["Peptide", "peptide", "IdentifiedPeptide", "PeptideSequence"],
                "",
            )
            mass_error_raw = _get_field_value(
                row,
                field_lookup,
                ["massErrors", "massError", "MassError", "Massdiff", "MassDiff"],
                "0",
            )
            isotopic_shift_raw = _get_field_value(
                row,
                field_lookup,
                ["isotopicMassWindowShifts", "isotopicMassWindowShift"],
                "0",
            )
            charge_raw = _get_field_value(
                row,
                field_lookup,
                ["parentCharges", "ParentCharge", "charge", "Charge"],
                "",
            )
            scan_nr_raw = _get_field_value(
                row,
                field_lookup,
                ["ScanNr", "Scan", "scan"],
                "",
            )
            label_raw = _get_field_value(row, field_lookup, ["Label"], "")
        else:
            idx = values[0] if len(values) > 0 else ""
            if len(values) > 1 and ("FT2" in values[1] or "_" in values[1]):
                idx = values[1]
            if len(values) > 0 and "FT2" in values[0] and "FT2" not in idx:
                idx = values[0]
            qvalue_raw = values[2] if len(values) > 2 else "0"
            peptide_raw = values[4] if len(values) > 4 else (values[1] if len(values) > 1 else "")
            mass_error_raw = "0"
            isotopic_shift_raw = "0"
            charge_raw = ""
            scan_nr_raw = ""
            label_raw = values[0] if len(values) > 0 else ""

        idx = idx.strip()
        label = _parse_label_value(label_raw)
        label_confidence = _compute_label_confidence(label, qvalue_raw)
        if idx:
            row_stats["rows_with_psm_id"] += 1
        if label == 1:
            row_stats["target_label_rows"] += 1
        elif label == 0:
            row_stats["decoy_label_rows"] += 1
        else:
            row_stats["unlabeled_rows"] += 1
        record_key = idx or f"__row_{row_index}"
        if record_key in record_key_counts:
            record_key_counts[record_key] += 1
            record_key = f"{record_key}__dup{record_key_counts[record_key]}"
        else:
            record_key_counts[record_key] = 0
        row_records.append(
            {
                "record_key": record_key,
                "psm_id": idx,
                "row_index": row_index,
                "row_values": values,
                "label": label,
                "label_raw": label_raw,
                "label_confidence": label_confidence,
            }
        )

        if not idx:
            continue

        fileidx, scannum = _parse_psm_id(idx)
        if not scannum:
            scannum = str(int(_to_float(scan_nr_raw, 0)))
        if not fileidx:
            fileidx = idx

        charge = str(int(_to_float(charge_raw, 0)))
        if charge == "0":
            parts = idx.split("_")
            if len(parts) >= 2 and parts[-2].isdigit():
                charge = parts[-2]
            else:
                charge = "2"
        charge_int = int(_to_float(charge, 0))

        qvalue = _to_float(qvalue_raw, 0.0)
        peptidestr = _clean_peptide(peptide_raw)
        if not peptidestr:
            row_stats["skipped_missing_peptide"] += 1
            continue
        row_stats["usable_peptide_rows"] += 1

        pep = peptide()
        pep.PSMId = idx
        pep.qvalue = qvalue
        pep.identified_pep = peptidestr
        pep.num_missed_cleavages = peptidestr[:-1].count("K") + peptidestr[:-1].count("R")
        pep.mono_mass = sum([AA_dict[aa] for aa in peptidestr]) + N_TERMINUS + C_TERMINUS
        pep.theory_mass = pep.mono_mass
        pep.mass_error = _to_float(mass_error_raw, 0.0)
        pep.isotopic_mass_window_shift = _to_float(isotopic_shift_raw, 0.0)
        pep.peplen = len(peptidestr)

        uniqueID = f"{fileidx}.{scannum}.{charge}"
        if uniqueID in psm_dict:
            psm_dict[uniqueID].add_pep(pep)
        else:
            one_scan = scan()
            one_scan.fidx = fileidx
            one_scan.scan_number = scannum
            one_scan.charge = charge
            one_scan.add_pep(pep)
            one_scan.ms1_scan = ms2_dict.get(scannum, {}).get("parent_scan", "")
            psm_dict[uniqueID] = one_scan
        psm_scan_map[idx] = {
            "ms2_scan": scannum,
            "ms1_scan": psm_dict[uniqueID].ms1_scan,
            "charge": charge_int,
        }
    return psm_scan_map, table_meta, row_records, row_stats

def feature_dict(f_dict):
    D_feature = dict()
    for psm in f_dict:
        for pep in f_dict[psm].pep_list:
            D_feature[pep.PSMId] = [
                pep.qvalue,
                pep.qn,
                pep.qnl,
                pep.mono_mass,
                pep.mass_error,
                pep.isotopic_mass_window_shift,
                pep.peplen,
                pep.num_missed_cleavages,
            ]
            if f_dict[psm].charge == '1':
                D_feature[pep.PSMId].extend([1, 0, 0])
            elif f_dict[psm].charge == '2':
                D_feature[pep.PSMId].extend([0, 1, 0])
            else:
                D_feature[pep.PSMId].extend([0, 0, 1])

    return D_feature


def pad_control_3d(data):
    data = sorted(data, key=lambda x: x[1], reverse=True)
    if len(data) > pairmaxlength:
        data = data[:pairmaxlength]
    else:
        while (len(data) < pairmaxlength):
            data.append([0, 0, 0])
    data = sorted(data, key=lambda x: x[0])
    return data

def _to_peak_array(peaks):
    if not peaks:
        return np.empty((0, 2), dtype=float)
    return np.asarray(peaks, dtype=float)


def _append_matches(exp_array, theory_array, x_features):
    if exp_array.shape[0] == 0 or theory_array.shape[0] == 0:
        return
    for mz in theory_array:
        index = np.where(
            np.logical_and(exp_array[:, 0] > mz[0] - diffDa, exp_array[:, 0] < mz[0] + diffDa)
        )[0]
        if len(index) > 0:
            for idx in index:
                x_features.append([exp_array[idx][0] - mz[0], exp_array[idx][1], mz[1]])


def _select_fragment_peaks_for_cnn(Xtheory, top_n=3):
    fragment = Xtheory.get("fragment", [])
    if len(fragment) == 0:
        return []
    if top_n <= 0:
        return sorted(fragment)

    fragment_kinds = Xtheory.get("fragment_kinds", [])
    fragment_positions = Xtheory.get("fragment_positions", [])
    n_meta = min(len(fragment), len(fragment_kinds), len(fragment_positions))
    if n_meta == 0:
        return sorted(fragment)

    grouped = {}
    for i in range(n_meta):
        kind = str(fragment_kinds[i]).lower()
        position = int(_to_float(fragment_positions[i], 0))
        key = (kind, position)
        grouped.setdefault(key, []).append(fragment[i])

    selected = []
    for peaks in grouped.values():
        selected.extend(sorted(peaks, key=lambda x: x[1], reverse=True)[:top_n])

    if len(fragment) > n_meta:
        selected.extend(fragment[n_meta:])
    return sorted(selected)


def _filter_ms1_window(ms1_peaks, precursor_theory, isolation_window_mz=10.0):
    if len(ms1_peaks) == 0 or len(precursor_theory) == 0:
        return ms1_peaks
    if isolation_window_mz <= 0:
        return ms1_peaks
    center_mz = _most_abundant_peak_mz(precursor_theory)
    half_window = isolation_window_mz / 2.0
    low = center_mz - half_window
    high = center_mz + half_window
    return [p for p in ms1_peaks if low <= p[0] <= high]


def IonExtract(
    ms1_peaks,
    ms2_peaks,
    Xtheory,
    X_add_feature,
    fragment_top_n=3,
):
    ms1_array = _to_peak_array(ms1_peaks)
    ms2_array = _to_peak_array(ms2_peaks)
    precursor_peaks = Xtheory.get("precursor", [])
    fragment_peaks = _select_fragment_peaks_for_cnn(Xtheory, fragment_top_n)
    theory_precursor = _to_peak_array(precursor_peaks)
    theory_fragment = _to_peak_array(fragment_peaks)
    theory_all = _to_peak_array(sorted(precursor_peaks + fragment_peaks))

    xFeatures = []
    _append_matches(ms1_array, theory_precursor, xFeatures)
    _append_matches(ms2_array, theory_fragment, xFeatures)

    if len(xFeatures) == 0:
        _append_matches(ms2_array, theory_all, xFeatures)

    xFeatures = np.asarray(pad_control_3d(xFeatures), dtype=float)

    transformer = StandardScaler()
    Norm = transformer.fit_transform(xFeatures)
    xFeatures[:, 1] = Norm[:, 1]
    xFeatures[:, 2] = Norm[:, 2]
    xFeatures = xFeatures.transpose()
    return [xFeatures, X_add_feature]


def IonExtract_Att(
    ms1_peaks,
    ms2_peaks,
    Xtheory,
    X_add_feature,
    isolation_window_mz=10.0,
):
    ms1_filtered = _filter_ms1_window(
        ms1_peaks,
        Xtheory.get("precursor", []),
        isolation_window_mz,
    )
    exp_all = ms1_filtered + ms2_peaks
    theory_all = Xtheory.get("all", [])

    if len(exp_all) == 0:
        exp_all = [[0.0, 0.0]]
    if len(theory_all) == 0:
        theory_all = [[0.0, 0.0]]

    Xexp = np.asarray(exp_all, dtype=float)
    Xtheory = np.asarray(theory_all, dtype=float)

    transformer = StandardScaler()
    Norm = transformer.fit_transform(Xexp)
    Xexp[:, 1] = Norm[:, 1]
    Norm = transformer.fit_transform(Xtheory)
    Xtheory[:, 1] = Norm[:, 1]
    return [Xexp, Xtheory]


def _set_worker_state(
    exp_ms1,
    exp_ms2,
    theory,
    feature,
    scan_map,
    mode,
    fragment_top_n,
    ms1_window_mz,
):
    global _WORKER_EXP_MS1
    global _WORKER_EXP_MS2
    global _WORKER_THEORY
    global _WORKER_FEATURE
    global _WORKER_SCAN_MAP
    global _WORKER_MODE
    global _WORKER_FRAGMENT_TOP_N
    global _WORKER_MS1_WINDOW_MZ

    _WORKER_EXP_MS1 = exp_ms1
    _WORKER_EXP_MS2 = exp_ms2
    _WORKER_THEORY = theory
    _WORKER_FEATURE = feature
    _WORKER_SCAN_MAP = scan_map
    _WORKER_MODE = mode
    _WORKER_FRAGMENT_TOP_N = fragment_top_n
    _WORKER_MS1_WINDOW_MZ = ms1_window_mz


def _extract_feature_for_key(key):
    scan_info = _WORKER_SCAN_MAP.get(key)
    if scan_info is None:
        return key, None

    ms2_scan = scan_info["ms2_scan"]
    ms1_scan = scan_info["ms1_scan"]
    ms2_peaks = _WORKER_EXP_MS2.get(ms2_scan, {}).get("peaks", [])
    ms1_peaks = _WORKER_EXP_MS1.get(ms1_scan, {}).get("peaks", [])
    theory = _WORKER_THEORY.get(key)
    features = _WORKER_FEATURE.get(key)

    if theory is None or features is None:
        return key, None

    if _WORKER_MODE == "cnn":
        model_input = IonExtract(
            ms1_peaks,
            ms2_peaks,
            theory,
            features,
            _WORKER_FRAGMENT_TOP_N,
        )
    else:
        model_input = IonExtract_Att(
            ms1_peaks,
            ms2_peaks,
            theory,
            features,
            _WORKER_MS1_WINDOW_MZ,
        )
    return key, model_input


def _iter_feature_keys(theory_dict, feature_dict_map, scan_map):
    for key in theory_dict:
        if key in feature_dict_map and key in scan_map:
            yield key


def print_usage():
    print("\n\nUsage:\n")
    print("-i\t tab delimited PSM file\n")
    print("-1\t FT1 file\n")
    print("-2\t FT2 file\n")
    print("-o\t Spectrum features output file\n")
    print("-t\t Number of threads\n")
    print("-f\t Attention mode or CNN mode\n")
    print("-c\t Sipros config file for theoretical spectra generator")
    print("-b\t SIP abundance percentage passed to sipros_theoretical_spectra")
    print("-w\t MS1 isolation window size (m/z, default 10)")
    print("-d\t m/z match tolerance (diffDa, default 0.01 for HRMS)")
    print("-n\t Top N peaks per fragment envelope for CNN match (default 3)")
    print("--max-peaks\t Number of top-intensity CNN matched peaks kept (default " + str(DEFAULT_MAX_PEAKS) + ")")
    print("Required input columns: PSMId or SpecId, and Peptide.")
    print("MS2IsotopicAbundances is required unless -b/--sip-abundance is provided.")


def main(argv=None):
    global diffDa
    global pairmaxlength
    if argv is None:
        argv = sys.argv[1:]
    argv = normalize_long_flag_aliases(
        argv,
        {
            "-max-peaks": "--max-peaks",
            "-sip-abundance": "--sip-abundance",
        },
    )
    argv = _normalize_sip_abundance_args(argv)
    try:
        opts, _ = getopt.getopt(argv, "hi:1:2:o:t:f:c:w:d:n:", ["max-peaks=", "sip-abundance="])
    except Exception:
        print("Error Option, using -h for help information.")
        return 1
    if len(opts) == 0:
        print_usage()
        return 1
    start_time = time.time()
    ft1_file = ""
    ft2_file = ""
    tsv_file = ""
    theoretical_file = ""
    output_file = ""
    mode = ""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_file = os.path.join(script_dir, "SIP.cfg")
    config_file = default_config_file if os.path.exists(default_config_file) else "SiprosConfig.cfg"
    num_cpus = str(os.cpu_count() or 1)
    ms1_isolation_window_mz = 10.0
    fragment_env_top_n = 3
    sip_abundance = ""
    pairmaxlength = DEFAULT_MAX_PEAKS
    for opt, arg in opts:
        if opt in ("-h"):
            print_usage()
            return 1
        elif opt in ("-i"):
            tsv_file = arg
        elif opt in ("-1"):
            ft1_file = arg
        elif opt in ("-2"):
            ft2_file = arg
        elif opt in ("-o"):
            output_file = arg
        elif opt in ("-t"):
            num_cpus = arg
        elif opt in ("-f"):
            mode = arg
        elif opt in ("-c"):
            config_file = arg
        elif opt in ("-b", "--sip-abundance"):
            sip_abundance = str(arg).strip()
        elif opt in ("-w"):
            ms1_isolation_window_mz = max(0.0, _to_float(arg, 10.0))
        elif opt in ("-d"):
            diffDa = max(0.0, _to_float(arg, 0.01))
        elif opt in ("-n"):
            fragment_env_top_n = max(1, int(_to_float(arg, 3)))
        elif opt == "--max-peaks":
            pairmaxlength = max(1, int(_to_float(arg, DEFAULT_MAX_PEAKS)))

    if len(ft2_file) == 0 or len(ft1_file) == 0:
        raise ValueError("Both FT1 and FT2 files are required. Use -1 and -2.")

    if len(mode) == 0:
        mode = "att"

    if mode == "cnn":
        print("Max peaks kept for CNN features: " + str(pairmaxlength))

    if len(tsv_file) == 0:
        raise ValueError("A tab delimited PSM file is required. Use -i.")
    if len(output_file) == 0:
        raise ValueError("An output pickle path is required. Use -o.")

    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)
    output_base_name = os.path.splitext(os.path.basename(output_file))[0]
    theoretical_base_name = output_base_name or os.path.splitext(os.path.basename(tsv_file))[0]
    theoretical_file = os.path.join(output_dir, theoretical_base_name + ".theory.txt")

    theoretical_bin = os.path.join(script_dir, "sipros_theoretical_spectra")
    if not os.path.isfile(theoretical_bin):
        raise FileNotFoundError(
            f"Cannot find sipros_theoretical_spectra binary at {theoretical_bin}"
        )

    D_exp_ms1 = FTtoDict(ft1_file, reduce_peak_charge_to_one=False)
    D_exp_ms2 = FTtoDict(ft2_file, reduce_peak_charge_to_one=True)
    print('Experimental spectra loaded (FT1 + FT2)!')

    psm_dict = dict()
    psm_scan_map, table_meta, row_records, row_stats = read_tsv(tsv_file, psm_dict, D_exp_ms2)
    print(
        "Input rows: "
        f"total={row_stats['total_rows']} "
        f"with_psm_id={row_stats['rows_with_psm_id']} "
        f"target_label=1:{row_stats['target_label_rows']} "
        f"decoy_label=-1:{row_stats['decoy_label_rows']} "
        f"unlabeled={row_stats['unlabeled_rows']} "
        f"usable_peptide={row_stats['usable_peptide_rows']} "
        f"skipped_missing_peptide={row_stats['skipped_missing_peptide']}"
    )

    cmd = [
        theoretical_bin,
        "-c",
        config_file,
        "-i",
        tsv_file,
        "-o",
        theoretical_file,
        "-t",
        str(num_cpus),
    ]
    if sip_abundance:
        cmd.extend(["-b", sip_abundance])
    subprocess.run(cmd, check=True)

    D_theory = theoryToDict(theoretical_file)
    print('Theoretical features loaded!')
    for psm in psm_dict:
        psm_dict[psm].get_features()
    D_feature = feature_dict(psm_dict)
    print('Additional features loaded!')

    worker_count = max(1, int(_to_float(num_cpus, 1)))
    feature_payload = {}
    feature_keys = _iter_feature_keys(D_theory, D_feature, psm_scan_map)
    _set_worker_state(
        D_exp_ms1,
        D_exp_ms2,
        D_theory,
        D_feature,
        psm_scan_map,
        mode,
        fragment_env_top_n,
        ms1_isolation_window_mz,
    )

    if worker_count == 1:
        for key in feature_keys:
            feature_key, model_input = _extract_feature_for_key(key)
            if model_input is not None:
                feature_payload[feature_key] = model_input
    else:
        start_method = "fork" if "fork" in get_all_start_methods() else None
        ctx = get_context(start_method) if start_method else get_context()
        pool_kwargs = {"processes": worker_count}
        if ctx.get_start_method() != "fork":
            pool_kwargs["initializer"] = _set_worker_state
            pool_kwargs["initargs"] = (
                D_exp_ms1,
                D_exp_ms2,
                D_theory,
                D_feature,
                psm_scan_map,
                mode,
                fragment_env_top_n,
                ms1_isolation_window_mz,
            )
        with ctx.Pool(**pool_kwargs) as pool:
            for feature_key, model_input in pool.imap_unordered(
                _extract_feature_for_key,
                feature_keys,
                chunksize=8,
            ):
                if model_input is not None:
                    feature_payload[feature_key] = model_input

    print('Features generated!')
    if os.path.exists(theoretical_file):
        os.remove(theoretical_file)

    rows_with_model_input = 0
    rows_without_model_input = 0
    final_payload = {
        PKL_META_KEY: {
            **table_meta,
            "mode": mode,
        }
    }
    for row_record in row_records:
        psm_id = row_record["psm_id"]
        model_input = feature_payload.get(psm_id) if psm_id else None
        if model_input is None:
            rows_without_model_input += 1
        else:
            rows_with_model_input += 1
        final_payload[row_record["record_key"]] = {
            "psm_id": psm_id,
            "model_input": model_input,
            "label": row_record["label"],
            "label_raw": row_record["label_raw"],
            "label_confidence": row_record["label_confidence"],
            "row_index": row_record["row_index"],
            "row_values": row_record["row_values"],
        }
    print(
        "PKL rows: "
        f"total={len(row_records)} "
        f"with_model_input={rows_with_model_input} "
        f"without_model_input={rows_without_model_input} "
        f"feature_keys={len(feature_payload)} "
        f"mode={mode}"
    )
    with open(output_file, 'wb') as f:
        pickle.dump(final_payload, f)
    print('time:' + str(time.time() - start_time))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
