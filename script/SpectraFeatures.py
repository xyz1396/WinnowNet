import csv
import getopt
import os
import pickle
import subprocess
import sys
import time
from multiprocessing import get_all_start_methods, get_context

import numpy as np
from pkl_utils import PKL_META_KEY, PKL_SCHEMA_VERSION, normalize_long_flag_aliases

DEFAULT_MAX_PEAKS = 128
CNN_INPUT_CHANNELS = 10
CNN_FEATURE_SCHEMA = "cnn_10ch_v2"
DEFAULT_MS1_PRECURSOR_TOP_N = 5
DEFAULT_MS2_FRAGMENT_TOP_N = 2
NEUTRON_MASS = 1.0033548378
pairmaxlength = DEFAULT_MAX_PEAKS
diffPPM = 10.0

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
_WORKER_MS1_WINDOW_MZ = 10.0
_WORKER_SIP_CONFIG = None
_WORKER_MS1_PRECURSOR_TOP_N = DEFAULT_MS1_PRECURSOR_TOP_N
_WORKER_MS2_FRAGMENT_TOP_N = DEFAULT_MS2_FRAGMENT_TOP_N

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


def _strip_config_comment(line):
    return line.split("#", 1)[0].strip()


def _parse_config_values(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_sip_config(config_file):
    element_list = []
    sip_element = ""
    residue_composition = {}

    with open(config_file) as fh:
        for raw_line in fh:
            line = _strip_config_comment(raw_line)
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            values = _parse_config_values(value)
            if key == "Element_List":
                element_list = values
            elif key == "SIP_Element":
                sip_element = values[0] if values else value.strip()
            elif key.startswith("Residue{") and "}" in key:
                residue = key.split("{", 1)[1].split("}", 1)[0]
                residue_composition[residue] = [_to_float(item, 0.0) for item in values]

    if not element_list:
        raise ValueError(f"{config_file} is missing Element_List.")
    if not sip_element:
        raise ValueError(f"{config_file} is missing SIP_Element.")
    if sip_element not in element_list:
        raise ValueError(
            f"SIP_Element {sip_element!r} is not present in Element_List from {config_file}."
        )

    sip_element_index = element_list.index(sip_element)
    return {
        "element_list": element_list,
        "sip_element": sip_element,
        "sip_element_index": sip_element_index,
        "residue_composition": residue_composition,
    }


def _residue_sip_atom_count(sip_config, residue, psm_id):
    residue_composition = sip_config.get("residue_composition", {})
    if residue not in residue_composition:
        raise ValueError(
            f"{psm_id}: residue/PTM symbol {residue!r} is missing from SIP config."
        )
    composition = residue_composition[residue]
    sip_element_index = int(sip_config.get("sip_element_index", -1))
    if sip_element_index < 0 or sip_element_index >= len(composition):
        return 0.0
    return float(composition[sip_element_index])


def _sequence_sip_atom_count(sip_config, residues, psm_id):
    return sum(_residue_sip_atom_count(sip_config, residue, psm_id) for residue in residues)


def _residue_mass(residue, psm_id):
    if residue not in AA_dict:
        raise ValueError(
            f"{psm_id}: residue/PTM symbol {residue!r} is missing from AA mass table."
        )
    return float(AA_dict[residue])


def _build_cnn_ion_metadata(sip_config, peptide_sequence, precursor_charge, precursor_group, psm_id):
    if sip_config is None:
        return {}

    peptide_sequence = str(peptide_sequence or "")
    residues = list(peptide_sequence)
    residue_masses = [_residue_mass(residue, psm_id) for residue in residues]
    residue_sip_counts = [
        _residue_sip_atom_count(sip_config, residue, psm_id)
        for residue in residues
    ]

    prefix_masses = [0.0]
    prefix_sip_counts = [_residue_sip_atom_count(sip_config, "Nterm", psm_id)]
    for residue_mass, residue_sip_count in zip(residue_masses, residue_sip_counts):
        prefix_masses.append(prefix_masses[-1] + residue_mass)
        prefix_sip_counts.append(prefix_sip_counts[-1] + residue_sip_count)

    full_residue_mass = prefix_masses[-1]
    precursor_neutral_mass = full_residue_mass + N_TERMINUS + C_TERMINUS
    precursor_sip_atom_count = (
        prefix_sip_counts[-1] + _residue_sip_atom_count(sip_config, "Cterm", psm_id)
    )
    precursor_charge = max(1, int(_to_float(precursor_charge, 1)))

    metadata = {
        precursor_group: {
            "mono_mz": (precursor_neutral_mass + precursor_charge * PROTON_MASS)
            / precursor_charge,
            "sip_atom_number": precursor_sip_atom_count,
        }
    }

    peptide_length = len(residues)
    for position in range(1, peptide_length + 1):
        metadata[("b", position)] = {
            "mono_mz": prefix_masses[position] + H,
            "sip_atom_number": prefix_sip_counts[position],
        }

        complement_prefix_position = peptide_length - position
        metadata[("y", position)] = {
            "mono_mz": precursor_neutral_mass + PROTON_MASS - prefix_masses[complement_prefix_position],
            "sip_atom_number": precursor_sip_atom_count
            - prefix_sip_counts[complement_prefix_position],
        }

    return metadata


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
    resolved_file_path = os.path.abspath(file_path)
    scan_id = None
    charge = 0
    parent_scan = ""
    isolation_center_mz = 0.0
    precursor_candidates = []
    peaks = []
    peak_metadata = []

    def _flush():
        if scan_id is None:
            return
        sorted_peaks = sorted(peaks, key=lambda x: x[0])
        sorted_peak_metadata = [
            metadata
            for _, metadata in sorted(
                zip(peaks, peak_metadata),
                key=lambda item: item[0][0],
            )
        ]
        msdict[scan_id] = {
            "isolation_center_mz__charge": f"{isolation_center_mz}_{charge}",
            "peaks": sorted_peaks,
            "peak_metadata": sorted_peak_metadata,
            "parent_scan": parent_scan,
            "isolation_center_mz": isolation_center_mz,
            "precursor_candidates": precursor_candidates.copy(),
            "source_file": resolved_file_path,
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
                    peak_metadata = []
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
                raw_mz = mz
                intensity = _to_float(vals[1])
                peak_charge = int(_to_float(vals[5], 0)) if len(vals) >= 6 else 0
                if reduce_peak_charge_to_one and peak_charge > 1:
                    mz = peak_charge * (mz - PROTON_MASS) + PROTON_MASS
                peaks.append([mz, intensity])
                peak_metadata.append(
                    {
                        "raw_mz": raw_mz,
                        "raw_charge": peak_charge,
                        "stored_mz": mz,
                    }
                )
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

    def _build_fragment_records(mz_vals, intensity_vals, kinds, positions):
        n = min(len(mz_vals), len(intensity_vals))
        records = []
        for idx in range(n):
            kind = kinds[idx] if idx < len(kinds) else ""
            position = positions[idx] if idx < len(positions) else 0
            records.append(
                {
                    "peak": [mz_vals[idx], intensity_vals[idx]],
                    "kind": kind,
                    "position": position,
                }
            )
        return sorted(records, key=lambda item: item["peak"][0])

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
        fragment_records = _build_fragment_records(
            fragment_mz_vals,
            fragment_int_vals,
            fragment_kinds,
            fragment_positions,
        )
        fragment_scan = [record["peak"] for record in fragment_records]
        fragment_kinds = [record["kind"] for record in fragment_records]
        fragment_positions = [record["position"] for record in fragment_records]
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
            exp_mass_raw = _get_field_value(
                row,
                field_lookup,
                ["ExpMass", "ExperimentalMass", "expmass"],
                "0",
            )
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
            exp_mass_raw = "0"
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
        exp_mass = _to_float(exp_mass_raw, 0.0)

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
            "peptide_sequence": peptidestr,
            "exp_mass": exp_mass,
            "precursor_mass_charge1": exp_mass + PROTON_MASS if exp_mass > 0 else 0.0,
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


def pad_control_cnn(data):
    data = sorted(data, key=lambda x: x["raw_exp_intensity"], reverse=True)[:pairmaxlength]
    data = sorted(data, key=lambda x: x["features"][0])
    padded = [item["features"] for item in data]
    while len(padded) < pairmaxlength:
        padded.append([0.0] * CNN_INPUT_CHANNELS)
    return padded

def _to_peak_array(peaks):
    if peaks is None:
        return np.empty((0, 2), dtype=float)
    peak_array = np.asarray(peaks, dtype=float)
    if peak_array.size == 0:
        return np.empty((0, 2), dtype=float)
    return peak_array


def _max_group_intensity(peaks):
    peak_array = _to_peak_array(peaks)
    if peak_array.shape[0] == 0:
        return 0.0
    intensities = np.maximum(peak_array[:, 1], 0.0)
    if intensities.shape[0] == 0:
        return 0.0
    return float(np.max(intensities))


def _normalize_peak_group(peaks):
    peak_array = _to_peak_array(peaks)
    if peak_array.shape[0] == 0:
        return []
    max_intensity = _max_group_intensity(peak_array)
    normalized = peak_array.copy()
    if max_intensity > 0:
        normalized[:, 1] = np.maximum(normalized[:, 1], 0.0) / max_intensity
    else:
        normalized[:, 1] = 0.0
    return normalized.tolist()


def _fragment_group_key(kind, position, fallback_idx):
    kind_text = str(kind or "").strip().lower()
    position_value = int(_to_float(position, 0))
    if kind_text and position_value > 0:
        return (kind_text, position_value)
    return ("fragment", int(fallback_idx))


def _fragment_peak_entries(peaks, fragment_kinds, fragment_positions):
    peak_array = _to_peak_array(peaks)
    if peak_array.shape[0] == 0:
        return []

    entries = []
    for idx, peak in enumerate(peak_array.tolist()):
        kind = fragment_kinds[idx] if idx < len(fragment_kinds) else ""
        position = fragment_positions[idx] if idx < len(fragment_positions) else 0
        group_key = _fragment_group_key(kind, position, idx)
        entries.append((peak, group_key))
    return entries


def _normalize_grouped_peak_entries(entries):
    grouped = {}
    for peak, group_key in entries:
        grouped.setdefault(group_key, []).append(peak)

    normalized_entries = []
    for group_key, peaks in grouped.items():
        for peak in _normalize_peak_group(peaks):
            normalized_entries.append((peak, group_key))
    normalized_entries.sort(key=lambda item: item[0][0])
    return normalized_entries


def _group_maxima_from_peak_entries(entries):
    maxima = {}
    for peak, group_key in entries:
        intensity = max(0.0, float(peak[1]))
        maxima[group_key] = max(maxima.get(group_key, 0.0), intensity)
    return maxima


def _build_group_info(entries):
    group_info = {}
    for peak, group_key in entries:
        mz = float(peak[0])
        intensity = max(0.0, float(peak[1]))
        info = group_info.setdefault(
            group_key,
            {
                "mono_mz": mz,
                "theory_max": 0.0,
            },
        )
        info["mono_mz"] = min(info["mono_mz"], mz)
        info["theory_max"] = max(info["theory_max"], intensity)
    return group_info


def _fragment_entries_in_exp_range(exp_peaks, Xtheory):
    exp_array = _to_peak_array(exp_peaks)
    fragment_entries = _fragment_peak_entries(
        Xtheory.get("fragment", []),
        Xtheory.get("fragment_kinds", []),
        Xtheory.get("fragment_positions", []),
    )
    if exp_array.shape[0] == 0 or len(fragment_entries) == 0:
        return []
    low = float(np.min(exp_array[:, 0]))
    high = float(np.max(exp_array[:, 0]))
    return [
        (peak, group_key)
        for peak, group_key in fragment_entries
        if low <= float(peak[0]) <= high
    ]


def _append_cnn_matches(exp_array, theory_entries, matches, source_group, top_n_to_try=1):
    if exp_array.shape[0] == 0 or len(theory_entries) == 0:
        return
    top_n_to_try = max(1, int(_to_float(top_n_to_try, 1)))
    grouped_entries = {}
    for peak, theory_group in theory_entries:
        grouped_entries.setdefault(theory_group, []).append(peak)

    for theory_group, peaks in grouped_entries.items():
        peaks_by_intensity = sorted(peaks, key=lambda x: (-float(x[1]), float(x[0])))
        found_match = False
        for peak_idx, mz in enumerate(peaks_by_intensity):
            theory_mz = float(mz[0])
            tolerance_mz = abs(theory_mz) * diffPPM * 1e-6
            index = np.where(
                np.logical_and(
                    exp_array[:, 0] > theory_mz - tolerance_mz,
                    exp_array[:, 0] < theory_mz + tolerance_mz,
                )
            )[0]
            if len(index) == 0:
                if found_match or peak_idx + 1 >= top_n_to_try:
                    break
                continue
            found_match = True
            for idx in index:
                exp_mz = float(exp_array[idx][0])
                matches.append(
                    {
                        "expmz": exp_mz,
                        "delta_mz": exp_mz - theory_mz,
                        "raw_exp_intensity": max(0.0, float(exp_array[idx][1])),
                        "theory_intensity": max(0.0, float(mz[1])),
                        "source_group": source_group,
                        "theory_group": theory_group,
                    }
                )


def _matched_source_maxima(matches):
    maxima = {}
    for match in matches:
        source_group = match["source_group"]
        maxima[source_group] = max(
            maxima.get(source_group, 0.0),
            float(match["raw_exp_intensity"]),
        )
    return maxima


def _envelope_enrich_ratios(matches, ion_metadata, sip_atom_counts, ion_charges):
    weighted_sums = {}
    weight_sums = {}
    for match in matches:
        group_key = match["theory_group"]
        sip_atom_number = float(sip_atom_counts.get(group_key, 0.0))
        if sip_atom_number <= 0:
            continue
        mono_mz = float(ion_metadata.get(group_key, {}).get("mono_mz", 0.0))
        ion_charge = float(ion_charges.get(group_key, 1))
        expected_neutron = (float(match["expmz"]) - mono_mz) * ion_charge / NEUTRON_MASS
        weight = max(0.0, float(match["raw_exp_intensity"]))
        if weight <= 0:
            continue
        weighted_sums[group_key] = weighted_sums.get(group_key, 0.0) + expected_neutron * weight
        weight_sums[group_key] = weight_sums.get(group_key, 0.0) + weight

    enrich_ratios = {}
    for group_key, weighted_sum in weighted_sums.items():
        weight_sum = weight_sums.get(group_key, 0.0)
        sip_atom_number = float(sip_atom_counts.get(group_key, 0.0))
        if weight_sum > 0 and sip_atom_number > 0:
            enrich_ratios[group_key] = (weighted_sum / weight_sum) / sip_atom_number
        else:
            enrich_ratios[group_key] = 0.0
    return enrich_ratios


def _normalize_cnn_matches(matches, source_maxima, group_info, ion_metadata, ion_charges):
    sip_atom_counts = {
        key: value.get("sip_atom_number", 0.0)
        for key, value in ion_metadata.items()
    }
    enrich_ratios = _envelope_enrich_ratios(
        matches,
        ion_metadata,
        sip_atom_counts,
        ion_charges,
    )
    normalized = []
    for match in matches:
        group_key = match["theory_group"]
        group_metadata = group_info.get(group_key, {})
        ion_info = ion_metadata.get(group_key)
        if ion_info is None:
            raise ValueError(
                f"Cannot compute CNN ion metadata for theory group {group_key!r}."
            )
        exp_max = float(source_maxima.get(match["source_group"], 0.0))
        theory_max = float(group_metadata.get("theory_max", 0.0))
        exp_value = max(0.0, float(match["raw_exp_intensity"]))
        theory_value = max(0.0, float(match["theory_intensity"]))
        if exp_max > 0:
            exp_value = exp_value / exp_max
        else:
            exp_value = 0.0
        if theory_max > 0:
            theory_value = theory_value / theory_max
        else:
            theory_value = 0.0
        ion_charge = float(ion_charges.get(group_key, 1.0))
        mono_mz = float(ion_info.get("mono_mz", 0.0))
        isotope_index = 0.0
        if ion_charge > 0:
            isotope_index = float(
                round((float(match["expmz"]) - mono_mz) * ion_charge / NEUTRON_MASS)
            )
        ms_level_flag = 1.0 if match["source_group"] == "precursor" else 0.0
        normalized.append(
            {
                "raw_exp_intensity": max(0.0, float(match["raw_exp_intensity"])),
                "features": [
                    float(match["expmz"]),
                    float(match["delta_mz"]),
                    mono_mz,
                    exp_value,
                    theory_value,
                    float(ion_info.get("sip_atom_number", 0.0)),
                    float(enrich_ratios.get(group_key, 0.0)),
                    ion_charge,
                    isotope_index,
                    ms_level_flag,
                ],
            }
        )
    return normalized


def _select_fragment_peaks_for_cnn(Xtheory, top_n=3):
    entries = _fragment_peak_entries(
        Xtheory.get("fragment", []),
        Xtheory.get("fragment_kinds", []),
        Xtheory.get("fragment_positions", []),
    )
    if len(entries) == 0:
        return []
    if top_n <= 0:
        return sorted(entries, key=lambda item: item[0][0])

    grouped = {}
    for peak, group_key in entries:
        grouped.setdefault(group_key, []).append(peak)

    grouped_selected = []
    for group_key, peaks in grouped.items():
        for peak in sorted(peaks, key=lambda x: x[1], reverse=True)[:top_n]:
            grouped_selected.append((peak, group_key))
    return sorted(grouped_selected, key=lambda item: item[0][0])


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


def _filter_ms2_peaks_by_precursor_mass(ms2_record, psm_id, scan_id, exp_mass, precursor_mass_charge1):
    peaks = ms2_record.get("peaks", [])
    if precursor_mass_charge1 <= 0 or len(peaks) == 0:
        return peaks, 0

    filtered_peaks = []
    removed_count = 0
    for peak in peaks:
        stored_mz = float(peak[0]) if len(peak) > 0 else 0.0
        if stored_mz > precursor_mass_charge1:
            removed_count += 1
            continue
        filtered_peaks.append(peak)
    return filtered_peaks, removed_count


def _trim_theory_to_exp_mz_range(exp_peaks, theory_peaks):
    exp_array = _to_peak_array(exp_peaks)
    theory_array = _to_peak_array(theory_peaks)
    if exp_array.shape[0] == 0 or theory_array.shape[0] == 0:
        return []
    low = float(np.min(exp_array[:, 0]))
    high = float(np.max(exp_array[:, 0]))
    return theory_array[(theory_array[:, 0] >= low) & (theory_array[:, 0] <= high)].tolist()


def IonExtract(
    ms1_peaks,
    ms2_peaks,
    Xtheory,
    scan_info,
    sip_config,
    psm_id,
    ms1_precursor_top_n=DEFAULT_MS1_PRECURSOR_TOP_N,
    ms2_fragment_top_n=DEFAULT_MS2_FRAGMENT_TOP_N,
):
    ms1_array = _to_peak_array(ms1_peaks)
    ms2_array = _to_peak_array(ms2_peaks)
    precursor_peaks = Xtheory.get("precursor", [])
    theory_precursor = _to_peak_array(precursor_peaks)
    precursor_group = ("precursor", 0)
    precursor_entries = [
        (peak, precursor_group)
        for peak in theory_precursor.tolist()
    ]
    fragment_entries = _fragment_entries_in_exp_range(ms2_array, Xtheory)
    all_theory_entries = precursor_entries + fragment_entries
    group_info = _build_group_info(all_theory_entries)

    matches = []
    _append_cnn_matches(
        ms1_array,
        precursor_entries,
        matches,
        source_group="precursor",
        top_n_to_try=ms1_precursor_top_n,
    )
    _append_cnn_matches(
        ms2_array,
        fragment_entries,
        matches,
        source_group="fragment",
        top_n_to_try=ms2_fragment_top_n,
    )

    if len(matches) == 0:
        xFeatures = np.asarray(pad_control_cnn([]), dtype=float)
        return [xFeatures.transpose()]

    peptide_sequence = scan_info.get("peptide_sequence", "")
    precursor_charge = max(1, int(_to_float(scan_info.get("charge", 1), 1)))
    ion_metadata_all = _build_cnn_ion_metadata(
        sip_config,
        peptide_sequence,
        precursor_charge,
        precursor_group,
        psm_id,
    )
    ion_metadata = {}
    ion_charges = {}
    for group_key in {match["theory_group"] for match in matches}:
        if group_key not in ion_metadata_all:
            raise ValueError(
                f"{psm_id}: cannot compute CNN ion metadata for theory group {group_key!r}."
            )
        ion_metadata[group_key] = ion_metadata_all[group_key]
        ion_charges[group_key] = precursor_charge if group_key == precursor_group else 1

    source_maxima = _matched_source_maxima(matches)
    xFeatures = _normalize_cnn_matches(
        matches,
        source_maxima,
        group_info,
        ion_metadata,
        ion_charges,
    )
    xFeatures = np.asarray(pad_control_cnn(xFeatures), dtype=float)
    xFeatures = xFeatures.transpose()
    return [xFeatures]


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
    exp_precursor = _normalize_peak_group(ms1_filtered)
    exp_fragment = _normalize_peak_group(ms2_peaks)
    theory_precursor = _normalize_peak_group(Xtheory.get("precursor", []))
    theory_fragment_entries = _fragment_peak_entries(
        _trim_theory_to_exp_mz_range(ms2_peaks, Xtheory.get("fragment", [])),
        Xtheory.get("fragment_kinds", []),
        Xtheory.get("fragment_positions", []),
    )
    theory_fragment = [
        peak for peak, _ in _normalize_grouped_peak_entries(theory_fragment_entries)
    ]
    exp_all = exp_precursor + exp_fragment
    theory_all = sorted(theory_precursor + theory_fragment)

    if len(exp_all) == 0:
        exp_all = [[0.0, 0.0]]
    if len(theory_all) == 0:
        theory_all = [[0.0, 0.0]]

    Xexp = np.asarray(exp_all, dtype=float)
    Xtheory = np.asarray(theory_all, dtype=float)
    return [Xexp, Xtheory]


def _set_worker_state(
    exp_ms1,
    exp_ms2,
    theory,
    feature,
    scan_map,
    mode,
    ms1_window_mz,
    sip_config,
    ms1_precursor_top_n,
    ms2_fragment_top_n,
):
    global _WORKER_EXP_MS1
    global _WORKER_EXP_MS2
    global _WORKER_THEORY
    global _WORKER_FEATURE
    global _WORKER_SCAN_MAP
    global _WORKER_MODE
    global _WORKER_MS1_WINDOW_MZ
    global _WORKER_SIP_CONFIG
    global _WORKER_MS1_PRECURSOR_TOP_N
    global _WORKER_MS2_FRAGMENT_TOP_N

    _WORKER_EXP_MS1 = exp_ms1
    _WORKER_EXP_MS2 = exp_ms2
    _WORKER_THEORY = theory
    _WORKER_FEATURE = feature
    _WORKER_SCAN_MAP = scan_map
    _WORKER_MODE = mode
    _WORKER_MS1_WINDOW_MZ = ms1_window_mz
    _WORKER_SIP_CONFIG = sip_config
    _WORKER_MS1_PRECURSOR_TOP_N = max(
        1, int(_to_float(ms1_precursor_top_n, DEFAULT_MS1_PRECURSOR_TOP_N))
    )
    _WORKER_MS2_FRAGMENT_TOP_N = max(
        1, int(_to_float(ms2_fragment_top_n, DEFAULT_MS2_FRAGMENT_TOP_N))
    )


def _extract_feature_for_key(key):
    scan_info = _WORKER_SCAN_MAP.get(key)
    if scan_info is None:
        return key, None, 0

    ms2_scan = scan_info["ms2_scan"]
    ms1_scan = scan_info["ms1_scan"]
    ms2_record = _WORKER_EXP_MS2.get(ms2_scan, {})
    ms2_peaks, removed_peak_count = _filter_ms2_peaks_by_precursor_mass(
        ms2_record,
        key,
        ms2_scan,
        scan_info.get("exp_mass", 0.0),
        scan_info.get("precursor_mass_charge1", 0.0),
    )
    ms1_peaks = _WORKER_EXP_MS1.get(ms1_scan, {}).get("peaks", [])
    theory = _WORKER_THEORY.get(key)
    features = _WORKER_FEATURE.get(key)

    if theory is None or features is None:
        return key, None, removed_peak_count

    if _WORKER_MODE == "cnn":
        model_input = IonExtract(
            ms1_peaks,
            ms2_peaks,
            theory,
            scan_info,
            _WORKER_SIP_CONFIG,
            key,
            _WORKER_MS1_PRECURSOR_TOP_N,
            _WORKER_MS2_FRAGMENT_TOP_N,
        )
    else:
        model_input = IonExtract_Att(
            ms1_peaks,
            ms2_peaks,
            theory,
            features,
            _WORKER_MS1_WINDOW_MZ,
        )
    return key, model_input, removed_peak_count


def _iter_feature_keys(theory_dict, feature_dict_map, scan_map):
    for key in theory_dict:
        if key in feature_dict_map and key in scan_map:
            yield key


def _split_path_arg(value):
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _default_base_from_tsv(tsv_file):
    directory = os.path.dirname(os.path.abspath(tsv_file))
    name = os.path.basename(tsv_file)
    suffix = "_filtered_psms.tsv"
    if name.endswith(suffix):
        stem = name[: -len(suffix)]
    else:
        stem = os.path.splitext(name)[0]
    return os.path.join(directory, stem)


def _collect_tsvs_from_directory(directory):
    matches = []
    for root, _, files in os.walk(directory):
        for file_name in sorted(files):
            if file_name.endswith("_filtered_psms.tsv"):
                matches.append(os.path.join(root, file_name))
    return matches


def _resolve_batch_tasks(input_value):
    items = _split_path_arg(input_value)
    if not items:
        return []

    any_directory = any(os.path.isdir(item) for item in items)
    batch_mode = any_directory or len(items) > 1
    if not batch_mode:
        return []

    tasks = []
    seen = set()
    for item in items:
        if os.path.isdir(item):
            tsv_files = _collect_tsvs_from_directory(item)
            if not tsv_files:
                raise ValueError(f"No *_filtered_psms.tsv files found under {item}.")
        else:
            tsv_files = [item]
        for tsv_file in tsv_files:
            tsv_path = os.path.abspath(tsv_file)
            if tsv_path in seen:
                continue
            seen.add(tsv_path)
            base_path = _default_base_from_tsv(tsv_path)
            ft1_path = base_path + ".FT1"
            ft2_path = base_path + ".FT2"
            if not os.path.isfile(ft1_path):
                raise FileNotFoundError(f"Missing FT1 file for {tsv_path}: {ft1_path}")
            if not os.path.isfile(ft2_path):
                raise FileNotFoundError(f"Missing FT2 file for {tsv_path}: {ft2_path}")
            tasks.append(
                {
                    "tsv": tsv_path,
                    "ft1": ft1_path,
                    "ft2": ft2_path,
                    "output": base_path + ".pkl",
                }
            )
    return tasks


def print_usage():
    print("\n\nUsage:\n")
    print("-i\t tab delimited PSM file, a directory, or a comma-separated list of files/directories\n")
    print("-1\t FT1 file (single-file mode only)\n")
    print("-2\t FT2 file (single-file mode only)\n")
    print("-o\t Spectrum features output file (single-file mode only)\n")
    print("-t\t Number of threads per conversion\n")
    print("-j\t Number of file conversions to run in parallel for directory/list input (default 1)\n")
    print("-f\t Attention mode or CNN mode\n")
    print("-c\t Sipros config file for theoretical spectra generator")
    print("-b\t SIP abundance percentage passed to sipros_theoretical_spectra")
    print("-w\t MS1 isolation window size (m/z, default 10)")
    print("-d\t m/z match tolerance in ppm (default 10)")
    print("--max-peaks\t Number of top-intensity matched peaks kept and tensor width (default " + str(DEFAULT_MAX_PEAKS) + ")")
    print("--ms1-topn\t Search the top N precursor isotopic peaks for the first match, then continue until the next miss (default " + str(DEFAULT_MS1_PRECURSOR_TOP_N) + ")")
    print("--ms2-topn\t Search the top N fragment isotopic peaks for the first match, then continue until the next miss (default " + str(DEFAULT_MS2_FRAGMENT_TOP_N) + ")")
    print("Required input columns: PSMId or SpecId, and Peptide.")
    print("MS2IsotopicAbundances is required unless -b/--sip-abundance is provided.")


def _run_single_conversion(
    tsv_file,
    ft1_file,
    ft2_file,
    output_file,
    config_file,
    num_cpus,
    mode,
    sip_abundance,
    ms1_isolation_window_mz,
    ms1_precursor_top_n,
    ms2_fragment_top_n,
):
    start_time = time.time()
    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)
    output_base_name = os.path.splitext(os.path.basename(output_file))[0]
    theoretical_base_name = output_base_name or os.path.splitext(os.path.basename(tsv_file))[0]
    theoretical_file = os.path.join(output_dir, theoretical_base_name + ".theory.txt")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    theoretical_bin = os.path.join(script_dir, "sipros_theoretical_spectra")
    if not os.path.isfile(theoretical_bin):
        raise FileNotFoundError(
            f"Cannot find sipros_theoretical_spectra binary at {theoretical_bin}"
        )

    D_exp_ms1 = FTtoDict(ft1_file, reduce_peak_charge_to_one=False)
    D_exp_ms2 = FTtoDict(ft2_file, reduce_peak_charge_to_one=True)
    print("Experimental spectra loaded (FT1 + FT2)!")

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
    print("Theoretical features loaded!")
    for psm in psm_dict:
        psm_dict[psm].get_features()
    D_feature = feature_dict(psm_dict)
    print("Additional features loaded!")
    sip_config = _parse_sip_config(config_file) if mode == "cnn" else None

    worker_count = max(1, int(_to_float(num_cpus, 1)))
    feature_payload = {}
    removed_peak_total = 0
    removed_psm_total = 0
    feature_keys = _iter_feature_keys(D_theory, D_feature, psm_scan_map)
    _set_worker_state(
        D_exp_ms1,
        D_exp_ms2,
        D_theory,
        D_feature,
        psm_scan_map,
        mode,
        ms1_isolation_window_mz,
        sip_config,
        ms1_precursor_top_n,
        ms2_fragment_top_n,
    )

    if worker_count == 1:
        for key in feature_keys:
            feature_key, model_input, removed_peak_count = _extract_feature_for_key(key)
            removed_peak_total += removed_peak_count
            if removed_peak_count > 0:
                removed_psm_total += 1
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
                ms1_isolation_window_mz,
                sip_config,
                ms1_precursor_top_n,
                ms2_fragment_top_n,
            )
        with ctx.Pool(**pool_kwargs) as pool:
            for feature_key, model_input, removed_peak_count in pool.imap_unordered(
                _extract_feature_for_key,
                feature_keys,
                chunksize=8,
            ):
                removed_peak_total += removed_peak_count
                if removed_peak_count > 0:
                    removed_psm_total += 1
                if model_input is not None:
                    feature_payload[feature_key] = model_input

    print("Features generated!")
    print(
        "FT2 precursor-mass filtering: "
        f"removed_peaks={removed_peak_total} "
        f"affected_psms={removed_psm_total}"
    )
    if os.path.exists(theoretical_file):
        os.remove(theoretical_file)

    rows_with_model_input = 0
    rows_without_model_input = 0
    output_meta = {
        **table_meta,
        "mode": mode,
    }
    if mode == "cnn":
        output_meta.update(
            {
                "feature_schema": CNN_FEATURE_SCHEMA,
                "input_channels": CNN_INPUT_CHANNELS,
                "max_peaks": pairmaxlength,
                "ms1_topn": ms1_precursor_top_n,
                "ms2_topn": ms2_fragment_top_n,
            }
        )
    final_payload = {
        PKL_META_KEY: output_meta
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
    with open(output_file, "wb") as f:
        pickle.dump(final_payload, f)
    print("time:" + str(time.time() - start_time))
    return 0


def _build_child_command(
    script_path,
    task,
    config_file,
    num_cpus,
    mode,
    sip_abundance,
    ms1_isolation_window_mz,
    diff_ppm,
    max_peaks,
    ms1_precursor_top_n,
    ms2_fragment_top_n,
):
    cmd = [
        sys.executable,
        script_path,
        "-i",
        task["tsv"],
        "-1",
        task["ft1"],
        "-2",
        task["ft2"],
        "-o",
        task["output"],
        "-t",
        str(num_cpus),
        "-f",
        mode,
        "-c",
        config_file,
        "-w",
        str(ms1_isolation_window_mz),
        "-d",
        str(diff_ppm),
        "--max-peaks",
        str(max_peaks),
        "--ms1-topn",
        str(ms1_precursor_top_n),
        "--ms2-topn",
        str(ms2_fragment_top_n),
    ]
    if sip_abundance:
        cmd.extend(["-b", sip_abundance])
    return cmd


def _run_batch_conversions(
    tasks,
    batch_jobs,
    config_file,
    num_cpus,
    mode,
    sip_abundance,
    ms1_isolation_window_mz,
    diff_ppm,
    max_peaks,
    ms1_precursor_top_n,
    ms2_fragment_top_n,
):
    script_path = os.path.abspath(__file__)
    max_parallel = max(1, int(_to_float(batch_jobs, 1)))
    pending = list(tasks)
    active = []
    print(
        f"Batch mode: files={len(tasks)} parallel_jobs={max_parallel} "
        f"threads_per_job={num_cpus} mode={mode}"
    )

    while pending or active:
        while pending and len(active) < max_parallel:
            task = pending.pop(0)
            print(
                f"[batch:start] input={task['tsv']} "
                f"output={task['output']}"
            )
            proc = subprocess.Popen(
                _build_child_command(
                    script_path,
                    task,
                    config_file,
                    num_cpus,
                    mode,
                    sip_abundance,
                    ms1_isolation_window_mz,
                    diff_ppm,
                    max_peaks,
                    ms1_precursor_top_n,
                    ms2_fragment_top_n,
                )
            )
            active.append((task, proc))

        if not active:
            continue

        time.sleep(0.2)
        still_active = []
        failed = None
        for task, proc in active:
            return_code = proc.poll()
            if return_code is None:
                still_active.append((task, proc))
                continue
            if return_code != 0:
                failed = (task, return_code)
                break
            print(f"[batch:done] output={task['output']}")
        if failed is not None:
            failed_task, failed_code = failed
            for task, proc in still_active:
                proc.terminate()
            for task, proc in still_active:
                proc.wait()
            raise RuntimeError(
                f"Batch conversion failed for {failed_task['tsv']} with exit code {failed_code}."
            )
        active = still_active

    print("Batch conversion complete.")
    return 0


def main(argv=None):
    global diffPPM
    global pairmaxlength
    if argv is None:
        argv = sys.argv[1:]
    argv = normalize_long_flag_aliases(
        argv,
        {
            "-max-peaks": "--max-peaks",
            "-jobs": "--jobs",
            "-sip-abundance": "--sip-abundance",
            "-ms1-topn": "--ms1-topn",
            "-ms2-topn": "--ms2-topn",
        },
    )
    argv = _normalize_sip_abundance_args(argv)
    try:
        opts, _ = getopt.getopt(
            argv,
            "hi:1:2:o:t:j:f:c:w:d:",
            ["max-peaks=", "jobs=", "sip-abundance=", "ms1-topn=", "ms2-topn="],
        )
    except Exception:
        print("Error Option, using -h for help information.")
        return 1
    if len(opts) == 0:
        print_usage()
        return 1
    ft1_file = ""
    ft2_file = ""
    tsv_file = ""
    output_file = ""
    mode = ""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_file = os.path.join(script_dir, "SIP.cfg")
    config_file = default_config_file if os.path.exists(default_config_file) else "SiprosConfig.cfg"
    num_cpus = str(os.cpu_count() or 1)
    batch_jobs = 1
    ms1_isolation_window_mz = 10.0
    sip_abundance = ""
    pairmaxlength = DEFAULT_MAX_PEAKS
    ms1_precursor_top_n = DEFAULT_MS1_PRECURSOR_TOP_N
    ms2_fragment_top_n = DEFAULT_MS2_FRAGMENT_TOP_N
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
        elif opt in ("-j", "--jobs"):
            batch_jobs = max(1, int(_to_float(arg, 1)))
        elif opt in ("-f"):
            mode = arg
        elif opt in ("-c"):
            config_file = arg
        elif opt in ("-b", "--sip-abundance"):
            sip_abundance = str(arg).strip()
        elif opt in ("-w"):
            ms1_isolation_window_mz = max(0.0, _to_float(arg, 10.0))
        elif opt in ("-d"):
            diffPPM = max(0.0, _to_float(arg, 10.0))
        elif opt == "--max-peaks":
            pairmaxlength = max(1, int(_to_float(arg, DEFAULT_MAX_PEAKS)))
        elif opt == "--ms1-topn":
            ms1_precursor_top_n = max(1, int(_to_float(arg, DEFAULT_MS1_PRECURSOR_TOP_N)))
        elif opt == "--ms2-topn":
            ms2_fragment_top_n = max(1, int(_to_float(arg, DEFAULT_MS2_FRAGMENT_TOP_N)))

    if len(mode) == 0:
        mode = "att"

    if mode == "cnn":
        print("Max peaks kept for CNN features: " + str(pairmaxlength))
        print("MS1 precursor top-N tries: " + str(ms1_precursor_top_n))
        print("MS2 fragment top-N tries: " + str(ms2_fragment_top_n))

    if len(tsv_file) == 0:
        raise ValueError("A tab delimited PSM file is required. Use -i.")
    batch_tasks = _resolve_batch_tasks(tsv_file)
    if batch_tasks:
        if ft1_file or ft2_file or output_file:
            raise ValueError(
                "Directory/list input auto-resolves FT1/FT2/output paths; do not pass -1, -2, or -o."
            )
        return _run_batch_conversions(
            batch_tasks,
            batch_jobs,
            config_file,
            num_cpus,
            mode,
            sip_abundance,
            ms1_isolation_window_mz,
            diffPPM,
            pairmaxlength,
            ms1_precursor_top_n,
            ms2_fragment_top_n,
        )

    if len(ft2_file) == 0 or len(ft1_file) == 0:
        raise ValueError("Both FT1 and FT2 files are required. Use -1 and -2.")
    if len(output_file) == 0:
        raise ValueError("An output pickle path is required. Use -o.")

    return _run_single_conversion(
        tsv_file,
        ft1_file,
        ft2_file,
        output_file,
        config_file,
        num_cpus,
        mode,
        sip_abundance,
        ms1_isolation_window_mz,
        ms1_precursor_top_n,
        ms2_fragment_top_n,
    )


if __name__ == "__main__":
    raise SystemExit(main())
