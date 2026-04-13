#!/usr/bin/env python3
"""Production pure_cnn_pct prediction wrapper for raw WinnowNet inputs."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_all_start_methods, get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL = REPO_ROOT / "data" / "pure_cnn_all_pct.pt"
DEFAULT_CONFIG = SCRIPT_DIR / "SIP.cfg"
DEFAULT_BATCH_SIZE = 1024
DEFAULT_THREADS_PER_JOB = 3
DEFAULT_RESERVE_CORES = 1
DEFAULT_MAX_PEAKS = 256
DEFAULT_MS1_WINDOW = 10.0
DEFAULT_PPM = 10.0
DECOY_LEFT_FRACTION = 0.01
TARGET_EXCLUDE_PROTEIN_PREFIXES = ("Decoy_", "Con_")
CNN_INPUT_CHANNELS = 7
CNN_FEATURE_SCHEMA = "cnn_7ch_v1"
CNN_ENRICH_RATIO_CHANNEL = 6
MODEL_ARCH_PURE_CNN_PCT = "pure_cnn_pct"
MS2_ISOTOPIC_ABUNDANCE_COLUMN = "MS2IsotopicAbundances"
MS2_ENRICH_RATIO_MEDIAN_COLUMN = "MS2isotopicAbundanceEvolopeMedian"
PREDICTED_13C_PCT_COLUMN = "Predicted13CPct"
OUTPUT_DROP_COLUMNS = {"posterior_error_prob"}
PKL_SCHEMA_VERSION = 2
NEUTRON_MASS = 1.0033548378
PROTON_MASS = 1.00727646688
H = 1.007825
O = 15.9949
N_TERMINUS = H
C_TERMINUS = O + H
pairmaxlength = DEFAULT_MAX_PEAKS
diffPPM = DEFAULT_PPM

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
    "C": 103.009185 + 57.0214637236,
    "F": 147.068414,
    "I": 113.084064,
    "A": 71.037114,
    "T": 101.047679,
    "W": 186.079313,
    "H": 137.058912,
    "D": 115.026943,
    "K": 128.094963,
    "~": 15.99491,
}

_WORKER_EXP_MS1 = None
_WORKER_EXP_MS2 = None
_WORKER_THEORY = None
_WORKER_FEATURE = None
_WORKER_SCAN_MAP = None
_WORKER_MODE = "cnn"
_WORKER_MS1_WINDOW_MZ = 10.0
_WORKER_SIP_CONFIG = None


@dataclass(frozen=True)
class RawTask:
    tsv: Path
    ft1: Path
    ft2: Path
    kind: str


@dataclass
class FeatureFile:
    task: RawTask
    meta: dict
    ordered_items: list
    feature_keys: list
    feature_batches: list


@dataclass
class ScoredFile:
    task: RawTask
    meta: dict
    ordered_items: list
    feature_keys: list
    feature_batches: list
    scores: list
    predicted_pct: list | None


class peptide:
    def __init__(self):
        self.identified_pep = ""
        self.qvalue = 0
        self.qn = 0
        self.qnl = 0
        self.num_missed_cleavages = 0
        self.mono_mass = 0
        self.mass_error = 0
        self.isotopic_mass_window_shift = 0
        self.theory_mass = 0
        self.peplen = 0
        self.PSMId = ""


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
            return
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


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _split_path_values(value):
    if not value:
        return []
    parts = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            parts.append(item)
    return parts


def _strip_config_comment(line):
    return line.split("#", 1)[0].strip()


def _parse_config_values(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _to_float(value, default=None):
    try:
        text = str(value).strip()
        if text.endswith("%"):
            text = text[:-1].strip()
        return float(text)
    except (TypeError, ValueError):
        if default is None:
            raise
        return default


def _normalize_isotope_token(token):
    text = str(token or "").strip()
    if text.upper() == "D":
        return "H", 2
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", text)
    if not match:
        raise ValueError(
            "SIP isotope must look like C13, N15, O18, H2, or D."
        )
    return match.group(1).capitalize(), int(match.group(2))


def _format_isotope(element, mass_number):
    if element.upper() == "H" and int(mass_number) == 2:
        return "H2/D"
    return f"{element}{int(mass_number)}"


def _parse_sip_atom_abundance_override(value):
    if not value:
        return None
    if "=" not in value:
        raise ValueError("--sip-atom-abundance must use <isotope>=<percent>, for example C13=1.07.")
    isotope, abundance = value.split("=", 1)
    element, mass_number = _normalize_isotope_token(isotope)
    percent = _to_float(abundance)
    if percent < 0:
        raise ValueError("--sip-atom-abundance percent must be non-negative.")
    return {
        "element": element,
        "mass_number": mass_number,
        "percent": percent,
        "source": "--sip-atom-abundance",
    }


def _parse_sip_atom_abundance_from_config(config_path):
    sip_element = ""
    sip_isotope = ""
    element_masses = {}
    element_percents = {}

    with open(config_path) as fh:
        for raw_line in fh:
            line = _strip_config_comment(raw_line)
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            values = _parse_config_values(value)
            if key == "SIP_Element":
                sip_element = values[0] if values else value.strip()
            elif key == "SIP_Element_Isotope":
                sip_isotope = values[0] if values else value.strip()
            else:
                match = re.fullmatch(r"Element_(Masses|Percent)\{([^}]+)\}", key)
                if match:
                    kind, element = match.groups()
                    parsed = [_to_float(item) for item in values]
                    if kind == "Masses":
                        element_masses[element.strip()] = parsed
                    else:
                        element_percents[element.strip()] = parsed

    if not sip_element:
        raise ValueError(f"{config_path} is missing SIP_Element.")
    if not sip_isotope:
        raise ValueError(f"{config_path} is missing SIP_Element_Isotope.")
    isotope_number = int(round(_to_float(sip_isotope)))
    masses = element_masses.get(sip_element)
    percents = element_percents.get(sip_element)
    if not masses or not percents:
        raise ValueError(f"{config_path} is missing masses or percents for SIP element {sip_element}.")
    if len(masses) != len(percents):
        raise ValueError(f"{config_path} has mismatched masses/percents for SIP element {sip_element}.")

    for mass, percent in zip(masses, percents):
        if int(round(mass)) == isotope_number:
            percent_units = percent * 100.0 if percent <= 1.0 else percent
            return {
                "element": sip_element.capitalize(),
                "mass_number": isotope_number,
                "percent": percent_units,
                "source": str(config_path),
            }

    isotope = _format_isotope(sip_element, isotope_number)
    raise ValueError(f"{config_path} does not define natural abundance for {isotope}.")


def _resolve_sip_atom_abundance(config_path, override):
    parsed_override = _parse_sip_atom_abundance_override(override)
    if parsed_override is not None:
        return parsed_override
    try:
        return _parse_sip_atom_abundance_from_config(config_path)
    except Exception as exc:
        raise ValueError(
            f"Could not read SIP atom abundance from {config_path}: {exc}. "
            "Pass --sip-atom-abundance, for example C13=1.07."
        ) from exc


def _most_abundant_peak_mz(peaks):
    if len(peaks) == 0:
        return 0.0
    return max(peaks, key=lambda x: x[1])[0]


def _parse_psm_id(psm_id):
    psm_id = psm_id.strip()
    if "." in psm_id:
        parts = psm_id.split(".")
        if len(parts) >= 3 and parts[-2].isdigit():
            return ".".join(parts[:-2]), parts[-2]
    parts = psm_id.split("_")
    if len(parts) >= 3 and parts[-3].isdigit():
        return "_".join(parts[:-3]), parts[-3]
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
        if field is not None:
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


def _split_proteins(value):
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    return [item.strip() for item in text.split(",") if item.strip()]


def _protein_matches_prefixes(protein, prefixes):
    return any(str(protein).startswith(prefix) for prefix in prefixes)


def _proteins_all_match_prefixes(proteins, prefixes):
    return bool(prefixes) and bool(proteins) and all(
        _protein_matches_prefixes(protein, prefixes) for protein in proteins
    )


def _parse_protein_prefixes(value):
    if value is None:
        return ()
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _parse_feature_sip_config(config_file):
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
        raise ValueError(f"SIP_Element {sip_element!r} is not present in Element_List from {config_file}.")
    return {
        "element_list": element_list,
        "sip_element": sip_element,
        "sip_element_index": element_list.index(sip_element),
        "residue_composition": residue_composition,
    }


def _residue_sip_atom_count(sip_config, residue, psm_id):
    residue_composition = sip_config.get("residue_composition", {})
    if residue not in residue_composition:
        raise ValueError(f"{psm_id}: residue/PTM symbol {residue!r} is missing from SIP config.")
    composition = residue_composition[residue]
    sip_element_index = int(sip_config.get("sip_element_index", -1))
    if sip_element_index < 0 or sip_element_index >= len(composition):
        return 0.0
    return float(composition[sip_element_index])


def _residue_mass(residue, psm_id):
    if residue not in AA_dict:
        raise ValueError(f"{psm_id}: residue/PTM symbol {residue!r} is missing from AA mass table.")
    return float(AA_dict[residue])


def _build_cnn_ion_metadata(sip_config, peptide_sequence, precursor_charge, precursor_group, psm_id):
    peptide_sequence = str(peptide_sequence or "")
    residues = list(peptide_sequence)
    residue_masses = [_residue_mass(residue, psm_id) for residue in residues]
    residue_sip_counts = [_residue_sip_atom_count(sip_config, residue, psm_id) for residue in residues]
    prefix_masses = [0.0]
    prefix_sip_counts = [_residue_sip_atom_count(sip_config, "Nterm", psm_id)]
    for residue_mass, residue_sip_count in zip(residue_masses, residue_sip_counts):
        prefix_masses.append(prefix_masses[-1] + residue_mass)
        prefix_sip_counts.append(prefix_sip_counts[-1] + residue_sip_count)
    full_residue_mass = prefix_masses[-1]
    precursor_neutral_mass = full_residue_mass + N_TERMINUS + C_TERMINUS
    precursor_sip_atom_count = prefix_sip_counts[-1] + _residue_sip_atom_count(sip_config, "Cterm", psm_id)
    precursor_charge = max(1, int(_to_float(precursor_charge, 1)))
    metadata = {
        precursor_group: {
            "mono_mz": (precursor_neutral_mass + precursor_charge * PROTON_MASS) / precursor_charge,
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
            "sip_atom_number": precursor_sip_atom_count - prefix_sip_counts[complement_prefix_position],
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


def _default_base_from_tsv(tsv_path):
    suffix = "_filtered_psms.tsv"
    name = tsv_path.name
    stem = name[: -len(suffix)] if name.endswith(suffix) else tsv_path.stem
    return tsv_path.with_name(stem)


def _expand_tsv_inputs(input_value):
    paths = []
    seen = set()
    for item in _split_path_values(input_value):
        matches = []
        if os.path.isdir(item):
            for root, _, files in os.walk(item):
                for file_name in sorted(files):
                    if file_name.endswith("_filtered_psms.tsv"):
                        matches.append(os.path.join(root, file_name))
        elif any(ch in item for ch in "*?["):
            for match in glob.glob(item):
                if os.path.isdir(match):
                    for root, _, files in os.walk(match):
                        for file_name in sorted(files):
                            if file_name.endswith("_filtered_psms.tsv"):
                                matches.append(os.path.join(root, file_name))
                else:
                    matches.append(match)
        else:
            matches = [item]

        for match in sorted(matches):
            path = Path(match).resolve()
            if not path.name.endswith("_filtered_psms.tsv"):
                raise ValueError(f"Input is not a *_filtered_psms.tsv file: {path}")
            if path not in seen:
                seen.add(path)
                paths.append(path)
    if not paths:
        raise ValueError(f"No *_filtered_psms.tsv files found in {input_value!r}.")
    return paths


def _tasks_from_input(input_value, label):
    tasks = []
    missing = []
    for tsv in _expand_tsv_inputs(input_value):
        base = _default_base_from_tsv(tsv)
        task = RawTask(tsv=tsv, ft1=Path(str(base) + ".FT1"), ft2=Path(str(base) + ".FT2"), kind=label)
        for path in (task.ft1, task.ft2):
            if not path.is_file():
                missing.append(f"{task.tsv}: missing {path}")
        tasks.append(task)
    if missing:
        raise FileNotFoundError(f"{label} input has missing raw matches:\n" + "\n".join(missing))
    return tasks


def _build_theory_file(task, config_path, theory_path, threads_per_job):
    theoretical_bin = SCRIPT_DIR / "sipros_theoretical_spectra"
    if not theoretical_bin.is_file():
        raise FileNotFoundError(f"Cannot find sipros_theoretical_spectra at {theoretical_bin}")
    command = [
        str(theoretical_bin),
        "-c",
        str(config_path),
        "-i",
        str(task.tsv),
        "-o",
        str(theory_path),
        "-t",
        str(threads_per_job),
    ]
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


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
            for _, metadata in sorted(zip(peaks, peak_metadata), key=lambda item: item[0][0])
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
                    isolation_center_mz = _to_float(tokens[2], 0.0) if len(tokens) > 2 else 0.0
                    charge = 0
                    parent_scan = ""
                    precursor_candidates = []
                    peaks = []
                    peak_metadata = []
                elif tokens[0] == "Z" and scan_id is not None:
                    selected_charge = int(_to_float(tokens[1], 0))
                    if selected_charge > 0:
                        charge = selected_charge
                    candidate_pairs = []
                    seen = set()
                    if len(tokens) > 3:
                        remain = tokens[3:]
                        for i in range(len(remain) // 2):
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
                mz = _to_float(vals[0], 0.0)
                raw_mz = mz
                intensity = _to_float(vals[1], 0.0)
                peak_charge = int(_to_float(vals[5], 0)) if len(vals) >= 6 else 0
                if reduce_peak_charge_to_one and peak_charge > 1:
                    mz = peak_charge * (mz - PROTON_MASS) + PROTON_MASS
                peaks.append([mz, intensity])
                peak_metadata.append({"raw_mz": raw_mz, "raw_charge": peak_charge, "stored_mz": mz})
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

    def _parse_int_list(line):
        values = []
        for token in line.split():
            try:
                values.append(int(float(token)))
            except ValueError:
                continue
        return values

    def _build_peaks(mz_vals, intensity_vals):
        n = min(len(mz_vals), len(intensity_vals))
        return sorted([[mz_vals[i], intensity_vals[i]] for i in range(n)])

    def _build_fragment_records(mz_vals, intensity_vals, kinds, positions):
        n = min(len(mz_vals), len(intensity_vals))
        records = []
        for idx in range(n):
            kind = kinds[idx] if idx < len(kinds) else ""
            position = positions[idx] if idx < len(positions) else 0
            records.append({"peak": [mz_vals[idx], intensity_vals[idx]], "kind": kind, "position": position})
        return sorted(records, key=lambda item: item["peak"][0])

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith(">"):
            i += 1
            continue
        if i + 6 >= len(lines):
            break
        psm_id = line[1:].strip()
        precursor_scan = _build_peaks(_parse_float_list(lines[i + 1]), _parse_float_list(lines[i + 2]))
        fragment_records = _build_fragment_records(
            _parse_float_list(lines[i + 3]),
            _parse_float_list(lines[i + 4]),
            lines[i + 5].split(),
            _parse_int_list(lines[i + 6]),
        )
        fragment_scan = [record["peak"] for record in fragment_records]
        theory_dic[psm_id] = {
            "all": sorted(precursor_scan + fragment_scan),
            "precursor": precursor_scan,
            "fragment": fragment_scan,
            "precursor_abundant_mz": _most_abundant_peak_mz(precursor_scan),
            "fragment_kinds": [record["kind"] for record in fragment_records],
            "fragment_positions": [record["position"] for record in fragment_records],
        }
        i += 7
    return theory_dic


def read_tsv(tsv_file, psm_dict, ms2_dict, exclude_protein_prefixes=(), exclude_decoy_label_rows=False):
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
        "skipped_excluded_protein_prefix": 0,
        "skipped_decoy_label_rows": 0,
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
        if use_header and exclude_protein_prefixes:
            protein_raw = _get_field_value(
                row,
                field_lookup,
                ["Proteins", "Protein", "ProteinName", "ProteinNames"],
                "",
            )
            if _proteins_all_match_prefixes(_split_proteins(protein_raw), exclude_protein_prefixes):
                row_stats["skipped_excluded_protein_prefix"] += 1
                continue
        if use_header:
            idx = (_get_field_value(row, field_lookup, ["PSMId", "SpecId"]) or "").strip()
            qvalue_raw = _get_field_value(row, field_lookup, ["q-value", "qvalue"], "")
            exp_mass_raw = _get_field_value(row, field_lookup, ["ExpMass", "ExperimentalMass", "expmass"], "0")
            peptide_raw = _get_field_value(row, field_lookup, ["Peptide", "peptide", "IdentifiedPeptide", "PeptideSequence"], "")
            mass_error_raw = _get_field_value(row, field_lookup, ["massErrors", "massError", "MassError", "Massdiff", "MassDiff"], "0")
            isotopic_shift_raw = _get_field_value(row, field_lookup, ["isotopicMassWindowShifts", "isotopicMassWindowShift"], "0")
            charge_raw = _get_field_value(row, field_lookup, ["parentCharges", "ParentCharge", "charge", "Charge"], "")
            scan_nr_raw = _get_field_value(row, field_lookup, ["ScanNr", "Scan", "scan"], "")
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
        if exclude_decoy_label_rows and label == 0:
            row_stats["skipped_decoy_label_rows"] += 1
            continue
        label_confidence = _compute_label_confidence(label, qvalue_raw)
        row_stats["rows_with_psm_id"] += 1 if idx else 0
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
            charge = parts[-2] if len(parts) >= 2 and parts[-2].isdigit() else "2"
        charge_int = int(_to_float(charge, 0))
        exp_mass = _to_float(exp_mass_raw, 0.0)
        peptidestr = _clean_peptide(peptide_raw)
        if not peptidestr:
            row_stats["skipped_missing_peptide"] += 1
            continue
        row_stats["usable_peptide_rows"] += 1

        pep = peptide()
        pep.PSMId = idx
        pep.qvalue = _to_float(qvalue_raw, 0.0)
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
    D_feature = {}
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
            if f_dict[psm].charge == "1":
                D_feature[pep.PSMId].extend([1, 0, 0])
            elif f_dict[psm].charge == "2":
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
        entries.append((peak, _fragment_group_key(kind, position, idx)))
    return entries


def _build_group_info(entries):
    group_info = {}
    for peak, group_key in entries:
        mz = float(peak[0])
        intensity = max(0.0, float(peak[1]))
        info = group_info.setdefault(group_key, {"mono_mz": mz, "theory_max": 0.0})
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
    return [(peak, group_key) for peak, group_key in fragment_entries if low <= float(peak[0]) <= high]


def _append_cnn_matches(exp_array, theory_entries, matches, source_group):
    if exp_array.shape[0] == 0 or len(theory_entries) == 0:
        return
    grouped_entries = {}
    for peak, theory_group in theory_entries:
        grouped_entries.setdefault(theory_group, []).append(peak)
    for theory_group, peaks in grouped_entries.items():
        peaks_by_intensity = sorted(peaks, key=lambda x: (-float(x[1]), float(x[0])))
        for mz in peaks_by_intensity:
            theory_mz = float(mz[0])
            tolerance_mz = abs(theory_mz) * diffPPM * 1e-6
            index = np.where(
                np.logical_and(
                    exp_array[:, 0] > theory_mz - tolerance_mz,
                    exp_array[:, 0] < theory_mz + tolerance_mz,
                )
            )[0]
            if len(index) == 0:
                break
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
        maxima[source_group] = max(maxima.get(source_group, 0.0), float(match["raw_exp_intensity"]))
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
        enrich_ratios[group_key] = (weighted_sum / weight_sum) / sip_atom_number if weight_sum > 0 and sip_atom_number > 0 else 0.0
    return enrich_ratios


def _normalize_cnn_matches(matches, source_maxima, group_info, ion_metadata, ion_charges):
    sip_atom_counts = {key: value.get("sip_atom_number", 0.0) for key, value in ion_metadata.items()}
    enrich_ratios = _envelope_enrich_ratios(matches, ion_metadata, sip_atom_counts, ion_charges)
    normalized = []
    for match in matches:
        group_key = match["theory_group"]
        group_metadata = group_info.get(group_key, {})
        ion_info = ion_metadata.get(group_key)
        if ion_info is None:
            raise ValueError(f"Cannot compute CNN ion metadata for theory group {group_key!r}.")
        exp_max = float(source_maxima.get(match["source_group"], 0.0))
        theory_max = float(group_metadata.get("theory_max", 0.0))
        exp_value = max(0.0, float(match["raw_exp_intensity"]))
        theory_value = max(0.0, float(match["theory_intensity"]))
        exp_value = exp_value / exp_max if exp_max > 0 else 0.0
        theory_value = theory_value / theory_max if theory_max > 0 else 0.0
        normalized.append(
            {
                "raw_exp_intensity": max(0.0, float(match["raw_exp_intensity"])),
                "features": [
                    float(match["expmz"]),
                    float(match["delta_mz"]),
                    float(ion_info.get("mono_mz", 0.0)),
                    exp_value,
                    theory_value,
                    float(ion_info.get("sip_atom_number", 0.0)),
                    float(enrich_ratios.get(group_key, 0.0)),
                ],
            }
        )
    return normalized


def _filter_ms2_peaks_by_precursor_mass(ms2_record, exp_mass, precursor_mass_charge1):
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


def IonExtract(ms1_peaks, ms2_peaks, Xtheory, scan_info, sip_config, psm_id):
    ms1_array = _to_peak_array(ms1_peaks)
    ms2_array = _to_peak_array(ms2_peaks)
    precursor_group = ("precursor", 0)
    precursor_entries = [(peak, precursor_group) for peak in _to_peak_array(Xtheory.get("precursor", [])).tolist()]
    fragment_entries = _fragment_entries_in_exp_range(ms2_array, Xtheory)
    group_info = _build_group_info(precursor_entries + fragment_entries)
    matches = []
    _append_cnn_matches(ms1_array, precursor_entries, matches, source_group="precursor")
    _append_cnn_matches(ms2_array, fragment_entries, matches, source_group="fragment")
    if len(matches) == 0:
        xFeatures = np.asarray(pad_control_cnn([]), dtype=float)
        return [xFeatures.transpose()]
    peptide_sequence = scan_info.get("peptide_sequence", "")
    precursor_charge = max(1, int(_to_float(scan_info.get("charge", 1), 1)))
    ion_metadata_all = _build_cnn_ion_metadata(sip_config, peptide_sequence, precursor_charge, precursor_group, psm_id)
    ion_metadata = {}
    ion_charges = {}
    for group_key in {match["theory_group"] for match in matches}:
        if group_key not in ion_metadata_all:
            raise ValueError(f"{psm_id}: cannot compute CNN ion metadata for theory group {group_key!r}.")
        ion_metadata[group_key] = ion_metadata_all[group_key]
        ion_charges[group_key] = precursor_charge if group_key == precursor_group else 1
    xFeatures = _normalize_cnn_matches(matches, _matched_source_maxima(matches), group_info, ion_metadata, ion_charges)
    xFeatures = np.asarray(pad_control_cnn(xFeatures), dtype=float)
    return [xFeatures.transpose()]


def _set_worker_state(exp_ms1, exp_ms2, theory, feature, scan_map, mode, ms1_window_mz, sip_config):
    global _WORKER_EXP_MS1, _WORKER_EXP_MS2, _WORKER_THEORY, _WORKER_FEATURE, _WORKER_SCAN_MAP
    global _WORKER_MODE, _WORKER_MS1_WINDOW_MZ, _WORKER_SIP_CONFIG
    _WORKER_EXP_MS1 = exp_ms1
    _WORKER_EXP_MS2 = exp_ms2
    _WORKER_THEORY = theory
    _WORKER_FEATURE = feature
    _WORKER_SCAN_MAP = scan_map
    _WORKER_MODE = mode
    _WORKER_MS1_WINDOW_MZ = ms1_window_mz
    _WORKER_SIP_CONFIG = sip_config


def _extract_feature_for_key(key):
    scan_info = _WORKER_SCAN_MAP.get(key)
    if scan_info is None:
        return key, None, 0
    ms2_record = _WORKER_EXP_MS2.get(scan_info["ms2_scan"], {})
    ms2_peaks, removed_peak_count = _filter_ms2_peaks_by_precursor_mass(
        ms2_record,
        scan_info.get("exp_mass", 0.0),
        scan_info.get("precursor_mass_charge1", 0.0),
    )
    ms1_peaks = _WORKER_EXP_MS1.get(scan_info["ms1_scan"], {}).get("peaks", [])
    theory = _WORKER_THEORY.get(key)
    features = _WORKER_FEATURE.get(key)
    if theory is None or features is None:
        return key, None, removed_peak_count
    model_input = IonExtract(ms1_peaks, ms2_peaks, theory, scan_info, _WORKER_SIP_CONFIG, key)
    return key, model_input, removed_peak_count


def _iter_feature_keys(theory_dict, feature_dict_map, scan_map):
    for key in theory_dict:
        if key in feature_dict_map and key in scan_map:
            yield key


def _extract_feature_payload(
    task,
    config_path,
    threads_per_job,
    max_peaks,
    ppm,
    ms1_window,
    exclude_protein_prefixes=(),
    exclude_decoy_label_rows=False,
):
    global pairmaxlength, diffPPM
    pairmaxlength = int(max_peaks)
    diffPPM = float(ppm)
    start = os.times()
    exp_ms1 = FTtoDict(str(task.ft1), reduce_peak_charge_to_one=False)
    exp_ms2 = FTtoDict(str(task.ft2), reduce_peak_charge_to_one=True)
    print(f"[features] loaded FT1/FT2 for {task.tsv}")

    psm_dict = {}
    psm_scan_map, table_meta, row_records, row_stats = read_tsv(
        str(task.tsv),
        psm_dict,
        exp_ms2,
        exclude_protein_prefixes=exclude_protein_prefixes,
        exclude_decoy_label_rows=exclude_decoy_label_rows,
    )
    print(
        f"[features] rows for {task.tsv}: total={row_stats['total_rows']} "
        f"with_psm_id={row_stats['rows_with_psm_id']} usable_peptide={row_stats['usable_peptide_rows']} "
        f"skipped_excluded_prefix={row_stats['skipped_excluded_protein_prefix']} "
        f"skipped_decoy_label={row_stats['skipped_decoy_label_rows']}"
    )

    with tempfile.TemporaryDirectory(prefix="winnownet_theory_") as temp_dir:
        theory_path = Path(temp_dir) / (task.tsv.stem + ".theory.txt")
        _build_theory_file(task, config_path, theory_path, threads_per_job)
        theory = theoryToDict(str(theory_path))

    for psm in psm_dict.values():
        psm.get_features()
    feature_map = feature_dict(psm_dict)
    sip_config = _parse_feature_sip_config(str(config_path))

    feature_payload = {}
    removed_peak_total = 0
    removed_psm_total = 0
    feature_keys = list(_iter_feature_keys(theory, feature_map, psm_scan_map))
    _set_worker_state(
        exp_ms1,
        exp_ms2,
        theory,
        feature_map,
        psm_scan_map,
        "cnn",
        float(ms1_window),
        sip_config,
    )

    if threads_per_job <= 1:
        for key in feature_keys:
            feature_key, model_input, removed_peak_count = _extract_feature_for_key(key)
            removed_peak_total += removed_peak_count
            if removed_peak_count > 0:
                removed_psm_total += 1
            if model_input is not None:
                feature_payload[feature_key] = model_input
    else:
        # Use fork-based workers when available; it avoids
        # serial CNN feature extraction without writing intermediate feature files.
        start_method = "fork" if "fork" in get_all_start_methods() else None
        ctx = get_context(start_method) if start_method else get_context()
        pool_kwargs = {"processes": int(threads_per_job)}
        if ctx.get_start_method() != "fork":
            pool_kwargs["initializer"] = _set_worker_state
            pool_kwargs["initargs"] = (
                exp_ms1,
                exp_ms2,
                theory,
                feature_map,
                psm_scan_map,
                "cnn",
                float(ms1_window),
                sip_config,
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

    output_meta = {
        **table_meta,
        "mode": "cnn",
        "feature_schema": CNN_FEATURE_SCHEMA,
        "input_channels": CNN_INPUT_CHANNELS,
        "max_peaks": int(max_peaks),
    }
    ordered_items = []
    rows_with_model_input = 0
    for row_record in row_records:
        psm_id = row_record["psm_id"]
        model_input = feature_payload.get(psm_id) if psm_id else None
        if model_input is not None:
            rows_with_model_input += 1
        ordered_items.append(
            (
                row_record["record_key"],
                {
                    "psm_id": psm_id,
                    "model_input": model_input,
                    "label": row_record["label"],
                    "label_raw": row_record["label_raw"],
                    "label_confidence": row_record["label_confidence"],
                    "row_index": row_record["row_index"],
                    "row_values": row_record["row_values"],
                },
            )
        )

    elapsed = (os.times().elapsed - start.elapsed) if hasattr(os.times(), "elapsed") else 0.0
    print(
        f"[features] extracted {task.tsv}: feature_rows={rows_with_model_input} "
        f"feature_keys={len(feature_payload)} removed_peaks={removed_peak_total} "
        f"affected_psms={removed_psm_total} time={elapsed:.1f}s"
    )
    return output_meta, ordered_items


def _resolve_parallelism(requested_jobs, total_cores, threads_per_job, task_count):
    cpu_cores = os.cpu_count() or 1
    tasks = max(1, int(task_count))
    if total_cores is None:
        total_cores = max(1, cpu_cores - DEFAULT_RESERVE_CORES)
    else:
        total_cores = max(1, int(total_cores))
    if total_cores > cpu_cores:
        print(
            f"Requested cores={total_cores} exceeds cpu_cores={cpu_cores}; using cores={cpu_cores}.",
            file=sys.stderr,
        )
        total_cores = cpu_cores

    threads = max(1, int(threads_per_job))
    if threads > total_cores:
        print(
            f"Requested cores_per_job={threads} exceeds cores={total_cores}; using cores_per_job={total_cores}.",
            file=sys.stderr,
        )
        threads = total_cores
    max_jobs = max(1, total_cores // threads)
    jobs = min(tasks, max_jobs) if requested_jobs is None else max(1, int(requested_jobs))
    if jobs > max_jobs:
        print(
            f"Requested jobs={jobs} with threads_per_job={threads} exceeds cores={total_cores}; "
            f"using jobs={max_jobs}.",
            file=sys.stderr,
        )
        jobs = max_jobs
    if jobs > tasks:
        print(
            f"Requested jobs={jobs} exceeds input files per scoring pass={tasks}; using jobs={tasks}.",
            file=sys.stderr,
        )
        jobs = tasks
    return jobs, threads, cpu_cores, total_cores


def _resolve_device(device_arg):
    value = str(device_arg or "auto").strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        value = "cuda:0"
    if value.startswith("cuda:"):
        try:
            index = int(value.split(":", 1)[1])
        except ValueError:
            print(f"Invalid CUDA device {device_arg!r}; using CPU.", file=sys.stderr)
            return torch.device("cpu")
        if not torch.cuda.is_available():
            print(f"CUDA requested as {device_arg!r}, but CUDA is unavailable; using CPU.", file=sys.stderr)
            return torch.device("cpu")
        if index < 0 or index >= torch.cuda.device_count():
            print(
                f"CUDA device {device_arg!r} does not exist; available device count is {torch.cuda.device_count()}; using CPU.",
                file=sys.stderr,
            )
            return torch.device("cpu")
        return torch.device(value)
    print(f"Unknown device {device_arg!r}; using CPU.", file=sys.stderr)
    return torch.device("cpu")


def _validate_cnn_features(features, label="xFeatures"):
    x_features = np.asarray(features, dtype=float)
    if x_features.ndim != 2:
        raise ValueError(f"{label} must be a 2D CNN feature tensor, got ndim={x_features.ndim}.")
    if x_features.shape[0] != CNN_INPUT_CHANNELS:
        raise ValueError(
            f"{label} must have {CNN_INPUT_CHANNELS} channels in the first dimension, got shape={x_features.shape}."
        )
    return x_features


def _extract_cnn_model_features(model_input, label="xFeatures"):
    if not isinstance(model_input, (list, tuple)) or len(model_input) != 1:
        raise ValueError(f"{label}: CNN model_input must be [xFeatures].")
    return _validate_cnn_features(model_input[0], label)


def _load_checkpoint_bundle(model_path):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint or "metadata" not in checkpoint:
        raise ValueError(f"Checkpoint {model_path} is missing state_dict or metadata.")
    metadata = checkpoint["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint {model_path} metadata must be a dict.")
    for key in ("checkpoint_format_version", "model_type", "best_decision_threshold"):
        if key not in metadata:
            raise ValueError(f"Checkpoint {model_path} metadata is missing required field: {key}")
    return checkpoint["state_dict"], metadata


def _load_checkpoint_metadata(model_path):
    _, metadata = _load_checkpoint_bundle(model_path)
    return metadata


def resolve_checkpoint_model_arch(metadata):
    return str((metadata or {}).get("model_arch", "")).strip().lower()


def _load_checkpoint_weights(model_path, expected_model_arch=None):
    state_dict, metadata = _load_checkpoint_bundle(model_path)
    if int(metadata.get("input_channels", 0)) != CNN_INPUT_CHANNELS:
        raise ValueError(f"Checkpoint {model_path} is not compatible with {CNN_INPUT_CHANNELS}-channel CNN input.")
    if metadata.get("feature_schema") != CNN_FEATURE_SCHEMA:
        raise ValueError(f"Checkpoint {model_path} feature_schema is {metadata.get('feature_schema')!r}, expected {CNN_FEATURE_SCHEMA!r}.")
    checkpoint_model_arch = resolve_checkpoint_model_arch(metadata)
    if expected_model_arch is not None and checkpoint_model_arch != expected_model_arch:
        raise ValueError(
            f"Checkpoint {model_path} was trained with model_arch={checkpoint_model_arch!r}, expected {expected_model_arch!r}."
        )
    return state_dict


class PureCNNPct(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(CNN_INPUT_CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.shared = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )
        self.classifier = nn.Linear(128, 2)
        self.pct_regressor = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x_max = torch.max(x, dim=2)[0]
        x_mean = torch.mean(x, dim=2)
        x = torch.cat([x_max, x_mean], dim=1)
        x = self.shared(x)
        return self.classifier(x), self.pct_regressor(x)


def build_cnn_model(model_arch):
    if model_arch != MODEL_ARCH_PURE_CNN_PCT:
        raise ValueError(f"Unsupported model_arch={model_arch!r}; this wrapper only supports pure_cnn_pct.")
    return PureCNNPct()


def _load_model(model_path, device):
    metadata = _load_checkpoint_metadata(str(model_path))
    model_arch = resolve_checkpoint_model_arch(metadata)
    if model_arch != MODEL_ARCH_PURE_CNN_PCT:
        raise ValueError(f"{model_path} is model_arch={model_arch!r}; production wrapper requires pure_cnn_pct.")
    state_dict = _load_checkpoint_weights(str(model_path), model_arch)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
    model = build_cnn_model(model_arch)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, metadata


def _feature_task(task, args):
    exclude_protein_prefixes = args.target_exclude_protein_prefixes if task.kind == "target" else ()
    meta, ordered_items = _extract_feature_payload(
        task,
        args.config,
        args.threads_per_job,
        args.max_peaks,
        args.ppm,
        args.ms1_window,
        exclude_protein_prefixes=exclude_protein_prefixes,
        exclude_decoy_label_rows=True,
    )
    feature_keys = []
    feature_batches = []
    for key, entry in ordered_items:
        model_input = entry.get("model_input")
        if model_input is None:
            continue
        x_features = _extract_cnn_model_features(model_input, f"{key}: xFeatures")
        feature_keys.append(key)
        feature_batches.append(x_features)
    return FeatureFile(
        task=task,
        meta=meta,
        ordered_items=ordered_items,
        feature_keys=feature_keys,
        feature_batches=feature_batches,
    )


def _feature_task_in_process(task, args):
    return _feature_task(task, args)


def _score_tasks(tasks, model, device, batch_size, jobs, args):
    if jobs <= 1 or len(tasks) <= 1:
        feature_files = [_feature_task(task, args) for task in tasks]
    else:
        feature_files = [None] * len(tasks)
        worker_count = min(jobs, len(tasks))
        print(f"Generating feature file jobs={worker_count}; cores_per_job={args.threads_per_job}", flush=True)
        executor_kwargs = {}
        if device.type != "cpu":
            executor_kwargs["mp_context"] = get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, **executor_kwargs) as executor:
            future_to_idx = {
                executor.submit(_feature_task_in_process, task, args): idx
                for idx, task in enumerate(tasks)
            }
            for future in as_completed(future_to_idx):
                feature_files[future_to_idx[future]] = future.result()

    all_feature_batches = []
    feature_slices = []
    for feature_file in feature_files:
        start_idx = len(all_feature_batches)
        all_feature_batches.extend(feature_file.feature_batches)
        feature_slices.append((start_idx, len(all_feature_batches)))

    print(
        f"Scoring feature rows={len(all_feature_batches)} files={len(feature_files)} on {device}",
        flush=True,
    )
    if all_feature_batches:
        scores, _labels, predicted_pct = _predict_scores(
            model,
            all_feature_batches,
            device,
            batch_size,
            0.0,
        )
    else:
        scores, predicted_pct = [], []

    scored_files = []
    for feature_file, (start_idx, end_idx) in zip(feature_files, feature_slices):
        scored_files.append(
            ScoredFile(
                task=feature_file.task,
                meta=feature_file.meta,
                ordered_items=feature_file.ordered_items,
                feature_keys=feature_file.feature_keys,
                feature_batches=feature_file.feature_batches,
                scores=scores[start_idx:end_idx],
                predicted_pct=predicted_pct[start_idx:end_idx],
            )
        )
    return scored_files


def _split_model_output(output):
    if isinstance(output, (tuple, list)):
        if len(output) != 2:
            raise ValueError(f"Expected CNN model output tuple of length 2, got {len(output)}.")
        return output[0], output[1]
    return output, None


class DefineDataset(Data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx])


def _predict_scores(model, feature_batches, device, batch_size, decision_threshold):
    test_loader = Data.DataLoader(DefineDataset(feature_batches), batch_size=batch_size)
    model.eval()
    y_pred_prob = []
    y_pred = []
    predicted_pct = []
    with torch.no_grad():
        for data1 in test_loader:
            data1 = data1.to(device)
            output = model(data1)
            logits, pred_log_pct = _split_model_output(output)
            pred_prob = torch.softmax(logits.data, dim=1).cpu().numpy()
            positive_scores = pred_prob[:, 1]
            y_pred_prob.extend(positive_scores.tolist())
            y_pred.extend((positive_scores >= decision_threshold).astype(int).tolist())
            if pred_log_pct is None:
                predicted_pct.extend([None] * len(positive_scores))
            else:
                pct_values = torch.clamp(torch.expm1(pred_log_pct.data), min=0.0).view(-1).cpu().numpy()
                predicted_pct.extend(pct_values.tolist())
    return y_pred_prob, y_pred, predicted_pct


def _all_scores(scored_files):
    values = []
    for scored in scored_files:
        values.extend(float(score) for score in scored.scores if score is not None)
    return values


def _threshold_from_target_decoy_scores(target_scores, decoy_scores):
    combined = []
    total_targets = 0
    total_decoys = 0
    for score in target_scores:
        if score is not None:
            combined.append((float(score), 0))
            total_targets += 1
    for score in decoy_scores:
        if score is not None:
            combined.append((float(score), 1))
            total_decoys += 1
    if total_decoys == 0:
        raise ValueError("No decoy scores were generated; cannot calibrate threshold.")
    if total_targets == 0:
        raise ValueError("No target scores were generated; cannot calibrate threshold.")

    combined.sort(key=lambda item: item[0], reverse=True)
    accepted_targets = 0
    accepted_decoys = 0
    best = None
    idx = 0
    while idx < len(combined):
        score = combined[idx][0]
        group_targets = 0
        group_decoys = 0
        while idx < len(combined) and combined[idx][0] == score:
            if combined[idx][1]:
                group_decoys += 1
            else:
                group_targets += 1
            idx += 1
        accepted_targets += group_targets
        accepted_decoys += group_decoys
        accepted_total = accepted_targets + accepted_decoys
        decoy_ratio = accepted_decoys / float(accepted_total)
        if decoy_ratio <= DECOY_LEFT_FRACTION:
            best = (score, accepted_targets, accepted_decoys, accepted_total, decoy_ratio)

    if best is None:
        return float("inf"), 0, 0, 0, 0.0, total_targets, total_decoys
    threshold, accepted_targets, accepted_decoys, accepted_total, decoy_ratio = best
    return threshold, accepted_targets, accepted_decoys, accepted_total, decoy_ratio, total_targets, total_decoys


def _field_lookup(fieldnames):
    return {str(field).lower(): field for field in fieldnames or []}


def _median_ms2_abundance_above_natural(tasks, natural_percent):
    values = []
    for task in tasks:
        with open(task.tsv, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            lookup = _field_lookup(reader.fieldnames)
            column = lookup.get(MS2_ISOTOPIC_ABUNDANCE_COLUMN.lower())
            if column is None:
                raise ValueError(f"{task.tsv} is missing {MS2_ISOTOPIC_ABUNDANCE_COLUMN}.")
            for row in reader:
                raw_value = str(row.get(column, "")).strip()
                if not raw_value:
                    continue
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                if value > natural_percent:
                    values.append(value)
    if not values:
        return None
    return float(statistics.median(values))


def _threshold_from_pct_metadata(metadata, median_pct):
    default_threshold = float(metadata.get("best_decision_threshold", 0.5))
    pct_thresholds = metadata.get("pct_decision_thresholds") or {}
    if median_pct is None or not pct_thresholds:
        return default_threshold, None
    closest_key = min(pct_thresholds, key=lambda key: abs(float(key) - float(median_pct)))
    entry = pct_thresholds[closest_key]
    if isinstance(entry, dict):
        return float(entry.get("threshold", default_threshold)), closest_key
    return float(entry), closest_key


def is_rich_entry(entry):
    return isinstance(entry, dict) and ("model_input" in entry or "row_values" in entry or "label" in entry)


def get_entry_model_input(entry):
    if is_rich_entry(entry):
        return entry.get("model_input")
    return entry


def get_entry_row_map(meta, psm_id, entry):
    if not is_rich_entry(entry):
        raise ValueError("The feature entry does not contain stored row metadata.")
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


def choose_output_column(columns, preferred_names, default_name):
    lower_to_actual = {column.lower(): column for column in columns}
    for name in preferred_names:
        actual = lower_to_actual.get(name.lower())
        if actual is not None:
            return actual
    return default_name


def format_label_value(label):
    if label is None:
        return ""
    return "1" if int(label) == 1 else "-1"


def _format_feature_value(value):
    if value == "":
        return ""
    if not np.isfinite(float(value)):
        return ""
    return f"{float(value):.10g}"


def _score_to_string(score):
    return f"{float(score):.10g}"


def _qvalue_to_string(qvalue):
    if qvalue == "":
        return ""
    return f"{float(qvalue):.10g}"


def _enrich_ratio_median_from_features(x_features):
    x_features = _validate_cnn_features(x_features, "xFeatures")
    if x_features.shape[1] == 0:
        return ""
    non_padded = np.any(np.abs(x_features) > 0, axis=0)
    enrich_ratios = x_features[CNN_ENRICH_RATIO_CHANNEL, non_padded]
    if enrich_ratios.shape[0] == 0:
        return ""
    return _format_feature_value(np.median(enrich_ratios) * 100.0)


def _get_entry_enrich_ratio_median(entry):
    model_input = get_entry_model_input(entry)
    if model_input is None:
        return ""
    x_features = _extract_cnn_model_features(model_input, "xFeatures")
    return _enrich_ratio_median_from_features(x_features)


def _insert_ms2_enrich_ratio_median_column(output_columns):
    if MS2_ISOTOPIC_ABUNDANCE_COLUMN not in output_columns:
        return
    if MS2_ENRICH_RATIO_MEDIAN_COLUMN in output_columns:
        output_columns.remove(MS2_ENRICH_RATIO_MEDIAN_COLUMN)
    insert_idx = output_columns.index(MS2_ISOTOPIC_ABUNDANCE_COLUMN) + 1
    output_columns.insert(insert_idx, MS2_ENRICH_RATIO_MEDIAN_COLUMN)


def _compute_qvalues(scores, labels):
    qvalues = [""] * len(scores)
    labeled_indices = [idx for idx, label in enumerate(labels) if label in (0, 1) and scores[idx] is not None]
    if len(labeled_indices) == 0:
        return qvalues
    ranked = sorted(labeled_indices, key=lambda idx: scores[idx], reverse=True)
    running_fdr = {}
    targets = 0
    decoys = 0
    for idx in ranked:
        if labels[idx] == 1:
            targets += 1
        else:
            decoys += 1
        running_fdr[idx] = 1.0 if targets == 0 else decoys / float(targets)
    best_fdr = 1.0
    for idx in reversed(ranked):
        best_fdr = min(best_fdr, running_fdr[idx])
        qvalues[idx] = best_fdr
    return qvalues


def _make_rescored_rows(meta, ordered_items, score_map, predicted_label_map, predicted_pct_map=None, source_file=None):
    labels = []
    scores = []
    rescored_rows = []
    for idx, (key, entry) in enumerate(ordered_items):
        score = score_map.get(key)
        label = predicted_label_map.get(key)
        labels.append(label)
        scores.append(score)
        rescored_rows.append(
            {
                "sort_score": float("-inf") if score is None else float(score),
                "row_index": idx,
                "key": key,
                "entry": entry,
                "score": score,
                "label": label,
                "predicted_pct": None if predicted_pct_map is None else predicted_pct_map.get(key),
                "meta": meta,
                "source_file": source_file,
            }
        )
    qvalues = _compute_qvalues(scores, labels)
    for row, qvalue in zip(rescored_rows, qvalues):
        row["qvalue"] = qvalue
    return rescored_rows


def _get_output_columns(metas):
    output_columns = []
    for meta in metas:
        for column in list(meta.get("columns", [])):
            if column not in output_columns:
                output_columns.append(column)
    _insert_ms2_enrich_ratio_median_column(output_columns)
    score_column = choose_output_column(output_columns, ["score"], "score")
    qvalue_column = choose_output_column(output_columns, ["q-value", "qvalue"], "q-value")
    label_column = choose_output_column(output_columns, ["Label"], "Label")
    for column in (score_column, qvalue_column, label_column):
        if column not in output_columns:
            output_columns.append(column)
    if PREDICTED_13C_PCT_COLUMN in output_columns:
        output_columns.remove(PREDICTED_13C_PCT_COLUMN)
    output_columns = [column for column in output_columns if column.lower() not in OUTPUT_DROP_COLUMNS]
    return output_columns, score_column, qvalue_column, label_column


def _write_rescored_rows(output_file, rescored_rows, output_columns, score_column, qvalue_column, label_column):
    ranked_rows = list(rescored_rows)
    ranked_rows.sort(key=lambda item: (-item["sort_score"], item["row_index"]))
    with open(output_file, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=output_columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in ranked_rows:
            key = row["key"]
            entry = row["entry"]
            row_map = get_entry_row_map(row["meta"], key, entry)
            score = row["score"]
            if score is not None:
                row_map[score_column] = _score_to_string(score)
            elif score_column not in row_map:
                row_map[score_column] = ""
            qvalue_text = _qvalue_to_string(row["qvalue"])
            if qvalue_text != "" or qvalue_column not in row_map:
                row_map[qvalue_column] = qvalue_text
            label_text = format_label_value(row["label"])
            if label_text != "" or label_column not in row_map:
                row_map[label_column] = label_text
            if MS2_ENRICH_RATIO_MEDIAN_COLUMN in output_columns:
                row_map[MS2_ENRICH_RATIO_MEDIAN_COLUMN] = _get_entry_enrich_ratio_median(entry)
            writer.writerow({column: row_map.get(column, "") for column in output_columns})


def _rescored_rows(scored_files, threshold):
    rows = []
    metas = []
    for scored in scored_files:
        metas.append(scored.meta)
        score_map = {key: score for key, score in zip(scored.feature_keys, scored.scores)}
        label_map = {
            key: 1 if float(score) >= threshold else 0
            for key, score in score_map.items()
            if score is not None
        }
        pct_map = None
        if scored.predicted_pct is not None:
            pct_map = {key: pct for key, pct in zip(scored.feature_keys, scored.predicted_pct)}
        rows.extend(
            _make_rescored_rows(
                scored.meta,
                scored.ordered_items,
                score_map,
                label_map,
                pct_map,
                source_file=str(scored.task.tsv),
            )
        )

    qvalues = _compute_qvalues([row["score"] for row in rows], [row["label"] for row in rows])
    for row, qvalue in zip(rows, qvalues):
        row["qvalue"] = qvalue
    return rows, metas


def _accepted_rows(rows, threshold):
    return [
        row
        for row in rows
        if row.get("score") is not None and float(row["score"]) >= threshold
    ]


def _set_row_labels(rows, label):
    for row in rows:
        row["label"] = label


def _write_output(output_path, rows, metas):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_columns, score_column, qvalue_column, label_column = _get_output_columns(metas)
    _write_rescored_rows(
        str(output_path),
        rows,
        output_columns,
        score_column,
        qvalue_column,
        label_column,
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Production pure_cnn_pct prediction for raw FT1/FT2/*_filtered_psms.tsv directories."
    )
    parser.add_argument("--target", required=True, help="Target *_filtered_psms.tsv input file/dir/glob/comma-list.")
    parser.add_argument("--decoy", default="", help="Optional decoy *_filtered_psms.tsv input file/dir/glob/comma-list.")
    parser.add_argument("-o", "--output", required=True, help="Accepted-only combined target TSV.")
    parser.add_argument("--decoy-output", default="", help="Optional accepted-only combined decoy audit TSV.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help=f"pure_cnn_pct checkpoint. Default: {DEFAULT_MODEL}")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG), help=f"SIP config file. Default: {DEFAULT_CONFIG}")
    parser.add_argument(
        "--sip-atom-abundance",
        default="",
        help="Optional no-decoy natural isotope abundance override, e.g. C13=1.07, N15=0.368, O18=0.205, D=0.0115.",
    )
    parser.add_argument(
        "--target-exclude-protein-prefixes",
        default=",".join(TARGET_EXCLUDE_PROTEIN_PREFIXES),
        help=(
            "Comma-separated protein prefixes to remove from --target before prediction. "
            "A row is removed only when all Proteins start with one of these prefixes. "
            "Use an empty value to disable. Default: Decoy_,Con_."
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N. Default: auto.")
    parser.add_argument(
        "--threads-per-job",
        "--cores-per-job",
        dest="threads_per_job",
        type=_positive_int,
        default=DEFAULT_THREADS_PER_JOB,
        help=f"CPU cores used inside each file job. Default: {DEFAULT_THREADS_PER_JOB}.",
    )
    parser.add_argument("--jobs", type=_positive_int, default=None, help="Maximum file jobs to run in parallel.")
    parser.add_argument(
        "--cores",
        type=_positive_int,
        default=None,
        help="Maximum total CPU cores to use. Default: all visible cores minus one.",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--ms1-window", type=_positive_float, default=DEFAULT_MS1_WINDOW)
    parser.add_argument("--ppm", type=_positive_float, default=DEFAULT_PPM)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    args.config = Path(args.config).resolve()
    args.model = Path(args.model).resolve()
    args.max_peaks = DEFAULT_MAX_PEAKS
    args.target_exclude_protein_prefixes = _parse_protein_prefixes(args.target_exclude_protein_prefixes)

    target_tasks = _tasks_from_input(args.target, "target")
    decoy_tasks = _tasks_from_input(args.decoy, "decoy") if args.decoy else []
    print(f"Matched target files={len(target_tasks)} decoy files={len(decoy_tasks)}")
    scoring_files = len(target_tasks) + len(decoy_tasks)

    jobs, threads_per_job, cpu_cores, total_cores = _resolve_parallelism(
        args.jobs,
        args.cores,
        args.threads_per_job,
        scoring_files,
    )
    args.threads_per_job = threads_per_job
    print(
        f"CPU cores={cpu_cores}; core_limit={total_cores}; jobs={jobs}; "
        f"cores_per_job={threads_per_job}"
    )
    print(f"Target exclude protein prefixes={','.join(args.target_exclude_protein_prefixes) or '<none>'}")

    device = _resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(total_cores)
    print(f"Using prediction device={device}")

    model, metadata = _load_model(args.model, device)
    print(f"Using model={args.model} arch={resolve_checkpoint_model_arch(metadata)} max_peaks={args.max_peaks}")

    scored = _score_tasks(target_tasks + decoy_tasks, model, device, args.batch_size, jobs, args)
    target_scored = scored[: len(target_tasks)]
    decoy_scored = scored[len(target_tasks) :]

    if decoy_scored:
        (
            threshold,
            accepted_targets,
            accepted_decoys,
            accepted_total,
            decoy_ratio,
            total_targets,
            total_decoys,
        ) = _threshold_from_target_decoy_scores(_all_scores(target_scored), _all_scores(decoy_scored))
        print(
            "Threshold source=target_decoy_1pct "
            f"threshold={threshold:.10g} target_psms={total_targets} decoy_psms={total_decoys} "
            f"accepted_targets={accepted_targets} accepted_decoys={accepted_decoys} "
            f"accepted_total={accepted_total} decoy_ratio={decoy_ratio:.10g}"
        )
    else:
        sip_abundance = _resolve_sip_atom_abundance(args.config, args.sip_atom_abundance)
        isotope = _format_isotope(sip_abundance["element"], sip_abundance["mass_number"])
        median_pct = _median_ms2_abundance_above_natural(target_tasks, sip_abundance["percent"])
        threshold, pct_bin = _threshold_from_pct_metadata(metadata, median_pct)
        print(
            "Threshold source=target_median_ms2_pct "
            f"isotope={isotope} natural_percent={sip_abundance['percent']:.10g} "
            f"abundance_source={sip_abundance['source']} median_above_natural={median_pct} "
            f"pct_bin={pct_bin} threshold={threshold:.10g}"
        )

    target_rows, target_metas = _rescored_rows(target_scored, threshold)
    accepted_target_rows = _accepted_rows(target_rows, threshold)
    _write_output(args.output, accepted_target_rows, target_metas)
    print(f"Wrote target accepted PSMs={len(accepted_target_rows)} -> {args.output}")

    if args.decoy_output:
        if not decoy_scored:
            raise ValueError("--decoy-output requires --decoy.")
        decoy_rows, decoy_metas = _rescored_rows(decoy_scored, threshold)
        accepted_decoy_rows = _accepted_rows(decoy_rows, threshold)
        _set_row_labels(accepted_decoy_rows, 0)
        _write_output(args.decoy_output, accepted_decoy_rows, decoy_metas)
        print(f"Wrote decoy accepted PSMs={len(accepted_decoy_rows)} -> {args.decoy_output}")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
