import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "script"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pkl_utils import get_entry_model_input, get_entry_row_map, load_feature_pickle


PROTON_MASS = 1.00727646688
MATCH_PPM = 10.0
MAX_PLOT_RECORDS = 5
CNN_INPUT_CHANNELS = 10
CNN_EXPMZ_IDX = 0
CNN_DELTA_MZ_IDX = 1
CNN_MONO_MZ_IDX = 2
CNN_EXP_INTENSITY_IDX = 3
CNN_THEORY_INTENSITY_IDX = 4
CNN_SIP_ATOM_NUMBER_IDX = 5
CNN_ENRICH_RATIO_IDX = 6
CNN_ION_CHARGE_IDX = 7
CNN_ISOTOPE_INDEX_IDX = 8
CNN_MS_LEVEL_FLAG_IDX = 9
PEPTIDE_COLUMNS = ["Peptide", "IdentifiedPeptide", "OriginalPeptide"]
SCORE_COLUMNS = ["score", "Score"]
EXPMASS_COLUMNS = ["ExpMass", "MeasuredParentMass"]
CHARGE_COLUMNS = ["parentCharges", "ParentCharge", "charge", "Charge"]


@dataclass
class PsmRecord:
    psm_id: str
    row_map: dict
    mode: str
    x_exp: np.ndarray | None
    x_theory: np.ndarray | None
    x_cnn: np.ndarray | None
    score: float
    peptide: str
    exp_mass: float | None
    charge: int | None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample PSMs from an ATT or CNN feature pickle and visualize the "
            "top-ranked sampled entries with precursor-centered MS1 and full-range MS2 views."
        )
    )
    parser.add_argument(
        "--input",
        default="data/pct1/Pan_062822_X3iso5.pkl",
        help="Input feature pickle path.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "att", "cnn"],
        default="auto",
        help="Feature pickle mode. Auto-detects by default.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of valid PSMs to sample before ranking.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top-scoring sampled PSMs to select for TSV export; plots at most 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--mz",
        type=float,
        default=10.0,
        help="Half-range around precursor m/z for the MS1 subplot.",
    )
    parser.add_argument(
        "--ppm",
        type=float,
        default=MATCH_PPM,
        help="PPM tolerance used only to highlight ATT matched peaks.",
    )
    parser.add_argument(
        "--output",
        default="test/Pan_062822_X3iso5_top3.pdf",
        help="Output plot path.",
    )
    parser.add_argument(
        "--tsv-output",
        default="",
        help="Output TSV path for selected feature rows. Defaults to the plot path with .tsv suffix.",
    )
    parser.add_argument(
        "--tsv-scope",
        choices=["selected", "sampled", "all"],
        default="selected",
        help="Which PSMs to write to the TSV.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    return parser.parse_args()


def _safe_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _safe_int(value):
    number = _safe_float(value)
    if number is None:
        return None
    charge = int(round(number))
    if charge <= 0:
        return None
    return charge


def _first_nonempty(row_map, candidates):
    for key in candidates:
        value = row_map.get(key, "")
        if str(value).strip():
            return value
    return ""


def _normalize_score(value):
    number = _safe_float(value)
    if number is None:
        return float("-inf")
    return number


def _is_att_entry(model_input):
    if not isinstance(model_input, (list, tuple)) or len(model_input) != 2:
        return False
    try:
        x_exp = np.asarray(model_input[0], dtype=float)
        x_theory = np.asarray(model_input[1], dtype=float)
    except Exception:
        return False
    return (
        x_exp.ndim == 2
        and x_theory.ndim == 2
        and x_exp.shape[1] == 2
        and x_theory.shape[1] == 2
    )


def _is_cnn_entry(model_input):
    if not isinstance(model_input, (list, tuple)) or len(model_input) != 1:
        return False
    try:
        x_features = np.asarray(model_input[0], dtype=float)
    except Exception:
        return False
    return (
        x_features.ndim == 2
        and x_features.shape[0] == CNN_INPUT_CHANNELS
    )


def _entry_mode(model_input):
    if _is_cnn_entry(model_input):
        return "cnn"
    if _is_att_entry(model_input):
        return "att"
    return None


def _load_records(path, requested_mode):
    meta, entries = load_feature_pickle(path)
    records = []

    for psm_id, entry in entries.items():
        model_input = get_entry_model_input(entry)
        if model_input is None:
            continue
        mode = _entry_mode(model_input)
        if mode is None:
            continue
        if requested_mode != "auto" and mode != requested_mode:
            continue

        if mode == "att":
            x_exp = np.asarray(model_input[0], dtype=float)
            x_theory = np.asarray(model_input[1], dtype=float)
            x_cnn = None
        else:
            x_exp = None
            x_theory = None
            x_cnn = np.asarray(model_input[0], dtype=float)

        row_map = get_entry_row_map(meta, psm_id, entry)
        score = _normalize_score(_first_nonempty(row_map, SCORE_COLUMNS))
        peptide = str(_first_nonempty(row_map, PEPTIDE_COLUMNS)).strip()
        exp_mass = _safe_float(_first_nonempty(row_map, EXPMASS_COLUMNS))
        charge = _safe_int(_first_nonempty(row_map, CHARGE_COLUMNS))

        records.append(
            PsmRecord(
                psm_id=str(psm_id),
                row_map=row_map,
                mode=mode,
                x_exp=x_exp,
                x_theory=x_theory,
                x_cnn=x_cnn,
                score=score,
                peptide=peptide,
                exp_mass=exp_mass,
                charge=charge,
            )
        )

    return records


def _compute_precursor_mz(record):
    if record.exp_mass is None:
        raise ValueError(f"{record.psm_id}: missing ExpMass/MeasuredParentMass in pickle row metadata.")
    if record.charge is None:
        raise ValueError(f"{record.psm_id}: missing parentCharges/ParentCharge in pickle row metadata.")
    return record.exp_mass / record.charge + PROTON_MASS


def _sample_records(records, sample_size, seed):
    if len(records) <= sample_size:
        return list(records)
    rng = random.Random(seed)
    return rng.sample(records, sample_size)


def _split_ms1_window(peaks, precursor_mz, mz_half_range):
    if peaks.size == 0:
        return peaks
    low = precursor_mz - mz_half_range
    high = precursor_mz + mz_half_range
    mask = (peaks[:, 0] >= low) & (peaks[:, 0] <= high)
    return peaks[mask]


def _exclude_ms1_window(peaks, precursor_mz, mz_half_range):
    if peaks.size == 0:
        return peaks
    low = precursor_mz - mz_half_range
    high = precursor_mz + mz_half_range
    mask = (peaks[:, 0] < low) | (peaks[:, 0] > high)
    return peaks[mask]


def _plot_peak_series(ax, peaks, color, label, direction=1.0):
    if peaks.size == 0:
        return
    ax.vlines(
        peaks[:, 0],
        0.0,
        direction * peaks[:, 1],
        color=color,
        alpha=0.65,
        linewidth=0.18 if color != "red" else 0.3,
        label=label,
        zorder=3 if color == "red" else 2,
    )


def _find_matched_masks(exp_peaks, theory_peaks, ppm_tolerance=MATCH_PPM):
    exp_mask = np.zeros(exp_peaks.shape[0], dtype=bool)
    theory_mask = np.zeros(theory_peaks.shape[0], dtype=bool)
    if exp_peaks.size == 0 or theory_peaks.size == 0:
        return exp_mask, theory_mask
    for theory_idx, theory_peak in enumerate(theory_peaks):
        tolerance_mz = abs(float(theory_peak[0])) * ppm_tolerance * 1e-6
        matched = np.where(np.abs(exp_peaks[:, 0] - theory_peak[0]) <= tolerance_mz)[0]
        if matched.size > 0:
            exp_mask[matched] = True
            theory_mask[theory_idx] = True
    return exp_mask, theory_mask


def _plot_matched_panel(ax, exp_peaks, theory_peaks, title, ppm_tolerance=MATCH_PPM, direction_labels=True):
    exp_match_mask, theory_match_mask = _find_matched_masks(exp_peaks, theory_peaks, ppm_tolerance)

    _plot_peak_series(ax, exp_peaks[~exp_match_mask], color="tab:blue", label="Real", direction=1.0)
    _plot_peak_series(
        ax,
        theory_peaks[~theory_match_mask],
        color="tab:orange",
        label="Theoretical",
        direction=-1.0,
    )
    _plot_peak_series(ax, exp_peaks[exp_match_mask], color="red", label="Matched", direction=1.0)
    _plot_peak_series(
        ax,
        theory_peaks[theory_match_mask],
        color="red",
        label="Matched",
        direction=-1.0,
    )
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative intensity")
    if direction_labels:
        ax.text(0.01, 0.96, "Real", transform=ax.transAxes, ha="left", va="top", fontsize=8)
        ax.text(0.01, 0.04, "Theoretical", transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
    if exp_peaks.size == 0 and theory_peaks.size == 0:
        ax.text(0.5, 0.5, "No peaks available", transform=ax.transAxes, ha="center")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        dedup = {}
        for handle, label in zip(handles, labels):
            dedup.setdefault(label, handle)
        ax.legend(dedup.values(), dedup.keys(), loc="upper right", fontsize=8)


def _cnn_matched_peak_pairs(x_cnn):
    if x_cnn is None or x_cnn.size == 0:
        empty = np.empty((0, 2), dtype=float)
        return empty, empty
    x_cnn = np.asarray(x_cnn, dtype=float)
    nonzero_mask = np.any(np.abs(x_cnn) > 0, axis=0)
    if not np.any(nonzero_mask):
        empty = np.empty((0, 2), dtype=float)
        return empty, empty

    matched = x_cnn[:, nonzero_mask]
    exp_mz = matched[CNN_EXPMZ_IDX, :]
    theory_mz = exp_mz - matched[CNN_DELTA_MZ_IDX, :]
    exp_intensity = matched[CNN_EXP_INTENSITY_IDX, :]
    theory_intensity = matched[CNN_THEORY_INTENSITY_IDX, :]
    x_exp = np.column_stack([exp_mz, exp_intensity])
    x_theory = np.column_stack([theory_mz, theory_intensity])
    order = np.argsort(x_exp[:, 0])
    return x_exp[order], x_theory[order]


def _cnn_nonzero_column_indices(x_cnn):
    if x_cnn is None or x_cnn.size == 0:
        return np.asarray([], dtype=int)
    x_cnn = np.asarray(x_cnn, dtype=float)
    nonzero_mask = np.any(np.abs(x_cnn) > 0, axis=0)
    return np.where(nonzero_mask)[0]


def _record_nonzero_peak_count(record):
    if record.mode == "cnn":
        return int(_cnn_nonzero_column_indices(record.x_cnn).size)

    exp_peaks = np.asarray(record.x_exp, dtype=float)
    theory_peaks = np.asarray(record.x_theory, dtype=float)
    return int(
        np.count_nonzero(np.any(np.abs(exp_peaks) > 0, axis=1))
        + np.count_nonzero(np.any(np.abs(theory_peaks) > 0, axis=1))
    )


def _nonzero_peak_summary(records):
    counts = np.asarray([_record_nonzero_peak_count(record) for record in records], dtype=float)
    if counts.size == 0:
        return None
    return {
        "min": float(np.min(counts)),
        "max": float(np.max(counts)),
        "median": float(np.median(counts)),
        "average": float(np.mean(counts)),
    }


def _plot_cnn_panel(ax, exp_peaks, theory_peaks, title, direction_labels=True):
    _plot_peak_series(ax, exp_peaks, color="red", label="Matched real", direction=1.0)
    _plot_peak_series(ax, theory_peaks, color="red", label="Matched theoretical", direction=-1.0)
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative intensity")
    if direction_labels:
        ax.text(0.01, 0.96, "Real", transform=ax.transAxes, ha="left", va="top", fontsize=8)
        ax.text(0.01, 0.04, "Theoretical", transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
    if exp_peaks.size == 0 and theory_peaks.size == 0:
        ax.text(0.5, 0.5, "No matched peaks available", transform=ax.transAxes, ha="center")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        dedup = {}
        for handle, label in zip(handles, labels):
            dedup.setdefault(label, handle)
        ax.legend(dedup.values(), dedup.keys(), loc="upper right", fontsize=8)


def _set_full_range_xlim(ax, arrays):
    non_empty = [arr for arr in arrays if arr.size > 0]
    if not non_empty:
        ax.set_xlim(0.0, 1.0)
        return
    min_mz = min(float(arr[:, 0].min()) for arr in non_empty)
    max_mz = max(float(arr[:, 0].max()) for arr in non_empty)
    if math.isclose(min_mz, max_mz):
        pad = max(1.0, 0.02 * max_mz)
        ax.set_xlim(min_mz - pad, max_mz + pad)
    else:
        pad = max(1.0, 0.02 * (max_mz - min_mz))
        ax.set_xlim(min_mz - pad, max_mz + pad)


def _format_score(score):
    if math.isfinite(score):
        return f"{score:.4f}"
    return "NA"


def _truncate_text(text, max_len=28):
    text = str(text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_tsv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.10g}"
        return ""
    return str(value)


def _default_tsv_output_path(output_path):
    return output_path.with_suffix(".tsv")


def _write_feature_tsv(records, precursor_values, output_path):
    fieldnames = [
        "record_rank",
        "psm_id",
        "mode",
        "score",
        "peptide",
        "exp_mass",
        "charge",
        "precursor_mz",
        "non_zero_peak_count",
        "feature_index",
        "peak_kind",
        "expmz",
        "theory_mz",
        "delta_mz",
        "mono_mz",
        "experimental_intensity",
        "theoretical_intensity",
        "SIPatomNumber",
        "EnrichRatio",
        "IonCharge",
        "IsotopeIndex",
        "MSLevelFlag",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for record_rank, (record, precursor_mz) in enumerate(zip(records, precursor_values), start=1):
            base_row = {
                "record_rank": record_rank,
                "psm_id": record.psm_id,
                "mode": record.mode,
                "score": _format_tsv_value(record.score if math.isfinite(record.score) else None),
                "peptide": record.peptide,
                "exp_mass": _format_tsv_value(record.exp_mass),
                "charge": _format_tsv_value(record.charge),
                "precursor_mz": _format_tsv_value(precursor_mz),
            }

            if record.mode == "cnn":
                x_cnn = np.asarray(record.x_cnn, dtype=float)
                nonzero_indices = _cnn_nonzero_column_indices(x_cnn)
                non_zero_peak_count = _record_nonzero_peak_count(record)
                for feature_index in nonzero_indices:
                    feature = x_cnn[:, feature_index]
                    expmz = float(feature[CNN_EXPMZ_IDX])
                    delta_mz = float(feature[CNN_DELTA_MZ_IDX])
                    row = dict(base_row)
                    row.update(
                        {
                            "non_zero_peak_count": non_zero_peak_count,
                            "feature_index": int(feature_index),
                            "peak_kind": "matched",
                            "expmz": _format_tsv_value(expmz),
                            "theory_mz": _format_tsv_value(expmz - delta_mz),
                            "delta_mz": _format_tsv_value(delta_mz),
                            "mono_mz": _format_tsv_value(float(feature[CNN_MONO_MZ_IDX])),
                            "experimental_intensity": _format_tsv_value(float(feature[CNN_EXP_INTENSITY_IDX])),
                            "theoretical_intensity": _format_tsv_value(float(feature[CNN_THEORY_INTENSITY_IDX])),
                            "SIPatomNumber": _format_tsv_value(float(feature[CNN_SIP_ATOM_NUMBER_IDX])),
                            "EnrichRatio": _format_tsv_value(float(feature[CNN_ENRICH_RATIO_IDX])),
                            "IonCharge": _format_tsv_value(float(feature[CNN_ION_CHARGE_IDX])),
                            "IsotopeIndex": _format_tsv_value(float(feature[CNN_ISOTOPE_INDEX_IDX])),
                            "MSLevelFlag": _format_tsv_value(float(feature[CNN_MS_LEVEL_FLAG_IDX])),
                        }
                    )
                    writer.writerow(row)
                    row_count += 1
                continue

            exp_peaks = np.asarray(record.x_exp, dtype=float)
            theory_peaks = np.asarray(record.x_theory, dtype=float)
            non_zero_peak_count = _record_nonzero_peak_count(record)
            for peak_kind, peaks in (("exp", exp_peaks), ("theory", theory_peaks)):
                for feature_index, peak in enumerate(peaks):
                    if not np.any(np.abs(peak) > 0):
                        continue
                    row = dict(base_row)
                    row.update(
                        {
                            "non_zero_peak_count": non_zero_peak_count,
                            "feature_index": feature_index,
                            "peak_kind": peak_kind,
                            "expmz": _format_tsv_value(float(peak[0]) if peak_kind == "exp" else None),
                            "theory_mz": _format_tsv_value(float(peak[0]) if peak_kind == "theory" else None),
                            "experimental_intensity": _format_tsv_value(float(peak[1]) if peak_kind == "exp" else None),
                            "theoretical_intensity": _format_tsv_value(float(peak[1]) if peak_kind == "theory" else None),
                        }
                    )
                    writer.writerow(row)
                    row_count += 1
    return row_count


def _plot_record(ms1_ax, ms2_ax, record, precursor_mz, mz_half_range, ppm_tolerance):
    if record.mode == "cnn":
        x_exp, x_theory = _cnn_matched_peak_pairs(record.x_cnn)
        panel_plotter = _plot_cnn_panel
        ms2_title = "MS2 Matched Peaks"
    else:
        x_exp = record.x_exp
        x_theory = record.x_theory
        panel_plotter = None
        ms2_title = "MS2 Full Range"

    x_exp_ms1 = _split_ms1_window(x_exp, precursor_mz, mz_half_range)
    x_theory_ms1 = _split_ms1_window(x_theory, precursor_mz, mz_half_range)
    x_exp_ms2 = _exclude_ms1_window(x_exp, precursor_mz, mz_half_range)
    x_theory_ms2 = _exclude_ms1_window(x_theory, precursor_mz, mz_half_range)

    if record.mode == "cnn":
        panel_plotter(ms1_ax, x_exp_ms1, x_theory_ms1, "MS1 Matched Peaks")
    else:
        _plot_matched_panel(ms1_ax, x_exp_ms1, x_theory_ms1, "MS1", ppm_tolerance)
    ms1_ax.set_xlim(precursor_mz - mz_half_range, precursor_mz + mz_half_range)
    if x_exp_ms1.size == 0 and x_theory_ms1.size == 0:
        ms1_ax.text(0.5, 0.5, "No peaks in window", transform=ms1_ax.transAxes, ha="center")

    if record.mode == "cnn":
        panel_plotter(ms2_ax, x_exp_ms2, x_theory_ms2, ms2_title)
    else:
        _plot_matched_panel(ms2_ax, x_exp_ms2, x_theory_ms2, ms2_title, ppm_tolerance)
    _set_full_range_xlim(ms2_ax, [x_exp_ms2, x_theory_ms2])

    peptide_text = _truncate_text(record.peptide or "NA", max_len=26)
    row_title = (
        f"{record.psm_id} | {record.mode.upper()} | pep={peptide_text} | score={_format_score(record.score)} | "
        f"z={record.charge} | prec={precursor_mz:.4f}"
    )
    ms1_ax.text(0.0, 1.08, row_title, transform=ms1_ax.transAxes, fontsize=10, va="bottom")


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    tsv_output_path = Path(args.tsv_output).resolve() if args.tsv_output else _default_tsv_output_path(output_path)

    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be positive.")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive.")
    if args.mz <= 0:
        raise SystemExit("--mz must be positive.")
    if args.ppm <= 0:
        raise SystemExit("--ppm must be positive.")

    try:
        records = _load_records(input_path, args.mode)
    except FileNotFoundError:
        raise SystemExit(f"Input pickle not found: {input_path}")
    except Exception as exc:
        raise SystemExit(f"Failed to load pickle {input_path}: {exc}")

    if not records:
        mode_text = "ATT/CNN" if args.mode == "auto" else args.mode.upper()
        raise SystemExit(f"No valid {mode_text} entries found in {input_path}")

    pkl_nonzero_summary = _nonzero_peak_summary(records)
    sampled = _sample_records(records, args.sample_size, args.seed)
    ranked = sorted(sampled, key=lambda record: (record.score, record.psm_id), reverse=True)
    selected = ranked[: min(args.top_k, len(ranked))]
    plotted = selected[: min(MAX_PLOT_RECORDS, len(selected))]

    if not selected:
        raise SystemExit("No sampled PSM entries available for plotting.")

    precursor_values = []
    for record in plotted:
        precursor_values.append(_compute_precursor_mz(record))

    if args.tsv_scope == "all":
        tsv_records = records
    elif args.tsv_scope == "sampled":
        tsv_records = sampled
    else:
        tsv_records = selected
    tsv_precursor_values = [_compute_precursor_mz(record) for record in tsv_records]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(plotted),
        2,
        figsize=(22, 6.4 * len(plotted)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.35, 2.15]},
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.05, hspace=0.42, wspace=0.18)

    for idx, record in enumerate(plotted):
        _plot_record(
            axes[idx][0],
            axes[idx][1],
            record,
            precursor_values[idx],
            args.mz,
            args.ppm,
        )

    fig.suptitle(
        f"Top {len(plotted)} plotted PSMs from {input_path.name} "
        f"(selected={len(selected)}, mode={args.mode}, sample_size={len(sampled)}, seed={args.seed})",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    tsv_row_count = _write_feature_tsv(tsv_records, tsv_precursor_values, tsv_output_path)

    print(f"input_file={input_path}")
    print(f"valid_entries={len(records)}")
    if pkl_nonzero_summary is not None:
        print(
            "pkl_non_zero_peak_count_"
            f"min={_format_tsv_value(pkl_nonzero_summary['min'])}\t"
            f"max={_format_tsv_value(pkl_nonzero_summary['max'])}\t"
            f"median={_format_tsv_value(pkl_nonzero_summary['median'])}\t"
            f"average={_format_tsv_value(pkl_nonzero_summary['average'])}"
        )
    print(f"sampled_entries={len(sampled)}")
    print(f"selected_entries={len(selected)}")
    print(f"plotted_entries={len(plotted)}")
    print(f"tsv_scope={args.tsv_scope}")
    print(f"tsv_feature_rows={tsv_row_count}")
    for record, precursor_mz in zip(plotted, precursor_values):
        print(
            f"plotted_psm={record.psm_id}\t"
            f"score={_format_score(record.score)}\t"
            f"charge={record.charge}\t"
            f"ExpMass={record.exp_mass:.6f}\t"
            f"precursor_mz={precursor_mz:.6f}"
        )
    print(f"output_plot={output_path}")
    print(f"output_tsv={tsv_output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
