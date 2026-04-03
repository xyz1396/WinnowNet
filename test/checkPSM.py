import argparse
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
MATCH_TOLERANCE = 0.01
PEPTIDE_COLUMNS = ["Peptide", "IdentifiedPeptide", "OriginalPeptide"]
SCORE_COLUMNS = ["score", "Score"]
EXPMASS_COLUMNS = ["ExpMass", "MeasuredParentMass"]
CHARGE_COLUMNS = ["parentCharges", "ParentCharge", "charge", "Charge"]


@dataclass
class PsmRecord:
    psm_id: str
    row_map: dict
    x_exp: np.ndarray
    x_theory: np.ndarray
    score: float
    peptide: str
    exp_mass: float | None
    charge: int | None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample PSMs from an ATT feature pickle and visualize the "
            "top-ranked sampled entries with precursor-centered MS1 and full-range MS2 views."
        )
    )
    parser.add_argument(
        "--input",
        default="data/pct1/Pan_062822_X3iso5.pkl",
        help="Input ATT feature pickle path.",
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
        help="Number of top-scoring sampled PSMs to visualize.",
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
        "--output",
        default="test/Pan_062822_X3iso5_top3.pdf",
        help="Output plot path.",
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


def _load_records(path):
    meta, entries = load_feature_pickle(path)
    records = []

    for psm_id, entry in entries.items():
        model_input = get_entry_model_input(entry)
        if model_input is None or not _is_att_entry(model_input):
            continue

        x_exp = np.asarray(model_input[0], dtype=float)
        x_theory = np.asarray(model_input[1], dtype=float)
        row_map = get_entry_row_map(meta, psm_id, entry)
        score = _normalize_score(_first_nonempty(row_map, SCORE_COLUMNS))
        peptide = str(_first_nonempty(row_map, PEPTIDE_COLUMNS)).strip()
        exp_mass = _safe_float(_first_nonempty(row_map, EXPMASS_COLUMNS))
        charge = _safe_int(_first_nonempty(row_map, CHARGE_COLUMNS))

        records.append(
            PsmRecord(
                psm_id=str(psm_id),
                row_map=row_map,
                x_exp=x_exp,
                x_theory=x_theory,
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


def _find_matched_masks(exp_peaks, theory_peaks, tolerance=MATCH_TOLERANCE):
    exp_mask = np.zeros(exp_peaks.shape[0], dtype=bool)
    theory_mask = np.zeros(theory_peaks.shape[0], dtype=bool)
    if exp_peaks.size == 0 or theory_peaks.size == 0:
        return exp_mask, theory_mask
    for theory_idx, theory_peak in enumerate(theory_peaks):
        matched = np.where(np.abs(exp_peaks[:, 0] - theory_peak[0]) <= tolerance)[0]
        if matched.size > 0:
            exp_mask[matched] = True
            theory_mask[theory_idx] = True
    return exp_mask, theory_mask


def _plot_matched_panel(ax, exp_peaks, theory_peaks, title, direction_labels=True):
    exp_match_mask, theory_match_mask = _find_matched_masks(exp_peaks, theory_peaks)

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


def _plot_record(ms1_ax, ms2_ax, record, precursor_mz, mz_half_range):
    x_exp_ms1 = _split_ms1_window(record.x_exp, precursor_mz, mz_half_range)
    x_theory_ms1 = _split_ms1_window(record.x_theory, precursor_mz, mz_half_range)
    x_exp_ms2 = _exclude_ms1_window(record.x_exp, precursor_mz, mz_half_range)
    x_theory_ms2 = _exclude_ms1_window(record.x_theory, precursor_mz, mz_half_range)

    _plot_matched_panel(ms1_ax, x_exp_ms1, x_theory_ms1, "MS1")
    ms1_ax.set_xlim(precursor_mz - mz_half_range, precursor_mz + mz_half_range)
    if x_exp_ms1.size == 0 and x_theory_ms1.size == 0:
        ms1_ax.text(0.5, 0.5, "No peaks in window", transform=ms1_ax.transAxes, ha="center")

    _plot_matched_panel(ms2_ax, x_exp_ms2, x_theory_ms2, "MS2 Full Range")
    _set_full_range_xlim(ms2_ax, [x_exp_ms2, x_theory_ms2])

    peptide_text = _truncate_text(record.peptide or "NA", max_len=26)
    row_title = (
        f"{record.psm_id} | pep={peptide_text} | score={_format_score(record.score)} | "
        f"z={record.charge} | prec={precursor_mz:.4f}"
    )
    ms1_ax.text(0.0, 1.08, row_title, transform=ms1_ax.transAxes, fontsize=10, va="bottom")


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be positive.")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive.")
    if args.mz <= 0:
        raise SystemExit("--mz must be positive.")

    try:
        records = _load_records(input_path)
    except FileNotFoundError:
        raise SystemExit(f"Input pickle not found: {input_path}")
    except Exception as exc:
        raise SystemExit(f"Failed to load pickle {input_path}: {exc}")

    if not records:
        raise SystemExit(f"No valid ATT entries with [Xexp, Xtheory] found in {input_path}")

    sampled = _sample_records(records, args.sample_size, args.seed)
    ranked = sorted(sampled, key=lambda record: (record.score, record.psm_id), reverse=True)
    selected = ranked[: min(args.top_k, len(ranked))]

    if not selected:
        raise SystemExit("No sampled PSM entries available for plotting.")

    precursor_values = []
    for record in selected:
        precursor_values.append(_compute_precursor_mz(record))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(selected),
        2,
        figsize=(22, 6.4 * len(selected)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.35, 2.15]},
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.05, hspace=0.42, wspace=0.18)

    for idx, record in enumerate(selected):
        _plot_record(
            axes[idx][0],
            axes[idx][1],
            record,
            precursor_values[idx],
            args.mz,
        )

    fig.suptitle(
        f"Top {len(selected)} sampled PSMs from {input_path.name} "
        f"(sample_size={len(sampled)}, seed={args.seed})",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"input_file={input_path}")
    print(f"valid_entries={len(records)}")
    print(f"sampled_entries={len(sampled)}")
    print(f"selected_entries={len(selected)}")
    for record, precursor_mz in zip(selected, precursor_values):
        print(
            f"selected_psm={record.psm_id}\t"
            f"score={_format_score(record.score)}\t"
            f"charge={record.charge}\t"
            f"ExpMass={record.exp_mass:.6f}\t"
            f"precursor_mz={precursor_mz:.6f}"
        )
    print(f"output_plot={output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
