import csv
import getopt
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

from WinnowNet_CNN import MODEL_ARCH_PURE_CNN_PCT, build_cnn_model, resolve_checkpoint_model_arch
from checkpoint_utils import load_checkpoint_bundle
from pkl_utils import (
    choose_output_column,
    expand_pickle_inputs,
    format_label_value,
    get_entry_model_input,
    get_entry_row_index,
    get_entry_row_map,
    load_feature_pickle,
    normalize_long_flag_aliases,
)

DEFAULT_BATCH_SIZE = 1024
DEFAULT_DECISION_THRESHOLD = 0.5
CNN_INPUT_CHANNELS = 7
CNN_FEATURE_SCHEMA = "cnn_7ch_v1"
CNN_ENRICH_RATIO_CHANNEL = 6
MS2_ISOTOPIC_ABUNDANCE_COLUMN = "MS2IsotopicAbundances"
MS2_ENRICH_RATIO_MEDIAN_COLUMN = "MS2isotopicAbundanceEvolopeMedian"
PREDICTED_13C_PCT_COLUMN = "Predicted13CPct"


def _format_pct_bucket(pct_label):
    try:
        pct_value = float(pct_label)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(pct_value):
        return ""
    return f"{pct_value:g}"


def _validate_cnn_features(features, label="xFeatures"):
    x_features = np.asarray(features, dtype=float)
    if x_features.ndim != 2:
        raise ValueError(
            f"{label} must be a 2D CNN feature tensor, got ndim={x_features.ndim}."
        )
    if x_features.shape[0] != CNN_INPUT_CHANNELS:
        raise ValueError(
            f"{label} must have {CNN_INPUT_CHANNELS} channels in the first dimension, "
            f"got shape={x_features.shape}. Regenerate CNN features with the 7-channel "
            "SpectraFeatures.py schema; legacy 3-channel CNN features are not supported."
        )
    return x_features


def _extract_cnn_model_features(model_input, label="xFeatures"):
    if not isinstance(model_input, (list, tuple)) or len(model_input) != 1:
        raise ValueError(
            f"{label}: CNN model_input must be [xFeatures] for the 7-channel schema."
        )
    return _validate_cnn_features(model_input[0], label)


def _format_feature_value(value):
    if value == "":
        return ""
    if not np.isfinite(float(value)):
        return ""
    return f"{float(value):.10g}"


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


class DefineDataset(Data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx])


def _score_to_string(score):
    return f"{float(score):.10g}"


def _split_model_output(output):
    if isinstance(output, (tuple, list)):
        if len(output) != 2:
            raise ValueError(f"Expected CNN model output tuple of length 2, got {len(output)}.")
        return output[0], output[1]
    return output, None


def _qvalue_to_string(qvalue):
    if qvalue == "":
        return ""
    return f"{float(qvalue):.10g}"


def _load_checkpoint_weights(model_path, expected_model_arch=None):
    state_dict, metadata = load_checkpoint_bundle(model_path)
    if int(metadata.get("input_channels", 0)) != CNN_INPUT_CHANNELS:
        raise ValueError(
            f"Checkpoint {model_path} is not compatible with the 7-channel CNN. "
            f"Expected metadata input_channels={CNN_INPUT_CHANNELS}, got "
            f"{metadata.get('input_channels')!r}. Old 3-channel checkpoints must be retrained."
        )
    if metadata.get("feature_schema") != CNN_FEATURE_SCHEMA:
        raise ValueError(
            f"Checkpoint {model_path} is not compatible with the 7-channel CNN. "
            f"Expected feature_schema={CNN_FEATURE_SCHEMA!r}, got "
            f"{metadata.get('feature_schema')!r}. Old 3-channel checkpoints must be retrained."
        )
    checkpoint_model_arch = resolve_checkpoint_model_arch(metadata)
    if expected_model_arch is not None and checkpoint_model_arch != expected_model_arch:
        raise ValueError(
            f"Checkpoint {model_path} was trained with model_arch={checkpoint_model_arch!r}, "
            f"but prediction constructed {expected_model_arch!r}."
        )
    return state_dict


def _load_model_state_dict(model, model_path, expected_model_arch=None):
    try:
        model.load_state_dict(_load_checkpoint_weights(model_path, expected_model_arch))
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load CNN checkpoint {model_path}. This model expects "
            f"{CNN_INPUT_CHANNELS}-channel inputs and is incompatible with old "
            "3-channel CNN checkpoints or checkpoints from a different CNN architecture; "
            "retrain the CNN checkpoint with regenerated 7-channel features and the requested architecture."
        ) from exc


def _load_checkpoint_metadata(model_path):
    _, metadata = load_checkpoint_bundle(model_path)
    return metadata


def _parse_probability(value, flag_name):
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"{flag_name} must be a float between 0 and 1.")
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{flag_name} must be between 0 and 1.")
    return parsed


def _parse_nonnegative_float(value, flag_name):
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"{flag_name} must be a non-negative number.")
    if parsed < 0.0:
        raise ValueError(f"{flag_name} must be non-negative.")
    return parsed


def _parse_positive_int(value, flag_name):
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"{flag_name} must be a positive integer.")
    if parsed <= 0:
        raise ValueError(f"{flag_name} must be a positive integer.")
    return parsed


def _compute_qvalues(scores, labels):
    qvalues = [""] * len(scores)
    labeled_indices = [
        idx for idx, label in enumerate(labels) if label in (0, 1) and scores[idx] is not None
    ]
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


def _threshold_for_pct(pct_value, pct_decision_thresholds, default_threshold):
    if not pct_decision_thresholds:
        return default_threshold
    pct_key = _format_pct_bucket(pct_value)
    threshold_entry = pct_decision_thresholds.get(pct_key)
    if threshold_entry is None:
        try:
            numeric_pct = float(pct_value)
            closest_key = min(
                pct_decision_thresholds,
                key=lambda key: abs(float(key) - numeric_pct),
            )
            threshold_entry = pct_decision_thresholds[closest_key]
        except (TypeError, ValueError):
            return default_threshold
    if isinstance(threshold_entry, dict):
        return float(threshold_entry.get("threshold", default_threshold))
    return float(threshold_entry)


def _predict_scores(
    model,
    feature_batches,
    device,
    batch_size,
    decision_threshold,
    predict_pct=False,
    pct_decision_thresholds=None,
    known_pct=None,
):
    test_data = DefineDataset(feature_batches)
    test_loader = Data.DataLoader(test_data, batch_size=batch_size)
    model.eval()

    y_pred_prob = []
    y_pred = []
    predicted_pct = [] if predict_pct else None
    with torch.no_grad():
        for data1 in test_loader:
            data1 = Variable(data1).to(device)
            output = model(data1)
            logits, pred_log_pct = _split_model_output(output)
            pred_prob = torch.softmax(logits.data, dim=1).cpu().numpy()
            positive_scores = pred_prob[:, 1]
            y_pred_prob.extend(positive_scores.tolist())
            if predict_pct:
                if pred_log_pct is None:
                    raise ValueError("Checkpoint requires pct prediction, but the model did not return a pct output.")
                pct_values = torch.clamp(torch.expm1(pred_log_pct.data), min=0.0).view(-1).cpu().numpy()
                predicted_pct.extend(pct_values.tolist())
            else:
                pct_values = [known_pct] * len(positive_scores)
            if pct_decision_thresholds:
                if known_pct is not None:
                    thresholds = np.asarray(
                        [_threshold_for_pct(known_pct, pct_decision_thresholds, decision_threshold)] * len(positive_scores),
                        dtype=float,
                    )
                else:
                    thresholds = np.asarray(
                        [_threshold_for_pct(pct_value, pct_decision_thresholds, decision_threshold) for pct_value in pct_values],
                        dtype=float,
                    )
                y_pred.extend((positive_scores >= thresholds).astype(int).tolist())
            else:
                y_pred.extend((positive_scores >= decision_threshold).astype(int).tolist())
    return y_pred_prob, y_pred, predicted_pct


def _default_output_file(input_file):
    input_dir = os.path.dirname(os.path.abspath(input_file))
    input_stem = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(input_dir, input_stem + "_SIP.tsv")


def _is_combined_output_file(output_arg):
    return bool(output_arg) and os.path.splitext(os.path.basename(output_arg.rstrip(os.sep)))[1].lower() in {
        ".tsv",
        ".txt",
        ".csv",
    }


def _resolve_output_file(input_file, output_arg):
    if not output_arg:
        return _default_output_file(input_file)
    if _is_combined_output_file(output_arg):
        return output_arg
    output_dir = output_arg
    os.makedirs(output_dir, exist_ok=True)
    input_stem = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(output_dir, input_stem + "_SIP.tsv")


def _load_prediction_rows(input_file):
    meta, entries = load_feature_pickle(input_file)
    ordered_items = sorted(
        entries.items(),
        key=lambda kv: get_entry_row_index(kv[1], 0),
    )

    feature_batches = []
    feature_keys = []
    for key, entry in ordered_items:
        model_input = get_entry_model_input(entry)
        if model_input is None:
            continue
        x_features = _extract_cnn_model_features(model_input, f"{key}: xFeatures")
        feature_keys.append(key)
        feature_batches.append(x_features)

    return meta, ordered_items, feature_keys, feature_batches


def _make_rescored_rows(
    meta,
    ordered_items,
    score_map,
    predicted_label_map,
    predicted_pct_map=None,
    source_file=None,
):
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


def _get_output_columns(metas, include_predicted_pct=False):
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
    if include_predicted_pct and PREDICTED_13C_PCT_COLUMN not in output_columns:
        output_columns.append(PREDICTED_13C_PCT_COLUMN)
    return output_columns, score_column, qvalue_column, label_column


def _write_rescored_rows(
    output_file,
    rescored_rows,
    output_columns,
    score_column,
    qvalue_column,
    label_column,
    predicted_pct_column=None,
):
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
            label_text = format_label_value(row["label"], None)
            if label_text != "" or label_column not in row_map:
                row_map[label_column] = label_text
            if predicted_pct_column is not None:
                predicted_pct = row.get("predicted_pct")
                row_map[predicted_pct_column] = "" if predicted_pct is None else _score_to_string(predicted_pct)
            if MS2_ENRICH_RATIO_MEDIAN_COLUMN in output_columns:
                row_map[MS2_ENRICH_RATIO_MEDIAN_COLUMN] = _get_entry_enrich_ratio_median(entry)
            writer.writerow({column: row_map.get(column, "") for column in output_columns})


if __name__ == "__main__":
    argv = normalize_long_flag_aliases(
        sys.argv[1:],
        {
            "-batch-size": "--batch-size",
            "-decision-threshold": "--decision-threshold",
            "-known-pct": "--known-pct",
        },
    )
    try:
        opts, args = getopt.getopt(argv, "hi:m:o:", ["batch-size=", "decision-threshold=", "known-pct="])
    except Exception:
        print("Error Option, using -h for help information.")
        sys.exit(1)
    if len(opts) == 0:
        print("\n\nUsage:\n")
        print("-i\t input representation pickle file(s), directory, glob, comma-separated, or repeated\n")
        print("-m\t Pre-trained model name\n")
        print("-o\t Output TSV file; for multiple inputs, a .tsv/.txt/.csv path writes one combined file, otherwise this is an output directory\n")
        print("--batch-size\t Prediction batch size (default: " + str(DEFAULT_BATCH_SIZE) + ")\n")
        print("--decision-threshold\t Predict target when p(target) >= threshold (default: best value from checkpoint)\n")
        print("--known-pct\t Known 13C pct for all input rows; pure_cnn_pct uses this to choose a pct-bin threshold\n")
        sys.exit(1)

    input_values = []
    model_name = ""
    output_file = ""
    batch_size = DEFAULT_BATCH_SIZE
    decision_threshold = None
    known_pct = None
    for opt, arg in opts:
        if opt in ("-h"):
            print("\n\nUsage:\n")
            print("-i\t input representation pickle file(s), directory, glob, comma-separated, or repeated\n")
            print("-m\t Pre-trained model name\n")
            print("-o\t Output TSV file; for multiple inputs, a .tsv/.txt/.csv path writes one combined file, otherwise this is an output directory\n")
            print("--batch-size\t Prediction batch size (default: " + str(DEFAULT_BATCH_SIZE) + ")\n")
            print("--decision-threshold\t Predict target when p(target) >= threshold (default: best value from checkpoint)\n")
            print("--known-pct\t Known 13C pct for all input rows; pure_cnn_pct uses this to choose a pct-bin threshold\n")
            sys.exit(1)
        elif opt in ("-i"):
            input_values.append(arg)
        elif opt in ("-m"):
            model_name = arg
        elif opt in ("-o"):
            output_file = arg
        elif opt == "--batch-size":
            batch_size = _parse_positive_int(arg, "--batch-size")
        elif opt == "--decision-threshold":
            decision_threshold = _parse_probability(arg, "--decision-threshold")
        elif opt == "--known-pct":
            known_pct = _parse_nonnegative_float(arg, "--known-pct")

    input_files = expand_pickle_inputs(input_values)
    if len(input_files) == 0:
        raise ValueError("No input pickle files found. Use -i with a .pkl file, directory, glob, or comma-separated list.")
    if len(model_name) == 0:
        raise ValueError("Use -m to provide a CNN checkpoint.")
    checkpoint_metadata = _load_checkpoint_metadata(model_name)
    model_arch = resolve_checkpoint_model_arch(checkpoint_metadata)
    predict_pct = model_arch == MODEL_ARCH_PURE_CNN_PCT
    print("Using CNN model architecture from checkpoint: " + model_arch)
    if "trainable_parameter_count" in checkpoint_metadata:
        print("Checkpoint trainable parameters: " + str(checkpoint_metadata["trainable_parameter_count"]))
    best_threshold = float(checkpoint_metadata["best_decision_threshold"])
    pct_decision_thresholds = checkpoint_metadata.get("pct_decision_thresholds", {}) if predict_pct else {}
    if decision_threshold is None:
        decision_threshold = best_threshold
        if pct_decision_thresholds:
            if known_pct is None:
                print(
                    "Using pct-bin decision thresholds from checkpoint with model-predicted pct bins; "
                    "fallback global threshold: " + str(decision_threshold)
                )
            else:
                known_threshold = _threshold_for_pct(known_pct, pct_decision_thresholds, decision_threshold)
                print(
                    "Using pct-bin decision threshold from checkpoint for known pct "
                    + str(known_pct)
                    + ": "
                    + str(known_threshold)
                )
        else:
            print("Using best decision threshold from checkpoint: " + str(decision_threshold))
    elif abs(decision_threshold - best_threshold) > 1e-9:
        print("WARNING: --decision-threshold " + str(decision_threshold) + " differs from checkpoint best " + str(best_threshold))
        pct_decision_thresholds = {}

    device = torch.device("cuda")
    model = build_cnn_model(model_arch)
    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)
    _load_model_state_dict(model, model_name, model_arch)

    combined_output = len(input_files) > 1 and _is_combined_output_file(output_file)
    combined_rows = []
    combined_metas = []
    for input_file in input_files:
        meta, ordered_items, feature_keys, feature_batches = _load_prediction_rows(input_file)
        if len(meta.get("columns", [])) == 0:
            raise ValueError(f"Input pickle {input_file} does not contain the original TSV/PIN row metadata.")
        predicted_scores, predicted_labels, predicted_pct = _predict_scores(
            model,
            feature_batches,
            device,
            batch_size,
            decision_threshold,
            predict_pct,
            pct_decision_thresholds,
            known_pct,
        )
        score_map = {key: score for key, score in zip(feature_keys, predicted_scores)}
        predicted_label_map = {key: label for key, label in zip(feature_keys, predicted_labels)}
        predicted_pct_map = None if predicted_pct is None else {
            key: pct for key, pct in zip(feature_keys, predicted_pct)
        }
        rescored_rows = _make_rescored_rows(
            meta,
            ordered_items,
            score_map,
            predicted_label_map,
            predicted_pct_map,
            source_file=input_file,
        )
        if combined_output:
            combined_rows.extend(rescored_rows)
            combined_metas.append(meta)
        else:
            output_path = _resolve_output_file(input_file, output_file)
            output_columns, score_column, qvalue_column, label_column = _get_output_columns([meta], predict_pct)
            print("Writing rescored TSV: " + output_path)
            _write_rescored_rows(
                output_path,
                rescored_rows,
                output_columns,
                score_column,
                qvalue_column,
                label_column,
                PREDICTED_13C_PCT_COLUMN if predict_pct else None,
            )
    if combined_output:
        labels = [row["label"] for row in combined_rows]
        scores = [row["score"] for row in combined_rows]
        qvalues = _compute_qvalues(scores, labels)
        for row, qvalue in zip(combined_rows, qvalues):
            row["qvalue"] = qvalue
        output_columns, score_column, qvalue_column, label_column = _get_output_columns(
            combined_metas,
            predict_pct,
        )
        print("Writing combined rescored TSV: " + output_file)
        _write_rescored_rows(
            output_file,
            combined_rows,
            output_columns,
            score_column,
            qvalue_column,
            label_column,
            PREDICTED_13C_PCT_COLUMN if predict_pct else None,
        )
    print("done")
