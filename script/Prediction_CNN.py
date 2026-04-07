import csv
import getopt
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

from WinnowNet_CNN import build_cnn_model, resolve_checkpoint_model_arch
from checkpoint_utils import load_checkpoint_bundle
from pkl_utils import (
    choose_output_column,
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


class DefineDataset(Data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx])


def _score_to_string(score):
    return f"{float(score):.10g}"


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


def _predict_scores(model, model_name, feature_batches, device, batch_size, decision_threshold, model_arch):
    test_data = DefineDataset(feature_batches)
    test_loader = Data.DataLoader(test_data, batch_size=batch_size)
    _load_model_state_dict(model, model_name, model_arch)
    model.eval()

    y_pred_prob = []
    y_pred = []
    for data1 in test_loader:
        data1 = Variable(data1).to(device)
        output = model(data1)
        pred_prob = torch.softmax(output.data, dim=1).cpu().numpy()
        positive_scores = pred_prob[:, 1]
        y_pred.extend((positive_scores >= decision_threshold).astype(int).tolist())
        y_pred_prob.extend(positive_scores.tolist())
    return y_pred_prob, y_pred


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


def _write_rescored_output(output_file, meta, ordered_items, score_map, predicted_label_map):
    base_columns = list(meta.get("columns", []))
    score_column = choose_output_column(base_columns, ["score"], "score")
    qvalue_column = choose_output_column(base_columns, ["q-value", "qvalue"], "q-value")
    label_column = choose_output_column(base_columns, ["Label"], "Label")

    output_columns = list(base_columns)
    for column in (score_column, qvalue_column, label_column):
        if column not in output_columns:
            output_columns.append(column)

    labels = [predicted_label_map.get(key) for key, _ in ordered_items]
    scores = [score_map.get(key) for key, _ in ordered_items]
    qvalues = _compute_qvalues(scores, labels)
    ranked_rows = []
    for idx, (key, entry) in enumerate(ordered_items):
        ranked_rows.append(
            {
                "sort_score": float("-inf") if scores[idx] is None else float(scores[idx]),
                "row_index": idx,
                "key": key,
                "entry": entry,
                "score": scores[idx],
                "qvalue": qvalues[idx],
                "label": labels[idx],
            }
        )
    ranked_rows.sort(key=lambda item: (-item["sort_score"], item["row_index"]))

    with open(output_file, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=output_columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in ranked_rows:
            key = row["key"]
            entry = row["entry"]
            row_map = get_entry_row_map(meta, key, entry)
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
            writer.writerow({column: row_map.get(column, "") for column in output_columns})


if __name__ == "__main__":
    argv = normalize_long_flag_aliases(
        sys.argv[1:],
        {
            "-batch-size": "--batch-size",
            "-decision-threshold": "--decision-threshold",
        },
    )
    try:
        opts, args = getopt.getopt(argv, "hi:m:o:", ["batch-size=", "decision-threshold="])
    except Exception:
        print("Error Option, using -h for help information.")
        sys.exit(1)
    if len(opts) == 0:
        print("\n\nUsage:\n")
        print("-i\t input representation file name\n")
        print("-m\t Pre-trained model name\n")
        print("-o\t Output rescored TSV file\n")
        print("--batch-size\t Prediction batch size (default: " + str(DEFAULT_BATCH_SIZE) + ")\n")
        print("--decision-threshold\t Predict target when p(target) >= threshold (default: best value from checkpoint)\n")
        sys.exit(1)

    input_file = ""
    model_name = ""
    output_file = ""
    batch_size = DEFAULT_BATCH_SIZE
    decision_threshold = None
    for opt, arg in opts:
        if opt in ("-h"):
            print("\n\nUsage:\n")
            print("-i\t input representation file name\n")
            print("-m\t Pre-trained model name\n")
            print("-o\t Output rescored TSV file\n")
            print("--batch-size\t Prediction batch size (default: " + str(DEFAULT_BATCH_SIZE) + ")\n")
            print("--decision-threshold\t Predict target when p(target) >= threshold (default: best value from checkpoint)\n")
            sys.exit(1)
        elif opt in ("-i"):
            input_file = arg
        elif opt in ("-m"):
            model_name = arg
        elif opt in ("-o"):
            output_file = arg
        elif opt == "--batch-size":
            batch_size = _parse_positive_int(arg, "--batch-size")
        elif opt == "--decision-threshold":
            decision_threshold = _parse_probability(arg, "--decision-threshold")

    meta, ordered_items, feature_keys, feature_batches = _load_prediction_rows(input_file)
    if len(meta.get("columns", [])) == 0:
        raise ValueError("Input pickle does not contain the original TSV/PIN row metadata.")
    checkpoint_metadata = _load_checkpoint_metadata(model_name)
    model_arch = resolve_checkpoint_model_arch(checkpoint_metadata)
    print("Using CNN model architecture from checkpoint: " + model_arch)
    if "trainable_parameter_count" in checkpoint_metadata:
        print("Checkpoint trainable parameters: " + str(checkpoint_metadata["trainable_parameter_count"]))
    best_threshold = float(checkpoint_metadata["best_decision_threshold"])
    if decision_threshold is None:
        decision_threshold = best_threshold
        print("Using best decision threshold from checkpoint: " + str(decision_threshold))
    elif abs(decision_threshold - best_threshold) > 1e-9:
        print("WARNING: --decision-threshold " + str(decision_threshold) + " differs from checkpoint best " + str(best_threshold))

    device = torch.device("cuda")
    model = build_cnn_model(model_arch)
    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)

    predicted_scores, predicted_labels = _predict_scores(
        model,
        model_name,
        feature_batches,
        device,
        batch_size,
        decision_threshold,
        model_arch,
    )
    score_map = {key: score for key, score in zip(feature_keys, predicted_scores)}
    predicted_label_map = {key: label for key, label in zip(feature_keys, predicted_labels)}
    _write_rescored_output(output_file, meta, ordered_items, score_map, predicted_label_map)
    print("done")
