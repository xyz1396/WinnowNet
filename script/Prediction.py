import csv
import getopt
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

from WinnowNet_Att import DEFAULT_MAX_PEAKS, DualPeakClassifier, pad_control
from pkl_utils import (
    choose_output_column,
    format_label_value,
    get_entry_model_input,
    get_entry_row_index,
    get_entry_row_map,
    load_feature_pickle,
    normalize_long_flag_aliases,
)

DEFAULT_BATCH_SIZE = 128


class DefineDataset(Data.Dataset):
    def __init__(self, X, max_peaks=DEFAULT_MAX_PEAKS):
        self.X = X
        self.max_peaks = max_peaks

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        xspectra1 = pad_control(self.X[idx][0], self.max_peaks)
        xspectra2 = pad_control(self.X[idx][1], self.max_peaks)
        xspectra1 = torch.FloatTensor(xspectra1)
        xspectra2 = torch.FloatTensor(xspectra2)
        return xspectra1, xspectra2


def _score_to_string(score):
    return f"{float(score):.10g}"


def _qvalue_to_string(qvalue):
    if qvalue == "":
        return ""
    return f"{float(qvalue):.10g}"


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


def _parse_positive_int(value, flag_name):
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"{flag_name} must be an integer.")
    if parsed <= 0:
        raise ValueError(f"{flag_name} must be greater than 0.")
    return parsed


def _load_checkpoint_weights(model_path):
    return torch.load(
        model_path,
        map_location=lambda storage, loc: storage,
        weights_only=True,
    )


def _predict_scores(model, model_name, feature_batches, device, max_peaks, batch_size):
    test_data = DefineDataset(feature_batches, max_peaks=max_peaks)
    test_loader = Data.DataLoader(test_data, batch_size=batch_size)
    model.load_state_dict(
        _load_checkpoint_weights(model_name)
    )
    model.eval()

    y_pred_prob = []
    y_pred = []
    for data1, data2 in test_loader:
        data1, data2 = Variable(data1), Variable(data2)
        data1, data2 = data1.to(device), data2.to(device)
        output = model(data1, data2)
        pred_prob = torch.softmax(output.data, dim=1).cpu().numpy()
        y_pred.extend(np.argmax(pred_prob, axis=1).tolist())
        y_pred_prob.extend(pred_prob[:, 1].tolist())
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
        feature_keys.append(key)
        feature_batches.append(model_input)

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
            "-max-peaks": "--max-peaks",
            "-batch-size": "--batch-size",
        },
    )
    try:
        opts, args = getopt.getopt(argv, "hi:m:o:", ["max-peaks=", "batch-size="])
    except Exception:
        print("Error Option, using -h for help information.")
        sys.exit(1)
    if len(opts) == 0:
        print("\n\nUsage:\n")
        print("-i\t input representation file name\n")
        print("-m\t Pre-trained model name\n")
        print("-o\t Output rescored TSV file\n")
        print("--batch-size\t Prediction batch size (default: " + str(DEFAULT_BATCH_SIZE) + ")\n")
        print("--max-peaks\t Number of top-intensity peaks kept per spectrum (default: " + str(DEFAULT_MAX_PEAKS) + ")\n")
        sys.exit(1)

    input_file = ""
    model_name = ""
    output_file = ""
    batch_size = DEFAULT_BATCH_SIZE
    max_peaks = DEFAULT_MAX_PEAKS
    for opt, arg in opts:
        if opt in ("-h"):
            print("\n\nUsage:\n")
            print("-i\t input representation file name\n")
            print("-m\t Pre-trained model name\n")
            print("-o\t Output rescored TSV file\n")
            print("--batch-size\t Prediction batch size (default: " + str(DEFAULT_BATCH_SIZE) + ")\n")
            print("--max-peaks\t Number of top-intensity peaks kept per spectrum (default: " + str(DEFAULT_MAX_PEAKS) + ")\n")
            sys.exit(1)
        elif opt in ("-i"):
            input_file = arg
        elif opt in ("-m"):
            model_name = arg
        elif opt in ("-o"):
            output_file = arg
        elif opt == "--batch-size":
            batch_size = _parse_positive_int(arg, "--batch-size")
        elif opt == "--max-peaks":
            max_peaks = _parse_positive_int(arg, "--max-peaks")

    meta, ordered_items, feature_keys, feature_batches = _load_prediction_rows(input_file)
    if len(meta.get("columns", [])) == 0:
        raise ValueError("Input pickle does not contain the original TSV/PIN row metadata.")

    device = torch.device("cuda")
    model = DualPeakClassifier(
        dim_model=256,
        n_heads=4,
        dim_feedforward=512,
        n_layers=4,
        dim_intensity=None,
        num_classes=2,
        dropout=0.3,
        max_len=max_peaks,
    )
    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)

    predicted_scores, predicted_labels = _predict_scores(
        model,
        model_name,
        feature_batches,
        device,
        max_peaks,
        batch_size,
    )
    score_map = {key: score for key, score in zip(feature_keys, predicted_scores)}
    predicted_label_map = {key: label for key, label in zip(feature_keys, predicted_labels)}
    _write_rescored_output(output_file, meta, ordered_items, score_map, predicted_label_map)
    print("done")
