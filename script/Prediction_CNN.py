import csv
import getopt
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

from WinnowNet_CNN import Net
from pkl_utils import (
    choose_output_column,
    format_label_value,
    get_entry_label,
    get_entry_label_raw,
    get_entry_model_input,
    get_entry_row_index,
    get_entry_row_map,
    load_feature_pickle,
)


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


def _predict_scores(model, model_name, feature_batches, device):
    test_data = DefineDataset(feature_batches)
    test_loader = Data.DataLoader(test_data, batch_size=32)
    model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
    model.eval()

    y_pred_prob = []
    for data1 in test_loader:
        data1 = Variable(data1).to(device)
        output = model(data1)
        pred_prob = torch.softmax(output.data, dim=1).cpu().numpy()
        y_pred_prob.extend(pred_prob[:, 1].tolist())
    return y_pred_prob


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
        feature_batches.append(model_input[0])

    return meta, ordered_items, feature_keys, feature_batches


def _write_rescored_output(output_file, meta, ordered_items, score_map):
    base_columns = list(meta.get("columns", []))
    score_column = choose_output_column(base_columns, ["score"], "score")
    qvalue_column = choose_output_column(base_columns, ["q-value", "qvalue"], "q-value")
    label_column = choose_output_column(base_columns, ["Label"], "Label")

    output_columns = list(base_columns)
    for column in (score_column, qvalue_column, label_column):
        if column not in output_columns:
            output_columns.append(column)

    labels = [get_entry_label(entry) for _, entry in ordered_items]
    scores = [score_map.get(key) for key, _ in ordered_items]
    qvalues = _compute_qvalues(scores, labels)

    with open(output_file, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=output_columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for idx, (key, entry) in enumerate(ordered_items):
            row_map = get_entry_row_map(meta, key, entry)
            score = scores[idx]
            if score is not None:
                row_map[score_column] = _score_to_string(score)
            elif score_column not in row_map:
                row_map[score_column] = ""
            qvalue_text = _qvalue_to_string(qvalues[idx])
            if qvalue_text != "" or qvalue_column not in row_map:
                row_map[qvalue_column] = qvalue_text
            label_text = format_label_value(
                labels[idx],
                get_entry_label_raw(entry),
            )
            if label_text != "" or label_column not in row_map:
                row_map[label_column] = label_text
            writer.writerow({column: row_map.get(column, "") for column in output_columns})


if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hi:m:o:")
    except Exception:
        print("Error Option, using -h for help information.")
        sys.exit(1)
    if len(opts) == 0:
        print("\n\nUsage:\n")
        print("-i\t input representation file name\n")
        print("-m\t Pre-trained model name\n")
        print("-o\t Output rescored TSV file\n")
        sys.exit(1)

    input_file = ""
    model_name = ""
    output_file = ""
    for opt, arg in opts:
        if opt in ("-h"):
            print("\n\nUsage:\n")
            print("-i\t input representation file name\n")
            print("-m\t Pre-trained model name\n")
            print("-o\t Output rescored TSV file\n")
            sys.exit(1)
        elif opt in ("-i"):
            input_file = arg
        elif opt in ("-m"):
            model_name = arg
        elif opt in ("-o"):
            output_file = arg

    meta, ordered_items, feature_keys, feature_batches = _load_prediction_rows(input_file)
    if len(meta.get("columns", [])) == 0:
        raise ValueError("Input pickle does not contain the original TSV/PIN row metadata.")

    device = torch.device("cuda")
    model = Net()
    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)

    predicted_scores = _predict_scores(model, model_name, feature_batches, device)
    score_map = {key: score for key, score in zip(feature_keys, predicted_scores)}
    _write_rescored_output(output_file, meta, ordered_items, score_map)
    print("done")
