import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.amp import autocast, GradScaler
import torch.utils.data as Data
import time
import sys
import getopt
from datetime import timedelta
from sklearn import metrics
import numpy as np
import glob
import os
from components.encoders import PeakEncoder
from checkpoint_utils import (
    build_checkpoint_metadata,
    load_checkpoint_bundle,
    save_checkpoint_bundle,
)
from pkl_utils import (
    expand_pickle_inputs,
    get_entry_label,
    get_entry_label_confidence,
    get_entry_model_input,
    get_entry_row_map,
    get_entry_spectrum_group_key,
    load_feature_pickle,
    normalize_long_flag_aliases,
    parse_prefix_filters,
    proteins_all_match_prefixes,
)

threshold=0.9
DEFAULT_TRAIN_BATCH_SIZE = 64
DEFAULT_EVAL_BATCH_SIZE = 128
DEFAULT_MAX_PEAKS = 300
DEFAULT_EPOCHS = 50
DEFAULT_SELECTION_MAX_FDR = 0.01


def _label_matches_expected(entry):
    label = get_entry_label(entry)
    if label is None:
        return True
    return label == 1


def _prepare_output_paths(model_name):
    if len(model_name) == 0:
        raise ValueError("Use -m to provide an output model path.")

    model_output_path = os.path.abspath(model_name)
    output_dir = os.path.dirname(model_output_path)
    if len(output_dir) == 0:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    model_stem = os.path.splitext(os.path.basename(model_output_path))[0] or "model"
    checkpoint_dir = os.path.join(output_dir, model_stem + "_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return model_output_path, checkpoint_dir


def _parse_positive_int(value, flag_name):
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"{flag_name} must be an integer.")
    if parsed <= 0:
        raise ValueError(f"{flag_name} must be greater than 0.")
    return parsed


def _prepare_training_pool(X, yweight, dataset_name="train"):
    pos_idx = [idx for idx, item in enumerate(yweight) if int(item[0]) == 1]
    neg_idx = [idx for idx, item in enumerate(yweight) if int(item[0]) == 0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError(f"[{dataset_name}] requires both target and decoy samples.")

    print(
        f"[{dataset_name}] using full training pool; "
        f"available_targets={len(pos_idx)} available_decoys={len(neg_idx)}"
    )
    return X, yweight


def _load_checkpoint_weights(model_path):
    state_dict, _ = load_checkpoint_bundle(model_path)
    return state_dict


def _compute_decoy_per_target(yweight):
    positive = sum(1 for item in yweight if int(item[0]) == 1)
    negative = sum(1 for item in yweight if int(item[0]) == 0)
    if positive == 0:
        raise ValueError("Training data must contain at least one target entry.")
    return negative / float(positive)


def _build_train_loader(train_data, yweight_train, batch_size):
    total_samples = len(yweight_train)
    natural_decoy_per_target = _compute_decoy_per_target(yweight_train)
    target_probability = 1.0 / (1.0 + natural_decoy_per_target)
    decoy_probability = natural_decoy_per_target / (1.0 + natural_decoy_per_target)
    target_count = int(round(total_samples * target_probability))
    decoy_count = total_samples - target_count
    print(
        f"[train] input target_count~{target_count} decoy_count~{decoy_count}"
    )
    return Data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
    )


def _collect_prediction_scores(data, model, loss, device, eval_batch_size):
    model.eval()
    data_loader = Data.DataLoader(
        data,
        batch_size=eval_batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
    )

    total_loss = 0.0
    y_true = []
    y_scores = []
    for input1, input2, label, weight in data_loader:
        input1 = input1.to(device, non_blocking=True)
        input2 = input2.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        output = model(input1, input2)
        losses = loss(output, label)
        total_loss += losses.data.item()
        pred_prob = torch.softmax(output.data, dim=1)[:, 1].detach().cpu().numpy()
        y_scores.extend(pred_prob.tolist())
        y_true.extend(label.data.cpu().numpy().tolist())

    data_len = max(1, len(data))
    return total_loss / data_len, np.asarray(y_true, dtype=int), np.asarray(y_scores, dtype=float)


def _select_best_prediction_defaults(y_true, y_scores):
    if len(y_scores) == 0:
        return 0.5, 0, 0.0

    candidate_thresholds = np.unique(np.asarray(y_scores, dtype=float))
    if len(candidate_thresholds) > 512:
        quantiles = np.linspace(0.0, 1.0, 512)
        candidate_thresholds = np.quantile(candidate_thresholds, quantiles)
    candidate_thresholds = np.unique(
        np.clip(np.concatenate(([1e-6, 0.5, 1.0 - 1e-6], candidate_thresholds)), 1e-6, 1.0 - 1e-6)
    )

    best_target_count = -1
    best_fdr = 1.0
    best_threshold = float(max(candidate_thresholds))
    for threshold_value in candidate_thresholds:
        accepted = y_scores >= threshold_value
        target_count = int(np.sum((y_true == 1) & accepted))
        decoy_count = int(np.sum((y_true == 0) & accepted))
        current_fdr = 0.0 if target_count == 0 else decoy_count / float(target_count)
        if current_fdr > DEFAULT_SELECTION_MAX_FDR:
            continue
        if target_count > best_target_count:
            best_target_count = target_count
            best_fdr = current_fdr
            best_threshold = float(threshold_value)
        elif target_count == best_target_count:
            if current_fdr < best_fdr - 1e-12:
                best_fdr = current_fdr
                best_threshold = float(threshold_value)
            elif abs(current_fdr - best_fdr) <= 1e-12 and threshold_value > best_threshold:
                best_threshold = float(threshold_value)

    return best_threshold, best_target_count, best_fdr


def _load_feature_records(feature_paths, force_label=None, dataset_name="dataset", exclude_protein_prefixes=None):
    L = []
    Yweight = []
    groups = []
    positive = 0
    negative = 0
    total_rows = 0
    kept_rows = 0
    skipped_non_one = 0
    unlabeled_rows = 0
    skipped_excluded_proteins = 0

    for feature_path in feature_paths:
        file_positive = 0
        file_negative = 0
        file_total_rows = 0
        file_kept_rows = 0
        file_skipped_non_one = 0
        file_unlabeled_rows = 0
        file_skipped_excluded_proteins = 0
        file_groups = set()
        meta, entries = load_feature_pickle(feature_path)
        for record_key, entry in entries.items():
            total_rows += 1
            file_total_rows += 1
            model_input = get_entry_model_input(entry)
            if model_input is None:
                continue
            psm_id = entry.get("psm_id", record_key) if isinstance(entry, dict) else record_key
            row_map = get_entry_row_map(meta, psm_id, entry) if isinstance(entry, dict) else {}
            if proteins_all_match_prefixes(row_map, exclude_protein_prefixes):
                skipped_excluded_proteins += 1
                file_skipped_excluded_proteins += 1
                continue
            group_key = get_entry_spectrum_group_key(meta, psm_id, entry)

            if force_label is None:
                label = get_entry_label(entry)
                confidence = get_entry_label_confidence(entry)
                if label is None:
                    continue
                if confidence is None:
                    confidence = 1.0 if label == 1 else 0.0
            else:
                if get_entry_label(entry) is None:
                    unlabeled_rows += 1
                    file_unlabeled_rows += 1
                elif not _label_matches_expected(entry):
                    skipped_non_one += 1
                    file_skipped_non_one += 1
                    continue
                label = force_label
                confidence = 1.0 if force_label == 1 else 0.0

            if label == 1:
                if confidence > threshold:
                    L.append(model_input)
                    Yweight.append([1, 1])
                    groups.append(group_key)
                    file_groups.add(group_key)
                    positive += 1
                    file_positive += 1
                    kept_rows += 1
                    file_kept_rows += 1
            else:
                L.append(model_input)
                Yweight.append([0, confidence])
                groups.append(group_key)
                file_groups.add(group_key)
                negative += 1
                file_negative += 1
                kept_rows += 1
                file_kept_rows += 1

        print(
            f"[{dataset_name}:file] path={os.path.abspath(feature_path)} "
            f"kept_psms={file_kept_rows} kept_targets={file_positive} kept_decoys={file_negative} "
            f"unique_groups={len(file_groups)} "
            f"total_rows={file_total_rows} unlabeled_rows={file_unlabeled_rows} "
            f"skipped_label_not_1={file_skipped_non_one} "
            f"skipped_excluded_proteins={file_skipped_excluded_proteins}"
        )

    print(
        f"[{dataset_name}] kept_targets={positive} kept_decoys={negative} "
        f"total_rows={total_rows} kept_rows={kept_rows} "
        f"unlabeled_rows={unlabeled_rows} skipped_label_not_1={skipped_non_one} "
        f"skipped_excluded_proteins={skipped_excluded_proteins}"
    )
    return L, Yweight, groups


def _collect_pickles_from_directory(directory):
    return sorted(glob.glob(os.path.join(directory, "*.pkl")))


def _resolve_training_inputs(input_directory, explicit_target, explicit_decoy):
    target_pickles = expand_pickle_inputs(explicit_target)
    decoy_pickles = expand_pickle_inputs(explicit_decoy)
    if target_pickles or decoy_pickles:
        if not target_pickles or not decoy_pickles:
            raise ValueError("Both -target and -decoy must be provided together.")
        return target_pickles, decoy_pickles, True

    pct1_dir = os.path.join(input_directory, "pct1")
    pct2_dir = os.path.join(input_directory, "pct2")
    if os.path.isdir(pct1_dir) and os.path.isdir(pct2_dir):
        return _collect_pickles_from_directory(pct2_dir), _collect_pickles_from_directory(pct1_dir), True

    return _collect_pickles_from_directory(input_directory), [], False


def pad_control(data,pairmaxlength):
    data = sorted(data, key=lambda x: x[1], reverse=True)
    width = 2
    if len(data) > 0:
        width = max(len(x) for x in data)
    zero_peak = [0.0] * width
    if len(data) > pairmaxlength:
        data = data[:pairmaxlength]
    else:
        while (len(data) < pairmaxlength):
            data.append(zero_peak.copy())
    for i in range(len(data)):
        if len(data[i]) < width:
            data[i] = list(data[i]) + [0.0] * (width - len(data[i]))
        elif len(data[i]) > width:
            data[i] = list(data[i])[:width]
    data = sorted(data, key=lambda x: x[0])
    return np.asarray(data,dtype=float)


def split_grouped(X, Yweight, groups, val_ratio=0.1, test_ratio=0.1, seed=10):
    if not (len(X) == len(Yweight) == len(groups)):
        raise ValueError("Grouped split requires X, Yweight, and groups to have the same length.")

    n = len(X)
    group_to_indices = {}
    for idx, group_key in enumerate(groups):
        group_to_indices.setdefault(group_key, []).append(idx)

    group_keys = list(group_to_indices.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(group_keys)

    test_n = max(1, int(n * test_ratio)) if n >= 10 else max(0, n // 5)
    val_n = max(1, int(n * val_ratio)) if n - test_n >= 10 else max(0, (n - test_n) // 5)

    test_idx = []
    val_idx = []
    train_idx = []
    test_count = 0
    val_count = 0

    for group_key in group_keys:
        indices = group_to_indices[group_key]
        if test_count < test_n:
            test_idx.extend(indices)
            test_count += len(indices)
        elif val_count < val_n:
            val_idx.extend(indices)
            val_count += len(indices)
        else:
            train_idx.extend(indices)

    def _pick(indices):
        return [X[i] for i in indices], [Yweight[i] for i in indices]

    train_groups = {groups[i] for i in train_idx}
    val_groups = {groups[i] for i in val_idx}
    test_groups = {groups[i] for i in test_idx}
    print(
        "[split] grouped by spectrum; "
        f"total_groups={len(group_to_indices)} "
        f"train_groups={len(train_groups)} val_groups={len(val_groups)} test_groups={len(test_groups)}"
    )

    return _pick(train_idx), _pick(val_idx), _pick(test_idx)


class DefineDataset(Data.Dataset):
    def __init__(self, X, yweight, max_peaks=DEFAULT_MAX_PEAKS):
        self.X = X
        self.yweight = yweight
        self.max_peaks = max_peaks

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        xspectra1 = pad_control(self.X[idx][0], self.max_peaks)
        xspectra2 = pad_control(self.X[idx][1], self.max_peaks)
        y = self.yweight[idx][0]
        weight = self.yweight[idx][1]
        xspectra1 = torch.FloatTensor(xspectra1)
        xspectra2 = torch.FloatTensor(xspectra2)

        return xspectra1, xspectra2, y, weight

class MS2Encoder(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_intensity: int,
        n_heads: int,
        dim_feedforward: int,
        n_layers: int,
        dropout: float = 0.1,
        max_len: int = 200
    ):
        super().__init__()
        self.peak_encoder = PeakEncoder(
            dim_model=dim_model,
            dim_intensity=dim_intensity,
            min_wavelength=0.001,
            max_wavelength=7000,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, spectra: torch.Tensor):
        src_key_padding_mask = spectra.sum(dim=2) == 0
        peaks = self.peak_encoder(spectra)
        out = self.transformer(peaks, src_key_padding_mask=src_key_padding_mask)
        return out, src_key_padding_mask


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_model)
        self.ff = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_model),
        )
        self.norm2 = nn.LayerNorm(dim_model)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_padding_mask: torch.Tensor,
        context_padding_mask: torch.Tensor,
    ):
        attended, _ = self.attention(
            query=query,
            key=context,
            value=context,
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )
        out = self.norm1(query + self.dropout(attended))
        out = self.norm2(out + self.dropout(self.ff(out)))
        if query_padding_mask is not None:
            out = out.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)
        return out


def _masked_mean_pool(sequence: torch.Tensor, padding_mask: torch.Tensor):
    if padding_mask is None:
        return sequence.mean(dim=1)
    valid_mask = (~padding_mask).unsqueeze(-1).type_as(sequence)
    pooled = (sequence * valid_mask).sum(dim=1)
    counts = valid_mask.sum(dim=1).clamp_min(1.0)
    return pooled / counts

class DualPeakClassifier(nn.Module):
    def __init__(
        self,
        dim_model: int = 256,
        dim_intensity: int = 128,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        n_layers: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
        max_len: int = 200,
    ):
        super().__init__()
        self.sequence_encoder = MS2Encoder(
            dim_model, dim_intensity, n_heads, dim_feedforward, n_layers, dropout, max_len
        )
        self.exp_from_theory = CrossAttentionBlock(
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.theory_from_exp = CrossAttentionBlock(
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * dim_model, 2 * dim_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim_model, num_classes),
        )

    def forward(
        self,
        spectra1: torch.Tensor,
        spectra2: torch.Tensor,
    ):
        exp_encoded, exp_padding_mask = self.sequence_encoder(spectra1)
        theory_encoded, theory_padding_mask = self.sequence_encoder(spectra2)

        exp_matched = self.exp_from_theory(
            query=exp_encoded,
            context=theory_encoded,
            query_padding_mask=exp_padding_mask,
            context_padding_mask=theory_padding_mask,
        )
        theory_matched = self.theory_from_exp(
            query=theory_encoded,
            context=exp_encoded,
            query_padding_mask=theory_padding_mask,
            context_padding_mask=exp_padding_mask,
        )

        rep1 = _masked_mean_pool(exp_matched, exp_padding_mask)
        rep2 = _masked_mean_pool(theory_matched, theory_padding_mask)
        abs_diff = torch.abs(rep1 - rep2)
        prod = rep1 * rep2
        joint = torch.cat([rep1, rep2, abs_diff, prod], dim=-1)
        outputs = self.classifier(joint)
        return outputs

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(data, model, loss, device, eval_batch_size):

    model.eval()
    data_loader = Data.DataLoader(
        data,
        batch_size=eval_batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
    )

    data_len = len(data)
    total_loss = 0.0
    y_true, y_pred = [], []

    for input1, input2, label, weight in data_loader:
        input1, input2, label = input1.to(device,non_blocking=True),input2.to(device,non_blocking=True),label.to(device,non_blocking=True)

        output = model(input1,input2)
        losses = loss(output, label)

        total_loss += losses.data.item()
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(label.data.cpu().numpy().tolist())

    acc = (np.array(y_true) == np.array(y_pred)).sum()
    positive_precision = metrics.precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    negative_precision = metrics.precision_score(y_true, y_pred, pos_label=0, zero_division=0)

    return acc / data_len, total_loss / data_len, positive_precision, negative_precision



def _format_checkpoint_label(model_path):
    base_name = os.path.basename(model_path)
    stem, _ = os.path.splitext(base_name)
    if stem.startswith("epoch"):
        epoch_text = stem[len("epoch"):]
        if epoch_text.isdigit():
            return f"{base_name} (Epoch {int(epoch_text) + 1})"
    return base_name


def test_model(model, test_data, device, model_str, eval_batch_size):
    print("Testing...")
    print("Checkpoint: " + _format_checkpoint_label(model_str))
    print("Checkpoint path: " + os.path.abspath(model_str))
    model.eval()
    start_time = time.time()
    test_loader = Data.DataLoader(test_data, batch_size=eval_batch_size)

    model.load_state_dict(_load_checkpoint_weights(model_str))

    y_true, y_pred, y_pred_prob = [], [], []
    for data1,addfeat,label, weight in test_loader:
        y_true.extend(label.data)
        data1,addfeat,label, weight = Variable(data1), Variable(addfeat),Variable(label), Variable(weight)
        data1,addfeat,label, weight = data1.to(device),Variable(addfeat),label.to(device), weight.to(device)

        output = model(data1,addfeat)
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        pred_prob = torch.softmax(output.data, dim=1).cpu()
        pred_prob = np.asarray(pred_prob, dtype=float)
        y_pred.extend(pred)
        y_pred_prob.extend(pred_prob[:, 1].tolist())

    test_acc = metrics.accuracy_score(y_true, y_pred)
    test_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    print(
        "Test accuracy: {0:>7.2%}, F1-Score: {1:>7.2%}".format(test_acc, test_f1))

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(
        y_true, y_pred, labels=[1, 0], target_names=['T', 'D'], zero_division=0))

    print('Confusion Matrix...')
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])
    print(cm)

    print("Time usage:", get_time_dif(start_time))


def train_model(
    X_train,
    X_val,
    X_test,
    yweight_train,
    yweight_val,
    yweight_test,
    model_name,
    pretrained_model,
    train_batch_size,
    eval_batch_size,
    max_peaks,
    epochs,
):
    LR = 1e-4
    train_data = DefineDataset(X_train, yweight_train, max_peaks=max_peaks)
    val_data = DefineDataset(X_val, yweight_val, max_peaks=max_peaks)
    test_data = DefineDataset(X_test, yweight_test, max_peaks=max_peaks)
    model_output_path, checkpoint_dir = _prepare_output_paths(model_name)
    checkpoint_paths = []
    print("Model output path: " + model_output_path)
    print("Checkpoint directory: " + checkpoint_dir)
    print("Train batch size: " + str(train_batch_size))
    print("Eval batch size: " + str(eval_batch_size))
    print("Max peaks kept per spectrum: " + str(max_peaks))
    print("Epochs: " + str(epochs))
    device = torch.device("cuda")
    model = DualPeakClassifier(dim_model=256,n_heads=4,dim_feedforward=512,n_layers=4,dim_intensity=None,num_classes=2,dropout=0.3,max_len=max_peaks)
    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)
    if len(pretrained_model)>0:
        print("loading pretrained_model")
        model.load_state_dict(_load_checkpoint_weights(pretrained_model))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    #model.load_state_dict(torch.load('cnn_pytorch.pt', map_location=lambda storage, loc: storage))
    #test_model(model, test_data, device)
    best_loss = float("inf")
    scaler = GradScaler("cuda")
    train_loader = _build_train_loader(
        train_data,
        yweight_train,
        train_batch_size,
    )
    for epoch in range(0, epochs):
        start_time = time.time()
        model.train()
        for input1, input2, y_batch, weight in train_loader:
            input1, input2, targets = input1.to(device,non_blocking=True), input2.to(device,non_blocking=True), y_batch.to(device,non_blocking=True)
            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = model(input1,input2)  # forward computation
                loss = criterion(outputs, targets)
            # backward propagation and update parameters
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        train_acc, train_loss, train_Posprec, train_Negprec = evaluate(
            train_data, model, criterion, device, eval_batch_size
        )
        val_acc, val_loss, val_PosPrec, val_Negprec = evaluate(
            val_data, model, criterion, device, eval_batch_size
        )
        _, val_y_true, val_y_scores = _collect_prediction_scores(
            val_data,
            model,
            criterion,
            device,
            eval_batch_size,
        )
        best_threshold, epoch_best_target_count, best_val_fdr = _select_best_prediction_defaults(
            val_y_true,
            val_y_scores,
        )
        checkpoint_metadata = build_checkpoint_metadata(
            model_type="att",
            best_decision_threshold=best_threshold,
            max_peaks=max_peaks,
        )
        checkpoint_path = os.path.join(checkpoint_dir, 'epoch' + str(epoch) + '.pt')
        save_checkpoint_bundle(checkpoint_path, model.state_dict(), checkpoint_metadata)
        checkpoint_paths.append(checkpoint_path)
        print("Saved checkpoint: " + checkpoint_path)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint_bundle(model_output_path, model.state_dict(), checkpoint_metadata)
            print("Saved best model: " + model_output_path)

        time_dif = get_time_dif(start_time)
        msg = "Epoch {0:3}, Train_loss: {1:>7.2}, Train_acc {2:>6.2%}, Train_Posprec {3:>6.2%}, Train_Negprec {" \
              "4:>6.2%}, " + "Val_loss: {5:>6.2}, Val_acc {6:>6.2%},Val_Posprec {7:6.2%}, Val_Negprec {8:6.2%}, " \
              "BestThreshold {9:.4f}, BestTargets@FDR<=1% {10}, BestValFDR {11:.4%} Time: {12} "
        print(msg.format(epoch + 1, train_loss, train_acc, train_Posprec, train_Negprec, val_loss, val_acc,
                         val_PosPrec, val_Negprec, best_threshold, epoch_best_target_count, best_val_fdr, time_dif))

    print("Best model path: " + model_output_path)
    for checkpoint_path in checkpoint_paths:
        test_model(model, test_data, device, checkpoint_path, eval_batch_size)


if __name__ == "__main__":
    argv = normalize_long_flag_aliases(
        sys.argv[1:],
        {
            "-target": "--target",
            "-decoy": "--decoy",
            "-train-batch-size": "--train-batch-size",
            "-eval-batch-size": "--eval-batch-size",
            "-max-peaks": "--max-peaks",
            "-epochs": "--epochs",
            "-exclude-protein-prefix": "--exclude-protein-prefix",
        },
    )
    try:
        opts, args = getopt.getopt(
            argv,
            "hi:m:p:",
            ["target=", "decoy=", "train-batch-size=", "eval-batch-size=", "max-peaks=", "epochs=", "exclude-protein-prefix="],
        )
    except:
        print("Error Option, using -h for help information.")
        sys.exit(1)
    if len(opts)==0:
        print("\n\nUsage:\n")
        print("-i\t Directory containing feature pickles with embedded labels\n")
        print("-m\t Output trained model name\n")
        print("-p\t Optional pretrained model name\n")
        print("-target\t Target feature pickle(s) or directory, comma-separated or repeated\n")
        print("-decoy\t Decoy feature pickle(s) or directory, comma-separated or repeated\n")
        print("--train-batch-size\t Training batch size (default: " + str(DEFAULT_TRAIN_BATCH_SIZE) + ")\n")
        print("--eval-batch-size\t Validation/test batch size (default: " + str(DEFAULT_EVAL_BATCH_SIZE) + ")\n")
        print("--max-peaks\t Number of top-intensity peaks kept per spectrum (default: " + str(DEFAULT_MAX_PEAKS) + ")\n")
        print("--epochs\t Number of training epochs (default: " + str(DEFAULT_EPOCHS) + ")\n")
        print("--exclude-protein-prefix\t Drop PSMs when all proteins start with one of the given prefixes, comma-separated (for example Con_)\n")
        sys.exit(1)
        start_time=time.time()
    input_directory=""
    model_name=""
    pretrained_model=""
    train_batch_size = DEFAULT_TRAIN_BATCH_SIZE
    eval_batch_size = DEFAULT_EVAL_BATCH_SIZE
    max_peaks = DEFAULT_MAX_PEAKS
    epochs = DEFAULT_EPOCHS
    exclude_protein_prefixes = []
    target_inputs = []
    decoy_inputs = []
    for opt, arg in opts:
        if opt in ("-h"):
            print("\n\nUsage:\n")
            print("-i\t Directory containing feature pickles with embedded labels\n")
            print("-m\t Output trained model name\n")
            print("-p\t Optional pretrained model name\n")
            print("-target\t Target feature pickle(s) or directory, comma-separated or repeated\n")
            print("-decoy\t Decoy feature pickle(s) or directory, comma-separated or repeated\n")
            print("--train-batch-size\t Training batch size (default: " + str(DEFAULT_TRAIN_BATCH_SIZE) + ")\n")
            print("--eval-batch-size\t Validation/test batch size (default: " + str(DEFAULT_EVAL_BATCH_SIZE) + ")\n")
            print("--max-peaks\t Number of top-intensity peaks kept per spectrum (default: " + str(DEFAULT_MAX_PEAKS) + ")\n")
            print("--epochs\t Number of training epochs (default: " + str(DEFAULT_EPOCHS) + ")\n")
            print("--exclude-protein-prefix\t Drop PSMs when all proteins start with one of the given prefixes, comma-separated (for example Con_)\n")
            sys.exit(1)
        elif opt in ("-i"):
            input_directory=arg
        elif opt in ("-m"):
            model_name=arg
        elif opt in ("-p"):
            pretrained_model=arg
        elif opt == "--target":
            target_inputs.append(arg)
        elif opt == "--decoy":
            decoy_inputs.append(arg)
        elif opt == "--train-batch-size":
            train_batch_size = _parse_positive_int(arg, "--train-batch-size")
        elif opt == "--eval-batch-size":
            eval_batch_size = _parse_positive_int(arg, "--eval-batch-size")
        elif opt == "--max-peaks":
            max_peaks = _parse_positive_int(arg, "--max-peaks")
        elif opt == "--epochs":
            epochs = _parse_positive_int(arg, "--epochs")
        elif opt == "--exclude-protein-prefix":
            exclude_protein_prefixes.extend(parse_prefix_filters(arg))
    start = time.time()
    exclude_protein_prefixes = list(dict.fromkeys(exclude_protein_prefixes))
    if exclude_protein_prefixes:
        print("Excluding PSMs whose proteins all match prefixes: " + ", ".join(exclude_protein_prefixes))
    split_mode = bool(target_inputs or decoy_inputs)
    if not split_mode and len(input_directory) == 0:
        raise ValueError("Use -i for embedded-label training or provide both -target and -decoy.")

    target_pickles, decoy_pickles, split_mode = _resolve_training_inputs(
        input_directory,
        target_inputs,
        decoy_inputs,
    )

    if split_mode:
        if len(target_pickles) == 0 or len(decoy_pickles) == 0:
            raise ValueError("Missing target/decoy pickle inputs.")
        X_pos, y_pos, group_pos = _load_feature_records(
            target_pickles,
            force_label=1,
            dataset_name="target",
            exclude_protein_prefixes=exclude_protein_prefixes,
        )
        X_neg, y_neg, group_neg = _load_feature_records(
            decoy_pickles,
            force_label=0,
            dataset_name="decoy",
            exclude_protein_prefixes=exclude_protein_prefixes,
        )
        X_all = X_pos + X_neg
        y_all = y_pos + y_neg
        group_all = group_pos + group_neg
        (X_train, yweight_train), (X_val, yweight_val), (X_test, yweight_test) = split_grouped(X_all, y_all, group_all)
    else:
        if len(target_pickles) == 0:
            raise ValueError(f"No feature pickles found under {input_directory}.")
        L, Yweight, groups = _load_feature_records(
            target_pickles,
            dataset_name="embedded",
            exclude_protein_prefixes=exclude_protein_prefixes,
        )
        (X_train, yweight_train), (X_val, yweight_val), (X_test, yweight_test) = split_grouped(L, Yweight, groups)

    X_train, yweight_train = _prepare_training_pool(
        X_train,
        yweight_train,
        dataset_name="train",
    )

    end = time.time()
    print('loading data: ' + str(end - start))
    print("length of training data: " + str(len(X_train)))
    print("length of validation data: " + str(len(X_val)))
    print("length of test data: " + str(len(X_test)))
    train_model(
        X_train,
        X_val,
        X_test,
        yweight_train,
        yweight_val,
        yweight_test,
        model_name,
        pretrained_model,
        train_batch_size,
        eval_batch_size,
        max_peaks,
        epochs,
    )
    print('done')
