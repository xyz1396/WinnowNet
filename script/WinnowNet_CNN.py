import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import time
from datetime import timedelta
from sklearn import metrics
import numpy as np
import sys
import getopt
import os
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
    get_entry_group_key,
    get_entry_row_map,
    load_feature_pickle,
    normalize_long_flag_aliases,
    parse_prefix_filters,
    proteins_all_match_prefixes,
)

threshold=0.9
DEFAULT_EPOCHS = 50
DEFAULT_TRAIN_BATCH_SIZE = 1024
DEFAULT_EVAL_BATCH_SIZE = 1024
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_SELECTION_MAX_FDR = 0.01
CNN_INPUT_CHANNELS = 7
CNN_FEATURE_SCHEMA = "cnn_7ch_v1"
MODEL_ARCH_TNET = "tnet"
MODEL_ARCH_PURE_CNN = "pure_cnn"
MODEL_ARCH_CHOICES = {MODEL_ARCH_TNET, MODEL_ARCH_PURE_CNN}
CLASS_WEIGHT_NONE = "none"
CLASS_WEIGHT_BALANCED = "balanced"
CLASS_WEIGHT_CHOICES = {CLASS_WEIGHT_NONE, CLASS_WEIGHT_BALANCED}


def _validate_model_arch(model_arch):
    model_arch = str(model_arch or MODEL_ARCH_TNET).strip().lower()
    if model_arch not in MODEL_ARCH_CHOICES:
        raise ValueError(
            f"Unknown CNN model architecture {model_arch!r}. "
            f"Choose one of: {', '.join(sorted(MODEL_ARCH_CHOICES))}."
        )
    return model_arch


def _validate_class_weight(class_weight):
    class_weight = str(class_weight or CLASS_WEIGHT_NONE).strip().lower()
    if class_weight not in CLASS_WEIGHT_CHOICES:
        raise ValueError(
            f"Unknown class weight mode {class_weight!r}. "
            f"Choose one of: {', '.join(sorted(CLASS_WEIGHT_CHOICES))}."
        )
    return class_weight


def resolve_checkpoint_model_arch(metadata):
    if metadata is None:
        return MODEL_ARCH_TNET
    return _validate_model_arch(metadata.get("model_arch", MODEL_ARCH_TNET))


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


def _label_matches_expected(entry):
    label = get_entry_label(entry)
    if label is None:
        return True
    return label == 1


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
    if expected_model_arch is not None and checkpoint_model_arch != _validate_model_arch(expected_model_arch):
        raise ValueError(
            f"Checkpoint {model_path} was trained with model_arch={checkpoint_model_arch!r}, "
            f"but the requested CNN model architecture is {expected_model_arch!r}."
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


def _compute_decoy_per_target(yweight):
    positive = sum(1 for item in yweight if int(item[0]) == 1)
    negative = sum(1 for item in yweight if int(item[0]) == 0)
    if positive == 0:
        raise ValueError("Training data must contain at least one target entry.")
    return negative / float(positive)


def _compute_class_weights(yweight):
    target_count = sum(1 for item in yweight if int(item[0]) == 1)
    decoy_count = sum(1 for item in yweight if int(item[0]) == 0)
    if target_count == 0 or decoy_count == 0:
        raise ValueError("Weighted CrossEntropyLoss requires both target and decoy samples.")
    total_count = target_count + decoy_count
    decoy_weight = total_count / (2.0 * decoy_count)
    target_weight = total_count / (2.0 * target_count)
    return [decoy_weight, target_weight], target_count, decoy_count


def _parse_positive_int(value, flag_name):
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"{flag_name} must be a positive integer.")
    if parsed <= 0:
        raise ValueError(f"{flag_name} must be a positive integer.")
    return parsed


def _parse_positive_float(value, flag_name):
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"{flag_name} must be a positive number.")
    if parsed <= 0:
        raise ValueError(f"{flag_name} must be a positive number.")
    return parsed


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
    return Data.DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True, drop_last=True)


def _collect_prediction_scores(data, model, loss, device, eval_batch_size):
    model.eval()
    data_loader = Data.DataLoader(data, batch_size=eval_batch_size)
    total_loss = 0.0
    y_true = []
    y_scores = []
    for data1, label, weight in data_loader:
        data1, label = Variable(data1), Variable(label)
        data1, label = data1.to(device), label.to(device)
        output = model(data1)
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
            x_features = _extract_cnn_model_features(model_input, f"{record_key}: xFeatures")
            psm_id = entry.get("psm_id", record_key) if isinstance(entry, dict) else record_key
            row_map = get_entry_row_map(meta, psm_id, entry) if isinstance(entry, dict) else {}
            if proteins_all_match_prefixes(row_map, exclude_protein_prefixes):
                skipped_excluded_proteins += 1
                file_skipped_excluded_proteins += 1
                continue
            group_key = get_entry_group_key(meta, psm_id, entry)

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
                    L.append(x_features)
                    Yweight.append([1, 1])
                    groups.append(group_key)
                    file_groups.add(group_key)
                    positive += 1
                    file_positive += 1
                    kept_rows += 1
                    file_kept_rows += 1
            else:
                L.append(x_features)
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


def _resolve_training_inputs(input_directory, explicit_target, explicit_decoy):
    target_pickles = expand_pickle_inputs(explicit_target)
    decoy_pickles = expand_pickle_inputs(explicit_decoy)
    if target_pickles or decoy_pickles:
        if not target_pickles or not decoy_pickles:
            raise ValueError("Both -target and -decoy must be provided together.")
        return target_pickles, decoy_pickles, True

    return expand_pickle_inputs([input_directory]), [], False


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
        "[split] grouped by peptide; "
        f"total_groups={len(group_to_indices)} "
        f"train_groups={len(train_groups)} val_groups={len(val_groups)} test_groups={len(test_groups)}"
    )

    return _pick(train_idx), _pick(val_idx), _pick(test_idx)


class DefineDataset(Data.Dataset):
    def __init__(self, X, yweight):
        self.X = X
        self.yweight = yweight

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        xFeatures = _validate_cnn_features(self.X[idx], f"sample {idx} xFeatures")
        y = self.yweight[idx][0]
        weight = self.yweight[idx][1]

        xFeatures = torch.FloatTensor(xFeatures)

        return xFeatures, y, weight



class T_Net(nn.Module):
    def __init__(self, k):
        super(T_Net, self).__init__()
        self.k = k
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity_matrix = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            identity_matrix = identity_matrix.cuda()
        x = x + identity_matrix
        x = x.view(-1, self.k, self.k)
        return x


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.stn = T_Net(k=CNN_INPUT_CHANNELS)
        self.fstn = T_Net(k=64)

        self.conv1 = nn.Conv1d(CNN_INPUT_CHANNELS, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        n_pts = x.size()[2]
        input_transform = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, input_transform)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)

        pointfeat=x

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x, input_transform, trans_feat



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, input_transform, feature_transform = self.transform(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        output = self.fc3(x)
        return output


class PureCNNNet(nn.Module):
    def __init__(self):
        super(PureCNNNet, self).__init__()
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
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x_max = torch.max(x, dim=2)[0]
        x_mean = torch.mean(x, dim=2)
        x = torch.cat([x_max, x_mean], dim=1)
        return self.classifier(x)


def build_cnn_model(model_arch=MODEL_ARCH_TNET):
    model_arch = _validate_model_arch(model_arch)
    if model_arch == MODEL_ARCH_PURE_CNN:
        return PureCNNNet()
    return Net()


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(data, model, loss, device, eval_batch_size):
    # Evaluation, return accuracy and loss

    model.eval()  # set mode to evaluation to disable dropout
    data_loader = Data.DataLoader(data, batch_size=eval_batch_size)

    data_len = len(data)
    total_loss = 0.0
    y_true, y_pred = [], []

    for data1, label, weight in data_loader:
        data1, label = Variable(data1), Variable(label)
        data1, label = data1.to(device),label.to(device)

        output = model(data1)
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


def test_model(model, test_data, device, model_str, model_arch, eval_batch_size):
    print("Testing...")
    print("Checkpoint: " + _format_checkpoint_label(model_str))
    print("Checkpoint path: " + os.path.abspath(model_str))
    model.eval()
    start_time = time.time()
    test_loader = Data.DataLoader(test_data, batch_size=eval_batch_size)

    _load_model_state_dict(model, model_str, model_arch)

    y_true, y_pred, y_pred_prob = [], [], []
    for data1,label, weight in test_loader:
        y_true.extend(label.data)
        data1,label, weight = Variable(data1), Variable(label), Variable(weight)
        data1,label, weight = data1.to(device),label.to(device), weight.to(device)

        output = model(data1)
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
    epochs,
    model_arch,
    train_batch_size,
    eval_batch_size,
    learning_rate,
    class_weight,
):
    train_data = DefineDataset(X_train, yweight_train)
    val_data = DefineDataset(X_val, yweight_val)
    test_data = DefineDataset(X_test, yweight_test)
    model_output_path, checkpoint_dir = _prepare_output_paths(model_name)
    checkpoint_paths = []
    print("Model output path: " + model_output_path)
    print("Checkpoint directory: " + checkpoint_dir)
    device = torch.device("cuda")
    model_arch = _validate_model_arch(model_arch)
    model = build_cnn_model(model_arch)
    trainable_parameter_count = count_trainable_parameters(model)
    print("CNN model architecture: " + model_arch)
    print("Trainable parameters: " + str(trainable_parameter_count))
    print("Train batch size: " + str(train_batch_size))
    print("Eval batch size: " + str(eval_batch_size))
    print("Learning rate: " + str(learning_rate))
    class_weight = _validate_class_weight(class_weight)
    class_weights = None
    if class_weight == CLASS_WEIGHT_BALANCED:
        class_weights, target_count, decoy_count = _compute_class_weights(yweight_train)
        print(
            "Weighted CrossEntropyLoss class weights "
            f"[decoy={class_weights[0]:.6g}, target={class_weights[1]:.6g}] "
            f"from train targets={target_count} decoys={decoy_count}"
        )
    else:
        print("CrossEntropyLoss class weight mode: none")
    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)
    if len(pretrained_model)>0:
        print("loading pretrained_model")
        _load_model_state_dict(model, pretrained_model, model_arch)
    if class_weights is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    #model.load_state_dict(
    #    torch.load('cnn_pytorch.pt', map_location=lambda storage, loc: storage))
    #test_model(model, test_data, device)
    best_loss = 10000
    print("Epochs: " + str(epochs))
    train_loader = _build_train_loader(train_data, yweight_train, train_batch_size)
    for epoch in range(0, epochs):
        start_time = time.time()
        best_epoch_loss = 10000
        # load the training data in batch
        batch_count = 0
        model.train()
        for x1_batch, y_batch, weight in train_loader:
            batch_count = batch_count + 1
            inputs, targets = Variable(x1_batch),Variable(y_batch)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  # forward computation
            loss = criterion(outputs, targets)
            # backward propagation and update parameters
            loss.backward()
            optimizer.step()
        train_acc, train_loss, train_Posprec, train_Negprec = evaluate(
            train_data, model, criterion, device, eval_batch_size
        )
        val_acc, val_loss, val_PosPrec, val_Negprec = evaluate(
            val_data, model, criterion, device, eval_batch_size
        )
        _, val_y_true, val_y_scores = _collect_prediction_scores(
            val_data, model, criterion, device, eval_batch_size
        )
        best_threshold, best_target_count, best_val_fdr = _select_best_prediction_defaults(
            val_y_true,
            val_y_scores,
        )
        checkpoint_metadata = build_checkpoint_metadata(
            model_type="cnn",
            best_decision_threshold=best_threshold,
            input_channels=CNN_INPUT_CHANNELS,
            feature_schema=CNN_FEATURE_SCHEMA,
            model_arch=model_arch,
            trainable_parameter_count=trainable_parameter_count,
            learning_rate=learning_rate,
            class_weight=class_weight,
            class_weights=class_weights,
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
                         val_PosPrec, val_Negprec, best_threshold, best_target_count, best_val_fdr, time_dif))

    print("Best model path: " + model_output_path)
    for checkpoint_path in checkpoint_paths:
        test_model(model, test_data, device, checkpoint_path, model_arch, eval_batch_size)



if __name__ == "__main__":
    argv = normalize_long_flag_aliases(
        sys.argv[1:],
        {
            "-target": "--target",
            "-decoy": "--decoy",
            "-epochs": "--epochs",
            "-train-batch-size": "--train-batch-size",
            "-eval-batch-size": "--eval-batch-size",
            "-learning-rate": "--learning-rate",
            "-class-weight": "--class-weight",
            "-exclude-protein-prefix": "--exclude-protein-prefix",
            "-model-arch": "--model-arch",
        },
    )
    try:
        opts, args = getopt.getopt(
            argv,
            "hi:m:p:t:",
            [
                "target=",
                "decoy=",
                "epochs=",
                "train-batch-size=",
                "eval-batch-size=",
                "learning-rate=",
                "class-weight=",
                "exclude-protein-prefix=",
                "model-arch=",
            ],
        )
    except:
        print("Error Option, using -h for help information.")
        sys.exit(1)
    if len(opts)==0:
        print("\n\nUsage:\n")
        print("-i\t Directory containing feature pickles with embedded labels\n")
        print("-m\t Output trained model name\n")
        print("-p\t Optional pretrained model name\n")
        print("-target\t Target feature pickle(s), comma-separated or repeated\n")
        print("-decoy\t Decoy feature pickle(s), comma-separated or repeated\n")
        print("--epochs\t Number of training epochs (default: " + str(DEFAULT_EPOCHS) + ")\n")
        print("--train-batch-size\t Training batch size (default: " + str(DEFAULT_TRAIN_BATCH_SIZE) + ")\n")
        print("--eval-batch-size\t Validation/test batch size (default: " + str(DEFAULT_EVAL_BATCH_SIZE) + ")\n")
        print("--learning-rate\t Adam learning rate (default: " + str(DEFAULT_LEARNING_RATE) + ")\n")
        print("--class-weight\t CrossEntropyLoss class weighting: none or balanced (default: " + CLASS_WEIGHT_NONE + ")\n")
        print("--model-arch\t CNN architecture: tnet or pure_cnn (default: " + MODEL_ARCH_TNET + ")\n")
        print("--exclude-protein-prefix\t Drop PSMs when all proteins start with one of the given prefixes, comma-separated (for example Con_)\n")
        sys.exit(1)
        start_time=time.time()
    input_directory=""
    model_name=""
    pretrained_model=""
    epochs = DEFAULT_EPOCHS
    exclude_protein_prefixes = []
    target_inputs = []
    decoy_inputs = []
    model_arch = MODEL_ARCH_TNET
    train_batch_size = DEFAULT_TRAIN_BATCH_SIZE
    eval_batch_size = DEFAULT_EVAL_BATCH_SIZE
    learning_rate = DEFAULT_LEARNING_RATE
    class_weight = CLASS_WEIGHT_NONE
    for opt, arg in opts:
        if opt in ("-h"):
            print("\n\nUsage:\n")
            print("-i\t Directory containing feature pickles with embedded labels\n")
            print("-m\t Output trained model name\n")
            print("-p\t Optional pretrained model name\n")
            print("-target\t Target feature pickle(s), comma-separated or repeated\n")
            print("-decoy\t Decoy feature pickle(s), comma-separated or repeated\n")
            print("--epochs\t Number of training epochs (default: " + str(DEFAULT_EPOCHS) + ")\n")
            print("--train-batch-size\t Training batch size (default: " + str(DEFAULT_TRAIN_BATCH_SIZE) + ")\n")
            print("--eval-batch-size\t Validation/test batch size (default: " + str(DEFAULT_EVAL_BATCH_SIZE) + ")\n")
            print("--learning-rate\t Adam learning rate (default: " + str(DEFAULT_LEARNING_RATE) + ")\n")
            print("--class-weight\t CrossEntropyLoss class weighting: none or balanced (default: " + CLASS_WEIGHT_NONE + ")\n")
            print("--model-arch\t CNN architecture: tnet or pure_cnn (default: " + MODEL_ARCH_TNET + ")\n")
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
        elif opt == "--epochs":
            epochs = _parse_positive_int(arg, "--epochs")
        elif opt == "--train-batch-size":
            train_batch_size = _parse_positive_int(arg, "--train-batch-size")
        elif opt == "--eval-batch-size":
            eval_batch_size = _parse_positive_int(arg, "--eval-batch-size")
        elif opt == "--learning-rate":
            learning_rate = _parse_positive_float(arg, "--learning-rate")
        elif opt == "--class-weight":
            class_weight = _validate_class_weight(arg)
        elif opt == "--model-arch":
            model_arch = _validate_model_arch(arg)
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
        epochs,
        model_arch,
        train_batch_size,
        eval_batch_size,
        learning_rate,
        class_weight,
    )
    print('done')
