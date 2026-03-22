import os

import torch


CHECKPOINT_FORMAT_VERSION = 1


def load_checkpoint_bundle(model_path):
    checkpoint = torch.load(
        model_path,
        map_location=lambda storage, loc: storage,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {model_path} is not in the required bundle format.")
    if "state_dict" not in checkpoint or "metadata" not in checkpoint:
        raise ValueError(f"Checkpoint {model_path} is missing state_dict or metadata.")
    metadata = checkpoint["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint {model_path} metadata must be a dict.")
    required_metadata = [
        "checkpoint_format_version",
        "model_type",
        "train_target_decoy_ratio",
        "best_prediction_target_decoy_ratio",
        "best_decision_threshold",
    ]
    for key in required_metadata:
        if key not in metadata:
            raise ValueError(f"Checkpoint {model_path} metadata is missing required field: {key}")
    return checkpoint["state_dict"], metadata


def save_checkpoint_bundle(model_path, state_dict, metadata=None):
    checkpoint = {
        "state_dict": state_dict,
        "metadata": {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            **(metadata or {}),
        },
    }
    torch.save(checkpoint, model_path)


def build_checkpoint_metadata(
    model_type,
    train_target_decoy_ratio,
    best_prediction_target_decoy_ratio,
    best_decision_threshold,
    max_peaks=None,
):
    metadata = {
        "model_type": model_type,
        "train_target_decoy_ratio": float(train_target_decoy_ratio),
        "best_prediction_target_decoy_ratio": float(best_prediction_target_decoy_ratio),
        "best_decision_threshold": float(best_decision_threshold),
    }
    if max_peaks is not None:
        metadata["max_peaks"] = int(max_peaks)
    return metadata


def checkpoint_display_name(model_path):
    return os.path.abspath(model_path)


def format_target_decoy_ratio(decoy_per_target):
    return "1:{0:.6g}".format(float(decoy_per_target))


def decision_threshold_from_ratios(train_decoy_per_target, prediction_decoy_per_target):
    train_decoy_per_target = float(train_decoy_per_target)
    prediction_decoy_per_target = float(prediction_decoy_per_target)
    if train_decoy_per_target <= 0 or prediction_decoy_per_target <= 0:
        raise ValueError("Target:decoy ratios must be greater than 0.")
    return prediction_decoy_per_target / (train_decoy_per_target + prediction_decoy_per_target)


def prediction_ratio_from_threshold(train_decoy_per_target, decision_threshold):
    train_decoy_per_target = float(train_decoy_per_target)
    decision_threshold = float(decision_threshold)
    if train_decoy_per_target <= 0:
        raise ValueError("Training target:decoy ratio must be greater than 0.")
    if decision_threshold <= 0.0:
        return 0.0
    if decision_threshold >= 1.0:
        return float("inf")
    return train_decoy_per_target * decision_threshold / (1.0 - decision_threshold)
