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
    best_decision_threshold,
    max_peaks=None,
    input_channels=None,
    feature_schema=None,
    model_arch=None,
    trainable_parameter_count=None,
    learning_rate=None,
    class_weight=None,
    class_weights=None,
):
    metadata = {
        "model_type": model_type,
        "best_decision_threshold": float(best_decision_threshold),
    }
    if max_peaks is not None:
        metadata["max_peaks"] = int(max_peaks)
    if input_channels is not None:
        metadata["input_channels"] = int(input_channels)
    if feature_schema is not None:
        metadata["feature_schema"] = str(feature_schema)
    if model_arch is not None:
        metadata["model_arch"] = str(model_arch)
    if trainable_parameter_count is not None:
        metadata["trainable_parameter_count"] = int(trainable_parameter_count)
    if learning_rate is not None:
        metadata["learning_rate"] = float(learning_rate)
    if class_weight is not None:
        metadata["class_weight"] = str(class_weight)
    if class_weights is not None:
        metadata["class_weights"] = [float(weight) for weight in class_weights]
    return metadata


def checkpoint_display_name(model_path):
    return os.path.abspath(model_path)
