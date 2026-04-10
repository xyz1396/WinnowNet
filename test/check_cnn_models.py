#!/usr/bin/env python3
"""Smoke-test the CNN backbones against the current feature schema."""

from pathlib import Path
import sys

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "script"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from WinnowNet_CNN import (  # noqa: E402
    CNN_INPUT_CHANNELS,
    _compute_model_loss,
    build_cnn_model,
)


def _assert_shape(actual, expected, label):
    if tuple(actual) != tuple(expected):
        raise AssertionError(f"{label} expected shape {expected}, got {tuple(actual)}")


def main():
    batch_size = 4
    peak_count = 128
    inputs = torch.randn(batch_size, CNN_INPUT_CHANNELS, peak_count)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    aux_target = torch.tensor([0.0, 0.2, 0.4, 0.6], dtype=torch.float32)
    aux_mask = torch.ones(batch_size, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss()

    pure_cnn = build_cnn_model("pure_cnn")
    pure_logits = pure_cnn(inputs)
    _assert_shape(pure_logits.shape, (batch_size, 2), "pure_cnn logits")

    tnet = build_cnn_model("tnet")
    tnet_output = tnet(inputs)
    if not isinstance(tnet_output, tuple) or len(tnet_output) != 2:
        raise AssertionError("tnet must return (logits, aux_prediction).")
    logits, aux_prediction = tnet_output
    _assert_shape(logits.shape, (batch_size, 2), "tnet logits")
    _assert_shape(aux_prediction.shape, (batch_size,), "tnet auxiliary prediction")

    total_loss, extracted_logits, auxiliary_loss = _compute_model_loss(
        tnet_output,
        labels,
        criterion,
        aux_target=aux_target,
        aux_mask=aux_mask,
    )
    _assert_shape(extracted_logits.shape, (batch_size, 2), "extracted logits")
    if total_loss.ndim != 0 or auxiliary_loss.ndim != 0:
        raise AssertionError("Expected scalar total and auxiliary losses.")

    print("pure_cnn and tnet smoke tests passed")


if __name__ == "__main__":
    main()
