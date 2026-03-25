import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
EPOCH_PATTERN = re.compile(
    r"Epoch\s+(\d+),\s+"
    rf"Train_loss:\s*({FLOAT_PATTERN}),\s*Train_acc\s*({FLOAT_PATTERN})%,\s*"
    rf"Train_Posprec\s*({FLOAT_PATTERN})%,\s*Train_Negprec\s*({FLOAT_PATTERN})%,\s*"
    rf"Val_loss:\s*({FLOAT_PATTERN}),\s*Val_acc\s*({FLOAT_PATTERN})%,\s*"
    rf"Val_Posprec\s*({FLOAT_PATTERN})%,\s*Val_Negprec\s*({FLOAT_PATTERN})%,\s*"
    rf"BestPredRatio\s*1:({FLOAT_PATTERN}),\s*BestThreshold\s*({FLOAT_PATTERN}),\s*"
    rf"BestTargets@FDR<=1%\s*(\d+),\s*BestValFDR\s*({FLOAT_PATTERN})%"
)
TEST_PATTERN = re.compile(
    rf"Checkpoint:\s+epoch(\d+)\.pt\s+\(Epoch\s+(\d+)\)\s+"
    rf"Checkpoint path:\s+.+?\s+"
    rf"Test accuracy:\s*({FLOAT_PATTERN})%,\s*F1-Score:\s*({FLOAT_PATTERN})%",
    re.DOTALL,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Draw training plots from a WinnowNet training log.")
    parser.add_argument("-i", "--input", required=True, help="Training log path.")
    parser.add_argument("-o", "--output", help="Output image path. Default: <log_stem>_training.png")
    parser.add_argument("--title", default="WinnowNet Training Summary", help="Figure title.")
    return parser.parse_args()


def load_rows(log_path):
    text = Path(log_path).read_text()
    rows = []
    for match in EPOCH_PATTERN.finditer(text):
        rows.append(
            {
                "epoch": int(match.group(1)),
                "train_loss": float(match.group(2)),
                "train_acc": float(match.group(3)),
                "train_posprec": float(match.group(4)),
                "train_negprec": float(match.group(5)),
                "val_loss": float(match.group(6)),
                "val_acc": float(match.group(7)),
                "val_posprec": float(match.group(8)),
                "val_negprec": float(match.group(9)),
                "best_pred_ratio": float(match.group(10)),
                "best_threshold": float(match.group(11)),
                "best_targets": int(match.group(12)),
                "best_val_fdr": float(match.group(13)),
            }
        )
    if not rows:
        raise ValueError("No epoch lines matched the WinnowNet log format.")

    test_rows = {}
    for match in TEST_PATTERN.finditer(text):
        epoch = int(match.group(2))
        checkpoint_epoch = int(match.group(1)) + 1
        if epoch != checkpoint_epoch:
            continue
        test_rows[epoch] = {
            "test_acc": float(match.group(3)),
            "test_f1": float(match.group(4)),
        }

    for row in rows:
        row.update(test_rows.get(row["epoch"], {}))
    return rows


def plot_rows(rows, output_path, title):
    epochs = [row["epoch"] for row in rows]
    best_targets = [row["best_targets"] for row in rows]
    best_val_fdr = [row["best_val_fdr"] for row in rows]
    train_loss = [row["train_loss"] for row in rows]
    val_loss = [row["val_loss"] for row in rows]
    train_acc = [row["train_acc"] for row in rows]
    val_acc = [row["val_acc"] for row in rows]
    best_threshold = [row["best_threshold"] for row in rows]
    best_pred_ratio = [row["best_pred_ratio"] for row in rows]
    has_test_metrics = any("test_acc" in row for row in rows)
    test_acc = [row.get("test_acc") for row in rows]
    test_f1 = [row.get("test_f1") for row in rows]

    best_target_idx = max(range(len(rows)), key=lambda idx: best_targets[idx])
    min_val_loss_idx = min(range(len(rows)), key=lambda idx: val_loss[idx])
    best_test_f1_idx = None
    if has_test_metrics:
        best_test_f1_idx = max(
            (idx for idx, value in enumerate(test_f1) if value is not None),
            key=lambda idx: test_f1[idx],
        )

    fig, axes = plt.subplots(4, 2, figsize=(14, 15), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="Train", color="#2563eb", linewidth=2)
    ax.plot(epochs, val_loss, label="Val", color="#ea580c", linewidth=2)
    ax.scatter([epochs[min_val_loss_idx]], [val_loss[min_val_loss_idx]], color="#dc2626", zorder=3)
    ax.annotate(
        f"min val_loss: e{epochs[min_val_loss_idx]} = {val_loss[min_val_loss_idx]:.3g}",
        (epochs[min_val_loss_idx], val_loss[min_val_loss_idx]),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=9,
    )
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.plot(epochs, train_acc, label="Train", color="#16a34a", linewidth=2)
    ax.plot(epochs, val_acc, label="Val", color="#ca8a04", linewidth=2)
    if has_test_metrics:
        ax.plot(epochs, test_acc, label="Test", color="#7c3aed", linewidth=2, alpha=0.9)
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.plot(epochs, best_targets, color="#0f766e", linewidth=2)
    ax.scatter([epochs[best_target_idx]], [best_targets[best_target_idx]], color="#dc2626", zorder=3)
    ax.annotate(
        f"best targets: e{epochs[best_target_idx]} = {best_targets[best_target_idx]}",
        (epochs[best_target_idx], best_targets[best_target_idx]),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=9,
    )
    ax.set_title("BestTargets@FDR<=1%")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Targets")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(epochs, best_val_fdr, color="#7c3aed", linewidth=2)
    ax.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1, label="1.0% limit")
    ax.set_title("BestValFDR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FDR (%)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[2, 0]
    ax.plot(epochs, best_threshold, color="#0891b2", linewidth=2)
    ax.set_title("BestThreshold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Threshold")
    ax.grid(True, alpha=0.25)

    ax = axes[2, 1]
    ax.plot(epochs, best_pred_ratio, color="#9333ea", linewidth=2)
    ax.set_title("BestPredRatio (1:x)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("x")
    ax.grid(True, alpha=0.25)

    ax = axes[3, 0]
    if has_test_metrics:
        ax.plot(epochs, test_f1, color="#0f766e", linewidth=2)
        ax.scatter([epochs[best_test_f1_idx]], [test_f1[best_test_f1_idx]], color="#dc2626", zorder=3)
        ax.annotate(
            f"best test F1: e{epochs[best_test_f1_idx]} = {test_f1[best_test_f1_idx]:.2f}%",
            (epochs[best_test_f1_idx], test_f1[best_test_f1_idx]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )
    else:
        ax.text(0.5, 0.5, "No per-checkpoint test metrics in log", ha="center", va="center")
    ax.set_title("Test F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 (%)" if has_test_metrics else "")
    ax.grid(True, alpha=0.25)

    ax = axes[3, 1]
    if has_test_metrics:
        ax.plot(epochs, test_acc, color="#be123c", linewidth=2)
    else:
        ax.text(0.5, 0.5, "No per-checkpoint test metrics in log", ha="center", va="center")
    ax.set_title("Test Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)" if has_test_metrics else "")
    ax.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=15)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_training.png")
    rows = load_rows(input_path)
    plot_rows(rows, output_path, args.title)

    best_target_row = max(rows, key=lambda row: row["best_targets"])
    min_val_loss_row = min(rows, key=lambda row: row["val_loss"])
    print(f"saved_plot={output_path}")
    print(f"epochs={len(rows)}")
    print(
        f"min_val_loss_epoch={min_val_loss_row['epoch']} "
        f"min_val_loss={min_val_loss_row['val_loss']:.4g}"
    )
    print(
        f"best_targets_epoch={best_target_row['epoch']} "
        f"best_targets={best_target_row['best_targets']} "
        f"best_threshold={best_target_row['best_threshold']:.4f} "
        f"best_pred_ratio=1:{best_target_row['best_pred_ratio']:.4f} "
        f"best_val_fdr={best_target_row['best_val_fdr']:.4f}%"
    )
    test_rows = [row for row in rows if "test_f1" in row]
    if test_rows:
        best_test_row = max(test_rows, key=lambda row: row["test_f1"])
        print(
            f"best_test_epoch={best_test_row['epoch']} "
            f"test_acc={best_test_row['test_acc']:.2f}% "
            f"test_f1={best_test_row['test_f1']:.2f}%"
        )


if __name__ == "__main__":
    main()
