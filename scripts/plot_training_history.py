from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training and validation curves from experiment histories."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiment directories, for example experiments/yolo11 experiments/yolo11-e3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directory for generated PNG figures.",
    )
    return parser.parse_args()


def _number(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        value = row.get(name, "")
        if value not in {"", None}:
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _load_history(experiment: Path) -> list[dict[str, float]]:
    path = experiment / "training_history.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Training history not found: {path}. "
            "Copy the Kaggle experiment folder locally or mount its parent directory first."
        )
    by_epoch: dict[float, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for index, source in enumerate(csv.DictReader(handle), start=1):
            epoch = _number(source, "epoch") or float(index)
            train_loss = _number(source, "train_loss", "train/box_loss", "loss")
            val_loss = _number(source, "val_loss", "val/box_loss", "eval_loss")
            map50 = _number(source, "mAP50", "metrics/mAP50(B)")
            map5095 = _number(source, "mAP50_95", "metrics/mAP50-95(B)")
            if train_loss is None and val_loss is None and map50 is None and map5095 is None:
                continue
            row = by_epoch.setdefault(
                epoch,
                {"epoch": epoch, "train_loss": None, "val_loss": None, "mAP50": None, "mAP50_95": None},
            )
            for key, value in (
                ("train_loss", train_loss),
                ("val_loss", val_loss),
                ("mAP50", map50),
                ("mAP50_95", map5095),
            ):
                if value is not None:
                    row[key] = value
    rows = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    if not rows:
        raise ValueError(f"No plottable history rows found in {path}")
    return rows


def _plot_curves(histories: dict[str, list[dict[str, float]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_path = output_dir / "training_validation_loss.png"
    metric_path = output_dir / "validation_metrics.png"

    figure, axis = plt.subplots(figsize=(10, 6))
    for label, rows in histories.items():
        epochs = [row["epoch"] for row in rows]
        train = [row["train_loss"] for row in rows]
        validation = [row["val_loss"] for row in rows]
        if any(value is not None for value in train):
            axis.plot(epochs, train, marker="o", label=f"{label} train loss")
        if any(value is not None for value in validation):
            axis.plot(epochs, validation, marker="x", linestyle="--", label=f"{label} validation loss")
    axis.set_title("Training and validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(loss_path, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    for label, rows in histories.items():
        epochs = [row["epoch"] for row in rows]
        map50 = [row["mAP50"] for row in rows]
        map5095 = [row["mAP50_95"] for row in rows]
        if any(value is not None for value in map50):
            axis.plot(epochs, map50, marker="o", label=f"{label} mAP50")
        if any(value is not None for value in map5095):
            axis.plot(epochs, map5095, marker="x", linestyle="--", label=f"{label} mAP50-95")
    axis.set_title("Validation metrics")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Metric")
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(metric_path, dpi=160)
    plt.close(figure)
    print(f"Loss figure: {loss_path}")
    print(f"Validation figure: {metric_path}")


def main() -> None:
    args = parse_args()
    histories = {}
    for experiment_arg in args.experiments:
        experiment = Path(experiment_arg)
        histories[experiment.name] = _load_history(experiment)
    _plot_curves(histories, args.output_dir)


if __name__ == "__main__":
    main()
