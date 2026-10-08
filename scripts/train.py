from __future__ import annotations

import argparse
import csv
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from ultralytics import YOLO

from surface_defect_detection.config import load_yaml_config, resolve_dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one of the five reproducible baseline detectors.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", choices=["yolo11", "rtdetr", "dino", "grounding-dino", "yolo-world"],
                        default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    return parser.parse_args()


ULTRALYTICS_MODELS = {
    "yolo11": "yolo11n.pt",
    "rtdetr": "rtdetr-l.pt",
    "yolo-world": "yolov8s-worldv2.pt",
}


def _write_data_yaml(root: Path, output: Path) -> None:
    output.write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val",
             "test": "images/test", "nc": 1, "names": {0: "defect"}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _copy_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    root = resolve_dataset_root(args.data_root)
    if root is None:
        raise FileNotFoundError("Dataset root could not be resolved.")
    config = load_yaml_config(args.config)
    training = config.get("training", {})
    model_key = args.model or str(training.get("baseline_model", "yolo11")).lower()
    if model_key in {"dino", "grounding-dino"}:
        from train_transformers import train_transformers_model
        train_transformers_model(
            model_key=model_key,
            data_root=root,
            config=config,
            experiment=args.experiment or model_key,
            device=args.device,
        )
        return
    if model_key not in ULTRALYTICS_MODELS:
        raise ValueError(f"Unsupported baseline model: {model_key}")
    experiment_name = args.experiment or model_key
    experiment_dir = Path("experiments") / experiment_name
    weights_dir = experiment_dir / "weights"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = experiment_dir / "dataset.yaml"
    _write_data_yaml(root, data_yaml)

    model_name = str(training.get("models", {}).get(model_key, ULTRALYTICS_MODELS[model_key]))
    model = YOLO(model_name)
    started = time.perf_counter()
    results = model.train(
        data=str(data_yaml),
        project=str(experiment_dir),
        name="run",
        exist_ok=True,
        epochs=int(training.get("epochs", 50)) if args.epochs is None else args.epochs,
        imgsz=int(training.get("image_size", 640)),
        batch=int(training.get("batch_size", 8)),
        optimizer=str(training.get("optimizer", "AdamW")),
        lr0=float(training.get("learning_rate", 0.001)),
        weight_decay=float(training.get("weight_decay", 0.0005)),
        cos_lr=str(training.get("scheduler", "cosine")).lower() == "cosine",
        warmup_epochs=3.0 if training.get("warmup", True) else 0.0,
        patience=int(training.get("early_stopping", {}).get("patience", 10)),
        seed=int(training.get("seed", 42)),
        deterministic=True,
        save=True,
        plots=True,
        device=None if args.device == "auto" else args.device,
    )
    elapsed = time.perf_counter() - started
    run_dir = Path(results.save_dir)
    _copy_checkpoint(run_dir / "weights" / "best.pt", weights_dir / "best.pt")
    _copy_checkpoint(run_dir / "weights" / "last.pt", weights_dir / "last.pt")

    history_source = run_dir / "results.csv"
    history_target = experiment_dir / "training_history.csv"
    with history_source.open(newline="", encoding="utf-8") as source, history_target.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        source_rows = list(csv.DictReader(source))
        fieldnames = ["epoch", "train_loss", "val_loss", "precision", "recall",
                      "mAP50", "mAP50_95", "learning_rate", "epoch_time_sec"]
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(source_rows, start=1):
            writer.writerow({
                "epoch": index,
                "train_loss": row.get("train/box_loss", ""),
                "val_loss": row.get("val/box_loss", ""),
                "precision": row.get("metrics/precision(B)", ""),
                "recall": row.get("metrics/recall(B)", ""),
                "mAP50": row.get("metrics/mAP50(B)", ""),
                "mAP50_95": row.get("metrics/mAP50-95(B)", ""),
                "learning_rate": row.get("lr/pg0", ""),
                "epoch_time_sec": row.get("time", ""),
            })

    best_row = max(source_rows, key=lambda row: float(row.get("metrics/mAP50-95(B)", 0) or 0))
    results_fields = [
        "experiment", "model", "seed", "epochs", "best_epoch", "image_size", "batch_size",
        "mAP50", "mAP50_95", "precision", "recall", "fps", "latency_ms", "params", "flops",
        "model_size_mb", "gpu_memory_mb", "training_time_sec", "checkpoint_path", "timestamp",
    ]
    with (experiment_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results_fields)
        writer.writeheader()
        writer.writerow({
            "experiment": experiment_name,
            "model": model_name,
            "seed": training.get("seed", 42),
            "epochs": len(source_rows),
            "best_epoch": source_rows.index(best_row) + 1,
            "image_size": training.get("image_size", 640),
            "batch_size": training.get("batch_size", 8),
            "mAP50": best_row.get("metrics/mAP50(B)", ""),
            "mAP50_95": best_row.get("metrics/mAP50-95(B)", ""),
            "precision": best_row.get("metrics/precision(B)", ""),
            "recall": best_row.get("metrics/recall(B)", ""),
            "fps": "", "latency_ms": "",
            "params": "", "flops": "",
            "model_size_mb": round((weights_dir / "best.pt").stat().st_size / 1024**2, 3),
            "gpu_memory_mb": "",
            "training_time_sec": round(elapsed, 3),
            "checkpoint_path": str(weights_dir / "best.pt"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    print(f"Training complete: {run_dir}")
    print(f"Elapsed seconds: {elapsed:.1f}")
    print(f"Best checkpoint: {weights_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
