from __future__ import annotations

import csv
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


class YoloDetectionDataset(Dataset):
    def __init__(self, root: Path, split: str, processor: Any, grounding: bool) -> None:
        self.image_dir = root / "images" / split
        self.label_dir = root / "labels" / split
        self.paths = sorted(
            path for path in self.image_dir.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        self.processor = processor
        self.grounding = grounding

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        annotations = []
        label_path = self.label_dir / f"{path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                values = line.split()
                if len(values) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, values)
                annotations.append({
                    "bbox": [(cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height],
                    "category_id": 0,
                    "area": bw * bh * width * height,
                    "iscrowd": 0,
                })
        target = {"image_id": index, "annotations": annotations}
        if self.grounding:
            return self.processor(images=image, text="defect", annotations=target, return_tensors="pt")
        return self.processor(images=image, annotations=target, return_tensors="pt")


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key].squeeze(0) for item in batch]
        if key == "labels":
            result[key] = values
        elif isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


def train_transformers_model(
    model_key: str,
    data_root: Path,
    config: dict[str, Any],
    experiment: str,
    device: str,
) -> None:
    try:
        from transformers import (
            AutoModelForObjectDetection,
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            TrainingArguments,
            Trainer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "DINO/Grounding DINO require transformers and accelerate. "
            "Install requirements.txt before training."
        ) from exc

    training = config.get("training", {})
    model_ids = training.get("models", {})
    default_id = (
        "IDEA-Research/dino-resnet-50"
        if model_key == "dino"
        else "IDEA-Research/grounding-dino-tiny"
    )
    model_id = str(model_ids.get(model_key, default_id))
    output_dir = Path("experiments") / experiment
    weights_dir = output_dir / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model_class = (
        AutoModelForZeroShotObjectDetection
        if model_key == "grounding-dino"
        else AutoModelForObjectDetection
    )
    model_kwargs = {"ignore_mismatched_sizes": True}
    if model_key == "dino":
        model_kwargs["num_labels"] = 1
    model = model_class.from_pretrained(model_id, **model_kwargs)
    train_set = YoloDetectionDataset(data_root, "train", processor, model_key == "grounding-dino")
    val_set = YoloDetectionDataset(data_root, "val", processor, model_key == "grounding-dino")
    use_cuda = device != "cpu" and torch.cuda.is_available()
    started = time.perf_counter()
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=float(training.get("epochs", 50)),
        per_device_train_batch_size=int(training.get("batch_size", 8)),
        per_device_eval_batch_size=int(training.get("batch_size", 8)),
        learning_rate=float(training.get("learning_rate", 0.001)),
        weight_decay=float(training.get("weight_decay", 0.0005)),
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=2,
        fp16=use_cuda,
        report_to="none",
        seed=int(training.get("seed", 42)),
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_set, eval_dataset=val_set, data_collator=_collate)
    trainer.train()
    trainer.save_model(str(weights_dir / "best"))
    processor.save_pretrained(str(weights_dir / "best"))
    shutil.copy2(weights_dir / "best" / "config.json", weights_dir / "config.json")
    from evaluate_transformers import evaluate_model
    metrics = evaluate_model(
        model, processor, data_root, "val", model_key, output_dir,
        int(training.get("image_size", 640)),
    )
    history = [row for row in trainer.state.log_history if "loss" in row or "eval_loss" in row]
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["epoch", "train_loss", "val_loss", "precision", "recall", "mAP50", "mAP50_95", "learning_rate", "epoch_time_sec"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in history:
            writer.writerow({
                "epoch": row.get("epoch", ""), "train_loss": row.get("loss", ""),
                "val_loss": row.get("eval_loss", ""), "learning_rate": row.get("learning_rate", ""),
                "precision": metrics["precision"], "recall": metrics["recall"],
                "mAP50": metrics["mAP50"], "mAP50_95": metrics["mAP50_95"],
                "epoch_time_sec": row.get("train_runtime", ""),
            })
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["experiment", "model", "seed", "epochs", "best_epoch", "image_size", "batch_size",
                  "mAP50", "mAP50_95", "precision", "recall", "fps", "latency_ms", "params", "flops",
                  "model_size_mb", "gpu_memory_mb", "training_time_sec", "checkpoint_path", "timestamp"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerow({
            "experiment": experiment, "model": model_id, "seed": training.get("seed", 42),
            "epochs": training.get("epochs", 50), "best_epoch": "", "image_size": training.get("image_size", 640),
            "batch_size": training.get("batch_size", 8), "mAP50": metrics["mAP50"],
            "mAP50_95": metrics["mAP50_95"], "precision": metrics["precision"],
            "recall": metrics["recall"], "fps": "", "latency_ms": "",
            "params": sum(p.numel() for p in model.parameters()),
            "flops": "", "model_size_mb": "", "gpu_memory_mb": "",
            "training_time_sec": round(time.perf_counter() - started, 3),
            "checkpoint_path": str(weights_dir / "best"), "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    print(f"Training complete: {output_dir}")
