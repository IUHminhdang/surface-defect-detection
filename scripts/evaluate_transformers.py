from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


def _read_targets(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    targets = []
    if not label_path.exists():
        return targets
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        _, cx, cy, bw, bh = map(float, values)
        targets.append({
            "box": [(cx - bw / 2) * width, (cy - bh / 2) * height,
                    (cx + bw / 2) * width, (cy + bh / 2) * height],
            "area_ratio": bw * bh,
        })
    return targets


def _post_process(processor: Any, outputs: Any, encoded: dict[str, Any],
                  size: tuple[int, int], grounding: bool) -> list[dict[str, Any]]:
    height, width = size
    target_size = torch.tensor([[height, width]], device=outputs.logits.device)
    if grounding and hasattr(processor, "post_process_grounded_object_detection"):
        processed = processor.post_process_grounded_object_detection(
            outputs, input_ids=encoded.get("input_ids"), threshold=0.0,
            text_threshold=0.0, target_sizes=target_size,
        )[0]
    else:
        processed = processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=target_size,
        )[0]
    return [
        {"box": [float(value) for value in box], "score": float(score)}
        for box, score in zip(processed["boxes"].cpu(), processed["scores"].cpu())
    ]


def _average_precision(predictions: list[dict[str, Any]], targets: dict[str, list[dict[str, Any]]],
                       iou_threshold: float, confidence: float = 0.0) -> float:
    candidates = sorted(
        [item for item in predictions if item["score"] >= confidence],
        key=lambda item: item["score"], reverse=True,
    )
    matched: dict[str, set[int]] = {image_id: set() for image_id in targets}
    true_positives, false_positives = [], []
    total_targets = sum(len(items) for items in targets.values())
    for prediction in candidates:
        image_targets = targets.get(prediction["image_id"], [])
        best_index, best_overlap = -1, 0.0
        for index, target in enumerate(image_targets):
            overlap = _iou(prediction["box"], target["box"])
            if index not in matched.setdefault(prediction["image_id"], set()) and overlap > best_overlap:
                best_index, best_overlap = index, overlap
        if best_overlap >= iou_threshold and best_index >= 0:
            matched[prediction["image_id"]].add(best_index)
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)
    if total_targets == 0:
        return 0.0
    tp = torch.tensor(true_positives, dtype=torch.float32).cumsum(0)
    fp = torch.tensor(false_positives, dtype=torch.float32).cumsum(0)
    recall = tp / total_targets
    precision = tp / torch.clamp(tp + fp, min=1)
    recall_points = torch.linspace(0, 1, 101)
    return float(torch.tensor([
        precision[recall >= point].max().item() if torch.any(recall >= point) else 0.0
        for point in recall_points
    ]).mean())


def evaluate_model(model: Any, processor: Any, data_root: Path, split: str,
                   model_key: str, output_dir: Path, image_size: int = 640) -> dict[str, float]:
    image_dir, label_dir = data_root / "images" / split, data_root / "labels" / split
    predictions, targets, prediction_rows, error_rows = [], {}, [], []
    model.eval()
    device = next(model.parameters()).device
    for image_id, image_path in enumerate(sorted(image_dir.rglob("*"))):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        target_items = _read_targets(label_dir / f"{image_path.stem}.txt", width, height)
        targets[image_path.stem] = target_items
        processor_kwargs = {"images": image, "return_tensors": "pt"}
        if model_key == "grounding-dino":
            processor_kwargs["text"] = "defect"
        encoded = processor(**processor_kwargs)
        encoded = {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()
                   if value is not None}
        with torch.no_grad():
            outputs = model(**encoded)
        image_predictions = _post_process(processor, outputs, encoded, (height, width),
                                           model_key == "grounding-dino")
        for prediction in image_predictions:
            prediction["image_id"] = image_path.stem
            predictions.append(prediction)
        matched = []
        for target_index, target in enumerate(target_items):
            overlaps = [(index, _iou(target["box"], prediction["box"]))
                        for index, prediction in enumerate(image_predictions)]
            best = max(overlaps, key=lambda item: item[1], default=(-1, 0.0))
            if best[1] >= 0.5:
                matched.append((target_index, best[0], best[1]))
        prediction_rows.append({
            "image_id": image_path.stem, "image_path": str(image_path), "domain": "unknown",
            "gt_count": len(target_items), "pred_count": len(image_predictions),
            "confidence": max((p["score"] for p in image_predictions), default=0.0),
            "iou": max((match[2] for match in matched), default=0.0),
            "tp": len(matched), "fp": max(0, len(image_predictions) - len(matched)),
            "fn": max(0, len(target_items) - len(matched)),
            "gt_bbox": json.dumps([item["box"] for item in target_items]),
            "pred_bbox": json.dumps([item["box"] for item in image_predictions]),
            "defect_size": json.dumps([
                "small" if item["area_ratio"] < 0.01 else
                "medium" if item["area_ratio"] < 0.05 else "large"
                for item in target_items
            ]),
        })
        matched_targets = {item[0] for item in matched}
        matched_predictions = {item[1] for item in matched}
        for index, target in enumerate(target_items):
            if index not in matched_targets:
                error_rows.append({
                    "image_id": image_path.stem, "domain": "unknown", "error_type": "FN",
                    "defect_size": "small" if target["area_ratio"] < 0.01 else
                    "medium" if target["area_ratio"] < 0.05 else "large",
                    "confidence": 0.0, "iou": 0.0, "gt_area_ratio": target["area_ratio"],
                    "pred_area_ratio": 0.0,
                })
        for index, prediction in enumerate(image_predictions):
            if index not in matched_predictions:
                error_rows.append({
                    "image_id": image_path.stem, "domain": "unknown", "error_type": "FP",
                    "defect_size": "", "confidence": prediction["score"], "iou": 0.0,
                    "gt_area_ratio": 0.0,
                    "pred_area_ratio": ((prediction["box"][2] - prediction["box"][0]) *
                                        (prediction["box"][3] - prediction["box"][1])) / (width * height),
                })
    map50 = _average_precision(predictions, targets, 0.5)
    map5095 = sum(_average_precision(predictions, targets, threshold)
                  for threshold in [0.5 + index * 0.05 for index in range(10)]) / 10
    tp = sum(row["tp"] for row in prediction_rows)
    fp = sum(row["fp"] for row in prediction_rows)
    fn = sum(row["fn"] for row in prediction_rows)
    metrics = {
        "mAP50": map50, "mAP50_95": map5095,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["image_id", "image_path", "domain", "gt_count", "pred_count", "confidence", "iou",
                  "tp", "fp", "fn", "gt_bbox", "pred_bbox", "defect_size"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(prediction_rows)
    with (output_dir / "error_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["image_id", "domain", "error_type", "defect_size", "confidence", "iou",
                  "gt_area_ratio", "pred_area_ratio"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(error_rows)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a DINO-family checkpoint.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model", choices=["dino", "grounding-dino"], required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    from transformers import AutoModelForObjectDetection, AutoModelForZeroShotObjectDetection, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.weights)
    model_class = AutoModelForZeroShotObjectDetection if args.model == "grounding-dino" else AutoModelForObjectDetection
    model = model_class.from_pretrained(args.weights)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_model(model, processor, Path(args.data_root), args.split, args.model,
                             Path("experiments") / args.experiment)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
