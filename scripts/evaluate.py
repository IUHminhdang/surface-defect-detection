from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO

from surface_defect_detection.config import load_yaml_config, resolve_dataset_root
from surface_defect_detection.dataset import _parse_yolo_label_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO detector and emit experiment CSVs.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--experiment", type=str, default="baseline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    return parser.parse_args()


def _xyxy(box: tuple[int, float, float, float, float], width: int, height: int) -> list[float]:
    _, cx, cy, bw, bh = box
    return [(cx - bw / 2) * width, (cy - bh / 2) * height,
            (cx + bw / 2) * width, (cy + bh / 2) * height]


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _ground_truth(label_path: Path, width: int, height: int) -> list[tuple[list[float], float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = _parse_yolo_label_row(line)
        if parsed:
            boxes.append((_xyxy(parsed, width, height), parsed[3] * parsed[4]))
    return boxes


def main() -> None:
    args = parse_args()
    root = resolve_dataset_root(args.data_root)
    if root is None:
        raise FileNotFoundError("Dataset root could not be resolved.")
    config = load_yaml_config(args.config)
    image_size = int(config.get("training", {}).get("image_size", 640))
    experiment_dir = Path("experiments") / args.experiment
    image_dir = root / "images" / args.split
    label_dir = root / "labels" / args.split
    model = YOLO(args.weights)
    prediction_rows, error_rows = [], []
    started = time.perf_counter()
    results = model.predict(source=str(image_dir), imgsz=image_size, conf=0.25, iou=0.5,
                            stream=True, verbose=False, save=False)
    for result in results:
        path = Path(result.path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        gt = _ground_truth(label_dir / f"{path.stem}.txt", width, height)
        predictions = []
        if result.boxes is not None:
            for box, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                predictions.append((box, float(confidence)))
        matched_gt, matched_pred = set(), set()
        matches = []
        candidates = sorted(
            ((_iou(g[0], p[0]), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(predictions)),
            reverse=True,
        )
        for overlap, gi, pi in candidates:
            if overlap >= 0.5 and gi not in matched_gt and pi not in matched_pred:
                matched_gt.add(gi); matched_pred.add(pi); matches.append((gi, pi, overlap))
        tp, fn, fp = len(matches), len(gt) - len(matches), len(predictions) - len(matches)
        best_conf = max((p[1] for p in predictions), default=0.0)
        best_iou = max((item[2] for item in matches), default=0.0)
        prediction_rows.append({
            "image_id": path.stem, "image_path": str(path), "domain": "unknown",
            "gt_count": len(gt), "pred_count": len(predictions), "confidence": best_conf,
            "iou": best_iou, "tp": tp, "fp": fp, "fn": fn,
            "gt_bbox": json.dumps([g[0] for g in gt]), "pred_bbox": json.dumps([p[0] for p in predictions]),
            "defect_size": json.dumps(["small" if g[1] < 0.01 else "medium" if g[1] < 0.05 else "large" for g in gt]),
        })
        for gi, g in enumerate(gt):
            if gi not in matched_gt:
                error_rows.append({"image_id": path.stem, "domain": "unknown", "error_type": "FN",
                                   "defect_size": "small" if g[1] < 0.01 else "medium" if g[1] < 0.05 else "large",
                                   "confidence": 0.0, "iou": 0.0, "gt_area_ratio": g[1], "pred_area_ratio": 0.0})
        for pi, p in enumerate(predictions):
            if pi not in matched_pred:
                error_rows.append({"image_id": path.stem, "domain": "unknown", "error_type": "FP",
                                   "defect_size": "", "confidence": p[1], "iou": 0.0,
                                   "gt_area_ratio": 0.0, "pred_area_ratio": ((p[0][2]-p[0][0])*(p[0][3]-p[0][1]))/(width*height)})
        for gi, pi, overlap in matches:
            if overlap < 0.75:
                error_rows.append({"image_id": path.stem, "domain": "unknown", "error_type": "LOCALIZATION_ERROR",
                                   "defect_size": "small" if gt[gi][1] < 0.01 else "medium" if gt[gi][1] < 0.05 else "large",
                                   "confidence": predictions[pi][1], "iou": overlap, "gt_area_ratio": gt[gi][1],
                                   "pred_area_ratio": ((predictions[pi][0][2]-predictions[pi][0][0]) *
                                                       (predictions[pi][0][3]-predictions[pi][0][1]))/(width*height)})
    elapsed = time.perf_counter() - started
    experiment_dir.mkdir(parents=True, exist_ok=True)
    prediction_fields = ["image_id", "image_path", "domain", "gt_count", "pred_count", "confidence", "iou", "tp", "fp", "fn", "gt_bbox", "pred_bbox", "defect_size"]
    error_fields = ["image_id", "domain", "error_type", "defect_size", "confidence", "iou", "gt_area_ratio", "pred_area_ratio"]
    with (experiment_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields); writer.writeheader(); writer.writerows(prediction_rows)
    with (experiment_dir / "error_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=error_fields); writer.writeheader(); writer.writerows(error_rows)
    print(f"Evaluated {len(prediction_rows)} images in {elapsed:.1f}s")
    print(f"Predictions: {experiment_dir / 'predictions.csv'}")
    print(f"Errors: {experiment_dir / 'error_analysis.csv'}")


if __name__ == "__main__":
    main()
