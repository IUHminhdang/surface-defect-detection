from __future__ import annotations

import argparse
import csv
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO

from surface_defect_detection.config import resolve_dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark inference speed for a trained defect detector.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name, for example yolo11 or rtdetr.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Deployment batch size. Use 1 for comparable latency.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", type=str, default="0",
                        help="Ultralytics device, for example 0 or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_dataset_root(args.data_root)
    if root is None:
        raise FileNotFoundError("Dataset root could not be resolved.")
    image_dir = root / "images" / "test"
    image_paths = sorted(
        path for path in image_dir.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {image_dir}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1.")

    model = YOLO(args.weights)
    total_batches = args.warmup + args.iterations
    sample_paths = image_paths * ((total_batches * args.batch_size) // len(image_paths) + 1)

    def batches(start: int, count: int):
        for offset in range(start, start + count):
            first = offset * args.batch_size
            yield [str(path) for path in sample_paths[first:first + args.batch_size]]

    for batch in batches(0, args.warmup):
        model.predict(batch, imgsz=args.image_size, device=args.device,
                      batch=args.batch_size, verbose=False)

    started = time.perf_counter()
    for batch in batches(args.warmup, args.iterations):
        model.predict(batch, imgsz=args.image_size, device=args.device,
                      batch=args.batch_size, verbose=False)
    elapsed = time.perf_counter() - started
    image_count = args.iterations * args.batch_size
    latency_ms = elapsed / args.iterations * 1000
    fps = image_count / elapsed
    output_path = Path("experiments") / args.experiment / "benchmark_speed.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "experiment", "weights", "device", "image_size", "batch_size",
            "warmup_iterations", "iterations", "images_processed", "fps",
            "latency_ms_per_batch", "latency_ms_per_image", "precision",
            "hardware", "timestamp", "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "experiment": args.experiment,
            "weights": str(Path(args.weights)),
            "device": args.device,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "warmup_iterations": args.warmup,
            "iterations": args.iterations,
            "images_processed": image_count,
            "fps": round(fps, 6),
            "latency_ms_per_batch": round(latency_ms, 6),
            "latency_ms_per_image": round(latency_ms / args.batch_size, 6),
            "precision": "fp32",
            "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": "inference_only=true;same_test_images=true",
        })
    print(f"Benchmark written to {output_path}")


if __name__ == "__main__":
    main()
