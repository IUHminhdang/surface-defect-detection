from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from ultralytics import YOLO

from surface_defect_detection.config import resolve_dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark inference speed for a trained defect detector.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
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
    model = YOLO(args.weights)
    sample_paths = (image_paths * ((args.warmup + args.iterations) // len(image_paths) + 1))
    for path in sample_paths[:args.warmup]:
        model.predict(str(path), imgsz=args.image_size, verbose=False)
    started = time.perf_counter()
    for path in sample_paths[args.warmup:args.warmup + args.iterations]:
        model.predict(str(path), imgsz=args.image_size, verbose=False)
    elapsed = time.perf_counter() - started
    latency_ms = elapsed / args.iterations * 1000
    fps = args.iterations / elapsed
    output_path = Path("experiments") / "baseline" / "benchmark_speed.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["weights", "image_size", "batch_size", "fps", "latency_ms", "notes"])
        writer.writerow([args.weights, args.image_size, args.batch_size, fps, latency_ms,
                         f"warmup={args.warmup};iterations={args.iterations};inference_only=true"])
    print(f"Benchmark written to {output_path}")


if __name__ == "__main__":
    main()
