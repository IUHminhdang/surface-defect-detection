from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from .config import DEFAULT_SPLITS, resolve_dataset_root


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _iter_image_paths(image_dir: Path) -> list[Path]:
    """Return all image files inside a YOLO split image directory."""
    paths: list[Path] = []

    for suffix in IMAGE_EXTENSIONS:
        paths.extend(image_dir.rglob(f"*{suffix}"))

    return sorted({path.resolve() for path in paths})


def _label_path_for_image(
    image_path: Path,
    label_dir: Path,
) -> Path | None:
    """Return the corresponding YOLO label file if it exists."""
    label_path = label_dir / f"{image_path.stem}.txt"

    if label_path.exists():
        return label_path

    return None


def _iter_label_paths(label_dir: Path) -> list[Path]:
    """Return label files in a split, including orphan labels."""
    if not label_dir.is_dir():
        return []
    return sorted(path.resolve() for path in label_dir.rglob("*.txt"))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_yolo_label_row(
    raw: str,
) -> tuple[int, float, float, float, float] | None:
    """
    Parse one YOLO annotation row.

    Expected format:
        class_id x_center y_center width height

    Bounding-box coordinates are normalized to [0, 1].
    """

    values = raw.strip().split()

    if len(values) != 5:
        return None

    try:
        class_id, x_center, y_center, width, height = [
            float(value) for value in values
        ]
    except ValueError:
        return None

    # Only one class: defect = 0
    if class_id != 0:
        return None

    # YOLO normalized bounding-box validation
    if not 0 <= x_center <= 1:
        return None

    if not 0 <= y_center <= 1:
        return None

    if not 0 < width <= 1:
        return None

    if not 0 < height <= 1:
        return None

    return (
        int(class_id),
        float(x_center),
        float(y_center),
        float(width),
        float(height),
    )


def _bbox_area_from_yolo(width: float, height: float) -> float:
    """
    Calculate normalized bounding-box area ratio.

    YOLO width and height are already normalized,
    therefore:

        area_ratio = width * height
    """
    return width * height


def validate_dataset(
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Validate YOLO dataset structure, images, and labels.

    Expected structure:

        root/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/

    Missing label files are treated as clean images.
    """

    root = Path(data_root) if data_root else resolve_dataset_root()

    if root is None:
        raise FileNotFoundError(
            "Dataset root could not be resolved. "
            "Pass --data-root or set DATASET_ROOT."
        )

    root = root.expanduser().resolve()

    summary: dict[str, Any] = {
        "dataset_root": str(root),
        "splits": {},
        "missing_labels": [],
        "orphan_labels": [],
        "invalid_labels": [],
        "corrupted_images": [],
        "empty_label_files": [],
        "duplicate_images": [],
        "cross_split_duplicates": [],
    }

    for split_name in DEFAULT_SPLITS:
        image_dir = root / "images" / split_name
        label_dir = root / "labels" / split_name

        if not image_dir.exists() or not label_dir.exists():
            summary["splits"][split_name] = {
                "exists": False,
                "image_count": 0,
                "label_count": 0,
            }
            continue

        image_paths = _iter_image_paths(image_dir)
        image_stems = {path.stem for path in image_paths}
        label_paths = _iter_label_paths(label_dir)
        orphan_labels = [
            str(path) for path in label_paths if path.stem not in image_stems
        ]

        label_files = 0
        invalid_rows: list[str] = []
        corrupted: list[str] = []
        empty_label_files: list[str] = []

        for image_path in image_paths:
            label_path = _label_path_for_image(
                image_path,
                label_dir,
            )

            # No label file is a valid clean image / negative sample.
            if label_path is None:
                summary["missing_labels"].append(str(image_path))
                try:
                    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                    if image is None or image.size == 0:
                        corrupted.append(str(image_path))
                except cv2.error:
                    corrupted.append(str(image_path))
                continue

            label_files += 1

            text = label_path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()

            if not text:
                # Empty label files are also valid clean-image annotations.
                continue

            for line in text.splitlines():
                if _parse_yolo_label_row(line) is None:
                    invalid_rows.append(
                        f"{label_path}:{line}"
                    )

            try:
                image = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_UNCHANGED,
                )

                if image is None or image.size == 0:
                    corrupted.append(str(image_path))

            except cv2.error:
                corrupted.append(str(image_path))

        summary["splits"][split_name] = {
            "exists": True,
            "image_count": len(image_paths),
            "label_count": label_files,
            "unlabeled_image_count": len(image_paths) - label_files,
            "invalid_label_count": len(invalid_rows),
            "corrupted_image_count": len(corrupted),
            "empty_label_count": len(empty_label_files),
            "orphan_label_count": len(orphan_labels),
        }

        summary["orphan_labels"].extend(orphan_labels)
        summary["invalid_labels"].extend(invalid_rows)
        summary["corrupted_images"].extend(corrupted)
        summary["empty_label_files"].extend(empty_label_files)

    hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for split_name in DEFAULT_SPLITS:
        image_dir = root / "images" / split_name
        for image_path in _iter_image_paths(image_dir):
            try:
                hashes[_file_hash(image_path)].append(
                    {"split": split_name, "image_path": str(image_path)}
                )
            except OSError:
                continue

    for members in hashes.values():
        if len(members) > 1:
            summary["duplicate_images"].append(members)
            if len({member["split"] for member in members}) > 1:
                summary["cross_split_duplicates"].append(members)

    summary["ready_for_training"] = (
        len(summary["invalid_labels"]) == 0
        and len(summary["corrupted_images"]) == 0
        and len(summary["orphan_labels"]) == 0
        and len(summary["cross_split_duplicates"]) == 0
    )

    return summary


def audit_dataset(
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Collect dataset audit metrics and save CSVs
    plus ground-truth sample visualizations.
    """

    root = Path(data_root) if data_root else resolve_dataset_root()

    if root is None:
        raise FileNotFoundError(
            "Dataset root could not be resolved. "
            "Pass --data-root or set DATASET_ROOT."
        )

    root = root.expanduser().resolve()

    validation = validate_dataset(root)

    reports_root = (
        Path(output_dir)
        if output_dir
        else root.parent / "reports"
    )

    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir = reports_root / "figures"
    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, Any]] = []
    bbox_rows: list[dict[str, Any]] = []
    defect_size_counter: Counter[str] = Counter()

    # ---------------------------------------------------------
    # Dataset statistics
    # ---------------------------------------------------------

    for split_name in DEFAULT_SPLITS:

        image_dir = root / "images" / split_name
        label_dir = root / "labels" / split_name

        if not image_dir.exists():
            continue

        for image_path in _iter_image_paths(image_dir):

            label_path = _label_path_for_image(
                image_path,
                label_dir,
            )

            bbox_count = 0

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_COLOR,
            )

            height = (
                image.shape[0]
                if image is not None
                else 0
            )

            width = (
                image.shape[1]
                if image is not None
                else 0
            )

            if label_path is not None:

                for line in label_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ).splitlines():

                    parsed = _parse_yolo_label_row(line)

                    if parsed is None:
                        continue

                    (
                        _,
                        x_center,
                        y_center,
                        bbox_w,
                        bbox_h,
                    ) = parsed

                    # YOLO width/height are normalized.
                    area_ratio = _bbox_area_from_yolo(
                        bbox_w,
                        bbox_h,
                    )

                    aspect_ratio = (
                        bbox_w / bbox_h
                        if bbox_h > 0
                        else 0.0
                    )

                    # Size buckets:
                    # small  < 0.01
                    # medium 0.01 - < 0.05
                    # large  >= 0.05

                    if area_ratio >= 0.05:
                        size_bucket = "large"
                    elif area_ratio >= 0.01:
                        size_bucket = "medium"
                    else:
                        size_bucket = "small"

                    defect_size_counter[
                        size_bucket
                    ] += 1

                    bbox_rows.append(
                        {
                            "split": split_name,
                            "image_id": image_path.stem,
                            "image_path": str(image_path),
                            "bbox_x_center_norm": x_center,
                            "bbox_y_center_norm": y_center,
                            "bbox_width_norm": bbox_w,
                            "bbox_height_norm": bbox_h,
                            "bbox_width_px": bbox_w * width,
                            "bbox_height_px": bbox_h * height,
                            "bbox_area_ratio": area_ratio,
                            "bbox_aspect_ratio": aspect_ratio,
                            "defect_size": size_bucket,
                        }
                    )

                    bbox_count += 1

            rows.append(
                {
                    "split": split_name,
                    "image_id": image_path.stem,
                    "image_path": str(image_path),
                    "is_defect": bbox_count > 0,
                    "gt_count": bbox_count,
                    "image_width": width,
                    "image_height": height,
                }
            )

    dataset_frame = pd.DataFrame(rows)
    bbox_frame = pd.DataFrame(bbox_rows)

    dataset_frame.to_csv(
        reports_root / "dataset_summary.csv",
        index=False,
    )

    bbox_frame.to_csv(
        reports_root / "dataset_bbox_stats.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = {
        "validation": validation,

        "split_summary": (
            dataset_frame.groupby("split")
            .agg(
                images=("image_id", "count"),
                defect_images=(
                    "is_defect",
                    lambda s: int(s.sum()),
                ),
                clean_images=(
                    "is_defect",
                    lambda s: int(
                        (~s.astype(bool)).sum()
                    ),
                ),
                total_boxes=("gt_count", "sum"),
            )
            .reset_index()
            .to_dict(orient="records")
        ),

        "bbox_summary": {
            "width_norm": (
                bbox_frame["bbox_width_norm"]
                .describe()
                .to_dict()
                if not bbox_frame.empty
                else {}
            ),

            "height_norm": (
                bbox_frame["bbox_height_norm"]
                .describe()
                .to_dict()
                if not bbox_frame.empty
                else {}
            ),

            "width_px": (
                bbox_frame["bbox_width_px"]
                .describe()
                .to_dict()
                if not bbox_frame.empty
                else {}
            ),

            "height_px": (
                bbox_frame["bbox_height_px"]
                .describe()
                .to_dict()
                if not bbox_frame.empty
                else {}
            ),

            "area_ratio": (
                bbox_frame["bbox_area_ratio"]
                .describe()
                .to_dict()
                if not bbox_frame.empty
                else {}
            ),

            "aspect_ratio": (
                bbox_frame["bbox_aspect_ratio"]
                .describe()
                .to_dict()
                if not bbox_frame.empty
                else {}
            ),
        },

        "defect_size_distribution": dict(
            defect_size_counter
        ),
    }

    # ---------------------------------------------------------
    # Ground-truth visualization
    # ---------------------------------------------------------

    for split_name in DEFAULT_SPLITS:

        image_dir = root / "images" / split_name
        label_dir = root / "labels" / split_name

        if not image_dir.exists():
            continue

        sample_path = next(
            iter(_iter_image_paths(image_dir)),
            None,
        )

        if sample_path is None:
            continue

        image = cv2.imread(
            str(sample_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            continue

        label_path = _label_path_for_image(
            sample_path,
            label_dir,
        )

        if label_path is not None:

            for line in label_path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines():

                parsed = _parse_yolo_label_row(line)

                if parsed is None:
                    continue

                _, cx, cy, bw, bh = parsed

                h, w = image.shape[:2]

                x1 = int(
                    (cx - bw / 2) * w
                )
                y1 = int(
                    (cy - bh / 2) * h
                )
                x2 = int(
                    (cx + bw / 2) * w
                )
                y2 = int(
                    (cy + bh / 2) * h
                )

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

        cv2.imwrite(
            str(
                figure_dir
                / f"gt_visualization_{split_name}.png"
            ),
            image,
        )

    return summary