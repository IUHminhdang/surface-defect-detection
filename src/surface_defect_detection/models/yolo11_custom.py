from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO
from ultralytics.nn import tasks

from .coordinate_attention import CoordinateAttention
from .multiscale import MultiScaleFusion
from .p2 import P2HighResolution


def register_custom_modules() -> None:
    """Register custom YAML layer names with the installed Ultralytics parser."""
    tasks.P2HighResolution = P2HighResolution
    tasks.MultiScaleFusion = MultiScaleFusion
    tasks.CoordinateAttention = CoordinateAttention


def _custom_yaml(ablation: str, weights: str) -> dict[str, Any]:
    baseline = copy.deepcopy(YOLO(weights).model.yaml)
    baseline["nc"] = 1
    baseline["scale"] = "n"
    baseline.pop("yaml_file", None)
    head: list[list[Any]] = []
    add = head.append

    add([-1, 1, "nn.Upsample", ["None", 2, "nearest"]])
    add([[-1, 6], 1, "Concat", [1]])
    add([-1, 2, "C3k2", [512, False]])
    add([-1, 1, "nn.Upsample", ["None", 2, "nearest"]])
    add([[-1, 4], 1, "Concat", [1]])

    p3_source = 15
    if ablation in {"E1", "E2", "E3"}:
        add([4, 1, "P2HighResolution", [128, 3, 1]])
        add([[-1, p3_source], 1, "Concat", [1]])
        add([-1, 2, "C3k2", [256, False]])
    else:
        add([-1, 2, "C3k2", [256, False]])
    p3_index = 18 if ablation in {"E1", "E2", "E3"} else 16

    if ablation in {"E2", "E3"}:
        add([-1, 1, "MultiScaleFusion", [64]])
        p3_index += 1
    if ablation == "E3":
        add([-1, 1, "CoordinateAttention", [64]])
        p3_index += 1

    add([-1, 1, "Conv", [256, 3, 2]])
    p4_downsample = len(head) + 10
    add([[-1, 13], 1, "Concat", [1]])
    add([-1, 2, "C3k2", [512, False]])
    p4_index = len(head) + 10
    add([-1, 1, "Conv", [512, 3, 2]])
    add([[-1, 10], 1, "Concat", [1]])
    add([-1, 2, "C3k2", [1024, True]])
    p5_index = len(head) + 10
    add([[p3_index, p4_index, p5_index], 1, "Detect", ["nc"]])

    baseline["head"] = head
    baseline["custom_ablation"] = ablation
    baseline["custom_pretrained_weights"] = str(Path(weights))
    return baseline


def build_yolo11_custom(
    ablation: str,
    output_dir: str | Path,
    weights: str = "yolo11n.pt",
) -> YOLO:
    if ablation not in {"E0", "E1", "E2", "E3"}:
        raise ValueError(f"Unsupported ablation: {ablation}")
    register_custom_modules()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    yaml_path = output_path / f"yolo11_{ablation.lower()}.yaml"
    yaml_path.write_text(yaml.safe_dump(_custom_yaml(ablation, weights), sort_keys=False), encoding="utf-8")
    model = YOLO(str(yaml_path))
    if ablation != "E0":
        model.load(weights)
    return model
