from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DATASET_ENV_VARS = ("DATASET_ROOT", "KAGGLE_INPUT_PATH", "KAGGLE_DATASET_PATH")
DEFAULT_SPLITS = ("train", "val", "test")
default_split_names = DEFAULT_SPLITS


def resolve_dataset_root(cli_value: str | None = None) -> Path | None:
    """Resolve the dataset root from CLI, environment variables, or common locations."""
    candidates: list[str] = []

    if cli_value:
        candidates.append(cli_value)

    for key in DATASET_ENV_VARS:
        value = os.getenv(key)
        if value:
            candidates.append(value)

    candidates.extend([
        "/kaggle/input",
        str(Path.cwd() / "data"),
        str(Path.cwd() / "dataset"),
    ])

    for candidate in candidates:
        path = Path(candidate).expanduser()

        if not path.exists():
            continue

        # YOLO dataset structure:
        # root/images/{train,val,test}
        # root/labels/{train,val,test}
        if (
            (path / "images").is_dir()
            and (path / "labels").is_dir()
            and all(
                (path / "images" / split).is_dir()
                and (path / "labels" / split).is_dir()
                for split in DEFAULT_SPLITS
            )
        ):
            return path

    if cli_value:
        return Path(cli_value).expanduser()

    return None


def load_yaml_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_output_dir(path: str | os.PathLike[str]) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
