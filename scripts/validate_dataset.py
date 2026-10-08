from __future__ import annotations

import argparse
import json
from pathlib import Path

from surface_defect_detection.dataset import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO-style surface-defect dataset.")
    parser.add_argument("--data-root", type=str, default=None, help="Path to the dataset root containing train/val/test folders.")
    parser.add_argument("--json-output", type=str, default=None, help="Optional path to write JSON results.")
    parser.add_argument("--fail-on-duplicates", action="store_true", help="Return a non-zero exit code for duplicate images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_dataset(args.data_root)
    print(json.dumps(result, indent=2, default=str))
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if not result["ready_for_training"] or (
        args.fail_on_duplicates and result["duplicate_images"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
