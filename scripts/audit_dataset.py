from __future__ import annotations

import argparse
import json
from pathlib import Path

from surface_defect_detection.dataset import audit_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a surface-defect dataset and save summary statistics.")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save dataset audit CSVs and figures.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path for audit thresholds.")
    parser.add_argument("--json-output", type=str, default=None, help="Optional path to write the audit summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_dataset(data_root=args.data_root, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, default=str))
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
