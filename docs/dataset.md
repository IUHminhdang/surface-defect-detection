# Dataset documentation

This project is designed for the combined dataset containing:

- KolektorSDD2
- Magnetic Tile Defect Dataset

## Split structure

The dataset is expected to be organized as:

```text
<dataset_root>/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Images can use a variety of standard file extensions, and labels are expected to follow YOLO object-detection format:

```text
class_id x_center y_center width height
```

For this project, the value is fixed to a single class:

```text
class_id = 0
```

## Validation rules

The `validate_dataset.py` entry point checks for:

- invalid or unreadable label rows
- label files that are empty
- corrupted or unreadable image files
- split-level integrity
- orphan label files
- duplicate image content, including duplicates crossing splits

Images without a label file are treated as valid clean/unlabeled images. An
empty label file is also valid and represents a clean image. Duplicate images
within a split are reported for review; duplicates crossing train, validation,
or test splits prevent the dataset from being marked ready for training.

## Audit metrics

The dataset audit summarizes:

- clean vs. defect image counts
- total boxes and boxes per image
- distribution of defect sizes
- normalized and pixel bbox widths/heights
- normalized bbox area ratios (`width_norm * height_norm`)
- aspect ratios
- image resolution summaries
- sample GT visualization outputs

Defect-size buckets are configured in `configs/default.yaml` using
`small_max` and `medium_max`: small is below `small_max`, medium is from
`small_max` up to (but excluding) `medium_max`, and large is at least
`medium_max`.

The outputs are saved under `reports/figures/` and `reports/tables/` where appropriate.

The validation JSON is written to `reports/validation.json`, and the complete
audit summary is written to `reports/audit_summary.json` when using the
commands in the README.
