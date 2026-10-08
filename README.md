# Surface Defect Detection

Research project for industrial surface defect object detection with a reproducible pipeline that runs locally and on Kaggle.

## Project goal

This repository is structured to support the full research workflow required by the project brief:

1. Dataset validation and audit
2. Baseline detection benchmarking
3. Error analysis
4. Proposed model improvement based on actual failure modes
5. Ablation studies
6. Final benchmark and CSV-based reporting

The repository intentionally avoids hard-coded local paths and supports both local dataset directories and Kaggle dataset roots.

## Repository layout

```text
surface-defect-detection/
├── README.md
├── requirements.txt
├── configs/
├── data/
├── notebooks/
├── src/
├── scripts/
├── experiments/
├── reports/
├── docs/
└── .gitignore
```

## Environment variables and dataset root

The project resolves the dataset root in this order:

1. `--data-root` CLI argument
2. `DATASET_ROOT`
3. `KAGGLE_INPUT_PATH`
4. `/kaggle/input` when running on Kaggle
5. local `data/` directory if present

Example:

```bash
export DATASET_ROOT=/path/to/dataset
python scripts/validate_dataset.py --data-root "$DATASET_ROOT"
```

## Dataset validation

```bash
python scripts/validate_dataset.py --data-root /path/to/dataset
```

This command checks for:

- missing or extra image/label pairs
- invalid YOLO label rows
- empty/negative/invalid boxes
- non-image files or corrupt images
- general split integrity

## Dataset audit

```bash
python scripts/audit_dataset.py --data-root /path/to/dataset --output-dir reports
```

The audit script produces:

- dataset summaries per split
- image resolution statistics
- defective vs. clean image counts
- box count statistics
- bbox width/height statistics
- bbox area ratio and aspect ratio summaries
- defect-size distribution summaries
- GT visualizations saved under `reports/figures/`

## Baseline training

The baseline entry point supports five model families. Ultralytics is used for
YOLO11, RT-DETR, and YOLO-World. Transformers is used for fine-tuning DINO and
Grounding DINO. Each run is independent and writes its own experiment folder.

```bash
python scripts/train.py --data-root /path/to/dataset --config configs/default.yaml \
  --model yolo11 --experiment yolo11

python scripts/train.py --data-root /path/to/dataset --config configs/default.yaml \
  --model rtdetr --experiment rtdetr

python scripts/train.py --data-root /path/to/dataset --config configs/default.yaml \
  --model yolo-world --experiment yolo-world

python scripts/train.py --data-root /path/to/dataset --config configs/default.yaml \
  --model dino --experiment dino

python scripts/train.py --data-root /path/to/dataset --config configs/default.yaml \
  --model grounding-dino --experiment grounding-dino

# Proposed YOLO11-P2-MSC-CA, with E1/E2/E3 selected in training.custom.ablation
python scripts/train.py --data-root /path/to/dataset --config configs/default.yaml \
  --model yolo11-p2-msc-ca --experiment yolo11-e3
```

For the Transformers models, training evaluates the validation split and
writes real `mAP50`, `mAP50-95`, `precision`, and `recall` values. Evaluate the
selected checkpoint on the held-out test split with:

```bash
python scripts/evaluate_transformers.py \
  --data-root /path/to/dataset \
  --weights experiments/dino/weights/best \
  --model dino --experiment dino --split test

python scripts/evaluate.py --data-root /path/to/dataset \
  --weights experiments/baseline/weights/best.pt --experiment baseline --split test

python scripts/benchmark_speed.py --data-root /path/to/dataset \
  --weights experiments/yolo11/weights/best.pt --experiment yolo11 \
  --image-size 640 --batch-size 1 --warmup 20 --iterations 100 --device 0
```

These five runs are baseline comparisons only. Do not add a proposed module
until all selected baseline runs have been evaluated with the same split and
reporting protocol.

The proposed model is named `YOLO11-P2-MSC-CA`. E0 is the unchanged YOLO11n
baseline; E1 adds P2 fusion, E2 adds multi-scale fusion, and E3 adds coordinate
attention. E4 and E5 isolate Multi-Scale Fusion and Coordinate Attention
without P2, respectively. See [docs/model_architecture.md](docs/model_architecture.md) and
[docs/ablation.md](docs/ablation.md) for the discovered tensor shapes and
insertion points.

## Evaluation

```bash
python scripts/evaluate.py --data-root /path/to/dataset --weights /path/to/best.pt
```

This script is designed to generate CSV summaries that match the required schema for:

- `results.csv`
- `training_history.csv`
- `predictions.csv`
- `error_analysis.csv`

## Training and validation curves

After training on Kaggle, download the experiment folders or mount them in
Colab and plot the recorded per-epoch history:

```bash
python scripts/plot_training_history.py \
  --experiments experiments/yolo11 experiments/yolo11-e1 \
  experiments/yolo11-e2 experiments/yolo11-e3 \
  --output-dir reports/figures
```

This creates:

- `reports/figures/training_validation_loss.png`
- `reports/figures/validation_metrics.png`

For YOLO/Ultralytics experiments, the loss curve uses box loss because that is
the validation loss recorded by the existing CSV pipeline. The figures do not
recompute metrics and do not use the test split.

## FPS benchmark

```bash
python scripts/benchmark_speed.py --weights /path/to/best.pt --data-root /path/to/dataset --image-size 640 --batch-size 1
```

This tracks the required deployment-latency benchmarking protocol with a warmup
phase and batch-size-1 inference timing. Each model writes independently to
`experiments/<experiment>/benchmark_speed.csv`.

## Error analysis

```bash
python scripts/error_analysis.py --data-root /path/to/dataset --weights /path/to/best.pt
```

This is meant to break down the failure modes required by the brief:

- false positive on clean images
- false negatives for small defects
- localization errors
- defect-size analysis
- source-domain analysis when labels permit it

## Kaggle workflow

A notebook has been included for the full Kaggle pipeline:

```text
notebooks/kaggle_pipeline.ipynb
```

Use the Kaggle config when working inside the Kaggle runtime:

```bash
python scripts/validate_dataset.py --data-root /kaggle/input
python scripts/audit_dataset.py --data-root /kaggle/input --output-dir /kaggle/working/reports
```

## Notes

- This repository creates the project skeleton and the necessary validation/audit tooling before model training.
- No fabricated benchmark numbers are included.
- All experiment metrics should be written to `experiments/<experiment>/` and generated from actual training/evaluation outputs.
