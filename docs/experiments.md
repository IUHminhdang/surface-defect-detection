# Experiment workflow

The project expects results to be written under `experiments/<experiment>/` with the exact CSV schema required by the brief.

Each experiment should produce:

- `results.csv`
- `training_history.csv`
- `predictions.csv`
- `error_analysis.csv`
- `weights/best.pt`
- `weights/last.pt`

Custom YOLO11 ablations use `--model yolo11-p2-msc-ca` and the
`training.custom.ablation` value (`E1`, `E2`, or `E3`) in the selected config.
The baseline E0 remains the existing `--model yolo11` path.

Comparison tables should be generated automatically under `reports/tables/` and figures under `reports/figures/`.

To draw loss and validation curves after a Kaggle run:

```bash
python scripts/plot_training_history.py \
  --experiments experiments/yolo11 experiments/yolo11-e1 \
  experiments/yolo11-e2 experiments/yolo11-e3
```
