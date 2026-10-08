# Experiment workflow

The project expects results to be written under `experiments/<experiment>/` with the exact CSV schema required by the brief.

Each experiment should produce:

- `results.csv`
- `training_history.csv`
- `predictions.csv`
- `error_analysis.csv`
- `weights/best.pt`
- `weights/last.pt`

Comparison tables should be generated automatically under `reports/tables/` and figures under `reports/figures/`.
