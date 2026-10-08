# Methodology

The implementation follows the required research sequence:

1. Data audit
2. Baseline benchmarking
3. Error analysis
4. Model selection
5. Improvement proposal based on actual errors
6. Ablation study
7. Final benchmark

The project intentionally does not proceed to a proposed model or training-stage improvement before the baseline and error-analysis stages are complete.

The workflow is designed to keep all experiments reproducible by writing structured metadata and CSV outputs into `experiments/<experiment>/` and periodic plotting under `reports/figures/`.
