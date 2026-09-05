# Phase 14 — Final Tables and Figures

Publication-ready tables and figures generated from exact Phase 7–13 results.
No values were invented or estimated.

## Tables

| # | File | Description |
|---|------|-------------|
| 1 | table1_performance_comparison.csv | Method comparison vs baselines (E1 pump, 4 IDs) |
| 2 | table2_multi_machine_evaluation.csv | Per-machine-type AUC + 95% CI + Cohen's d |
| 3 | table3_ablation_study.csv | Ablation: component contribution (pump, 4 IDs) |
| 4 | table4_statistical_results.csv | AUC + 95% bootstrap CI per machine type |
| 5 | table5_seed_stability.csv | Multi-seed stability (seeds 42, 123, 2026) |
| 6 | table6_runtime_benchmark.csv | CPU runtime benchmark (cache-hit path) |

## Figures

| # | File | Description |
|---|------|-------------|
| 1 | fig1_overall_metric_comparison.png | Bar chart: method vs baselines (mean AUC) |
| 2 | fig2_per_machine_type_performance.png | Bar chart: AUC per machine type with 95% CI |
| 3 | fig3_ablation_comparison.png | Bar chart: ablation component analysis |
| 4 | fig4_seed_stability.png | Dual panel: AUC and Cohen's d across 3 seeds |

## Key Results

- **Overall ROC-AUC** (Phase 9, seed=42): 0.7875
  95% CI: [0.7773, 0.8012]
- **Overall Cohen's d**: 1.0613
- **Seed stability** (3 seeds): AUC = 0.7803 ± 0.0081
- **Best machine type**: Slider (AUC=0.8813)
- **Inference latency** (cache-hit): 5.86 ms mean

## Data Sources

| Table/Figure | Source file(s) |
|---|---|
| Table 1, Fig 1 | experiments/results/e1/baseline_comparison/consolidated_comparison.csv |
| Table 2, Fig 2 | experiments/results/phase13/final_method_config.json |
| Table 3, Fig 3 | experiments/results/e1/ablation_study/ablation_results.csv |
| Table 4       | experiments/results/phase9/comparison_e1/step10_bootstrap_ci.csv |
| Table 5, Fig 4 | experiments/results/phase11/phase11_results.csv |
| Table 6       | experiments/results/phase12/phase12_timing_summary.csv |