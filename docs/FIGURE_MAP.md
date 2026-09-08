# Figure and table map

Which code produces each figure and table in the paper. Compiled by tracing the
Snakemake rules and figure scripts in this repository.

Figures 1–5 are **composites assembled by hand** from the component plots listed
below; the panel schematics (Fig. 1A, Fig. 2A–B/D, Fig. 4A–B, the Fig. 3C table)
are drawn in a vector editor, not generated. Only the data panels come from code.

## Main text

| Paper item | Produced by | Component outputs |
|---|---|---|
| Fig. 1A | drawn by hand | — |
| Fig. 1B | `workflow/rules/sr_comparison.smk` → `src/pipelines/sr_comparison/` | `integrated_results_plot.png`, complexity-vs-accuracy plot (`docs/figures/sr_complexity_vs_accuracy_two_step_enzyme.png`) |
| Fig. 2A, 2B, 2D | drawn by hand | — |
| Fig. 2C | `sr_comparison.smk` across enzyme mechanisms and settings | `all_pysr_formulas.txt` per mechanism |
| Fig. 3A | `src/pipelines/regimes/{noise,dataset_size,mm_deviation}_regimes.py` | `log_mae_*_lineplot.png`, `error_landscape.png` |
| Fig. 3B | the same three, run with `--variants tQSSA` | as above, tQSSA variants |
| Fig. 3C | `src/pipelines/regimes/kinetic_regimes.py`; the condition/expectation table is typed by hand | `all_pysr_formulas_sQSSA.txt` |
| Fig. 4C–F | `workflow/rules/experimental.smk` | `fig_4d_sr_vs_linreg.png`, `fig_4ef_accuracy_and_parsimony.png`, `fig_4_closure_cost.png`, `fig_4_joint_linreg_parsimony.png` |
| Fig. 5 | `experimental.smk`, per marker and seed | `fig_5_fan_{marker}_s{seed}.png`, `fig_5_perbin_{marker}_s{seed}.png` |
| Table 1 | `sr_comparison.smk`, two/three/four-step × closed/open | `all_pysr_formulas.txt` |

## Supporting information

| Paper item | Produced by | Output |
|---|---|---|
| Fig. S1 | `src/pipelines/experimental/sr_pipeline/figures/plot_fig_s1_loss_ablation.py` | `fig_s1_loss_ablation.png`, `fig_s1_loss_ablation_metrics.csv` |
| Fig. S2 | `.../figures/plot_fig_s2_threshold_examples.py` | `fig_s2_threshold_examples.png` |
| Fig. S3 | `.../figures/plot_fig_s3_neural_ode_selection.py` | `fig_s3_neural_ode_selection.png` |
| Tables S1–S3 | `regimes/{noise,dataset_size,mm_deviation}_regimes.py` (sQSSA) | `all_pysr_formulas_sQSSA.txt` |
| Tables S4–S6 | the same three with `--variants tQSSA` | `all_pysr_formulas_tQSSA.txt` |
| Table S7 | `regimes/kinetic_regimes.py` | `all_pysr_formulas_sQSSA.txt` |
| Table S8 | `.../figures/plot_fig4_composite_complexity.py` | `results/published/table_s8_sr_per_context.csv` |
| Tables S9–S10 | `.../figures/make_supplementary_tables.py` | `results/published/table_s9_*.csv`, `table_s10_*.csv` |
| Table S11 | `experimental.smk` (retained expression per context) | — |
| Tables S12–S13 | rule `tables_s12_s13_threshold_sensitivity` | `results/published/table_s12_*.csv`, `table_s13_*.csv` |
| Table S14 | rule `table_s14_lambda_calibration` | `results/published/table_s14_lambda_calibration.csv` |

The CSVs behind Tables S8–S14 are committed under `results/published/`, so those
numbers can be checked without re-running the pipeline.

## Outputs not used in the paper

The workflow also emits `fig_s4_sr_vs_linreg_matched.png`, `fig_s5_parsimony_all40.png`,
`fig_threshold_calibration.png` and `fig_threshold_error_tradeoff.png`. These were
exploratory or belong to an earlier draft and are not in the submitted manuscript.
