# Global configuration for feature selection, data type, dataset sizes, and timeout duration
import os
import shlex
from collections import OrderedDict

from snakemake.shell import shell
shell.executable("/bin/bash")
shell.prefix("set -euo pipefail; ")

configfile: "config.yaml"

SR_VARIANTS = OrderedDict([
    ("sQSSA", "sQSSA"),
    ("tQSSA", "tQSSA"),
])


def _filename_without_ext(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem


def _add_model_suffix(path: str, slug: str) -> str:
    if not slug:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{slug}{ext}"


def _relative_basename(path: str, root: str) -> str:
    rel_path = os.path.relpath(path, root)
    stem, _ = os.path.splitext(rel_path)
    return stem.replace("\\", "/")

# Define data type and parameters from configuration
data_type = config["data_type"]  # Example: 'dynamic'
enzyme_model = config["enzyme_model"]
base_models = config["models"]  # Example: ['pysindy', 'aifeynman', 'dso', 'kan', 'pysr']
output_extension = config["output_extension"]
features = config["features"][data_type]


def _variant_suffix(variant_key):
    if not variant_key:
        return ""
    return f"_{variant_key}"


variant_keys = list(SR_VARIANTS.keys())
variant_pattern = "|".join(variant_keys)

global_wildcard_constraints = {"variant": variant_pattern}

model_details = []
for base_model in base_models:
    if base_model == "pysr":
        for variant_key in variant_keys:
            suffix = _variant_suffix(variant_key)
            name = f"{base_model}{suffix}"
            temp_path = (
                f"data/{enzyme_model}/{data_type}/sr_comparison/temp/hall_of_fame{suffix}.csv"
            )
            formula_path = (
                f"data/{enzyme_model}/{data_type}/sr_comparison/results/formula_{base_model}{suffix}.txt"
            )
            model_details.append(
                {
                    "name": name,
                    "base": base_model,
                    "variant": variant_key,
                    "temp": temp_path,
                    "formula": formula_path,
                }
            )
    else:
        suffix = ""
        name = base_model
        ext = output_extension.get(base_model, "txt")
        temp_path = (
            f"data/{enzyme_model}/{data_type}/sr_comparison/temp/temp_formula_{base_model}{suffix}.{ext}"
        )
        formula_path = (
            f"data/{enzyme_model}/{data_type}/sr_comparison/results/formula_{base_model}{suffix}.txt"
        )
        model_details.append(
            {
                "name": name,
                "base": base_model,
                "variant": None,
                "temp": temp_path,
                "formula": formula_path,
            }
        )

models = [entry["name"] for entry in model_details]

# Experimental configuration for functional marker groups
experimental_cfg = config.get("experimental", {})
exp_raw_time_course = experimental_cfg.get("raw_time_course", "data/experimental/raw/lun_2019.csv")
exp_processed_dir = experimental_cfg.get("processed_dir", "data/experimental/processed").rstrip("/")
exp_output_prefix = experimental_cfg.get("output_prefix", "markers")
exp_gfp_bins = experimental_cfg.get("gfp_bins", 50)
exp_min_points = experimental_cfg.get("min_points", 5)
exp_extra_times = experimental_cfg.get("extra_times", [1, 3])
exp_extra_times_args = "--extra-times " + " ".join(str(t) for t in exp_extra_times) if exp_extra_times else ""
exp_force_rebin = bool(experimental_cfg.get("force_rebin", False))
exp_force_rebin_arg = "--force-rebin" if exp_force_rebin else ""

exp_group_definitions_csv = experimental_cfg.get("group_definitions_csv")
if not exp_group_definitions_csv:
    default_marker_meta = experimental_cfg.get(
        "marker_metadata_csv",
        f"{exp_processed_dir}/markers.csv",
    )
    if os.path.exists(default_marker_meta):
        exp_group_definitions_csv = default_marker_meta
    else:
        fallback_group_defs = f"{exp_processed_dir}/functional_groups.csv"
        exp_group_definitions_csv = (
            fallback_group_defs
            if os.path.exists(fallback_group_defs)
            else default_marker_meta
        )

marker_base_dir = exp_processed_dir

exp_runs_root_cfg = experimental_cfg.get("runs_root") or experimental_cfg.get("sr_output_dir")
if exp_runs_root_cfg:
    exp_runs_root = exp_runs_root_cfg.rstrip("/")
else:
    exp_runs_root = "data/experimental/runs"
exp_sr_output_dir = exp_runs_root
exp_runs_aggregated = f"{exp_runs_root}/aggregated"
exp_runs_seeds = f"{exp_runs_root}/seeds"
exp_reports_root = f"{exp_runs_aggregated}/reports"
exp_metrics_root = f"{exp_runs_aggregated}/metrics"
exp_trajectories_root = f"{exp_runs_aggregated}/trajectories"
exp_plots_root = f"{exp_runs_aggregated}/plots"

# Pipeline now supports only 'all' feature mode
exp_feature_modes = ["all"]
exp_feature_modes_args = " ".join(exp_feature_modes)
exp_gfp_columns = experimental_cfg.get(
    "gfp_columns",
    ["GFP", "p-ERK1-2", "p-MEK1-2", "p-ERK1-2_min", "p-MEK1-2_min"],
)
exp_confidence_threshold = experimental_cfg.get("confidence_threshold", 4.0)
exp_models = [model.lower() for model in experimental_cfg.get("models", ["pysr"])]
exp_per_minute_max_time = float(experimental_cfg.get("per_minute_max_time", 30.0))
exp_measured_timepoints = experimental_cfg.get(
    "measured_timepoints",
    [0, 5, 10, 15, 30, 60],
)
allowed_exp_model_codes = {"pysr", "linreg"}
for _model_code in exp_models:
    if _model_code not in allowed_exp_model_codes:
        raise ValueError(f"Unsupported experimental model '{_model_code}'. Allowed: {sorted(allowed_exp_model_codes)}")
exp_models_args = " ".join(exp_models)
exp_model_name_map = {
    "pysr": "PySR",
    "linreg": "Linear Regression",
}
exp_model_slug_map = {
    "pysr": "",
    "linreg": "linear_regression",
}
run_pysr = "pysr" in exp_models
run_linreg = "linreg" in exp_models

exp_log10_cutoff = float(experimental_cfg.get("log10_cutoff", -3.0))
exp_fit_groups = experimental_cfg.get(
    "fit_groups",
    ["RTK_penta_full", "DUSP_trio_noDUSP7", "Controls"],
)
exp_fit_groups_args = " ".join(shlex.quote(group) for group in exp_fit_groups)
exp_fit_grid_columns = int(experimental_cfg.get("fit_grid_columns", 5))
exp_fit_output_subdir = experimental_cfg.get("fit_output_subdir", "pysr/fits")
exp_fit_sanity_markers = experimental_cfg.get("fit_sanity_markers", [])
exp_fit_sanity_markers_args = " ".join(shlex.quote(marker) for marker in exp_fit_sanity_markers)
exp_fit_sanity_protein = experimental_cfg.get("fit_sanity_protein", "p-ERK1-2")
exp_fit_sanity_max_columns = int(experimental_cfg.get("fit_sanity_max_columns", 4))

# Diagnostic trade-off evaluation settings
exp_tradeoff_groups = experimental_cfg.get("tradeoff_groups", exp_fit_groups)
exp_tradeoff_groups_args = " ".join(shlex.quote(group) for group in exp_tradeoff_groups)
exp_tradeoff_caps = experimental_cfg.get(
    "tradeoff_caps",
    [1500, 1200, 1000, 750, 500, 350],
)
exp_tradeoff_caps_args = " ".join(str(int(cap)) for cap in exp_tradeoff_caps)
exp_tradeoff_include_uncapped = bool(experimental_cfg.get("tradeoff_include_uncapped", True))
exp_tradeoff_run_pysr = bool(experimental_cfg.get("tradeoff_run_pysr", True))
exp_tradeoff_output_root = experimental_cfg.get(
    "tradeoff_output_root",
    f"{exp_runs_root}/tradeoff_runs",
)
exp_tradeoff_skip_existing = bool(experimental_cfg.get("tradeoff_skip_existing", True))
exp_tradeoff_uncapped_max = int(experimental_cfg.get("tradeoff_uncapped_max", 1_000_000))
exp_tradeoff_seed = int(experimental_cfg.get("tradeoff_seed", 42))
exp_tradeoff_min_bin_samples = int(experimental_cfg.get("tradeoff_min_bin_samples", 500))
exp_tradeoff_eval_feature_mode = experimental_cfg.get("tradeoff_feature_mode_eval", "gfp")
exp_tradeoff_eval_model_name = experimental_cfg.get("tradeoff_model_name", "pysr")
exp_tradeoff_include_linreg = bool(experimental_cfg.get("tradeoff_include_linreg", True))
exp_per_minute_sampling_strategy = experimental_cfg.get(
    "per_minute_sampling_strategy",
    "early_plus_sparse_late",
)
exp_late_sample_window = tuple(experimental_cfg.get("late_sample_window", [30.0, 60.0]))
exp_late_sample_points = int(experimental_cfg.get("late_sample_points", 15))
exp_custom_loss_ablation_cfg = experimental_cfg.get("custom_loss_ablation", {})
exp_custom_loss_ablation_enabled = bool(
    exp_custom_loss_ablation_cfg.get("enabled", False)
)
exp_custom_loss_ablation_seed = int(
    exp_custom_loss_ablation_cfg.get("seed", 42)
)
exp_custom_loss_ablation_metric = str(
    exp_custom_loss_ablation_cfg.get("metric_column", "ode_integ_r2_median")
)
exp_custom_loss_ablation_dataset_mode = str(
    exp_custom_loss_ablation_cfg.get("metric_dataset_mode", "per_minute")
)
exp_custom_loss_ablation_feature_mode = str(
    exp_custom_loss_ablation_cfg.get("metric_feature_mode", "all")
)
exp_custom_loss_ablation_metric_source = str(
    exp_custom_loss_ablation_cfg.get("metric_source", "integration")
)

weighting_cfg = {}

pysr_grid_cfg = experimental_cfg.get("pysr_grid", {})
pysr_grid_seed = int(pysr_grid_cfg.get("seed", 1337))
pysr_grid_base_niterations = int(pysr_grid_cfg.get("base_niterations", 300))
pysr_grid_base_population_size = int(pysr_grid_cfg.get("base_population_size", 30))
pysr_grid_base_populations = int(pysr_grid_cfg.get("base_populations", 30))
pysr_grid_lower_scale = float(pysr_grid_cfg.get("lower_scale", 0.5))
pysr_grid_upper_scale = float(pysr_grid_cfg.get("upper_scale", 2.0))
pysr_grid_min_bin_samples = int(pysr_grid_cfg.get("min_bin_samples", 500))
pysr_grid_max_bin_samples = int(pysr_grid_cfg.get("max_bin_samples", 1000))

marker_output_root = f"{marker_base_dir}/{exp_output_prefix}"
marker_time_series_csv = f"{marker_output_root}/markers_time_series.csv"
marker_feature_matrix_csv = f"{marker_output_root}/markers_feature_matrix.csv"
marker_fit_snapshot_csv = f"{marker_output_root}/markers_fit_snapshot.csv"
marker_per_minute_csv = f"{marker_output_root}/markers_per_minute_fit.csv"
reports_summary_output = f"{exp_reports_root}/summary/marker_summary.csv"
reports_summary_seed_output = f"{exp_reports_root}/summary/marker_summary_seed.csv"
# Per-minute outputs still live under a subdir for plots convenience
marker_formula_files = [
    f"{exp_reports_root}/formulas/{mode}_{suffix}.txt"
    for mode in exp_feature_modes
    for suffix in ("snapshot", "per_minute")
]
marker_summary_output = reports_summary_output
marker_summary_seed_output = reports_summary_seed_output

plots_root = exp_plots_root
plots_overlays_dir = f"{plots_root}/overlays"
plots_metrics_dir = f"{plots_root}/metrics"
plots_metrics_scatter_dir = f"{plots_metrics_dir}/scatter"
plots_metrics_lines_dir = f"{plots_metrics_dir}/lines"
plots_metrics_features_dir = f"{plots_metrics_dir}/features"
plots_metrics_boxes_dir = f"{plots_metrics_dir}/boxes"

# Overlay plot outputs (explicit per dataset mode)
overlay_snapshot_plot = f"{plots_overlays_dir}/marker_overlay_snapshot.png"
overlay_snapshot_metrics = f"{exp_metrics_root}/marker_overlay_metrics_snapshot.csv"
overlay_snapshot_plot_svg = f"{plots_overlays_dir}/marker_overlay_snapshot.svg"
overlay_per_minute_plot = f"{plots_overlays_dir}/marker_overlay_per_minute.png"
overlay_per_minute_metrics = f"{exp_metrics_root}/marker_overlay_metrics_per_minute.csv"
overlay_per_minute_plot_svg = f"{plots_overlays_dir}/marker_overlay_per_minute.svg"
overlay_snapshot_linreg_plot = f"{plots_overlays_dir}/marker_overlay_linear_regression_snapshot.png"
overlay_snapshot_linreg_plot_svg = f"{plots_overlays_dir}/marker_overlay_linear_regression_snapshot.svg"
overlay_snapshot_linreg_metrics = f"{exp_metrics_root}/marker_overlay_metrics_linear_regression_snapshot.csv"
overlay_per_minute_linreg_plot = f"{plots_overlays_dir}/marker_overlay_linear_regression_per_minute.png"
overlay_per_minute_linreg_plot_svg = f"{plots_overlays_dir}/marker_overlay_linear_regression_per_minute.svg"
overlay_per_minute_linreg_metrics = f"{exp_metrics_root}/marker_overlay_metrics_linear_regression_per_minute.csv"
integration_snapshot_metrics = f"{exp_metrics_root}/marker_integration_metrics_snapshot.csv"
integration_per_minute_metrics = f"{exp_metrics_root}/marker_integration_metrics_per_minute.csv"
integration_snapshot_traj = f"{exp_trajectories_root}/marker_integration_trajectories_snapshot.csv"
integration_per_minute_traj = f"{exp_trajectories_root}/marker_integration_trajectories_per_minute.csv"
predicted_snapshot_traj = f"{exp_trajectories_root}/predicted_trajectories_snapshot.csv"
predicted_per_minute_traj = f"{exp_trajectories_root}/predicted_trajectories_per_minute.csv"
integration_snapshot_metrics_all_seeds = f"{exp_metrics_root}/marker_integration_metrics_snapshot_all_seeds.csv"
integration_per_minute_metrics_all_seeds = f"{exp_metrics_root}/marker_integration_metrics_per_minute_all_seeds.csv"
integration_snapshot_metrics_all_seeds_mean = f"{exp_metrics_root}/marker_integration_metrics_snapshot_all_seeds_mean.csv"
integration_per_minute_metrics_all_seeds_mean = f"{exp_metrics_root}/marker_integration_metrics_per_minute_all_seeds_mean.csv"

select_k_dir = f"{exp_runs_root}/select_k"
select_k_metrics = f"{select_k_dir}/select_k_metrics.csv"
select_k_metrics_agg = f"{select_k_dir}/select_k_metrics_agg.csv"
select_k_importance = f"{select_k_dir}/select_k_importance.csv"
select_k_importance_agg = f"{select_k_dir}/select_k_importance_agg.csv"
select_k_boxplot_dt = f"{select_k_dir}/boxplot_dt_r2.png"
select_k_boxplot_integ = f"{select_k_dir}/boxplot_integ_r2.png"
select_k_boxplot_ode = f"{select_k_dir}/boxplot_ode_r2.png"
select_k_ribbon_coef = f"{select_k_dir}/ribbon_coef_importance.png"
select_k_ribbon_variance = f"{select_k_dir}/ribbon_variance_importance.png"
select_k_panel_a = f"{select_k_dir}/pysr_selectk_panel_a.png"
select_k_panel_b = f"{select_k_dir}/pysr_selectk_panel_b.png"
select_k_panel_a_box = f"{select_k_dir}/pysr_selectk_panel_a_box.png"
select_k_panel_a_dt = f"{select_k_dir}/pysr_selectk_panel_a_dt.png"
select_k_panel_b_dt = f"{select_k_dir}/pysr_selectk_panel_b_dt.png"
select_k_panel_a_box_dt = f"{select_k_dir}/pysr_selectk_panel_a_box_dt.png"
select_k_panel_a_linreg_bar = f"{select_k_dir}/pysr_selectk_panel_a_linreg_bar.png"
select_k_panel_a_relmae = f"{select_k_dir}/pysr_selectk_panel_a_relmae.png"
select_k_panel_b_relmae = f"{select_k_dir}/pysr_selectk_panel_b_relmae.png"
select_k_panel_a_box_relmae = f"{select_k_dir}/pysr_selectk_panel_a_relmae_box.png"
select_k_panel_a_relmae_dt = f"{select_k_dir}/pysr_selectk_panel_a_relmae_dt.png"
select_k_panel_b_relmae_dt = f"{select_k_dir}/pysr_selectk_panel_b_relmae_dt.png"
select_k_panel_a_box_relmae_dt = f"{select_k_dir}/pysr_selectk_panel_a_relmae_box_dt.png"
select_k_panel_a_linreg_bar_relmae = f"{select_k_dir}/pysr_selectk_panel_a_linreg_bar_relmae.png"
select_k_variability_dt = f"{select_k_dir}/pysr_selectk_variability_dt.png"
select_k_variability_perk = f"{select_k_dir}/pysr_selectk_variability_perk.png"
select_k_pysr_k_vs_r2 = f"{select_k_dir}/pysr_selectk_pysr_k_vs_r2.png"
select_k_heatmap = f"{select_k_dir}/pysr_selectk_heatmap.png"
select_k_trajectory_all_k = f"{select_k_dir}/pysr_selectk_trajectory_all_k.png"
select_k_baseline_pysr_vs_linreg = f"{select_k_dir}/pysr_selectk_baseline_pysr_vs_linreg.png"
select_k_trajectory_all_k = f"{select_k_dir}/pysr_selectk_trajectory_all_k.png"

custom_loss_ablation_dir = f"{exp_runs_root}/custom_loss_ablation"
custom_loss_ablation_metrics = f"{custom_loss_ablation_dir}/pysr_custom_loss_ablation_metrics.csv"
custom_loss_ablation_boxplot = f"{custom_loss_ablation_dir}/pysr_custom_loss_ablation_boxplot.png"
custom_loss_ablation_boxplot_svg = f"{custom_loss_ablation_dir}/pysr_custom_loss_ablation_boxplot.svg"

neural_ode_dir = f"{exp_runs_root}/neural_ode"
neural_ode_full_dir = f"{neural_ode_dir}/full"
neural_ode_matched_dir = f"{neural_ode_dir}/matched"

neural_ode_metrics_full = f"{neural_ode_full_dir}/neural_ode_metrics.csv"
neural_ode_metrics_agg_full = f"{neural_ode_full_dir}/neural_ode_metrics_agg.csv"
neural_ode_boxplot_dt_full = f"{neural_ode_full_dir}/boxplot_dt_r2.png"
neural_ode_boxplot_integ_full = f"{neural_ode_full_dir}/boxplot_integ_r2.png"
neural_ode_boxplot_ode_full = f"{neural_ode_full_dir}/boxplot_ode_r2.png"
neural_ode_panel_ode_full = f"{neural_ode_full_dir}/pysr_neuralode_panel_ode.png"
neural_ode_panel_dt_full = f"{neural_ode_full_dir}/pysr_neuralode_panel_dt.png"
neural_ode_panel_ode_relmae_full = f"{neural_ode_full_dir}/pysr_neuralode_panel_ode_relmae.png"
neural_ode_panel_dt_relmae_full = f"{neural_ode_full_dir}/pysr_neuralode_panel_dt_relmae.png"
neural_ode_baseline_full = f"{neural_ode_full_dir}/pysr_neuralode_baseline.png"
neural_ode_pysr_k_vs_r2_full = f"{neural_ode_full_dir}/pysr_neuralode_pysr_k_vs_r2.png"
neural_ode_sweep_full = f"{neural_ode_full_dir}/neural_ode_sweep.csv"
neural_ode_quadrant_bar_full = f"{neural_ode_full_dir}/pysr_neuralode_quadrant_bar.png"

neural_ode_metrics_matched = f"{neural_ode_matched_dir}/neural_ode_metrics.csv"
neural_ode_metrics_agg_matched = f"{neural_ode_matched_dir}/neural_ode_metrics_agg.csv"
neural_ode_boxplot_dt_matched = f"{neural_ode_matched_dir}/boxplot_dt_r2.png"
neural_ode_boxplot_integ_matched = f"{neural_ode_matched_dir}/boxplot_integ_r2.png"
neural_ode_boxplot_ode_matched = f"{neural_ode_matched_dir}/boxplot_ode_r2.png"
neural_ode_panel_ode_matched = f"{neural_ode_matched_dir}/pysr_neuralode_panel_ode.png"
neural_ode_panel_dt_matched = f"{neural_ode_matched_dir}/pysr_neuralode_panel_dt.png"
neural_ode_panel_ode_relmae_matched = f"{neural_ode_matched_dir}/pysr_neuralode_panel_ode_relmae.png"
neural_ode_panel_dt_relmae_matched = f"{neural_ode_matched_dir}/pysr_neuralode_panel_dt_relmae.png"
neural_ode_baseline_matched = f"{neural_ode_matched_dir}/pysr_neuralode_baseline.png"
neural_ode_pysr_k_vs_r2_matched = f"{neural_ode_matched_dir}/pysr_neuralode_pysr_k_vs_r2.png"
neural_ode_sweep_matched = f"{neural_ode_matched_dir}/neural_ode_sweep.csv"
neural_ode_quadrant_bar_matched = f"{neural_ode_matched_dir}/pysr_neuralode_quadrant_bar.png"

# Diffrax true-NeuralODE baseline (trajectory-loss + backprop-through-solver).
# Sits in parallel with the pipeline NN's `neural_ode/` directory; shares the
# same plot script. Note: this script does NOT compute rel-MAE columns, so the
# plotting rule below does not request the rel-MAE panels.
neural_ode_diffrax_dir = f"{exp_runs_root}/neural_ode_diffrax"
neural_ode_diffrax_full_dir = f"{neural_ode_diffrax_dir}/full"
neural_ode_diffrax_metrics_full = f"{neural_ode_diffrax_full_dir}/neural_ode_metrics.csv"
neural_ode_diffrax_metrics_agg_full = f"{neural_ode_diffrax_full_dir}/neural_ode_metrics_agg.csv"
neural_ode_diffrax_boxplot_dt_full = f"{neural_ode_diffrax_full_dir}/boxplot_dt_r2.png"
neural_ode_diffrax_boxplot_integ_full = f"{neural_ode_diffrax_full_dir}/boxplot_integ_r2.png"
neural_ode_diffrax_boxplot_ode_full = f"{neural_ode_diffrax_full_dir}/boxplot_ode_r2.png"
neural_ode_diffrax_panel_ode_full = f"{neural_ode_diffrax_full_dir}/pysr_neuralode_panel_ode.png"
neural_ode_diffrax_panel_dt_full = f"{neural_ode_diffrax_full_dir}/pysr_neuralode_panel_dt.png"
neural_ode_diffrax_baseline_full = f"{neural_ode_diffrax_full_dir}/pysr_neuralode_baseline.png"
neural_ode_diffrax_pysr_k_vs_r2_full = f"{neural_ode_diffrax_full_dir}/pysr_neuralode_pysr_k_vs_r2.png"
neural_ode_diffrax_quadrant_bar_full = f"{neural_ode_diffrax_full_dir}/pysr_neuralode_quadrant_bar.png"

# Causal analysis (Jacobian-mass driver/brake split + PR distribution + Hessian
# gating heatmaps). Reads per-(seed, marker) checkpoints from the baseline's
# `_seeds/` staging dir — those only exist when the baseline was run with
# `--save-models`.
neural_ode_diffrax_causal_dir = f"{neural_ode_diffrax_dir}/causal"
neural_ode_diffrax_causal_summary = f"{neural_ode_diffrax_causal_dir}/causal_summary.csv"
neural_ode_diffrax_causal_pr_csv = f"{neural_ode_diffrax_causal_dir}/participation_ratio_per_seed.csv"
neural_ode_diffrax_causal_pr_plot = f"{neural_ode_diffrax_causal_dir}/participation_ratio.png"

# Jacobian-sparsity regularisation sweep: trains a handful of markers under a
# log-spaced ladder of L1 strengths so we can pick the elbow at which the test
# trajectory R² starts dropping but participation ratio has collapsed.
neural_ode_diffrax_lambda_sweep_dir = f"{neural_ode_diffrax_dir}/lambda_sweep"
neural_ode_diffrax_lambda_sweep_csv = f"{neural_ode_diffrax_lambda_sweep_dir}/lambda_sweep.csv"

# OOD variant: same trainer + same λ, but the GFP-bin split policy is
# top_gfp_bins -> the highest-dose bins are held out as test, so generalisation
# is genuine extrapolation rather than dose-interpolation. Outputs mirror the
# random-bins structure under a _ood suffix.
neural_ode_diffrax_full_ood_dir = f"{neural_ode_diffrax_dir}/full_ood"
neural_ode_diffrax_metrics_full_ood = f"{neural_ode_diffrax_full_ood_dir}/neural_ode_metrics.csv"
neural_ode_diffrax_metrics_agg_full_ood = f"{neural_ode_diffrax_full_ood_dir}/neural_ode_metrics_agg.csv"
neural_ode_diffrax_boxplot_dt_full_ood = f"{neural_ode_diffrax_full_ood_dir}/boxplot_dt_r2.png"
neural_ode_diffrax_boxplot_integ_full_ood = f"{neural_ode_diffrax_full_ood_dir}/boxplot_integ_r2.png"
neural_ode_diffrax_boxplot_ode_full_ood = f"{neural_ode_diffrax_full_ood_dir}/boxplot_ode_r2.png"
neural_ode_diffrax_panel_ode_full_ood = f"{neural_ode_diffrax_full_ood_dir}/pysr_neuralode_panel_ode.png"
neural_ode_diffrax_panel_dt_full_ood = f"{neural_ode_diffrax_full_ood_dir}/pysr_neuralode_panel_dt.png"
neural_ode_diffrax_baseline_full_ood = f"{neural_ode_diffrax_full_ood_dir}/pysr_neuralode_baseline.png"
neural_ode_diffrax_pysr_k_vs_r2_full_ood = f"{neural_ode_diffrax_full_ood_dir}/pysr_neuralode_pysr_k_vs_r2.png"
neural_ode_diffrax_quadrant_bar_full_ood = f"{neural_ode_diffrax_full_ood_dir}/pysr_neuralode_quadrant_bar.png"
neural_ode_diffrax_causal_ood_dir = f"{neural_ode_diffrax_dir}/causal_ood"
neural_ode_diffrax_causal_ood_summary = f"{neural_ode_diffrax_causal_ood_dir}/causal_summary.csv"
neural_ode_diffrax_causal_ood_pr_csv = f"{neural_ode_diffrax_causal_ood_dir}/participation_ratio_per_seed.csv"
neural_ode_diffrax_causal_ood_pr_plot = f"{neural_ode_diffrax_causal_ood_dir}/participation_ratio.png"

# Paper-figure sweeps: three OOD regulariser variants used in main text + appendix.
# In the main text "Neural ODE" refers to L21 (λ_jac = 3.0, group-sparse). L1 and
# C-NODE are reported in the appendix.
neural_ode_diffrax_l1_seeds_dir    = f"{neural_ode_diffrax_dir}/_seeds_ood_l1_archive"
neural_ode_diffrax_l21_seeds_dir   = f"{neural_ode_diffrax_dir}/_seeds_ood_l21j_only"
neural_ode_diffrax_cnode_seeds_dir = f"{neural_ode_diffrax_dir}/_seeds_ood_pathreg"
neural_ode_diffrax_l1_done    = f"{neural_ode_diffrax_l1_seeds_dir}/.done"
neural_ode_diffrax_l21_done   = f"{neural_ode_diffrax_l21_seeds_dir}/.done"
neural_ode_diffrax_cnode_done = f"{neural_ode_diffrax_cnode_seeds_dir}/.done"

# Main-text + appendix paper figures.
paper_fig_dir = f"{exp_runs_root}/paper_figures"
paper_fig_parsimony_tradeoff = f"{paper_fig_dir}/scatter_parsimony_tradeoff.png"
paper_fig_cutoff_robustness  = f"{paper_fig_dir}/cutoff_robustness.png"
paper_fig_nn_appendix        = f"{paper_fig_dir}/nn_appendix_comparison.png"

random_forest_dir = f"{exp_runs_root}/random_forest"
random_forest_full_dir = f"{random_forest_dir}/full"
random_forest_matched_dir = f"{random_forest_dir}/matched"

random_forest_metrics_full = f"{random_forest_full_dir}/random_forest_metrics.csv"
random_forest_metrics_agg_full = f"{random_forest_full_dir}/random_forest_metrics_agg.csv"
random_forest_boxplot_r2_full = f"{random_forest_full_dir}/boxplot_state_r2.png"
random_forest_boxplot_relmae_full = f"{random_forest_full_dir}/boxplot_state_relmae.png"
random_forest_panel_r2_full = f"{random_forest_full_dir}/pysr_randomforest_panel_r2.png"
random_forest_panel_r2_full_svg = f"{random_forest_full_dir}/pysr_randomforest_panel_r2.svg"
random_forest_panel_relmae_full = f"{random_forest_full_dir}/pysr_randomforest_panel_relmae.png"
random_forest_panel_relmae_full_svg = f"{random_forest_full_dir}/pysr_randomforest_panel_relmae.svg"

random_forest_metrics_matched = f"{random_forest_matched_dir}/random_forest_metrics.csv"
random_forest_metrics_agg_matched = f"{random_forest_matched_dir}/random_forest_metrics_agg.csv"
random_forest_boxplot_r2_matched = f"{random_forest_matched_dir}/boxplot_state_r2.png"
random_forest_boxplot_relmae_matched = f"{random_forest_matched_dir}/boxplot_state_relmae.png"
random_forest_panel_r2_matched = f"{random_forest_matched_dir}/pysr_randomforest_panel_r2.png"
random_forest_panel_r2_matched_svg = f"{random_forest_matched_dir}/pysr_randomforest_panel_r2.svg"
random_forest_panel_relmae_matched = f"{random_forest_matched_dir}/pysr_randomforest_panel_relmae.png"
random_forest_panel_relmae_matched_svg = f"{random_forest_matched_dir}/pysr_randomforest_panel_relmae.svg"

random_forest_dt_dir = f"{exp_runs_root}/random_forest_dt_top_gfp_ood"
random_forest_dt_direct_dir = f"{random_forest_dt_dir}/direct"
random_forest_dt_minus_erk_dir = f"{random_forest_dt_dir}/minus_erk"

random_forest_dt_metrics_direct = f"{random_forest_dt_direct_dir}/random_forest_dt_metrics.csv"
random_forest_dt_metrics_agg_direct = f"{random_forest_dt_direct_dir}/random_forest_dt_metrics_agg.csv"
random_forest_dt_boxplot_dt_direct = f"{random_forest_dt_direct_dir}/boxplot_dt_r2.png"
random_forest_dt_boxplot_integ_direct = f"{random_forest_dt_direct_dir}/boxplot_integ_r2.png"
random_forest_dt_boxplot_ode_direct = f"{random_forest_dt_direct_dir}/boxplot_ode_r2.png"
random_forest_dt_importance_direct = f"{random_forest_dt_direct_dir}/random_forest_dt_feature_importance.csv"
random_forest_dt_importance_agg_direct = f"{random_forest_dt_direct_dir}/random_forest_dt_feature_importance_agg.csv"
random_forest_dt_importance_by_marker_direct = f"{random_forest_dt_direct_dir}/feature_importance_by_marker.png"

random_forest_dt_metrics_minus_erk = f"{random_forest_dt_minus_erk_dir}/random_forest_dt_metrics.csv"
random_forest_dt_metrics_agg_minus_erk = f"{random_forest_dt_minus_erk_dir}/random_forest_dt_metrics_agg.csv"
random_forest_dt_boxplot_dt_minus_erk = f"{random_forest_dt_minus_erk_dir}/boxplot_dt_r2.png"
random_forest_dt_boxplot_integ_minus_erk = f"{random_forest_dt_minus_erk_dir}/boxplot_integ_r2.png"
random_forest_dt_boxplot_ode_minus_erk = f"{random_forest_dt_minus_erk_dir}/boxplot_ode_r2.png"
random_forest_dt_importance_minus_erk = f"{random_forest_dt_minus_erk_dir}/random_forest_dt_feature_importance.csv"
random_forest_dt_importance_agg_minus_erk = f"{random_forest_dt_minus_erk_dir}/random_forest_dt_feature_importance_agg.csv"
random_forest_dt_importance_by_marker_minus_erk = f"{random_forest_dt_minus_erk_dir}/feature_importance_by_marker.png"
random_forest_dt_pysr_feature_usage_dir = f"{random_forest_dt_minus_erk_dir}/feature_usage_comparison"
random_forest_dt_pysr_feature_usage_csv = f"{random_forest_dt_pysr_feature_usage_dir}/rf_vs_pysr_feature_usage_by_marker.csv"
random_forest_dt_pysr_feature_usage_heatmap = f"{random_forest_dt_pysr_feature_usage_dir}/rf_vs_pysr_feature_usage_heatmap.png"
random_forest_dt_pysr_feature_usage_heatmap_svg = f"{random_forest_dt_pysr_feature_usage_dir}/rf_vs_pysr_feature_usage_heatmap.svg"
random_forest_dt_pysr_feature_overlap = f"{random_forest_dt_pysr_feature_usage_dir}/rf_vs_pysr_feature_overlap_by_marker.png"
random_forest_dt_pysr_feature_overlap_svg = f"{random_forest_dt_pysr_feature_usage_dir}/rf_vs_pysr_feature_overlap_by_marker.svg"
random_forest_dt_pysr_threshold_feature_usage_heatmap = f"{random_forest_dt_pysr_feature_usage_dir}/rf_vs_pysr_threshold_feature_usage_heatmap.png"
random_forest_dt_pysr_threshold_feature_usage_heatmap_svg = f"{random_forest_dt_pysr_feature_usage_dir}/rf_vs_pysr_threshold_feature_usage_heatmap.svg"
random_forest_dt_pysr_threshold_feature_overlap = f"{random_forest_dt_pysr_feature_usage_dir}/rf_vs_pysr_threshold_feature_overlap_by_marker.png"
random_forest_dt_pysr_threshold_feature_overlap_svg = f"{random_forest_dt_pysr_feature_usage_dir}/rf_vs_pysr_threshold_feature_overlap_by_marker.svg"
random_forest_dt_rf_r2_by_pysr_perk_dependence = f"{random_forest_dt_pysr_feature_usage_dir}/rf_ood_r2_by_pysr_perk_dependence.png"
random_forest_dt_rf_r2_by_pysr_perk_dependence_svg = f"{random_forest_dt_pysr_feature_usage_dir}/rf_ood_r2_by_pysr_perk_dependence.svg"
random_forest_dt_rf_r2_by_pysr_perk_dependence_csv = f"{random_forest_dt_pysr_feature_usage_dir}/rf_ood_r2_by_pysr_perk_dependence.csv"
random_forest_dt_rf_r2_by_pysr_perk_dependence_stats_csv = f"{random_forest_dt_pysr_feature_usage_dir}/rf_ood_r2_by_pysr_perk_dependence_stats.csv"
random_forest_dt_rf_r2_by_pysr_perk_dependence_high_r2 = f"{random_forest_dt_pysr_feature_usage_dir}/rf_ood_r2_by_pysr_perk_dependence_pysr_gt_0p6.png"
random_forest_dt_rf_r2_by_pysr_perk_dependence_high_r2_svg = f"{random_forest_dt_pysr_feature_usage_dir}/rf_ood_r2_by_pysr_perk_dependence_pysr_gt_0p6.svg"
random_forest_dt_rf_r2_by_pysr_perk_dependence_high_r2_csv = f"{random_forest_dt_pysr_feature_usage_dir}/rf_ood_r2_by_pysr_perk_dependence_pysr_gt_0p6.csv"
random_forest_dt_rf_r2_by_pysr_perk_dependence_high_r2_stats_csv = f"{random_forest_dt_pysr_feature_usage_dir}/rf_ood_r2_by_pysr_perk_dependence_pysr_gt_0p6_stats.csv"
random_forest_dt_best_seed_rf_vs_pysr_r2 = f"{random_forest_dt_pysr_feature_usage_dir}/best_seed_rf_vs_pysr_ood_r2_by_perk_dependence.png"
random_forest_dt_best_seed_rf_vs_pysr_r2_svg = f"{random_forest_dt_pysr_feature_usage_dir}/best_seed_rf_vs_pysr_ood_r2_by_perk_dependence.svg"
random_forest_dt_minus_erk_analysis_dir = f"{random_forest_dt_minus_erk_dir}/trajectory_analysis"
random_forest_dt_minus_erk_analysis_summary_dir = f"{random_forest_dt_minus_erk_analysis_dir}/summary"
random_forest_dt_minus_erk_analysis_perk_dir = f"{random_forest_dt_minus_erk_analysis_dir}/plots/perk"
random_forest_dt_minus_erk_analysis_target_dir = f"{random_forest_dt_minus_erk_analysis_dir}/plots/target"
random_forest_dt_minus_erk_analysis_branch_dir = f"{random_forest_dt_minus_erk_analysis_dir}/plots/branch_panels"

random_forest_dt_perk_nn_minus_erk_csv = f"{random_forest_dt_minus_erk_analysis_summary_dir}/perk_trajectory_nn_vs_seedavg_ode_r2.csv"
random_forest_dt_perk_nn_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_nn_vs_seedavg_ode_r2.png"
random_forest_dt_perk_nn_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_nn_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_nn_median_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_median_nn_vs_seedavg_ode_r2.png"
random_forest_dt_perk_nn_median_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_median_nn_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_nn_p90_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_p90_nn_vs_seedavg_ode_r2.png"
random_forest_dt_perk_nn_p90_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_p90_nn_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_quantile_band_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_quantile_band_positive_mean_vs_seedavg_ode_r2.png"
random_forest_dt_perk_quantile_band_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_quantile_band_positive_mean_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_mean_train_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_mean_train_distance_vs_seedavg_ode_r2.png"
random_forest_dt_perk_mean_train_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_mean_train_distance_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_dist_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_distribution_distance_vs_seedavg_ode_r2.png"
random_forest_dt_perk_dist_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_distribution_distance_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_variance_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_variance_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_perk_variance_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_variance_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_top20_variance_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_top20_gfp_variance_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_perk_top20_variance_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_top20_gfp_variance_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_top15_variance_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_top15_gfp_variance_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_perk_top15_variance_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_top15_gfp_variance_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_amplitude_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_amplitude_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_perk_amplitude_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_amplitude_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_top20_amplitude_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_top20_gfp_amplitude_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_perk_top20_amplitude_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_top20_gfp_amplitude_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_top15_amplitude_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_top15_gfp_amplitude_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_perk_top15_amplitude_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_top15_gfp_amplitude_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_amplitude_ratio_sum_top15_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_amplitude_ratio_plus_top15_vs_seedavg_ode_r2.png"
random_forest_dt_perk_amplitude_ratio_sum_top15_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_amplitude_ratio_plus_top15_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_amplitude_ratio_top15_over_full_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_amplitude_ratio_top15_over_full_vs_seedavg_ode_r2.png"
random_forest_dt_perk_amplitude_ratio_top15_over_full_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_amplitude_ratio_top15_over_full_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_top20_trace_cov_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_top20_gfp_trace_cov_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_perk_top20_trace_cov_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_top20_gfp_trace_cov_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_top15_trace_cov_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_top15_gfp_trace_cov_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_perk_top15_trace_cov_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_top15_gfp_trace_cov_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_top20_pairwise_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_top20_gfp_pairwise_sq_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_perk_top20_pairwise_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_top20_gfp_pairwise_sq_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_top15_pairwise_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_top15_gfp_pairwise_sq_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_perk_top15_pairwise_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_top15_gfp_pairwise_sq_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_tail_shape_continuation_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_tail_shape_continuation_vs_seedavg_ode_r2.png"
random_forest_dt_perk_tail_shape_continuation_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_tail_shape_continuation_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_tail_envelope_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_tail_envelope_vs_seedavg_ode_r2.png"
random_forest_dt_perk_tail_envelope_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_tail_envelope_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_tail_raw_continuation_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_tail_raw_continuation_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_perk_tail_raw_continuation_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_tail_raw_continuation_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_tail_local_pca_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_tail_local_pca_vs_seedavg_ode_r2.png"
random_forest_dt_perk_tail_local_pca_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_tail_local_pca_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_boundary_jump_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_boundary_jump_vs_seedavg_ode_r2.png"
random_forest_dt_perk_boundary_jump_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_boundary_jump_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_tail_convex_hull_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_tail_convex_hull_vs_seedavg_ode_r2.png"
random_forest_dt_perk_tail_convex_hull_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_tail_convex_hull_vs_seedavg_ode_r2.svg"
random_forest_dt_perk_tail_timepoint_continuation_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_tail_timepoint_continuation_vs_seedavg_ode_r2.png"
random_forest_dt_perk_tail_timepoint_continuation_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_perk_dir}/perk_trajectory_tail_timepoint_continuation_vs_seedavg_ode_r2.svg"
random_forest_dt_target_nn_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_nn_vs_seedavg_ode_r2.png"
random_forest_dt_target_nn_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_nn_vs_seedavg_ode_r2.svg"
random_forest_dt_target_nn_median_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_median_nn_vs_seedavg_ode_r2.png"
random_forest_dt_target_nn_median_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_median_nn_vs_seedavg_ode_r2.svg"
random_forest_dt_target_nn_p90_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_p90_nn_vs_seedavg_ode_r2.png"
random_forest_dt_target_nn_p90_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_p90_nn_vs_seedavg_ode_r2.svg"
random_forest_dt_target_quantile_band_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_quantile_band_positive_mean_vs_seedavg_ode_r2.png"
random_forest_dt_target_quantile_band_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_quantile_band_positive_mean_vs_seedavg_ode_r2.svg"
random_forest_dt_target_mean_train_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_mean_train_distance_vs_seedavg_ode_r2.png"
random_forest_dt_target_mean_train_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_mean_train_distance_vs_seedavg_ode_r2.svg"
random_forest_dt_target_dist_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_distribution_distance_vs_seedavg_ode_r2.png"
random_forest_dt_target_dist_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_distribution_distance_vs_seedavg_ode_r2.svg"
random_forest_dt_target_variance_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_variance_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_target_variance_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_variance_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_target_top20_variance_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_top20_gfp_variance_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_target_top20_variance_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_top20_gfp_variance_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_target_top15_variance_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_top15_gfp_variance_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_target_top15_variance_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_top15_gfp_variance_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_target_amplitude_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_amplitude_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_target_amplitude_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_amplitude_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_target_top20_amplitude_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_top20_gfp_amplitude_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_target_top20_amplitude_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_top20_gfp_amplitude_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_target_top15_amplitude_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_top15_gfp_amplitude_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_target_top15_amplitude_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_top15_gfp_amplitude_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_target_amplitude_ratio_sum_top15_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_amplitude_ratio_plus_top15_vs_seedavg_ode_r2.png"
random_forest_dt_target_amplitude_ratio_sum_top15_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_amplitude_ratio_plus_top15_vs_seedavg_ode_r2.svg"
random_forest_dt_target_amplitude_ratio_top15_over_full_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_amplitude_ratio_top15_over_full_vs_seedavg_ode_r2.png"
random_forest_dt_target_amplitude_ratio_top15_over_full_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_amplitude_ratio_top15_over_full_vs_seedavg_ode_r2.svg"
random_forest_dt_target_top20_trace_cov_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_top20_gfp_trace_cov_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_target_top20_trace_cov_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_top20_gfp_trace_cov_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_target_top15_trace_cov_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_top15_gfp_trace_cov_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_target_top15_trace_cov_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_top15_gfp_trace_cov_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_target_top20_pairwise_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_top20_gfp_pairwise_sq_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_target_top20_pairwise_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_top20_gfp_pairwise_sq_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_target_top15_pairwise_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_top15_gfp_pairwise_sq_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_target_top15_pairwise_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_top15_gfp_pairwise_sq_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_target_tail_shape_continuation_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_tail_shape_continuation_vs_seedavg_ode_r2.png"
random_forest_dt_target_tail_shape_continuation_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_tail_shape_continuation_vs_seedavg_ode_r2.svg"
random_forest_dt_target_tail_envelope_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_tail_envelope_vs_seedavg_ode_r2.png"
random_forest_dt_target_tail_envelope_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_tail_envelope_vs_seedavg_ode_r2.svg"
random_forest_dt_target_tail_raw_continuation_ratio_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_tail_raw_continuation_ratio_vs_seedavg_ode_r2.png"
random_forest_dt_target_tail_raw_continuation_ratio_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_tail_raw_continuation_ratio_vs_seedavg_ode_r2.svg"
random_forest_dt_target_tail_local_pca_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_tail_local_pca_vs_seedavg_ode_r2.png"
random_forest_dt_target_tail_local_pca_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_tail_local_pca_vs_seedavg_ode_r2.svg"
random_forest_dt_target_boundary_jump_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_boundary_jump_vs_seedavg_ode_r2.png"
random_forest_dt_target_boundary_jump_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_boundary_jump_vs_seedavg_ode_r2.svg"
random_forest_dt_target_tail_convex_hull_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_tail_convex_hull_vs_seedavg_ode_r2.png"
random_forest_dt_target_tail_convex_hull_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_tail_convex_hull_vs_seedavg_ode_r2.svg"
random_forest_dt_target_tail_timepoint_continuation_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_tail_timepoint_continuation_vs_seedavg_ode_r2.png"
random_forest_dt_target_tail_timepoint_continuation_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/perk_dt_trajectory_tail_timepoint_continuation_vs_seedavg_ode_r2.svg"
random_forest_dt_tail_timepoint_worstcase_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/tail_timepoint_continuation_worstcase_vs_seedavg_ode_r2.png"
random_forest_dt_tail_timepoint_worstcase_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/tail_timepoint_continuation_worstcase_vs_seedavg_ode_r2.svg"
random_forest_dt_branch_break_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_target_dir}/tail_branch_break_score_vs_seedavg_ode_r2.png"
random_forest_dt_branch_break_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_target_dir}/tail_branch_break_score_vs_seedavg_ode_r2.svg"
random_forest_dt_tail_branch_panels_minus_erk_plot = f"{random_forest_dt_minus_erk_analysis_branch_dir}/rf_minus_erk_tail_branch_panels.png"
random_forest_dt_tail_branch_panels_minus_erk_plot_svg = f"{random_forest_dt_minus_erk_analysis_branch_dir}/rf_minus_erk_tail_branch_panels.svg"
random_forest_dt_tail_branch_panels_minus_erk_csv = f"{random_forest_dt_minus_erk_analysis_summary_dir}/rf_minus_erk_tail_branch_panels.csv"

# Metrics scatter outputs
metrics_models_r2 = f"{plots_metrics_scatter_dir}/dt/metrics_models_scatter_r2.png"
metrics_models_r2_svg = f"{plots_metrics_scatter_dir}/dt/metrics_models_scatter_r2.svg"
metrics_models_relmae = f"{plots_metrics_scatter_dir}/dt/metrics_models_scatter_relmae.png"
metrics_models_relmae_svg = f"{plots_metrics_scatter_dir}/dt/metrics_models_scatter_relmae.svg"
metrics_pysr_modes_r2 = f"{plots_metrics_scatter_dir}/dt/metrics_pysr_modes_r2.png"
metrics_pysr_modes_r2_svg = f"{plots_metrics_scatter_dir}/dt/metrics_pysr_modes_r2.svg"
metrics_pysr_modes_relmae = f"{plots_metrics_scatter_dir}/dt/metrics_pysr_modes_relmae.png"
metrics_pysr_modes_relmae_svg = f"{plots_metrics_scatter_dir}/dt/metrics_pysr_modes_relmae.svg"
metrics_linreg_modes_r2 = f"{plots_metrics_scatter_dir}/dt/metrics_linreg_modes_r2.png"
metrics_linreg_modes_r2_svg = f"{plots_metrics_scatter_dir}/dt/metrics_linreg_modes_r2.svg"
metrics_linreg_modes_relmae = f"{plots_metrics_scatter_dir}/dt/metrics_linreg_modes_relmae.png"
metrics_linreg_modes_relmae_svg = f"{plots_metrics_scatter_dir}/dt/metrics_linreg_modes_relmae.svg"
metrics_r2_lineplot = f"{plots_metrics_lines_dir}/metrics_r2_by_marker.png"
metrics_r2_lineplot_svg = f"{plots_metrics_lines_dir}/metrics_r2_by_marker.svg"
metrics_integrated_r2_lineplot = f"{plots_metrics_lines_dir}/metrics_integrated_r2_by_marker.png"
metrics_integrated_r2_lineplot_svg = f"{plots_metrics_lines_dir}/metrics_integrated_r2_by_marker.svg"
metrics_integrated_ode_r2_lineplot = f"{plots_metrics_lines_dir}/metrics_integrated_ode_r2_by_marker.png"
metrics_integrated_ode_r2_lineplot_svg = f"{plots_metrics_lines_dir}/metrics_integrated_ode_r2_by_marker.svg"
metrics_pysr_r2_threshold_sanity = f"{plots_metrics_lines_dir}/pysr_r2_threshold_sanity.png"
metrics_pysr_r2_threshold_sanity_svg = f"{plots_metrics_lines_dir}/pysr_r2_threshold_sanity.svg"
metrics_pysr_r2_threshold_sanity_csv = f"{exp_metrics_root}/pysr_r2_threshold_sanity_examples.csv"
metrics_r2_boxplot = f"{plots_metrics_boxes_dir}/metrics_r2_boxplot.png"
metrics_r2_boxplot_svg = f"{plots_metrics_boxes_dir}/metrics_r2_boxplot.svg"
metrics_integrated_r2_boxplot = f"{plots_metrics_boxes_dir}/metrics_integrated_r2_boxplot.png"
metrics_integrated_r2_boxplot_svg = f"{plots_metrics_boxes_dir}/metrics_integrated_r2_boxplot.svg"
metrics_integrated_ode_r2_boxplot = f"{plots_metrics_boxes_dir}/metrics_integrated_ode_r2_boxplot.png"
metrics_integrated_ode_r2_boxplot_svg = f"{plots_metrics_boxes_dir}/metrics_integrated_ode_r2_boxplot.svg"
feature_usage_plots_png = [
    f"{plots_metrics_features_dir}/feature_usage_pysr_snapshot.png",
    f"{plots_metrics_features_dir}/feature_usage_pysr_per_minute.png",
    f"{plots_metrics_features_dir}/feature_usage_linear_regression_snapshot.png",
    f"{plots_metrics_features_dir}/feature_usage_linear_regression_per_minute.png",
]
feature_usage_plots_svg = [
    f"{plots_metrics_features_dir}/feature_usage_pysr_snapshot.svg",
    f"{plots_metrics_features_dir}/feature_usage_pysr_per_minute.svg",
    f"{plots_metrics_features_dir}/feature_usage_linear_regression_snapshot.svg",
    f"{plots_metrics_features_dir}/feature_usage_linear_regression_per_minute.svg",
]
feature_importance_plots_png = [
    f"{plots_metrics_features_dir}/feature_importance_total_snapshot.png",
    f"{plots_metrics_features_dir}/feature_importance_stacked_snapshot.png",
    f"{plots_metrics_features_dir}/feature_importance_coef_stacked_snapshot.png",
    f"{plots_metrics_features_dir}/feature_importance_box_snapshot.png",
    f"{plots_metrics_features_dir}/feature_importance_total_per_minute.png",
    f"{plots_metrics_features_dir}/feature_importance_stacked_per_minute.png",
    f"{plots_metrics_features_dir}/feature_importance_coef_stacked_per_minute.png",
    f"{plots_metrics_features_dir}/feature_importance_box_per_minute.png",
]
feature_importance_plots_svg = [
    f"{plots_metrics_features_dir}/feature_importance_total_snapshot.svg",
    f"{plots_metrics_features_dir}/feature_importance_stacked_snapshot.svg",
    f"{plots_metrics_features_dir}/feature_importance_coef_stacked_snapshot.svg",
    f"{plots_metrics_features_dir}/feature_importance_box_snapshot.svg",
    f"{plots_metrics_features_dir}/feature_importance_total_per_minute.svg",
    f"{plots_metrics_features_dir}/feature_importance_stacked_per_minute.svg",
    f"{plots_metrics_features_dir}/feature_importance_coef_stacked_per_minute.svg",
    f"{plots_metrics_features_dir}/feature_importance_box_per_minute.svg",
]

fit_sanity_dir = f"{plots_root}/fit_sanity"
fit_sanity_done = f"{fit_sanity_dir}/.done"

pysr_grid_output_dir = f"{plots_root}/pysr_grid"
pysr_grid_results_csv = f"{pysr_grid_output_dir}/pysr_hyperparam_grid_results.csv"

# Precompute paths for temporary and final formula files
temp_files = [entry["temp"] for entry in model_details]
formula_files = [entry["formula"] for entry in model_details]

sweeps_enabled = config.get("enable_sr_sweeps", True)

if sweeps_enabled:
    sweep_best_configs = {
        "pysindy": f"data/{enzyme_model}/{data_type}/sr_comparison/sweeps/pysindy/best_config.json",
        "aifeynman": f"data/{enzyme_model}/{data_type}/sr_comparison/sweeps/aifeynman/best_config.json",
        "dso": f"data/{enzyme_model}/{data_type}/sr_comparison/sweeps/dso/best_config.json",
        "kan": f"data/{enzyme_model}/{data_type}/sr_comparison/sweeps/kan/best_config.json",
        "pysr": f"data/{enzyme_model}/{data_type}/sr_comparison/sweeps/pysr/best_config.json",
    }
else:
    sweep_best_configs = {}

SR_SWEEP_SEARCH_SPACE = "sweeps/sr_default_search.yaml"

def sweep_input(method):
    return sweep_best_configs.get(method, [])

timepoint_outputs = []
if data_type == "dynamic":
    timepoint_base = f"data/{enzyme_model}/{data_type}/timepoint_regimes"
    timepoint_outputs = [
        f"{timepoint_base}/shared/plots/log_mae/log_mae_timepoint_lineplot.png",
        f"{timepoint_base}/shared/plots/log_mae/log_mae_timepoint_lineplot_template.png",
        f"{timepoint_base}/shared/plots/log_mae/log_mae_horizontal_boxplot.png",
        f"{timepoint_base}/shared/plots/log_mae/log_mae_horizontal_boxplot_no_outliers.png",
        f"{timepoint_base}/shared/plots/log_mae/log_mae_vertical_boxplot.png",
        f"{timepoint_base}/shared/plots/log_mae/log_mae_vertical_boxplot_no_outliers.png",
        f"{timepoint_base}/shared/plots/log_mae/log_mae_error_distributions.png",
        f"{timepoint_base}/shared/plots/error_landscape.png",
        f"{timepoint_base}/shared/plots/feature_error_correlation_grid.png",
        f"{timepoint_base}/shared/plots/feature_error_correlation_overall.png",
        f"{timepoint_base}/shared/plots/model_error_correlation_grid.png",
        f"{timepoint_base}/shared/plots/model_error_correlation_overall.png",
        f"{timepoint_base}/shared/results/pysr/all_pysr_formulas.txt",
    ]
# Precompute NN model outputs for each variant
nn_model_outputs = [
    f"data/{enzyme_model}/{data_type}/sr_comparison/models/nn/model_nn_{variant}.pth"
    for variant in variant_keys
]

nn_model_arg_str = " ".join(
    f"--nn-model {variant}={path}"
    for variant, path in zip(variant_keys, nn_model_outputs)
)

nn_dataset_size_value = config["dataset_sizes"].get("nn")
nn_dataset_size_arg = (
    f"--nn-dataset-size {nn_dataset_size_value}" if nn_dataset_size_value else ""
)


# Define output files for the entire workflow (Snakemake will skip existing ones)
if enzyme_model == "experimental":

    experimental_rule_all_inputs = [
        marker_time_series_csv,
        marker_feature_matrix_csv,
        marker_fit_snapshot_csv,
        marker_per_minute_csv,
        marker_summary_output,
        overlay_snapshot_plot,
        overlay_snapshot_plot_svg,
        overlay_snapshot_metrics,
        overlay_per_minute_plot,
        overlay_per_minute_plot_svg,
        overlay_per_minute_metrics,
        integration_snapshot_metrics,
        integration_per_minute_metrics,
        integration_snapshot_traj,
        integration_per_minute_traj,
        select_k_metrics,
        select_k_metrics_agg,
        select_k_importance,
        select_k_importance_agg,
        select_k_boxplot_dt,
        select_k_boxplot_integ,
        select_k_boxplot_ode,
        select_k_ribbon_coef,
        select_k_ribbon_variance,
        select_k_panel_a_box,
        select_k_panel_a,
        select_k_panel_b,
        select_k_panel_a_dt,
        select_k_panel_b_dt,
        select_k_panel_a_box_dt,
        select_k_panel_a_linreg_bar,
        select_k_panel_a_relmae,
        select_k_panel_b_relmae,
        select_k_panel_a_box_relmae,
        select_k_panel_a_relmae_dt,
        select_k_panel_b_relmae_dt,
        select_k_panel_a_box_relmae_dt,
        select_k_panel_a_linreg_bar_relmae,
        select_k_variability_dt,
        select_k_variability_perk,
        select_k_pysr_k_vs_r2,
        select_k_heatmap,
        metrics_models_r2,
        metrics_models_r2_svg,
        metrics_models_relmae,
        metrics_models_relmae_svg,
        metrics_pysr_modes_r2,
        metrics_pysr_modes_r2_svg,
        metrics_pysr_modes_relmae,
        metrics_pysr_modes_relmae_svg,
        metrics_linreg_modes_r2,
        metrics_linreg_modes_r2_svg,
        metrics_linreg_modes_relmae,
        metrics_linreg_modes_relmae_svg,
    ]

    experimental_rule_all_inputs.extend(marker_formula_files)
    experimental_rule_all_inputs.append(fit_sanity_done)
    if exp_custom_loss_ablation_enabled:
        experimental_rule_all_inputs.extend(
            [
                custom_loss_ablation_metrics,
                custom_loss_ablation_boxplot,
                custom_loss_ablation_boxplot_svg,
            ]
        )

    rule all:
        input:
            experimental_rule_all_inputs

    rule experimental_marker_inputs:
        input:
            raw=exp_raw_time_course
        output:
            time_series=marker_time_series_csv,
            feature_matrix=marker_feature_matrix_csv,
            fit_snapshot=marker_fit_snapshot_csv,
            per_minute=marker_per_minute_csv
        conda:
            "../../envs/pysr.yaml"
        params:
            output_dir=marker_base_dir,
            prefix=exp_output_prefix,
            gfp_bins=exp_gfp_bins,
            min_points=exp_min_points,
            extra_times=exp_extra_times_args,
            force_rebin=exp_force_rebin_arg,
        shell:
            """
            mkdir -p {params.output_dir}
            if [ -f {output.time_series} ] && [ -f {output.feature_matrix} ] && [ -f {output.fit_snapshot} ] && [ -f {output.per_minute} ]; then
                echo "Using existing functional group inputs; delete outputs to regenerate."
                touch {output.time_series} {output.feature_matrix} {output.fit_snapshot} {output.per_minute}
            else
                python src/pipelines/experimental/data_prep/marker_inputs.py \
                    --time-course {input.raw} \
                    --output-dir {params.output_dir} \
                    --output-prefix {params.prefix} \
                    --gfp-bins {params.gfp_bins} \
                    --min-points {params.min_points} {params.extra_times} {params.force_rebin}
                touch {output.time_series} {output.feature_matrix} {output.fit_snapshot} {output.per_minute}
            fi
            """

    rule experimental_fit_sanity_plots:
        input:
            time=marker_time_series_csv,
            dense=marker_per_minute_csv
        output:
            done=fit_sanity_done
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=fit_sanity_dir,
            protein=exp_fit_sanity_protein,
            marker_flag=(f"--markers {exp_fit_sanity_markers_args}" if exp_fit_sanity_markers_args else ""),
            max_cols=exp_fit_sanity_max_columns,
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/data_prep/fit_sanity_plots.py \
                --time-trajectories {input.time} \
                --fit-trajectories {input.dense} \
                --output-dir {params.out_dir} \
                --protein "{params.protein}" \
                --max-columns {params.max_cols} \
                {params.marker_flag}
            touch {output.done}
            """

    rule experimental_marker_sr:
        input:
            dataset=marker_fit_snapshot_csv,
            per_minute=marker_per_minute_csv
        output:
            summary=marker_summary_output,
            seed_summary=marker_summary_seed_output,
            formulas=marker_formula_files
        conda:
            "../../envs/pysr.yaml"
        params:
            output_dir=exp_sr_output_dir,
            group_definitions_csv=exp_group_definitions_csv,
            models=exp_models_args,
            measured=" ".join(str(t) for t in exp_measured_timepoints)
        shell:
            """
            mkdir -p {params.output_dir}
            python src/pipelines/experimental/sr_pipeline/run_markers.py \
                --dataset {input.dataset} \
                --per-minute-dataset {input.per_minute} \
                --output-dir {params.output_dir} \
                --group-definitions-csv {params.group_definitions_csv} \
                --models {params.models} \
                --measured-timepoints {params.measured}
            """

    rule experimental_custom_loss_ablation:
        input:
            dataset=marker_fit_snapshot_csv,
            per_minute=marker_per_minute_csv
        output:
            metrics=custom_loss_ablation_metrics,
            boxplot=custom_loss_ablation_boxplot,
            boxplot_svg=custom_loss_ablation_boxplot_svg
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=custom_loss_ablation_dir,
            group_definitions_csv=exp_group_definitions_csv,
            seed=exp_custom_loss_ablation_seed,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            max_time=exp_per_minute_max_time,
            strategy=exp_per_minute_sampling_strategy,
            late_window_start=exp_late_sample_window[0],
            late_window_end=exp_late_sample_window[1],
            late_points=exp_late_sample_points,
            metric_column=exp_custom_loss_ablation_metric,
            metric_dataset_mode=exp_custom_loss_ablation_dataset_mode,
            metric_feature_mode=exp_custom_loss_ablation_feature_mode,
            metric_source=exp_custom_loss_ablation_metric_source,
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/experiments/pysr_custom_loss_ablation.py \
                --dataset {input.dataset} \
                --per-minute-dataset {input.per_minute} \
                --group-definitions-csv {params.group_definitions_csv} \
                --output-dir {params.out_dir} \
                --seed {params.seed} \
                --measured-timepoints {params.measured} \
                --per-minute-max-time {params.max_time} \
                --per-minute-sampling-strategy {params.strategy} \
                --late-sample-window {params.late_window_start} {params.late_window_end} \
                --late-sample-points {params.late_points} \
                --metric-column {params.metric_column} \
                --metric-dataset-mode {params.metric_dataset_mode} \
                --metric-feature-mode {params.metric_feature_mode} \
                --metric-source {params.metric_source} \
                --output-csv {output.metrics} \
                --output-plot {output.boxplot} \
                --output-plot-svg {output.boxplot_svg}
            """

    rule experimental_select_k_linreg:
        input:
            per_minute=marker_per_minute_csv
        output:
            metrics=select_k_metrics,
            metrics_agg=select_k_metrics_agg,
            importance=select_k_importance,
            importance_agg=select_k_importance_agg,
            boxplot_dt=select_k_boxplot_dt,
            boxplot_integ=select_k_boxplot_integ,
            boxplot_ode=select_k_boxplot_ode,
            ribbon_coef=select_k_ribbon_coef,
            ribbon_variance=select_k_ribbon_variance
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=select_k_dir,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            max_time=exp_per_minute_max_time,
            strategy=exp_per_minute_sampling_strategy if 'exp_per_minute_sampling_strategy' in globals() else "early_plus_sparse_late",
            late_window_start=exp_late_sample_window[0] if 'exp_late_sample_window' in globals() else 30.0,
            late_window_end=exp_late_sample_window[1] if 'exp_late_sample_window' in globals() else 60.0,
            late_points=exp_late_sample_points if 'exp_late_sample_points' in globals() else 15,
            seeds="42 43 44",
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/sr_pipeline/select_k_linreg_per_minute.py \
                --dataset {input.per_minute} \
                --output-dir {params.out_dir} \
                --per-minute-max-time {params.max_time} \
                --per-minute-sampling-strategy {params.strategy} \
                --late-sample-window {params.late_window_start} {params.late_window_end} \
                --late-sample-points {params.late_points} \
                --seeds {params.seeds} \
                --measured-timepoints {params.measured}
            """

    rule experimental_neural_ode_baseline:
        input:
            per_minute=marker_per_minute_csv
        output:
            metrics=neural_ode_metrics_full,
            metrics_agg=neural_ode_metrics_agg_full,
            boxplot_dt=neural_ode_boxplot_dt_full,
            boxplot_integ=neural_ode_boxplot_integ_full,
            boxplot_ode=neural_ode_boxplot_ode_full
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=neural_ode_full_dir,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            max_time=exp_per_minute_max_time,
            strategy=exp_per_minute_sampling_strategy if 'exp_per_minute_sampling_strategy' in globals() else "early_plus_sparse_late",
            late_window_start=exp_late_sample_window[0] if 'exp_late_sample_window' in globals() else 30.0,
            late_window_end=exp_late_sample_window[1] if 'exp_late_sample_window' in globals() else 60.0,
            late_points=exp_late_sample_points if 'exp_late_sample_points' in globals() else 15,
            seeds="42 43 44",
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/sr_pipeline/neural_ode_baseline_per_minute.py \
                --dataset {input.per_minute} \
                --output-dir {params.out_dir} \
                --per-minute-max-time {params.max_time} \
                --per-minute-sampling-strategy {params.strategy} \
                --late-sample-window {params.late_window_start} {params.late_window_end} \
                --late-sample-points {params.late_points} \
                --seeds {params.seeds} \
                --measured-timepoints {params.measured} \
                --sweep-output {neural_ode_sweep_full}
            """

    rule experimental_neural_ode_sweep:
        input:
            per_minute=marker_per_minute_csv
        output:
            sweep=neural_ode_sweep_full
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=neural_ode_full_dir,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            max_time=exp_per_minute_max_time,
            strategy=exp_per_minute_sampling_strategy if 'exp_per_minute_sampling_strategy' in globals() else "early_plus_sparse_late",
            late_window_start=exp_late_sample_window[0] if 'exp_late_sample_window' in globals() else 30.0,
            late_window_end=exp_late_sample_window[1] if 'exp_late_sample_window' in globals() else 60.0,
            late_points=exp_late_sample_points if 'exp_late_sample_points' in globals() else 15,
            seeds="42 43 44",
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/sr_pipeline/neural_ode_baseline_per_minute.py \
                --dataset {input.per_minute} \
                --output-dir {params.out_dir} \
                --per-minute-max-time {params.max_time} \
                --per-minute-sampling-strategy {params.strategy} \
                --late-sample-window {params.late_window_start} {params.late_window_end} \
                --late-sample-points {params.late_points} \
                --seeds {params.seeds} \
                --measured-timepoints {params.measured} \
                --sweep-output {output.sweep} \
                --sweep-only
            """

    rule experimental_neural_ode_baseline_matched:
        input:
            per_minute=marker_per_minute_csv,
            pysr=integration_per_minute_metrics
        output:
            metrics=neural_ode_metrics_matched,
            metrics_agg=neural_ode_metrics_agg_matched,
            boxplot_dt=neural_ode_boxplot_dt_matched,
            boxplot_integ=neural_ode_boxplot_integ_matched,
            boxplot_ode=neural_ode_boxplot_ode_matched
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=neural_ode_matched_dir,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            max_time=exp_per_minute_max_time,
            strategy=exp_per_minute_sampling_strategy if 'exp_per_minute_sampling_strategy' in globals() else "early_plus_sparse_late",
            late_window_start=exp_late_sample_window[0] if 'exp_late_sample_window' in globals() else 30.0,
            late_window_end=exp_late_sample_window[1] if 'exp_late_sample_window' in globals() else 60.0,
            late_points=exp_late_sample_points if 'exp_late_sample_points' in globals() else 15,
            seeds="42 43 44",
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/sr_pipeline/neural_ode_baseline_per_minute.py \
                --dataset {input.per_minute} \
                --output-dir {params.out_dir} \
                --per-minute-max-time {params.max_time} \
                --per-minute-sampling-strategy {params.strategy} \
                --late-sample-window {params.late_window_start} {params.late_window_end} \
                --late-sample-points {params.late_points} \
                --seeds {params.seeds} \
                --measured-timepoints {params.measured} \
                --sweep-output {neural_ode_sweep_matched} \
                --pysr-metrics {input.pysr} \
                --tag matched_k
            """

    rule experimental_neural_ode_sweep_matched:
        input:
            per_minute=marker_per_minute_csv,
            pysr=integration_per_minute_metrics
        output:
            sweep=neural_ode_sweep_matched
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=neural_ode_matched_dir,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            max_time=exp_per_minute_max_time,
            strategy=exp_per_minute_sampling_strategy if 'exp_per_minute_sampling_strategy' in globals() else "early_plus_sparse_late",
            late_window_start=exp_late_sample_window[0] if 'exp_late_sample_window' in globals() else 30.0,
            late_window_end=exp_late_sample_window[1] if 'exp_late_sample_window' in globals() else 60.0,
            late_points=exp_late_sample_points if 'exp_late_sample_points' in globals() else 15,
            seeds="42 43 44",
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/sr_pipeline/neural_ode_baseline_per_minute.py \
                --dataset {input.per_minute} \
                --output-dir {params.out_dir} \
                --per-minute-max-time {params.max_time} \
                --per-minute-sampling-strategy {params.strategy} \
                --late-sample-window {params.late_window_start} {params.late_window_end} \
                --late-sample-points {params.late_points} \
                --seeds {params.seeds} \
                --measured-timepoints {params.measured} \
                --sweep-output {output.sweep} \
                --pysr-metrics {input.pysr} \
                --tag matched_k \
                --sweep-only
            """

    rule experimental_random_forest_baseline:
        input:
            per_minute=marker_per_minute_csv
        output:
            metrics=random_forest_metrics_full,
            metrics_agg=random_forest_metrics_agg_full,
            boxplot_r2=random_forest_boxplot_r2_full,
            boxplot_relmae=random_forest_boxplot_relmae_full
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=random_forest_full_dir,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            max_time=exp_per_minute_max_time,
            strategy=exp_per_minute_sampling_strategy if 'exp_per_minute_sampling_strategy' in globals() else "early_plus_sparse_late",
            late_window_start=exp_late_sample_window[0] if 'exp_late_sample_window' in globals() else 30.0,
            late_window_end=exp_late_sample_window[1] if 'exp_late_sample_window' in globals() else 60.0,
            late_points=exp_late_sample_points if 'exp_late_sample_points' in globals() else 15,
            seeds="42 43 44",
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/sr_pipeline/random_forest_baseline_per_minute.py \
                --dataset {input.per_minute} \
                --output-dir {params.out_dir} \
                --per-minute-max-time {params.max_time} \
                --per-minute-sampling-strategy {params.strategy} \
                --late-sample-window {params.late_window_start} {params.late_window_end} \
                --late-sample-points {params.late_points} \
                --seeds {params.seeds} \
                --measured-timepoints {params.measured}
            """

    rule experimental_random_forest_baseline_matched:
        input:
            per_minute=marker_per_minute_csv
        output:
            metrics=random_forest_metrics_matched,
            metrics_agg=random_forest_metrics_agg_matched,
            boxplot_r2=random_forest_boxplot_r2_matched,
            boxplot_relmae=random_forest_boxplot_relmae_matched
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=random_forest_matched_dir,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            max_time=exp_per_minute_max_time,
            strategy=exp_per_minute_sampling_strategy if 'exp_per_minute_sampling_strategy' in globals() else "early_plus_sparse_late",
            late_window_start=exp_late_sample_window[0] if 'exp_late_sample_window' in globals() else 30.0,
            late_window_end=exp_late_sample_window[1] if 'exp_late_sample_window' in globals() else 60.0,
            late_points=exp_late_sample_points if 'exp_late_sample_points' in globals() else 15,
            seeds="42 43 44",
            pysr_metrics=integration_per_minute_metrics,
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/sr_pipeline/random_forest_baseline_per_minute.py \
                --dataset {input.per_minute} \
                --output-dir {params.out_dir} \
                --per-minute-max-time {params.max_time} \
                --per-minute-sampling-strategy {params.strategy} \
                --late-sample-window {params.late_window_start} {params.late_window_end} \
                --late-sample-points {params.late_points} \
                --seeds {params.seeds} \
                --measured-timepoints {params.measured} \
                --pysr-metrics {params.pysr_metrics} \
                --tag matched_k
            """

    rule experimental_random_forest_all:
        input:
            random_forest_metrics_agg_full,
            random_forest_metrics_agg_matched,
            random_forest_panel_r2_full,
            random_forest_panel_r2_matched
        run:
            pass

    rule experimental_random_forest_dt_baseline:
        input:
            per_minute=marker_per_minute_csv
        output:
            metrics=random_forest_dt_metrics_direct,
            metrics_agg=random_forest_dt_metrics_agg_direct,
            boxplot_dt=random_forest_dt_boxplot_dt_direct,
            boxplot_integ=random_forest_dt_boxplot_integ_direct,
            boxplot_ode=random_forest_dt_boxplot_ode_direct,
            importance=random_forest_dt_importance_direct,
            importance_agg=random_forest_dt_importance_agg_direct,
            importance_by_marker=random_forest_dt_importance_by_marker_direct
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=random_forest_dt_direct_dir,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            max_time=exp_per_minute_max_time,
            strategy=exp_per_minute_sampling_strategy if 'exp_per_minute_sampling_strategy' in globals() else "early_plus_sparse_late",
            late_window_start=exp_late_sample_window[0] if 'exp_late_sample_window' in globals() else 30.0,
            late_window_end=exp_late_sample_window[1] if 'exp_late_sample_window' in globals() else 60.0,
            late_points=exp_late_sample_points if 'exp_late_sample_points' in globals() else 15,
            seeds="42 43 44",
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/sr_pipeline/random_forest_dt_baseline_per_minute.py \
                --dataset {input.per_minute} \
                --output-dir {params.out_dir} \
                --dt-model-variant rf_dt \
                --test-split-policy top_gfp_bins \
                --per-minute-max-time {params.max_time} \
                --per-minute-sampling-strategy {params.strategy} \
                --late-sample-window {params.late_window_start} {params.late_window_end} \
                --late-sample-points {params.late_points} \
                --seeds {params.seeds} \
                --measured-timepoints {params.measured}
            """

    rule experimental_random_forest_dt_baseline_minus_erk:
        input:
            per_minute=marker_per_minute_csv
        output:
            metrics=random_forest_dt_metrics_minus_erk,
            metrics_agg=random_forest_dt_metrics_agg_minus_erk,
            boxplot_dt=random_forest_dt_boxplot_dt_minus_erk,
            boxplot_integ=random_forest_dt_boxplot_integ_minus_erk,
            boxplot_ode=random_forest_dt_boxplot_ode_minus_erk,
            importance=random_forest_dt_importance_minus_erk,
            importance_agg=random_forest_dt_importance_agg_minus_erk,
            importance_by_marker=random_forest_dt_importance_by_marker_minus_erk
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=random_forest_dt_minus_erk_dir,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            max_time=exp_per_minute_max_time,
            strategy=exp_per_minute_sampling_strategy if 'exp_per_minute_sampling_strategy' in globals() else "early_plus_sparse_late",
            late_window_start=exp_late_sample_window[0] if 'exp_late_sample_window' in globals() else 30.0,
            late_window_end=exp_late_sample_window[1] if 'exp_late_sample_window' in globals() else 60.0,
            late_points=exp_late_sample_points if 'exp_late_sample_points' in globals() else 15,
            seeds="42 43 44",
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/sr_pipeline/random_forest_dt_baseline_per_minute.py \
                --dataset {input.per_minute} \
                --output-dir {params.out_dir} \
                --dt-model-variant minus_erk_plus_rf \
                --test-split-policy top_gfp_bins \
                --per-minute-max-time {params.max_time} \
                --per-minute-sampling-strategy {params.strategy} \
                --late-sample-window {params.late_window_start} {params.late_window_end} \
                --late-sample-points {params.late_points} \
                --seeds {params.seeds} \
                --measured-timepoints {params.measured}
            """

    rule experimental_random_forest_dt_minus_erk_perk_trajectory_nn:
        input:
            per_minute=marker_per_minute_csv,
            rf_metrics=random_forest_dt_metrics_minus_erk
        output:
            summary=random_forest_dt_perk_nn_minus_erk_csv,
            plot=random_forest_dt_perk_nn_minus_erk_plot,
            plot_svg=random_forest_dt_perk_nn_minus_erk_plot_svg,
            median_nn_plot=random_forest_dt_perk_nn_median_minus_erk_plot,
            median_nn_plot_svg=random_forest_dt_perk_nn_median_minus_erk_plot_svg,
            p90_nn_plot=random_forest_dt_perk_nn_p90_minus_erk_plot,
            p90_nn_plot_svg=random_forest_dt_perk_nn_p90_minus_erk_plot_svg,
            quantile_band_plot=random_forest_dt_perk_quantile_band_minus_erk_plot,
            quantile_band_plot_svg=random_forest_dt_perk_quantile_band_minus_erk_plot_svg,
            mean_train_plot=random_forest_dt_perk_mean_train_minus_erk_plot,
            mean_train_plot_svg=random_forest_dt_perk_mean_train_minus_erk_plot_svg,
            distribution_plot=random_forest_dt_perk_dist_minus_erk_plot,
            distribution_plot_svg=random_forest_dt_perk_dist_minus_erk_plot_svg,
            variance_ratio_plot=random_forest_dt_perk_variance_ratio_minus_erk_plot,
            variance_ratio_plot_svg=random_forest_dt_perk_variance_ratio_minus_erk_plot_svg,
            top20_variance_ratio_plot=random_forest_dt_perk_top20_variance_ratio_minus_erk_plot,
            top20_variance_ratio_plot_svg=random_forest_dt_perk_top20_variance_ratio_minus_erk_plot_svg,
            top15_variance_ratio_plot=random_forest_dt_perk_top15_variance_ratio_minus_erk_plot,
            top15_variance_ratio_plot_svg=random_forest_dt_perk_top15_variance_ratio_minus_erk_plot_svg,
            amplitude_ratio_plot=random_forest_dt_perk_amplitude_ratio_minus_erk_plot,
            amplitude_ratio_plot_svg=random_forest_dt_perk_amplitude_ratio_minus_erk_plot_svg,
            top20_amplitude_ratio_plot=random_forest_dt_perk_top20_amplitude_ratio_minus_erk_plot,
            top20_amplitude_ratio_plot_svg=random_forest_dt_perk_top20_amplitude_ratio_minus_erk_plot_svg,
            top15_amplitude_ratio_plot=random_forest_dt_perk_top15_amplitude_ratio_minus_erk_plot,
            top15_amplitude_ratio_plot_svg=random_forest_dt_perk_top15_amplitude_ratio_minus_erk_plot_svg,
            amplitude_ratio_sum_top15_plot=random_forest_dt_perk_amplitude_ratio_sum_top15_minus_erk_plot,
            amplitude_ratio_sum_top15_plot_svg=random_forest_dt_perk_amplitude_ratio_sum_top15_minus_erk_plot_svg,
            amplitude_ratio_top15_over_full_plot=random_forest_dt_perk_amplitude_ratio_top15_over_full_minus_erk_plot,
            amplitude_ratio_top15_over_full_plot_svg=random_forest_dt_perk_amplitude_ratio_top15_over_full_minus_erk_plot_svg,
            top20_trace_cov_ratio_plot=random_forest_dt_perk_top20_trace_cov_ratio_minus_erk_plot,
            top20_trace_cov_ratio_plot_svg=random_forest_dt_perk_top20_trace_cov_ratio_minus_erk_plot_svg,
            top15_trace_cov_ratio_plot=random_forest_dt_perk_top15_trace_cov_ratio_minus_erk_plot,
            top15_trace_cov_ratio_plot_svg=random_forest_dt_perk_top15_trace_cov_ratio_minus_erk_plot_svg,
            top20_pairwise_ratio_plot=random_forest_dt_perk_top20_pairwise_ratio_minus_erk_plot,
            top20_pairwise_ratio_plot_svg=random_forest_dt_perk_top20_pairwise_ratio_minus_erk_plot_svg,
            top15_pairwise_ratio_plot=random_forest_dt_perk_top15_pairwise_ratio_minus_erk_plot,
            top15_pairwise_ratio_plot_svg=random_forest_dt_perk_top15_pairwise_ratio_minus_erk_plot_svg,
            tail_shape_continuation_plot=random_forest_dt_perk_tail_shape_continuation_minus_erk_plot,
            tail_shape_continuation_plot_svg=random_forest_dt_perk_tail_shape_continuation_minus_erk_plot_svg,
            tail_envelope_plot=random_forest_dt_perk_tail_envelope_minus_erk_plot,
            tail_envelope_plot_svg=random_forest_dt_perk_tail_envelope_minus_erk_plot_svg,
            tail_raw_continuation_ratio_plot=random_forest_dt_perk_tail_raw_continuation_ratio_minus_erk_plot,
            tail_raw_continuation_ratio_plot_svg=random_forest_dt_perk_tail_raw_continuation_ratio_minus_erk_plot_svg,
            tail_local_pca_plot=random_forest_dt_perk_tail_local_pca_minus_erk_plot,
            tail_local_pca_plot_svg=random_forest_dt_perk_tail_local_pca_minus_erk_plot_svg,
            boundary_jump_plot=random_forest_dt_perk_boundary_jump_minus_erk_plot,
            boundary_jump_plot_svg=random_forest_dt_perk_boundary_jump_minus_erk_plot_svg,
            tail_convex_hull_plot=random_forest_dt_perk_tail_convex_hull_minus_erk_plot,
            tail_convex_hull_plot_svg=random_forest_dt_perk_tail_convex_hull_minus_erk_plot_svg,
            tail_timepoint_continuation_plot=random_forest_dt_perk_tail_timepoint_continuation_minus_erk_plot,
            tail_timepoint_continuation_plot_svg=random_forest_dt_perk_tail_timepoint_continuation_minus_erk_plot_svg,
            target_plot=random_forest_dt_target_nn_minus_erk_plot,
            target_plot_svg=random_forest_dt_target_nn_minus_erk_plot_svg,
            target_median_nn_plot=random_forest_dt_target_nn_median_minus_erk_plot,
            target_median_nn_plot_svg=random_forest_dt_target_nn_median_minus_erk_plot_svg,
            target_p90_nn_plot=random_forest_dt_target_nn_p90_minus_erk_plot,
            target_p90_nn_plot_svg=random_forest_dt_target_nn_p90_minus_erk_plot_svg,
            target_quantile_band_plot=random_forest_dt_target_quantile_band_minus_erk_plot,
            target_quantile_band_plot_svg=random_forest_dt_target_quantile_band_minus_erk_plot_svg,
            target_mean_train_plot=random_forest_dt_target_mean_train_minus_erk_plot,
            target_mean_train_plot_svg=random_forest_dt_target_mean_train_minus_erk_plot_svg,
            target_distribution_plot=random_forest_dt_target_dist_minus_erk_plot,
            target_distribution_plot_svg=random_forest_dt_target_dist_minus_erk_plot_svg,
            target_variance_ratio_plot=random_forest_dt_target_variance_ratio_minus_erk_plot,
            target_variance_ratio_plot_svg=random_forest_dt_target_variance_ratio_minus_erk_plot_svg,
            target_top20_variance_ratio_plot=random_forest_dt_target_top20_variance_ratio_minus_erk_plot,
            target_top20_variance_ratio_plot_svg=random_forest_dt_target_top20_variance_ratio_minus_erk_plot_svg,
            target_top15_variance_ratio_plot=random_forest_dt_target_top15_variance_ratio_minus_erk_plot,
            target_top15_variance_ratio_plot_svg=random_forest_dt_target_top15_variance_ratio_minus_erk_plot_svg,
            target_amplitude_ratio_plot=random_forest_dt_target_amplitude_ratio_minus_erk_plot,
            target_amplitude_ratio_plot_svg=random_forest_dt_target_amplitude_ratio_minus_erk_plot_svg,
            target_top20_amplitude_ratio_plot=random_forest_dt_target_top20_amplitude_ratio_minus_erk_plot,
            target_top20_amplitude_ratio_plot_svg=random_forest_dt_target_top20_amplitude_ratio_minus_erk_plot_svg,
            target_top15_amplitude_ratio_plot=random_forest_dt_target_top15_amplitude_ratio_minus_erk_plot,
            target_top15_amplitude_ratio_plot_svg=random_forest_dt_target_top15_amplitude_ratio_minus_erk_plot_svg,
            target_amplitude_ratio_sum_top15_plot=random_forest_dt_target_amplitude_ratio_sum_top15_minus_erk_plot,
            target_amplitude_ratio_sum_top15_plot_svg=random_forest_dt_target_amplitude_ratio_sum_top15_minus_erk_plot_svg,
            target_amplitude_ratio_top15_over_full_plot=random_forest_dt_target_amplitude_ratio_top15_over_full_minus_erk_plot,
            target_amplitude_ratio_top15_over_full_plot_svg=random_forest_dt_target_amplitude_ratio_top15_over_full_minus_erk_plot_svg,
            target_top20_trace_cov_ratio_plot=random_forest_dt_target_top20_trace_cov_ratio_minus_erk_plot,
            target_top20_trace_cov_ratio_plot_svg=random_forest_dt_target_top20_trace_cov_ratio_minus_erk_plot_svg,
            target_top15_trace_cov_ratio_plot=random_forest_dt_target_top15_trace_cov_ratio_minus_erk_plot,
            target_top15_trace_cov_ratio_plot_svg=random_forest_dt_target_top15_trace_cov_ratio_minus_erk_plot_svg,
            target_top20_pairwise_ratio_plot=random_forest_dt_target_top20_pairwise_ratio_minus_erk_plot,
            target_top20_pairwise_ratio_plot_svg=random_forest_dt_target_top20_pairwise_ratio_minus_erk_plot_svg,
            target_top15_pairwise_ratio_plot=random_forest_dt_target_top15_pairwise_ratio_minus_erk_plot,
            target_top15_pairwise_ratio_plot_svg=random_forest_dt_target_top15_pairwise_ratio_minus_erk_plot_svg,
            target_tail_shape_continuation_plot=random_forest_dt_target_tail_shape_continuation_minus_erk_plot,
            target_tail_shape_continuation_plot_svg=random_forest_dt_target_tail_shape_continuation_minus_erk_plot_svg,
            target_tail_envelope_plot=random_forest_dt_target_tail_envelope_minus_erk_plot,
            target_tail_envelope_plot_svg=random_forest_dt_target_tail_envelope_minus_erk_plot_svg,
            target_tail_raw_continuation_ratio_plot=random_forest_dt_target_tail_raw_continuation_ratio_minus_erk_plot,
            target_tail_raw_continuation_ratio_plot_svg=random_forest_dt_target_tail_raw_continuation_ratio_minus_erk_plot_svg,
            target_tail_local_pca_plot=random_forest_dt_target_tail_local_pca_minus_erk_plot,
            target_tail_local_pca_plot_svg=random_forest_dt_target_tail_local_pca_minus_erk_plot_svg,
            target_boundary_jump_plot=random_forest_dt_target_boundary_jump_minus_erk_plot,
            target_boundary_jump_plot_svg=random_forest_dt_target_boundary_jump_minus_erk_plot_svg,
            target_tail_convex_hull_plot=random_forest_dt_target_tail_convex_hull_minus_erk_plot,
            target_tail_convex_hull_plot_svg=random_forest_dt_target_tail_convex_hull_minus_erk_plot_svg,
            target_tail_timepoint_continuation_plot=random_forest_dt_target_tail_timepoint_continuation_minus_erk_plot,
            target_tail_timepoint_continuation_plot_svg=random_forest_dt_target_tail_timepoint_continuation_minus_erk_plot_svg,
            tail_timepoint_worstcase_plot=random_forest_dt_tail_timepoint_worstcase_minus_erk_plot,
            tail_timepoint_worstcase_plot_svg=random_forest_dt_tail_timepoint_worstcase_minus_erk_plot_svg,
            branch_break_plot=random_forest_dt_branch_break_minus_erk_plot,
            branch_break_plot_svg=random_forest_dt_branch_break_minus_erk_plot_svg
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=random_forest_dt_minus_erk_dir
        shell:
            """
            mkdir -p {params.out_dir}
            MPLCONFIGDIR={params.out_dir}/.mplcache conda run -n pysr_env python src/pipelines/experimental/sr_pipeline/analyze_rf_perk_trajectory_nn.py \
                --dataset {input.per_minute} \
                --rf-metrics {input.rf_metrics} \
                --output-csv {output.summary} \
                --output-plot {output.plot} \
                --output-median-nn-plot {output.median_nn_plot} \
                --output-p90-nn-plot {output.p90_nn_plot} \
                --output-quantile-band-plot {output.quantile_band_plot} \
                --output-mean-train-plot {output.mean_train_plot} \
                --output-distribution-plot {output.distribution_plot} \
                --output-variance-ratio-plot {output.variance_ratio_plot} \
                --output-top20-variance-ratio-plot {output.top20_variance_ratio_plot} \
                --output-top15-variance-ratio-plot {output.top15_variance_ratio_plot} \
                --output-amplitude-ratio-plot {output.amplitude_ratio_plot} \
                --output-top20-amplitude-ratio-plot {output.top20_amplitude_ratio_plot} \
                --output-top15-amplitude-ratio-plot {output.top15_amplitude_ratio_plot} \
                --output-amplitude-ratio-sum-top15-plot {output.amplitude_ratio_sum_top15_plot} \
                --output-amplitude-ratio-top15-over-full-plot {output.amplitude_ratio_top15_over_full_plot} \
                --output-top20-trace-cov-ratio-plot {output.top20_trace_cov_ratio_plot} \
                --output-top15-trace-cov-ratio-plot {output.top15_trace_cov_ratio_plot} \
                --output-top20-pairwise-ratio-plot {output.top20_pairwise_ratio_plot} \
                --output-top15-pairwise-ratio-plot {output.top15_pairwise_ratio_plot} \
                --output-tail-shape-continuation-plot {output.tail_shape_continuation_plot} \
                --output-tail-envelope-plot {output.tail_envelope_plot} \
                --output-tail-raw-continuation-ratio-plot {output.tail_raw_continuation_ratio_plot} \
                --output-tail-local-pca-plot {output.tail_local_pca_plot} \
                --output-boundary-jump-plot {output.boundary_jump_plot} \
                --output-tail-convex-hull-plot {output.tail_convex_hull_plot} \
                --output-tail-timepoint-continuation-plot {output.tail_timepoint_continuation_plot} \
                --output-target-plot {output.target_plot} \
                --output-target-median-nn-plot {output.target_median_nn_plot} \
                --output-target-p90-nn-plot {output.target_p90_nn_plot} \
                --output-target-quantile-band-plot {output.target_quantile_band_plot} \
                --output-target-mean-train-plot {output.target_mean_train_plot} \
                --output-target-distribution-plot {output.target_distribution_plot} \
                --output-target-variance-ratio-plot {output.target_variance_ratio_plot} \
                --output-target-top20-variance-ratio-plot {output.target_top20_variance_ratio_plot} \
                --output-target-top15-variance-ratio-plot {output.target_top15_variance_ratio_plot} \
                --output-target-amplitude-ratio-plot {output.target_amplitude_ratio_plot} \
                --output-target-top20-amplitude-ratio-plot {output.target_top20_amplitude_ratio_plot} \
                --output-target-top15-amplitude-ratio-plot {output.target_top15_amplitude_ratio_plot} \
                --output-target-amplitude-ratio-sum-top15-plot {output.target_amplitude_ratio_sum_top15_plot} \
                --output-target-amplitude-ratio-top15-over-full-plot {output.target_amplitude_ratio_top15_over_full_plot} \
                --output-target-top20-trace-cov-ratio-plot {output.target_top20_trace_cov_ratio_plot} \
                --output-target-top15-trace-cov-ratio-plot {output.target_top15_trace_cov_ratio_plot} \
                --output-target-top20-pairwise-ratio-plot {output.target_top20_pairwise_ratio_plot} \
                --output-target-top15-pairwise-ratio-plot {output.target_top15_pairwise_ratio_plot} \
                --output-target-tail-shape-continuation-plot {output.target_tail_shape_continuation_plot} \
                --output-target-tail-envelope-plot {output.target_tail_envelope_plot} \
                --output-target-tail-raw-continuation-ratio-plot {output.target_tail_raw_continuation_ratio_plot} \
                --output-target-tail-local-pca-plot {output.target_tail_local_pca_plot} \
                --output-target-boundary-jump-plot {output.target_boundary_jump_plot} \
                --output-target-tail-convex-hull-plot {output.target_tail_convex_hull_plot} \
                --output-target-tail-timepoint-continuation-plot {output.target_tail_timepoint_continuation_plot} \
                --output-tail-timepoint-worstcase-plot {output.tail_timepoint_worstcase_plot} \
                --output-branch-break-plot {output.branch_break_plot} \
                --test-split-policy top_gfp_bins
            """

    rule experimental_random_forest_dt_all:
        input:
            random_forest_dt_metrics_agg_direct,
            random_forest_dt_metrics_agg_minus_erk,
            random_forest_dt_importance_agg_direct,
            random_forest_dt_importance_agg_minus_erk,
            random_forest_dt_pysr_feature_usage_csv,
            random_forest_dt_pysr_feature_usage_heatmap,
            random_forest_dt_pysr_feature_overlap,
            random_forest_dt_pysr_threshold_feature_usage_heatmap,
            random_forest_dt_pysr_threshold_feature_overlap,
            random_forest_dt_rf_r2_by_pysr_perk_dependence,
            random_forest_dt_rf_r2_by_pysr_perk_dependence_csv,
            random_forest_dt_rf_r2_by_pysr_perk_dependence_stats_csv,
            random_forest_dt_rf_r2_by_pysr_perk_dependence_high_r2,
            random_forest_dt_rf_r2_by_pysr_perk_dependence_high_r2_csv,
            random_forest_dt_rf_r2_by_pysr_perk_dependence_high_r2_stats_csv,
            random_forest_dt_best_seed_rf_vs_pysr_r2
        run:
            pass

    rule experimental_random_forest_dt_minus_erk_vs_ood_pysr_feature_usage:
        input:
            rf_importance=random_forest_dt_importance_minus_erk,
            rf_metrics=random_forest_dt_metrics_minus_erk,
            pysr_metrics=[
                f"{exp_runs_root}/top_gfp_ood_pysr/seeds/seed_42/metrics/marker_integration_metrics_per_minute.csv",
                f"{exp_runs_root}/top_gfp_ood_pysr/seeds/seed_43/metrics/marker_integration_metrics_per_minute.csv",
                f"{exp_runs_root}/top_gfp_ood_pysr/seeds/seed_44/metrics/marker_integration_metrics_per_minute.csv",
            ]
        output:
            csv=random_forest_dt_pysr_feature_usage_csv,
            heatmap=random_forest_dt_pysr_feature_usage_heatmap,
            heatmap_svg=random_forest_dt_pysr_feature_usage_heatmap_svg,
            overlap=random_forest_dt_pysr_feature_overlap,
            overlap_svg=random_forest_dt_pysr_feature_overlap_svg,
            threshold_heatmap=random_forest_dt_pysr_threshold_feature_usage_heatmap,
            threshold_heatmap_svg=random_forest_dt_pysr_threshold_feature_usage_heatmap_svg,
            threshold_overlap=random_forest_dt_pysr_threshold_feature_overlap,
            threshold_overlap_svg=random_forest_dt_pysr_threshold_feature_overlap_svg,
            perk_dependence_plot=random_forest_dt_rf_r2_by_pysr_perk_dependence,
            perk_dependence_plot_svg=random_forest_dt_rf_r2_by_pysr_perk_dependence_svg,
            perk_dependence_csv=random_forest_dt_rf_r2_by_pysr_perk_dependence_csv,
            perk_dependence_stats_csv=random_forest_dt_rf_r2_by_pysr_perk_dependence_stats_csv,
            perk_dependence_high_r2_plot=random_forest_dt_rf_r2_by_pysr_perk_dependence_high_r2,
            perk_dependence_high_r2_plot_svg=random_forest_dt_rf_r2_by_pysr_perk_dependence_high_r2_svg,
            perk_dependence_high_r2_csv=random_forest_dt_rf_r2_by_pysr_perk_dependence_high_r2_csv,
            perk_dependence_high_r2_stats_csv=random_forest_dt_rf_r2_by_pysr_perk_dependence_high_r2_stats_csv,
            best_seed_scatter=random_forest_dt_best_seed_rf_vs_pysr_r2,
            best_seed_scatter_svg=random_forest_dt_best_seed_rf_vs_pysr_r2_svg
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=random_forest_dt_pysr_feature_usage_dir,
            mpl_cache=f"{random_forest_dt_pysr_feature_usage_dir}/.mplcache"
        shell:
            """
            mkdir -p {params.out_dir}
            MPLCONFIGDIR={params.mpl_cache} PYTHONPATH=src python src/pipelines/experimental/sr_pipeline/compare_rf_pysr_feature_usage.py \
                --rf-importance {input.rf_importance} \
                --rf-metrics {input.rf_metrics} \
                --pysr-metrics {input.pysr_metrics} \
                --output-dir {params.out_dir} \
                --importance-threshold 0.20 \
                --rf-best-metric ode_integ_r2_median
            """

    rule experimental_random_forest_dt_minus_erk_tail_branch_panels:
        input:
            per_minute=marker_per_minute_csv,
            rf_metrics=random_forest_dt_metrics_minus_erk
        output:
            png=random_forest_dt_tail_branch_panels_minus_erk_plot,
            svg=random_forest_dt_tail_branch_panels_minus_erk_plot_svg,
            csv=random_forest_dt_tail_branch_panels_minus_erk_csv
        params:
            markers="MAPK1 MAPK3 PTPN7 MAP2K2 PIP5K3 TBK1 ERBB2 DUSP16",
            out_dir=random_forest_dt_minus_erk_dir
        shell:
            """
            mkdir -p {params.out_dir}
            MPLCONFIGDIR={params.out_dir}/.mplcache conda run -n pysr_env python src/pipelines/experimental/sr_pipeline/plot_rf_tail_branch_panels.py \
                --dataset {input.per_minute} \
                --rf-metrics {input.rf_metrics} \
                --output-png {output.png} \
                --output-svg {output.svg} \
                --output-csv {output.csv} \
                --markers {params.markers}
            """

    rule experimental_plot_pysr_vs_random_forest:
        input:
            baseline=random_forest_metrics_agg_full
        output:
            panel_r2=random_forest_panel_r2_full,
            panel_r2_svg=random_forest_panel_r2_full_svg,
            panel_relmae=random_forest_panel_relmae_full,
            panel_relmae_svg=random_forest_panel_relmae_full_svg
        conda:
            "../../envs/pysr.yaml"
        params:
            pysr_metrics=integration_per_minute_metrics
        shell:
            """
            python src/pipelines/experimental/sr_pipeline/plot_pysr_vs_random_forest.py \
                --baseline-metrics {input.baseline} \
                --pysr-metrics {params.pysr_metrics} \
                --output-r2 {output.panel_r2} \
                --output-relmae {output.panel_relmae}
            """

    rule experimental_plot_pysr_vs_random_forest_matched:
        input:
            baseline=random_forest_metrics_agg_matched
        output:
            panel_r2=random_forest_panel_r2_matched,
            panel_r2_svg=random_forest_panel_r2_matched_svg,
            panel_relmae=random_forest_panel_relmae_matched,
            panel_relmae_svg=random_forest_panel_relmae_matched_svg
        conda:
            "../../envs/pysr.yaml"
        params:
            pysr_metrics=integration_per_minute_metrics
        shell:
            """
            python src/pipelines/experimental/sr_pipeline/plot_pysr_vs_random_forest.py \
                --baseline-metrics {input.baseline} \
                --pysr-metrics {params.pysr_metrics} \
                --output-r2 {output.panel_r2} \
                --output-relmae {output.panel_relmae}
            """

    rule experimental_neural_ode_all:
        input:
            neural_ode_metrics_agg_full,
            neural_ode_metrics_agg_matched,
            neural_ode_diffrax_metrics_agg_full

    # True Neural ODE baseline (diffrax, trajectory loss + backprop-through-solver).
    # Trains one model per (marker, seed) with diffrax + Dopri5 and writes
    # outputs in the same canonical schema as the pipeline NN baseline so the
    # PySR comparison plot script can be reused unchanged. Resumes from CSV
    # per (seed, marker) — re-running is cheap when artifacts already exist.
    rule experimental_neural_ode_diffrax_baseline:
        input:
            per_minute=marker_per_minute_csv
        output:
            metrics=neural_ode_diffrax_metrics_full,
            metrics_agg=neural_ode_diffrax_metrics_agg_full,
            boxplot_dt=neural_ode_diffrax_boxplot_dt_full,
            boxplot_integ=neural_ode_diffrax_boxplot_integ_full,
            boxplot_ode=neural_ode_diffrax_boxplot_ode_full
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=neural_ode_diffrax_full_dir,
            seeds_dir=f"{exp_runs_root}/neural_ode_diffrax/_seeds",
            runs_root=exp_runs_root,
            seeds="42 43 44",
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            max_time=exp_per_minute_max_time,
            strategy=exp_per_minute_sampling_strategy if 'exp_per_minute_sampling_strategy' in globals() else "early_plus_sparse_late",
            late_window_start=exp_late_sample_window[0] if 'exp_late_sample_window' in globals() else 30.0,
            late_window_end=exp_late_sample_window[1] if 'exp_late_sample_window' in globals() else 60.0,
            late_points=exp_late_sample_points if 'exp_late_sample_points' in globals() else 15,
        shell:
            """
            mkdir -p {params.out_dir} {params.seeds_dir}
            for seed in {params.seeds}; do
                mkdir -p {params.seeds_dir}/seed_${{seed}}
                python src/pipelines/experimental/sr_pipeline/neural_ode_diffrax_baseline.py \
                    --dataset {input.per_minute} \
                    --output-dir {params.seeds_dir}/seed_${{seed}} \
                    --seeds ${{seed}} \
                    --epochs 200 --patience 20 \
                    --per-minute-max-time {params.max_time} \
                    --per-minute-sampling-strategy {params.strategy} \
                    --late-sample-window {params.late_window_start} {params.late_window_end} \
                    --late-sample-points {params.late_points} \
                    --measured-timepoints {params.measured} \
                    --save-models \
                    --jac-reg 1.0 \
                    --tag diffrax_seed${{seed}}_jac1
            done
            python src/pipelines/experimental/sr_pipeline/neural_ode_diffrax_promote.py \
                --source-dir {params.seeds_dir} \
                --runs-root {params.runs_root} \
                --seeds {params.seeds}
            """

    rule experimental_neural_ode_diffrax_baseline_ood:
        # Same trainer as `experimental_neural_ode_diffrax_baseline` but with
        # --test-split-policy top_gfp_bins: the highest-GFP-dose bins are held
        # out so generalisation is genuine extrapolation rather than dose
        # interpolation. Models + per-seed CSVs stage to _seeds_ood/.
        input:
            per_minute=marker_per_minute_csv
        output:
            metrics=neural_ode_diffrax_metrics_full_ood,
            metrics_agg=neural_ode_diffrax_metrics_agg_full_ood,
            boxplot_dt=neural_ode_diffrax_boxplot_dt_full_ood,
            boxplot_integ=neural_ode_diffrax_boxplot_integ_full_ood,
            boxplot_ode=neural_ode_diffrax_boxplot_ode_full_ood
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=neural_ode_diffrax_full_ood_dir,
            seeds_dir=f"{exp_runs_root}/neural_ode_diffrax/_seeds_ood",
            runs_root=exp_runs_root,
            seeds="42 43 44",
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            max_time=exp_per_minute_max_time,
            strategy=exp_per_minute_sampling_strategy if 'exp_per_minute_sampling_strategy' in globals() else "early_plus_sparse_late",
            late_window_start=exp_late_sample_window[0] if 'exp_late_sample_window' in globals() else 30.0,
            late_window_end=exp_late_sample_window[1] if 'exp_late_sample_window' in globals() else 60.0,
            late_points=exp_late_sample_points if 'exp_late_sample_points' in globals() else 15,
        shell:
            """
            mkdir -p {params.out_dir} {params.seeds_dir}
            for seed in {params.seeds}; do
                mkdir -p {params.seeds_dir}/seed_${{seed}}
                python src/pipelines/experimental/sr_pipeline/neural_ode_diffrax_baseline.py \
                    --dataset {input.per_minute} \
                    --output-dir {params.seeds_dir}/seed_${{seed}} \
                    --seeds ${{seed}} \
                    --epochs 200 --patience 20 \
                    --per-minute-max-time {params.max_time} \
                    --per-minute-sampling-strategy {params.strategy} \
                    --late-sample-window {params.late_window_start} {params.late_window_end} \
                    --late-sample-points {params.late_points} \
                    --measured-timepoints {params.measured} \
                    --save-models \
                    --jac-reg 1.0 \
                    --test-split-policy top_gfp_bins \
                    --tag diffrax_seed${{seed}}_jac1_ood
            done
            python src/pipelines/experimental/sr_pipeline/neural_ode_diffrax_promote.py \
                --source-dir {params.seeds_dir} \
                --runs-root {params.runs_root} \
                --out-subdir neural_ode_diffrax/full_ood \
                --seeds {params.seeds}
            """

    rule experimental_neural_ode_diffrax_causal_ood:
        input:
            metrics_agg=neural_ode_diffrax_metrics_agg_full_ood
        output:
            summary=neural_ode_diffrax_causal_ood_summary,
            pr_csv=neural_ode_diffrax_causal_ood_pr_csv,
            pr_plot=neural_ode_diffrax_causal_ood_pr_plot
        conda:
            "../../envs/pysr.yaml"
        params:
            seeds_dir=f"{exp_runs_root}/neural_ode_diffrax/_seeds_ood",
            out_dir=neural_ode_diffrax_causal_ood_dir,
            seeds="42 43 44",
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/sr_pipeline/neural_ode_diffrax_causal.py \
                --source-dir {params.seeds_dir} \
                --output-dir {params.out_dir} \
                --seeds {params.seeds} \
                --consensus-only
            """

    # ---------------------------------------------------------------------
    # Paper figures: three OOD regulariser sweeps + plotting rules.
    # Each sweep trains a model per (marker, seed) under top_gfp_bins split
    # and saves the model checkpoints + per-seed metrics CSVs needed by the
    # paper figure scripts.
    # ---------------------------------------------------------------------
    def _diffrax_sweep_shell(seeds_dir, tag, extra_flags):
        return f"""
        mkdir -p {{params.out_dir}}
        for seed in {{params.seeds}}; do
            mkdir -p {{params.out_dir}}/seed_${{{{seed}}}}
            python src/pipelines/experimental/sr_pipeline/neural_ode_diffrax_baseline.py \\
                --dataset {{input.per_minute}} \\
                --output-dir {{params.out_dir}}/seed_${{{{seed}}}} \\
                --seeds ${{{{seed}}}} \\
                --epochs 200 --patience 20 \\
                --per-minute-max-time {{params.max_time}} \\
                --per-minute-sampling-strategy {{params.strategy}} \\
                --late-sample-window {{params.late_window_start}} {{params.late_window_end}} \\
                --late-sample-points {{params.late_points}} \\
                --measured-timepoints {{params.measured}} \\
                --save-models \\
                --test-split-policy top_gfp_bins \\
                {extra_flags} \\
                --tag {tag}_seed${{{{seed}}}}
        done
        touch {{output.done}}
        """

    _diffrax_sweep_params = dict(
        runs_root=exp_runs_root,
        seeds="42 43 44",
        measured=" ".join(str(t) for t in exp_measured_timepoints),
        max_time=exp_per_minute_max_time,
        strategy=exp_per_minute_sampling_strategy if 'exp_per_minute_sampling_strategy' in globals() else "early_plus_sparse_late",
        late_window_start=exp_late_sample_window[0] if 'exp_late_sample_window' in globals() else 30.0,
        late_window_end=exp_late_sample_window[1] if 'exp_late_sample_window' in globals() else 60.0,
        late_points=exp_late_sample_points if 'exp_late_sample_points' in globals() else 15,
    )

    rule experimental_neural_ode_diffrax_ood_l1:
        # L1 element-wise Jacobian sparsity, λ_jac = 1.0 (appendix variant).
        input:
            per_minute=marker_per_minute_csv
        output:
            done=neural_ode_diffrax_l1_done
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=neural_ode_diffrax_l1_seeds_dir,
            **_diffrax_sweep_params,
        shell:
            _diffrax_sweep_shell(
                neural_ode_diffrax_l1_seeds_dir,
                "diffrax_ood_l1_lam1",
                "--jac-reg 1.0 --jac-reg-mode l1 --hess-reg 0.0",
            )

    rule experimental_neural_ode_diffrax_ood_l21:
        # L21 group-sparse Jacobian, λ_jac = 3.0 — main-text "Neural ODE".
        # Calibrated on VAL R²; see scripts/calib_l21jac_then_decide.sh.
        input:
            per_minute=marker_per_minute_csv
        output:
            done=neural_ode_diffrax_l21_done
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=neural_ode_diffrax_l21_seeds_dir,
            **_diffrax_sweep_params,
        shell:
            _diffrax_sweep_shell(
                neural_ode_diffrax_l21_seeds_dir,
                "diffrax_ood_l21j3_valcalib",
                "--jac-reg 3.0 --jac-reg-mode l21 --hess-reg 0.0",
            )

    rule experimental_neural_ode_diffrax_ood_cnode:
        # C-NODE path-product regulariser (Aliee/Theis/Kilbertus 2022), λ = 0.01.
        # Calibrated on VAL R²; see scripts/calib_pathreg_then_sweep.sh.
        input:
            per_minute=marker_per_minute_csv
        output:
            done=neural_ode_diffrax_cnode_done
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=neural_ode_diffrax_cnode_seeds_dir,
            **_diffrax_sweep_params,
        shell:
            _diffrax_sweep_shell(
                neural_ode_diffrax_cnode_seeds_dir,
                "diffrax_ood_cnode_lam0p01",
                "--path-reg 0.01",
            )

    rule experimental_paper_fig_parsimony_tradeoff:
        # Main-text figure 1: PySR vs Neural ODE (L21) OOD R² scatter + PR boxplot.
        input:
            nn_done=neural_ode_diffrax_l21_done,
            dataset=marker_per_minute_csv
        output:
            fig=paper_fig_parsimony_tradeoff
        conda:
            "../../envs/pysr.yaml"
        params:
            nn_dir=neural_ode_diffrax_l21_seeds_dir,
            pysr_dir=f"{exp_runs_root}/top_gfp_ood_pysr/seeds",
        shell:
            """
            python src/pipelines/experimental/sr_pipeline/paper_figures/plot_parsimony_tradeoff.py \
                --nn-dir {params.nn_dir} \
                --pysr-dir {params.pysr_dir} \
                --dataset {input.dataset} \
                --output {output.fig}
            """

    rule experimental_paper_fig_cutoff_robustness:
        # Main-text figure 2: top-X% mass cutoff sweep — seed robustness +
        # between-method agreement (L21 vs PySR) + driver set size vs cutoff.
        input:
            nn_done=neural_ode_diffrax_l21_done,
            dataset=marker_per_minute_csv
        output:
            fig=paper_fig_cutoff_robustness
        conda:
            "../../envs/pysr.yaml"
        params:
            nn_dir=neural_ode_diffrax_l21_seeds_dir,
            pysr_dir=f"{exp_runs_root}/top_gfp_ood_pysr/seeds",
        shell:
            """
            python src/pipelines/experimental/sr_pipeline/paper_figures/plot_cutoff_robustness.py \
                --nn-dir {params.nn_dir} \
                --pysr-dir {params.pysr_dir} \
                --dataset {input.dataset} \
                --output {output.fig}
            """

    rule experimental_paper_fig_nn_appendix:
        # Appendix figure: L1 vs L21 vs C-NODE accuracy + parsimony comparison.
        input:
            l1=neural_ode_diffrax_l1_done,
            l21=neural_ode_diffrax_l21_done,
            cnode=neural_ode_diffrax_cnode_done
        output:
            fig=paper_fig_nn_appendix
        conda:
            "../../envs/pysr.yaml"
        params:
            l1_dir=neural_ode_diffrax_l1_seeds_dir,
            l21_dir=neural_ode_diffrax_l21_seeds_dir,
            cnode_dir=neural_ode_diffrax_cnode_seeds_dir,
        shell:
            """
            python src/pipelines/experimental/sr_pipeline/paper_figures/plot_nn_appendix_comparison.py \
                --l1-dir {params.l1_dir} \
                --l21-dir {params.l21_dir} \
                --cnode-dir {params.cnode_dir} \
                --output {output.fig}
            """

    rule experimental_paper_figures_all:
        # Aggregator: build everything for the manuscript (main text + appendix).
        # Run with:  snakemake -j 1 experimental_paper_figures_all
        input:
            paper_fig_parsimony_tradeoff,
            paper_fig_cutoff_robustness,
            paper_fig_nn_appendix,

    rule experimental_plot_pysr_vs_neural_ode_diffrax_ood:
        input:
            baseline=neural_ode_diffrax_metrics_agg_full_ood,
            pysr=integration_per_minute_metrics
        output:
            panel_ode=neural_ode_diffrax_panel_ode_full_ood,
            panel_dt=neural_ode_diffrax_panel_dt_full_ood,
            baseline=neural_ode_diffrax_baseline_full_ood,
            pysr_k_vs_r2=neural_ode_diffrax_pysr_k_vs_r2_full_ood,
            quadrant_bar=neural_ode_diffrax_quadrant_bar_full_ood
        conda:
            "../../envs/pysr.yaml"
        params:
            measured=" ".join(str(t) for t in exp_measured_timepoints)
        shell:
            """
            python src/pipelines/experimental/sr_pipeline/plot_pysr_vs_neural_ode.py \
                --baseline-metrics {input.baseline} \
                --pysr-metrics {input.pysr} \
                --output-panel-ode {output.panel_ode} \
                --output-panel-dt {output.panel_dt} \
                --output-baseline {output.baseline} \
                --output-pysr-k-vs-r2 {output.pysr_k_vs_r2} \
                --output-quadrant-bar {output.quadrant_bar} \
                --measured-timepoints {params.measured}
            """

    rule experimental_neural_ode_diffrax_lambda_sweep:
        # Calibration sweep for the Jacobian L1 regulariser (`--jac-reg`) on a
        # few representative markers. Produces a CSV + per-marker plots; the
        # full causal analysis can then be re-run with the chosen lambda.
        input:
            per_minute=marker_per_minute_csv
        output:
            sweep_csv=neural_ode_diffrax_lambda_sweep_csv
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=neural_ode_diffrax_lambda_sweep_dir,
            markers="ABL1 AKT3 ALPK2 MAPK1",
            lambdas="0 0.001 0.01 0.1 1.0",
            seeds="42",
        shell:
            """
            python src/pipelines/experimental/sr_pipeline/neural_ode_diffrax_lambda_sweep.py \
                --output-base {params.out_dir} \
                --dataset {input.per_minute} \
                --markers {params.markers} \
                --lambdas {params.lambdas} \
                --seeds {params.seeds}
            """

    rule experimental_neural_ode_diffrax_causal:
        # Reads the per-(seed, marker) model checkpoints staged under
        # `_seeds/seed_{42,43,44}/models/` by the baseline rule when run with
        # --save-models. If checkpoints are absent the script errors with a
        # helpful message; rerun the baseline rule to repopulate them.
        input:
            metrics_agg=neural_ode_diffrax_metrics_agg_full
        output:
            summary=neural_ode_diffrax_causal_summary,
            pr_csv=neural_ode_diffrax_causal_pr_csv,
            pr_plot=neural_ode_diffrax_causal_pr_plot
        conda:
            "../../envs/pysr.yaml"
        params:
            seeds_dir=f"{exp_runs_root}/neural_ode_diffrax/_seeds",
            out_dir=neural_ode_diffrax_causal_dir,
            seeds="42 43 44",
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/sr_pipeline/neural_ode_diffrax_causal.py \
                --source-dir {params.seeds_dir} \
                --output-dir {params.out_dir} \
                --seeds {params.seeds} \
                --consensus-only
            """

    rule experimental_plot_pysr_vs_neural_ode_diffrax:
        input:
            baseline=neural_ode_diffrax_metrics_agg_full,
            pysr=integration_per_minute_metrics
        output:
            panel_ode=neural_ode_diffrax_panel_ode_full,
            panel_dt=neural_ode_diffrax_panel_dt_full,
            baseline=neural_ode_diffrax_baseline_full,
            pysr_k_vs_r2=neural_ode_diffrax_pysr_k_vs_r2_full,
            quadrant_bar=neural_ode_diffrax_quadrant_bar_full
        conda:
            "../../envs/pysr.yaml"
        params:
            measured=" ".join(str(t) for t in exp_measured_timepoints)
        shell:
            """
            python src/pipelines/experimental/sr_pipeline/plot_pysr_vs_neural_ode.py \
                --baseline-metrics {input.baseline} \
                --pysr-metrics {input.pysr} \
                --output-panel-ode {output.panel_ode} \
                --output-panel-dt {output.panel_dt} \
                --output-baseline {output.baseline} \
                --output-pysr-k-vs-r2 {output.pysr_k_vs_r2} \
                --output-quadrant-bar {output.quadrant_bar} \
                --measured-timepoints {params.measured}
            """

    rule experimental_plot_pysr_vs_selectk:
        input:
            selectk=select_k_metrics_agg,
            pysr=integration_per_minute_metrics
        output:
            panel_a_box=select_k_panel_a_box,
            panel_a=select_k_panel_a,
            panel_b=select_k_panel_b,
            panel_a_dt=select_k_panel_a_dt,
            panel_b_dt=select_k_panel_b_dt,
            panel_a_box_dt=select_k_panel_a_box_dt,
            linreg_bar=select_k_panel_a_linreg_bar,
            panel_a_relmae=select_k_panel_a_relmae,
            panel_b_relmae=select_k_panel_b_relmae,
            panel_a_box_relmae=select_k_panel_a_box_relmae,
            panel_a_relmae_dt=select_k_panel_a_relmae_dt,
            panel_b_relmae_dt=select_k_panel_b_relmae_dt,
            panel_a_box_relmae_dt=select_k_panel_a_box_relmae_dt,
            linreg_bar_relmae=select_k_panel_a_linreg_bar_relmae,
            variability_dt=select_k_variability_dt,
            variability_perk=select_k_variability_perk,
            pysr_k_vs_r2=select_k_pysr_k_vs_r2,
            trajectory_all_k=select_k_trajectory_all_k,
            baseline=select_k_baseline_pysr_vs_linreg,
            heatmap=select_k_heatmap
        conda:
            "../../envs/pysr.yaml"
        params:
            measured=" ".join(str(t) for t in exp_measured_timepoints)
        shell:
            """
            python src/pipelines/experimental/sr_pipeline/plot_pysr_vs_selectk.py \
                --selectk-metrics {input.selectk} \
                --pysr-metrics {input.pysr} \
                --output-panel-a {output.panel_a} \
                --output-panel-a-box {output.panel_a_box} \
                --output-panel-b {output.panel_b} \
                --output-panel-a-dt {output.panel_a_dt} \
                --output-panel-a-dt-box {output.panel_a_box_dt} \
                --output-panel-b-dt {output.panel_b_dt} \
                --output-linreg-compare-bar {output.linreg_bar} \
                --output-panel-a-relmae {output.panel_a_relmae} \
                --output-panel-a-relmae-box {output.panel_a_box_relmae} \
                --output-panel-b-relmae {output.panel_b_relmae} \
                --output-panel-a-relmae-dt {output.panel_a_relmae_dt} \
                --output-panel-a-relmae-dt-box {output.panel_a_box_relmae_dt} \
                --output-panel-b-relmae-dt {output.panel_b_relmae_dt} \
                --output-linreg-compare-bar-relmae {output.linreg_bar_relmae} \
                --variability-dt-output {output.variability_dt} \
                --variability-perk-output {output.variability_perk} \
                --output-pysr-k-vs-r2 {output.pysr_k_vs_r2} \
                --output-trajectory-all-k {output.trajectory_all_k} \
                --output-baseline-pysr-vs-linreg {output.baseline} \
                --trajectories {integration_per_minute_traj} \
                --heatmap-output {output.heatmap} \
                --measured-timepoints {params.measured}
            """

    rule experimental_plot_pysr_vs_neural_ode:
        input:
            baseline=neural_ode_metrics_agg_full,
            pysr=integration_per_minute_metrics
        output:
            panel_ode=neural_ode_panel_ode_full,
            panel_dt=neural_ode_panel_dt_full,
            panel_ode_relmae=neural_ode_panel_ode_relmae_full,
            panel_dt_relmae=neural_ode_panel_dt_relmae_full,
            baseline=neural_ode_baseline_full,
            pysr_k_vs_r2=neural_ode_pysr_k_vs_r2_full,
            quadrant_bar=neural_ode_quadrant_bar_full
        conda:
            "../../envs/pysr.yaml"
        params:
            measured=" ".join(str(t) for t in exp_measured_timepoints)
        shell:
            """
            python src/pipelines/experimental/sr_pipeline/plot_pysr_vs_neural_ode.py \
                --baseline-metrics {input.baseline} \
                --pysr-metrics {input.pysr} \
                --output-panel-ode {output.panel_ode} \
                --output-panel-dt {output.panel_dt} \
                --output-panel-ode-relmae {output.panel_ode_relmae} \
                --output-panel-dt-relmae {output.panel_dt_relmae} \
                --output-baseline {output.baseline} \
                --output-pysr-k-vs-r2 {output.pysr_k_vs_r2} \
                --output-quadrant-bar {output.quadrant_bar} \
                --measured-timepoints {params.measured}
            """

    rule experimental_plot_pysr_vs_neural_ode_matched:
        input:
            baseline=neural_ode_metrics_agg_matched,
            pysr=integration_per_minute_metrics
        output:
            panel_ode=neural_ode_panel_ode_matched,
            panel_dt=neural_ode_panel_dt_matched,
            panel_ode_relmae=neural_ode_panel_ode_relmae_matched,
            panel_dt_relmae=neural_ode_panel_dt_relmae_matched,
            baseline=neural_ode_baseline_matched,
            pysr_k_vs_r2=neural_ode_pysr_k_vs_r2_matched,
            quadrant_bar=neural_ode_quadrant_bar_matched
        conda:
            "../../envs/pysr.yaml"
        params:
            measured=" ".join(str(t) for t in exp_measured_timepoints)
        shell:
            """
            python src/pipelines/experimental/sr_pipeline/plot_pysr_vs_neural_ode.py \
                --baseline-metrics {input.baseline} \
                --pysr-metrics {input.pysr} \
                --output-panel-ode {output.panel_ode} \
                --output-panel-dt {output.panel_dt} \
                --output-panel-ode-relmae {output.panel_ode_relmae} \
                --output-panel-dt-relmae {output.panel_dt_relmae} \
                --output-baseline {output.baseline} \
                --output-pysr-k-vs-r2 {output.pysr_k_vs_r2} \
                --output-quadrant-bar {output.quadrant_bar} \
                --measured-timepoints {params.measured}
            """

    rule experimental_marker_integration:
        input:
            summary_mean=marker_summary_output,
            summary_seed=marker_summary_seed_output,
            snapshot=marker_fit_snapshot_csv,
            per_minute=marker_per_minute_csv
        output:
            integration_snapshot=integration_snapshot_metrics,
            integration_per_minute=integration_per_minute_metrics,
            integration_snapshot_traj=integration_snapshot_traj,
            integration_per_minute_traj=integration_per_minute_traj,
            predicted_snapshot=predicted_snapshot_traj,
            predicted_per_minute=predicted_per_minute_traj,
        conda:
            "../../envs/pysr.yaml"
        params:
            metrics_dir=exp_metrics_root,
            seeds_dir=exp_runs_seeds,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            trajectories_dir=exp_trajectories_root,
        shell:
            """
            mkdir -p {params.metrics_dir} {params.trajectories_dir}
            seed_summaries=($(ls {params.seeds_dir}/seed_*/summary/marker_summary.csv 2>/dev/null || true))

            rm -f {output.integration_snapshot} {output.integration_per_minute} \
                  {output.integration_snapshot_traj} {output.integration_per_minute_traj} \
                  {output.predicted_snapshot} {output.predicted_per_minute} \
                  {integration_snapshot_metrics_all_seeds} {integration_per_minute_metrics_all_seeds} \
                  {integration_snapshot_metrics_all_seeds_mean} {integration_per_minute_metrics_all_seeds_mean}

            if [ "${{#seed_summaries[@]}}" -gt 0 ]; then
                for summary_path in "${{seed_summaries[@]}}"; do
                    seed_id="$(basename "$(dirname "$(dirname "$summary_path")")" | sed 's/seed_//')"
                    seed_metrics_dir="{params.seeds_dir}/seed_${{seed_id}}/metrics"
                    mkdir -p "$seed_metrics_dir"
                    seed_snapshot="$seed_metrics_dir/marker_integration_metrics_snapshot.csv"
                    seed_per_minute="$seed_metrics_dir/marker_integration_metrics_per_minute.csv"
                    seed_snapshot_traj="$seed_metrics_dir/marker_integration_trajectories_snapshot.csv"
                    seed_per_minute_traj="$seed_metrics_dir/marker_integration_trajectories_per_minute.csv"
                    python src/pipelines/experimental/sr_pipeline/compute_marker_integration.py \
                        --dataset {input.snapshot} \
                        --summary "$summary_path" \
                        --sr-trajectories "{params.seeds_dir}/seed_${{seed_id}}/metrics/predicted_trajectories_snapshot.csv" \
                        --dataset-mode snapshot \
                        --output "$seed_snapshot" \
                        --trajectories-output "$seed_snapshot_traj" \
                        --aggregate-output {integration_snapshot_metrics_all_seeds} \
                        --measured-timepoints {params.measured} \
                        --seed "$seed_id"

                    python src/pipelines/experimental/sr_pipeline/compute_marker_integration.py \
                        --dataset {input.per_minute} \
                        --summary "$summary_path" \
                        --sr-trajectories "{params.seeds_dir}/seed_${{seed_id}}/metrics/predicted_trajectories_per_minute.csv" \
                        --dataset-mode per_minute \
                        --output "$seed_per_minute" \
                        --trajectories-output "$seed_per_minute_traj" \
                        --aggregate-output {integration_per_minute_metrics_all_seeds} \
                        --measured-timepoints {params.measured} \
                        --seed "$seed_id"
                done

                # Aggregate integration trajectories across seeds
                for mode in snapshot per_minute; do
                    out_file="{params.trajectories_dir}/marker_integration_trajectories_${{mode}}.csv"
                    rm -f "$out_file"
                    for summary_path in "${{seed_summaries[@]}}"; do
                        seed_id="$(basename "$(dirname "$(dirname "$summary_path")")" | sed 's/seed_//')"
                        src_file="{params.seeds_dir}/seed_${{seed_id}}/metrics/marker_integration_trajectories_${{mode}}.csv"
                        if [ -f "$src_file" ]; then
                            if [ ! -f "$out_file" ]; then
                                cp "$src_file" "$out_file"
                            else
                                tail -n +2 "$src_file" >> "$out_file"
                            fi
                        fi
                    done
                done

            else
                # Single-seed fallback: write seed metrics/trajectories into a seed_single directory
                seed_id="single"
                seed_metrics_dir="{params.seeds_dir}/seed_${{seed_id}}/metrics"
                mkdir -p "$seed_metrics_dir"
                python src/pipelines/experimental/sr_pipeline/compute_marker_integration.py \
                    --dataset {input.snapshot} \
                    --summary {input.summary_seed} \
                    --sr-trajectories "$seed_metrics_dir/predicted_trajectories_snapshot.csv" \
                    --dataset-mode snapshot \
                    --output "$seed_metrics_dir/marker_integration_metrics_snapshot.csv" \
                    --trajectories-output "$seed_metrics_dir/marker_integration_trajectories_snapshot.csv" \
                    --aggregate-output {integration_snapshot_metrics_all_seeds} \
                    --measured-timepoints {params.measured}
                python src/pipelines/experimental/sr_pipeline/compute_marker_integration.py \
                    --dataset {input.per_minute} \
                    --summary {input.summary_seed} \
                    --sr-trajectories "$seed_metrics_dir/predicted_trajectories_per_minute.csv" \
                    --dataset-mode per_minute \
                    --output "$seed_metrics_dir/marker_integration_metrics_per_minute.csv" \
                    --trajectories-output "$seed_metrics_dir/marker_integration_trajectories_per_minute.csv" \
                    --aggregate-output {integration_per_minute_metrics_all_seeds} \
                    --measured-timepoints {params.measured}

                cp "$seed_metrics_dir/marker_integration_trajectories_snapshot.csv" {output.integration_snapshot_traj}
                cp "$seed_metrics_dir/marker_integration_trajectories_per_minute.csv" {output.integration_per_minute_traj}
                if [ -f "$seed_metrics_dir/predicted_trajectories_snapshot.csv" ]; then
                    cp "$seed_metrics_dir/predicted_trajectories_snapshot.csv" {output.predicted_snapshot}
                fi
                if [ -f "$seed_metrics_dir/predicted_trajectories_per_minute.csv" ]; then
                    cp "$seed_metrics_dir/predicted_trajectories_per_minute.csv" {output.predicted_per_minute}
                fi
            fi

            # Ensure trajectory outputs are present for Snakemake
            touch {output.integration_snapshot_traj} {output.integration_per_minute_traj}

            # Copy mean/all-seed aggregates to canonical output locations
            if [ -f {integration_snapshot_metrics_all_seeds_mean} ]; then
                cp {integration_snapshot_metrics_all_seeds_mean} {output.integration_snapshot}
            elif [ -f {integration_snapshot_metrics_all_seeds} ]; then
                cp {integration_snapshot_metrics_all_seeds} {output.integration_snapshot}
            fi
            if [ -f {integration_per_minute_metrics_all_seeds_mean} ]; then
                cp {integration_per_minute_metrics_all_seeds_mean} {output.integration_per_minute}
            elif [ -f {integration_per_minute_metrics_all_seeds} ]; then
                cp {integration_per_minute_metrics_all_seeds} {output.integration_per_minute}
            fi

            # Aggregate predicted trajectories across seeds for phase mapping/plots
            rm -f {output.predicted_snapshot} {output.predicted_per_minute}
            if [ "${{#seed_summaries[@]}}" -gt 0 ]; then
                for summary_path in "${{seed_summaries[@]}}"; do
                    seed_dir="$(dirname "$summary_path")/../metrics"
                    for mode in snapshot per_minute; do
                        src_file="$seed_dir/predicted_trajectories_${{mode}}.csv"
                        dest_file="{params.trajectories_dir}/predicted_trajectories_${{mode}}.csv"
                        if [ -f "$src_file" ]; then
                            if [ ! -f "$dest_file" ]; then
                                cp "$src_file" "$dest_file"
                            else
                                tail -n +2 "$src_file" >> "$dest_file"
                            fi
                        fi
                    done
                done
            fi

            # Ensure predicted trajectory outputs exist even when snapshot predictions are absent.
            if [ ! -f {output.predicted_snapshot} ]; then
                touch {output.predicted_snapshot}
            fi
            if [ ! -f {output.predicted_per_minute} ]; then
                touch {output.predicted_per_minute}
            fi
            """

    rule experimental_plots:
        input:
            summary_mean=marker_summary_output,
            summary_seed=marker_summary_seed_output,
            snapshot=marker_fit_snapshot_csv,
            per_minute=marker_per_minute_csv,
            integration_snapshot=integration_snapshot_metrics,
            integration_per_minute=integration_per_minute_metrics,
            integration_snapshot_traj=integration_snapshot_traj,
            integration_per_minute_traj=integration_per_minute_traj
        output:
            metrics_models_r2=metrics_models_r2,
            metrics_models_r2_svg=metrics_models_r2_svg,
            metrics_models_relmae=metrics_models_relmae,
            metrics_models_relmae_svg=metrics_models_relmae_svg,
            metrics_pysr_modes_r2=metrics_pysr_modes_r2,
            metrics_pysr_modes_r2_svg=metrics_pysr_modes_r2_svg,
            metrics_pysr_modes_relmae=metrics_pysr_modes_relmae,
            metrics_pysr_modes_relmae_svg=metrics_pysr_modes_relmae_svg,
            metrics_linreg_modes_r2=metrics_linreg_modes_r2,
            metrics_linreg_modes_r2_svg=metrics_linreg_modes_r2_svg,
            metrics_linreg_modes_relmae=metrics_linreg_modes_relmae,
            metrics_linreg_modes_relmae_svg=metrics_linreg_modes_relmae_svg,
            metrics_r2_lineplot=metrics_r2_lineplot,
            metrics_r2_lineplot_svg=metrics_r2_lineplot_svg,
            metrics_integrated_r2_lineplot=metrics_integrated_r2_lineplot,
            metrics_integrated_r2_lineplot_svg=metrics_integrated_r2_lineplot_svg,
            metrics_integrated_ode_r2_lineplot=metrics_integrated_ode_r2_lineplot,
            metrics_integrated_ode_r2_lineplot_svg=metrics_integrated_ode_r2_lineplot_svg,
            metrics_r2_boxplot=metrics_r2_boxplot,
            metrics_r2_boxplot_svg=metrics_r2_boxplot_svg,
            metrics_integrated_r2_boxplot=metrics_integrated_r2_boxplot,
            metrics_integrated_r2_boxplot_svg=metrics_integrated_r2_boxplot_svg,
            metrics_integrated_ode_r2_boxplot=metrics_integrated_ode_r2_boxplot,
            metrics_integrated_ode_r2_boxplot_svg=metrics_integrated_ode_r2_boxplot_svg,
            feature_usage_png=feature_usage_plots_png,
            feature_usage_svg=feature_usage_plots_svg,
            feature_importance_png=feature_importance_plots_png,
            feature_importance_svg=feature_importance_plots_svg,
            overlay_snapshot=overlay_snapshot_plot,
            overlay_snapshot_svg=overlay_snapshot_plot_svg,
            overlay_snapshot_metrics=overlay_snapshot_metrics,
            overlay_per_minute=overlay_per_minute_plot,
            overlay_per_minute_svg=overlay_per_minute_plot_svg,
            overlay_per_minute_metrics=overlay_per_minute_metrics
        conda:
            "../../envs/pysr.yaml"
        params:
            overlays_dir=exp_runs_seeds,
            metrics_plots_dir=plots_metrics_dir,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            metrics_dir=exp_metrics_root,
            trajectories_dir=exp_trajectories_root,
            seeds_dir=exp_runs_seeds,
        shell:
            """
            mkdir -p {params.overlays_dir} {params.metrics_plots_dir} {params.metrics_dir} {params.trajectories_dir}
            seed_summaries=($(ls {params.seeds_dir}/seed_*/summary/marker_summary.csv 2>/dev/null || true))

            if [ "${{#seed_summaries[@]}}" -gt 0 ]; then
                # Per-seed overlays only
                for summary_path in "${{seed_summaries[@]}}"; do
                    seed_id="$(basename "$(dirname "$(dirname "$summary_path")")" | sed 's/seed_//')"
                    seed_metrics_dir="{params.seeds_dir}/seed_${{seed_id}}/metrics"
                    seed_overlay_dir="{params.overlays_dir}/seed_${{seed_id}}"
                    mkdir -p "$seed_overlay_dir" "$seed_metrics_dir"

                    python src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
                        --dataset {input.snapshot} \
                        --summary "$summary_path" \
                        --summary-seed "$summary_path" \
                        --output-dir "$seed_overlay_dir" \
                        --dataset-mode snapshot \
                        --filter-dataset-mode snapshot \
                        --measured-timepoints {params.measured} \
                        --fig-base marker_overlay_snapshot \
                        --metrics-csv marker_overlay_metrics_snapshot.csv \
                        --metrics-output-dir "$seed_metrics_dir" \
                        --integration-trajectories "$seed_metrics_dir/marker_integration_trajectories_snapshot.csv" \
                        --integration-metrics "$seed_metrics_dir/marker_integration_metrics_snapshot.csv"
                    python src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
                        --dataset {input.snapshot} \
                        --summary "$summary_path" \
                        --summary-seed "$summary_path" \
                        --output-dir "$seed_overlay_dir" \
                        --dataset-mode snapshot \
                        --filter-dataset-mode snapshot \
                        --measured-timepoints {params.measured} \
                        --fig-base marker_overlay_linear_regression_snapshot \
                        --metrics-csv marker_overlay_metrics_linear_regression_snapshot.csv \
                        --metrics-output-dir "$seed_metrics_dir" \
                        --integration-trajectories "$seed_metrics_dir/marker_integration_trajectories_snapshot.csv" \
                        --integration-metrics "$seed_metrics_dir/marker_integration_metrics_snapshot.csv" \
                        --model "Linear Regression"

                    python src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
                        --dataset {input.per_minute} \
                        --summary "$summary_path" \
                        --summary-seed "$summary_path" \
                        --output-dir "$seed_overlay_dir" \
                        --dataset-mode per_minute \
                        --filter-dataset-mode per_minute \
                        --measured-timepoints {params.measured} \
                        --fig-base marker_overlay_per_minute \
                        --metrics-csv marker_overlay_metrics_per_minute.csv \
                        --metrics-output-dir "$seed_metrics_dir" \
                        --integration-trajectories "$seed_metrics_dir/marker_integration_trajectories_per_minute.csv" \
                        --integration-metrics "$seed_metrics_dir/marker_integration_metrics_per_minute.csv"
                    python src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
                        --dataset {input.per_minute} \
                        --summary "$summary_path" \
                        --summary-seed "$summary_path" \
                        --output-dir "$seed_overlay_dir" \
                        --dataset-mode per_minute \
                        --filter-dataset-mode per_minute \
                        --measured-timepoints {params.measured} \
                        --fig-base marker_overlay_linear_regression_per_minute \
                        --metrics-csv marker_overlay_metrics_linear_regression_per_minute.csv \
                        --metrics-output-dir "$seed_metrics_dir" \
                        --integration-trajectories "$seed_metrics_dir/marker_integration_trajectories_per_minute.csv" \
                        --integration-metrics "$seed_metrics_dir/marker_integration_metrics_per_minute.csv" \
                        --model "Linear Regression"
                done
            else
                # Single-seed case: render overlays once
                python src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
                    --dataset {input.snapshot} \
                    --summary {input.summary_seed} \
                    --summary-seed {input.summary_seed} \
                    --output-dir {params.overlays_dir} \
                    --dataset-mode snapshot \
                    --filter-dataset-mode snapshot \
                    --measured-timepoints {params.measured} \
                    --fig-base marker_overlay_snapshot \
                    --metrics-csv marker_overlay_metrics_snapshot.csv \
                    --metrics-output-dir {params.metrics_dir} \
                    --integration-trajectories {input.integration_snapshot_traj} \
                    --integration-metrics {input.integration_snapshot}
                python src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
                    --dataset {input.snapshot} \
                    --summary {input.summary_seed} \
                    --summary-seed {input.summary_seed} \
                    --output-dir {params.overlays_dir} \
                    --dataset-mode snapshot \
                    --filter-dataset-mode snapshot \
                    --measured-timepoints {params.measured} \
                    --fig-base marker_overlay_linear_regression_snapshot \
                    --metrics-csv marker_overlay_metrics_linear_regression_snapshot.csv \
                    --metrics-output-dir {params.metrics_dir} \
                    --integration-trajectories {input.integration_snapshot_traj} \
                    --integration-metrics {input.integration_snapshot} \
                    --model "Linear Regression"
                python src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
                    --dataset {input.per_minute} \
                    --summary {input.summary_seed} \
                    --summary-seed {input.summary_seed} \
                    --output-dir {params.overlays_dir} \
                    --dataset-mode per_minute \
                    --filter-dataset-mode per_minute \
                    --measured-timepoints {params.measured} \
                    --fig-base marker_overlay_per_minute \
                    --metrics-csv marker_overlay_metrics_per_minute.csv \
                    --metrics-output-dir {params.metrics_dir} \
                    --integration-trajectories {input.integration_per_minute_traj} \
                    --integration-metrics {input.integration_per_minute}
                python src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
                    --dataset {input.per_minute} \
                    --summary {input.summary_seed} \
                    --summary-seed {input.summary_seed} \
                    --output-dir {params.overlays_dir} \
                    --dataset-mode per_minute \
                    --filter-dataset-mode per_minute \
                    --measured-timepoints {params.measured} \
                    --fig-base marker_overlay_linear_regression_per_minute \
                    --metrics-csv marker_overlay_metrics_linear_regression_per_minute.csv \
                    --metrics-output-dir {params.metrics_dir} \
                    --integration-trajectories {input.integration_per_minute_traj} \
                    --integration-metrics {input.integration_per_minute} \
                    --model "Linear Regression"
            fi

            # Metrics scatter/KDE (all rows, both models)
            python src/pipelines/experimental/sr_pipeline/plot_metrics_summary.py \
                --summary {input.summary_mean} \
                --output-dir {params.metrics_plots_dir} \
                --integration-metrics-dir {params.metrics_dir} \
                --trajectories-dir {params.trajectories_dir} \
                --snapshot-dataset {input.snapshot} \
                --per-minute-dataset {input.per_minute}

            # Copy one set of overlay plots/metrics into the aggregated outputs Snakemake expects.
            mkdir -p "$(dirname {output.overlay_snapshot})" "$(dirname {output.overlay_snapshot_svg})" \
                     "$(dirname {output.overlay_per_minute})" "$(dirname {output.overlay_per_minute_svg})" \
                     "$(dirname {output.overlay_snapshot_metrics})" "$(dirname {output.overlay_per_minute_metrics})"
            if [ "${{#seed_summaries[@]}}" -gt 0 ]; then
                first_seed="$(basename "$(dirname "$(dirname "${{seed_summaries[0]}}")")" | sed 's/seed_//')"
                src_overlay_dir="{params.overlays_dir}/seed_${{first_seed}}"
                src_metrics_dir="{params.seeds_dir}/seed_${{first_seed}}/metrics"
            else
                src_overlay_dir="{params.overlays_dir}"
                src_metrics_dir="{params.metrics_dir}"
            fi
            for mode in snapshot per_minute; do
                png_out="{output.overlay_snapshot}" ; svg_out="{output.overlay_snapshot_svg}" ; csv_out="{output.overlay_snapshot_metrics}"
                if [ "$mode" = "per_minute" ]; then
                    png_out="{output.overlay_per_minute}" ; svg_out="{output.overlay_per_minute_svg}" ; csv_out="{output.overlay_per_minute_metrics}"
                fi
                if [ -f "$src_overlay_dir/marker_overlay_${{mode}}.png" ]; then
                    cp "$src_overlay_dir/marker_overlay_${{mode}}.png" "$png_out"
                else
                    touch "$png_out"
                fi
                if [ -f "$src_overlay_dir/marker_overlay_${{mode}}.svg" ]; then
                    cp "$src_overlay_dir/marker_overlay_${{mode}}.svg" "$svg_out"
                else
                    touch "$svg_out"
                fi
                if [ -f "$src_metrics_dir/marker_overlay_metrics_${{mode}}.csv" ]; then
                    cp "$src_metrics_dir/marker_overlay_metrics_${{mode}}.csv" "$csv_out"
                else
                    touch "$csv_out"
                fi
            done
            """

    rule experimental_pysr_r2_threshold_sanity:
        output:
            plot=metrics_pysr_r2_threshold_sanity,
            plot_svg=metrics_pysr_r2_threshold_sanity_svg,
            examples_csv=metrics_pysr_r2_threshold_sanity_csv
        conda:
            "../../envs/pysr.yaml"
        params:
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            trajectories=integration_per_minute_traj,
            raw_dataset=marker_time_series_csv,
            override_065="0.65,DUSP10 (P2),22,44",
            override_070="0.70,DYRK3,5,43"
        shell:
            """
            python src/pipelines/experimental/sr_pipeline/plot_pysr_r2_threshold_sanity.py \
                --trajectories {params.trajectories} \
                --raw-dataset {params.raw_dataset} \
                --output-plot {output.plot} \
                --output-csv {output.examples_csv} \
                --override-example {params.override_065} \
                --override-example {params.override_070} \
                --examples-per-target 3 \
                --measured-timepoints {params.measured}
            """

if run_pysr:
        rule experimental_marker_pysr_grid:
            input:
                summary=marker_summary_output,
                dataset=marker_fit_snapshot_csv,
                groups=exp_group_definitions_csv
            output:
                results=pysr_grid_results_csv
            conda:
                "../../envs/pysr.yaml"
            params:
                output_dir=pysr_grid_output_dir,
                seed=pysr_grid_seed,
                base_niterations=pysr_grid_base_niterations,
                base_population_size=pysr_grid_base_population_size,
                base_populations=pysr_grid_base_populations,
                lower_scale=pysr_grid_lower_scale,
                upper_scale=pysr_grid_upper_scale,
                min_bin=pysr_grid_min_bin_samples,
                max_bin=pysr_grid_max_bin_samples,
                log10_cutoff=exp_log10_cutoff
            shell:
                """
                mkdir -p {params.output_dir}
                python src/pipelines/experimental/experiments/pysr_hyperparam_grid.py \
                    --summary {input.summary} \
                    --dataset {input.dataset} \
                    --group-definitions-csv {input.groups} \
                    --output-dir {params.output_dir} \
                    --seed {params.seed} \
                    --base-niterations {params.base_niterations} \
                    --base-population-size {params.base_population_size} \
                    --base-populations {params.base_populations} \
                    --lower-scale {params.lower_scale} \
                    --upper-scale {params.upper_scale} \
                    --min-bin-samples {params.min_bin} \
                    --max-bin-samples {params.max_bin} \
                    --log10-cutoff {params.log10_cutoff}
            """

else:

    rule all:
        input:
            f"data/{enzyme_model}/{data_type}/sr_comparison/plots/results_plot.png",
            f"data/{enzyme_model}/{data_type}/sr_comparison/plots/integrated_results_plot.png",
            f"data/{enzyme_model}/{data_type}/sr_comparison/results/loss_comparison.csv",
            f"data/{enzyme_model}/{data_type}/sr_comparison/plots/log_mae_timepoint_lineplot.png",
            f"data/{enzyme_model}/{data_type}/sr_comparison/plots/log_mae_timepoint_lineplot_template.png",
            *nn_model_outputs,
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/error_landscape.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/feature_error_correlation_grid.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/feature_error_correlation_overall.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/model_error_correlation_grid.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/model_error_correlation_overall.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/results/pysr/all_pysr_formulas.txt",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae/log_mae_horizontal_boxplot.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae/log_mae_horizontal_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae/log_mae_vertical_boxplot.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae/log_mae_vertical_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae/log_mae_regime_lineplot.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae/log_mae_regime_lineplot_template.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae/log_mae_error_distributions.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/error_landscape.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/feature_error_correlation_grid.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/feature_error_correlation_overall.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/model_error_correlation_grid.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/model_error_correlation_overall.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/results/pysr/all_pysr_formulas.txt",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae/log_mae_horizontal_boxplot.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae/log_mae_horizontal_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae/log_mae_vertical_boxplot.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae/log_mae_vertical_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae/log_mae_deviation_lineplot.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae/log_mae_deviation_lineplot_template.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae/log_mae_error_distributions.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/error_landscape.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/feature_error_correlation_grid.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/feature_error_correlation_overall.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/model_error_correlation_grid.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/model_error_correlation_overall.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/results/pysr/all_pysr_formulas.txt",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae/log_mae_horizontal_boxplot.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae/log_mae_horizontal_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae/log_mae_vertical_boxplot.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae/log_mae_vertical_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae/log_mae_noise_regime_lineplot.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae/log_mae_noise_regime_lineplot_template.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae/log_mae_error_distributions.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/error_landscape.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/results/pysr/all_pysr_formulas.txt",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/feature_error_correlation_grid.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/feature_error_correlation_overall.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/model_error_correlation_grid.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/model_error_correlation_overall.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/log_mae/log_mae_horizontal_boxplot.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/log_mae/log_mae_horizontal_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/log_mae/log_mae_vertical_boxplot.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/log_mae/log_mae_vertical_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/log_mae/log_mae_dataset_size_lineplot.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/log_mae/log_mae_dataset_size_lineplot_template.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/log_mae/log_mae_error_distributions.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/relative_mae/relative_mae_horizontal_boxplot.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/relative_mae/relative_mae_horizontal_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/relative_mae/relative_mae_vertical_boxplot.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/relative_mae/relative_mae_vertical_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/relative_mae/relative_mae_dataset_size_lineplot.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/relative_mae/relative_mae_dataset_size_lineplot_template.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/relative_mae/relative_mae_error_distributions.png",
            *timepoint_outputs,
            "data/panmodel_plots/symbolic_model_r2_scores_bar.png",
            "data/panmodel_plots/symbolic_model_r2_scores_scatter.png",
            "data/panmodel_plots/all_symbolic_formulas.txt"

    rule generate_data_model:
        output:
            train=f"data/{enzyme_model}/{data_type}/raw/train_data.csv",
            test=f"data/{enzyme_model}/{data_type}/raw/test_data.csv",
            valid=f"data/{enzyme_model}/{data_type}/raw/val_data.csv"
        shell:
            """
            source amici_env/bin/activate 
            python src/synthetic/generate_data.py {enzyme_model} {data_type}
            deactivate
            """

    rule preprocessing_model:
        input:
            train=f"data/{enzyme_model}/{data_type}/raw/train_data.csv",
            test=f"data/{enzyme_model}/{data_type}/raw/test_data.csv",
            valid=f"data/{enzyme_model}/{data_type}/raw/val_data.csv"
        output:
            merged=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv",
            train_split=f"data/{enzyme_model}/{data_type}/processed/data_train.csv",
            test_split=f"data/{enzyme_model}/{data_type}/processed/data_test.csv"
        conda:
            "../../envs/base.yaml"
        params:
            target_feature=config["target_feature"]
        shell:
            """
            python src/preprocessing.py --train {input.train} --test {input.test} --valid {input.valid} --output {output.merged} --train-output {output.train_split} --test-output {output.test_split} --target_feature {params.target_feature}
            """

if enzyme_model != "experimental":
    if "pysindy" in sweep_best_configs:
        rule sweep_pysindy:
            input:
                train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv"
            output:
                config_file=sweep_best_configs["pysindy"]
            conda:
                "../../envs/pysindy.yaml"
            params:
                dataset_size=config["dataset_sizes"]["pysindy"],
                features=features,
                search_space=SR_SWEEP_SEARCH_SPACE,
                timeout=config["timeout_duration"],
                max_trials=50
            shell:
                """
                mkdir -p $(dirname {output.config_file})
                python src/sr_sweep/agent.py \
                    --dataset {input.train} \
                    --dataset_size {params.dataset_size} \
                    --features {params.features} \
                    --methods pysindy \
                    --search-space {params.search_space} \
                    --max-runtime-seconds {params.timeout} \
                    --max-workers 2 \
                    --cpus-per-run 1 \
                    --max-trials {params.max_trials} \
                    --output-dir $(dirname {output.config_file})
                """

    if "aifeynman" in sweep_best_configs:
        rule sweep_aifeynman:
            input:
                train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv"
            output:
                config_file=sweep_best_configs["aifeynman"]
            conda:
                "../../envs/aifeynman.yaml"
            params:
                dataset_size=config["dataset_sizes"]["aifeynman"],
                features=features,
                search_space=SR_SWEEP_SEARCH_SPACE,
                timeout=config["timeout_duration"],
                max_trials=50
            shell:
                """
                mkdir -p $(dirname {output.config_file})
                python src/sr_sweep/agent.py \
                    --dataset {input.train} \
                    --dataset_size {params.dataset_size} \
                    --features {params.features} \
                    --methods aifeynman \
                    --search-space {params.search_space} \
                    --max-runtime-seconds {params.timeout} \
                    --max-workers 2 \
                    --cpus-per-run 1 \
                    --max-trials {params.max_trials} \
                    --output-dir $(dirname {output.config_file})
                """

    if "dso" in sweep_best_configs:
        rule sweep_dso:
            input:
                train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv"
            output:
                config_file=sweep_best_configs["dso"]
            conda:
                "../../envs/dso.yaml"
            params:
                dataset_size=config["dataset_sizes"]["dso"],
                features=features,
                search_space=SR_SWEEP_SEARCH_SPACE,
                timeout=config["timeout_duration"],
                max_trials=50
            shell:
                """
                mkdir -p $(dirname {output.config_file})
                python src/sr_sweep/agent.py \
                    --dataset {input.train} \
                    --dataset_size {params.dataset_size} \
                    --features {params.features} \
                    --methods dso \
                    --search-space {params.search_space} \
                    --max-runtime-seconds {params.timeout} \
                    --max-workers 2 \
                    --cpus-per-run 1 \
                    --max-trials {params.max_trials} \
                    --output-dir $(dirname {output.config_file})
                """

    if "kan" in sweep_best_configs:
        rule sweep_kan:
            input:
                train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv"
            output:
                config_file=sweep_best_configs["kan"]
            conda:
                "../../envs/kan.yaml"
            params:
                dataset_size=config["dataset_sizes"]["kan"],
                features=features,
                search_space=SR_SWEEP_SEARCH_SPACE,
                timeout=config["timeout_duration"],
                max_trials=50
            shell:
                """
                mkdir -p $(dirname {output.config_file})
                python src/sr_sweep/agent.py \
                    --dataset {input.train} \
                    --dataset_size {params.dataset_size} \
                    --features {params.features} \
                    --methods kan \
                    --search-space {params.search_space} \
                    --max-runtime-seconds {params.timeout} \
                    --max-workers 2 \
                    --cpus-per-run 1 \
                    --max-trials {params.max_trials} \
                    --output-dir $(dirname {output.config_file})
                """

    if "pysr" in sweep_best_configs:
        rule sweep_pysr:
            input:
                train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv"
            output:
                config_file=sweep_best_configs["pysr"]
            conda:
                "../../envs/pysr.yaml"
            params:
                dataset_size=config["dataset_sizes"]["pysr"],
                features=features,
                search_space=SR_SWEEP_SEARCH_SPACE,
                timeout=config["timeout_duration"],
                max_trials=50
            shell:
                """
                mkdir -p $(dirname {output.config_file})
                python src/sr_sweep/agent.py \
                    --dataset {input.train} \
                    --dataset_size {params.dataset_size} \
                    --features {params.features} \
                    --methods pysr \
                    --search-space {params.search_space} \
                    --max-runtime-seconds {params.timeout} \
                    --max-workers 2 \
                    --cpus-per-run 1 \
                    --max-trials {params.max_trials} \
                    --output-dir $(dirname {output.config_file})
                """

    # Function to generate shell commands for symbolic regression with timeout and optional installs
    def symbolic_regression_rule(model, dataset, dataset_size, features, temp_file, variant=None):
        install_cmds = {
            # AI-Feynman ships f2py Fortran extensions built via the deprecated
            # numpy.distutils, which breaks with setuptools>=60 and needs the
            # conda toolchain (SDKROOT) to link -lSystem on macOS. Pin setuptools,
            # skip build isolation so the env's numpy is visible, and export the
            # SDK path (no-op on Linux where xcrun is absent).
            "aifeynman": (
                "pip install 'setuptools<60' wheel && "
                "export SDKROOT=\"$(xcrun --show-sdk-path 2>/dev/null || echo)\" && "
                "pip install --no-deps --no-build-isolation aifeynman"
            ),
            "dso": "pip install absl-py==0.7.0 && pip install numpy==1.18 && pip install -e src/dso",
            "pysindy": "pip install -e src/pysindy && pip install cvxpy && pip install tensorflow"
        }
        install_command = install_cmds.get(model, "")
        separator = ";" if install_command else ""
        variant_flag = f" --variant {variant}" if variant else ""
        seed_flag = " --seed 42" if model == "pysr" else ""
        # Force the activated conda env's bin to the front of PATH. snakemake
        # --use-conda logs "Activating conda environment ..." and sets
        # CONDA_PREFIX, but on some nodes the loaded Anaconda *module* python
        # still shadows the env's python/pip (pip then does a user-site install
        # into the wrong interpreter and the build/import fails). Prepending
        # $CONDA_PREFIX/bin is a no-op when activation already worked and a fix
        # when it didn't.
        env_path_fix = '[ -n "$CONDA_PREFIX" ] && export PATH="$CONDA_PREFIX/bin:$PATH";'
        return f"""
            {env_path_fix} {install_command}{separator} timeout {config["timeout_duration"]} python src/sr_models/{model}_model.py --dataset {dataset} --dataset_size {dataset_size} --features {features} --temp_file {temp_file}{variant_flag}{seed_flag} || test -s {temp_file}
        """

    # Rules for symbolic regression for each model
    rule pysindy:
        input:
            train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv",
            sweep=lambda wildcards: sweep_input("pysindy")
        output:
            temp_file=temporary(f"data/{enzyme_model}/{data_type}/sr_comparison/temp/temp_formula_pysindy.txt")
        conda:
            "../../envs/pysindy.yaml"
        params:
            dataset_size=config["dataset_sizes"]["pysindy"],
            features=features
        shell:
            symbolic_regression_rule("pysindy", "{input.train}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

    rule aifeynman:
        input:
            train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv",
            sweep=lambda wildcards: sweep_input("aifeynman")
        output:
            temp_file=temporary(f"data/{enzyme_model}/{data_type}/sr_comparison/temp/temp_formula_aifeynman.txt")
        conda:
            "../../envs/aifeynman.yaml"
        params:
            dataset_size=config["dataset_sizes"]["aifeynman"],
            features=features
        shell:
            symbolic_regression_rule("aifeynman", "{input.train}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

    rule dso:
        input:
            train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv",
            sweep=lambda wildcards: sweep_input("dso")
        output:
            temp_file=temporary(f"data/{enzyme_model}/{data_type}/sr_comparison/temp/temp_formula_dso.txt")
        conda:
            "../../envs/dso.yaml"
        params:
            dataset_size=config["dataset_sizes"]["dso"],
            features=features
        shell:
            symbolic_regression_rule("dso", "{input.train}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

    rule kan:
        input:
            train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv",
            sweep=lambda wildcards: sweep_input("kan")
        output:
            temp_file=temporary(f"data/{enzyme_model}/{data_type}/sr_comparison/temp/temp_formula_kan.txt")
        conda:
            "../../envs/kan.yaml"
        params:
            dataset_size=config["dataset_sizes"]["kan"],
            features=features
        shell:
            symbolic_regression_rule("kan", "{input.train}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

    rule pysr:
        input:
            train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv",
            sweep=lambda wildcards: sweep_input("pysr")
        output:
            temp_file=temporary(f"data/{enzyme_model}/{data_type}/sr_comparison/temp/hall_of_fame_{{variant}}.csv")
        conda:
            "../../envs/pysr.yaml"
        params:
            dataset_size=config["dataset_sizes"]["pysr"],
            features=features,
            variant=lambda wildcards: wildcards.variant
        shell:
            symbolic_regression_rule("pysr", "{input.train}", "{params.dataset_size}", "{params.features}", "{output.temp_file}", variant="{params.variant}")

    rule nn:
        input:
            train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv"
        output:
            output=f"data/{enzyme_model}/{data_type}/sr_comparison/models/nn/model_nn_{{variant}}.pth"
        conda:
            "../../envs/nn.yaml"
        params:
            dataset_size=config["dataset_sizes"]["nn"],
            features=features,
            variant=lambda wildcards: wildcards.variant
        shell:
            """
            python src/nn_model.py --dataset {input.train} --dataset_size {params.dataset_size} --features {params.features} --variant {params.variant} --output {output} || true
            """

    # Rule to extract the best formula from each temporary result file
    rule get_best_formula:
        input:
            temp_files=temp_files
        output:
            formula_files=formula_files
        conda:
            "../../envs/base.yaml"
        params:
            methods=" ".join(models),
            features=features
        shell:
            """
            echo "Extracting best formulas for methods: {params.methods}"
            PYTHONPATH=src python src/pipelines/sr_comparison/get_best_formula.py --methods {params.methods} --hall_of_fame {input.temp_files} --save {output.formula_files} --features {params.features}
            echo "Best formulas saved to: {output.formula_files}"
            """

    # Rule to integrate and plot results based on formulas from all models
    rule integrate_and_plot_results:
        input:
            dataset=lambda wildcards: f"data/{enzyme_model}/{data_type}/processed/data_test.csv",
            formulas=formula_files
        output:
            csv=f"data/{enzyme_model}/{data_type}/sr_comparison/results/loss_comparison.csv",
            plot=f"data/{enzyme_model}/{data_type}/sr_comparison/plots/integrated_results_plot.png"
        conda:
            "../../envs/base.yaml"
        params:
            methods=" ".join(models),
            data_proportion=config["data_proportion"],
            features=features,
            discovery_scales="'" + config["discovery_scales"] + "'",
            trajectory_column="condition_id"
        shell:
            """
            echo "Integrating and plotting results for methods: {params.methods}"
            PYTHONPATH=src python src/pipelines/sr_comparison/integrate_and_plot_methods.py --dataset {input.dataset} --formulas {input.formulas} --methods {params.methods}  --discovery-scales {params.discovery_scales} --data-proportion {params.data_proportion} --trajectory-column {params.trajectory_column} --output {output.csv} --plot {output.plot}
            echo "Integrated results saved to {output.csv}, plot saved to {output.plot}"
            """

    # Rule to generate a comparison plot of all methods
    rule plot_methods:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_test.csv",
            formulas=formula_files
        output:
            f"data/{enzyme_model}/{data_type}/sr_comparison/plots/results_plot.png"
        conda:
            "../../envs/base.yaml"
        params:
            discovery_scales="'" + config["discovery_scales"] + "'"
        shell:
            """
            echo "Generating comparison plot for all methods."
            PYTHONPATH=src python src/pipelines/sr_comparison/plot_methods.py --dataset {input.dataset} --formulas {input.formulas} --output {output} --discovery-scales {params.discovery_scales}
            echo "Comparison plot saved to {output}"
            """


    rule sr_timepoint_lineplot:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_test.csv",
            train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv",
            formulas=formula_files,
            nn_models=nn_model_outputs
        output:
            plot=f"data/{enzyme_model}/{data_type}/sr_comparison/plots/log_mae_timepoint_lineplot.png",
            template=f"data/{enzyme_model}/{data_type}/sr_comparison/plots/log_mae_timepoint_lineplot_template.png"
        conda:
            "../../envs/base.yaml"
        params:
            methods=" ".join(models),
            features=features,
            discovery_scales="'" + config["discovery_scales"] + "'",
            nn_model_args=nn_model_arg_str,
            nn_dataset_size_arg=nn_dataset_size_arg,
            seed=42
        shell:
            """
            PYTHONPATH=src python src/pipelines/sr_comparison/plot_sr_timepoint_lineplot.py \
                --dataset {input.dataset} \
                --train-dataset {input.train} \
                --features {params.features} \
                --formulas {input.formulas} \
                --methods {params.methods} \
                --discovery-scales {params.discovery_scales} \
                {params.nn_model_args} \
                {params.nn_dataset_size_arg} \
                --seed {params.seed} \
                --output {output.plot} \
                --template-output {output.template}
            """

    rule kinetic_regimes:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
        output:
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/results/pysr/all_pysr_formulas.txt",
        conda:
            "../../envs/pysr.yaml"
        params:
            dataset_size=config["dataset_sizes"]["pysr"],
            features=features
        shell:
            """
            echo "Running PySR on different kinetic regimes (no plots)."
            python src/pipelines/regimes/kinetic_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --no-plots
            echo "PySR biochemical regime evaluation completed."
            """

    rule kinetic_regime_plots:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
        output:
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/error_landscape.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/feature_error_correlation_grid.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/feature_error_correlation_overall.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/model_error_correlation_grid.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/model_error_correlation_overall.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae/log_mae_horizontal_boxplot.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae/log_mae_horizontal_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae/log_mae_vertical_boxplot.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae/log_mae_vertical_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae/log_mae_regime_lineplot.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae/log_mae_regime_lineplot_template.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae/log_mae_error_distributions.png",
        conda:
            "../../envs/pysr.yaml"
        params:
            dataset_size=config["dataset_sizes"]["pysr"],
            features=features
        shell:
            """
            echo "Running PySR on different kinetic regimes (plots-only target)."
            python src/pipelines/regimes/kinetic_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --plots-only
            echo "PySR biochemical regime plotting completed."
            """

    rule kinetic_regimes_tqssa:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv",
            sqssa=f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/results/pysr/all_pysr_formulas_sQSSA.txt"
        output:
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/results/pysr/all_pysr_formulas_tQSSA.txt",
        conda:
            "../../envs/pysr.yaml"
        params:
            dataset_size=config["dataset_sizes"]["pysr"],
            features=features,
            output_dir=f"data/{enzyme_model}/{data_type}/kinetic_regimes"
        shell:
            """
            echo "Refreshing tQSSA PySR on different kinetic regimes (no plots)."
            python src/pipelines/regimes/kinetic_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --variants tQSSA --output-dir {params.output_dir} --no-plots
            echo "tQSSA PySR biochemical regime refresh completed."
            """

    rule kinetic_regime_plots_tqssa:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv",
            sqssa=f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/results/pysr/all_pysr_formulas_sQSSA.txt",
            trained=f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/results/pysr/all_pysr_formulas_tQSSA.txt"
        output:
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/tqssa_refresh.done",
        conda:
            "../../envs/pysr.yaml"
        params:
            dataset_size=config["dataset_sizes"]["pysr"],
            features=features,
            output_dir=f"data/{enzyme_model}/{data_type}/kinetic_regimes"
        shell:
            """
            echo "Refreshing shared kinetic regime plots after tQSSA rerun."
            python src/pipelines/regimes/kinetic_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --output-dir {params.output_dir} --plots-only
            mkdir -p "$(dirname {output})"
            touch {output}
            echo "Shared kinetic regime plotting refresh completed."
            """

    rule noise_regimes_full:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
        output:
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/results/pysr/all_pysr_formulas.txt",
        conda:
            "../../envs/pysr.yaml"
        params:
            dataset_size=config["dataset_sizes"]["pysr"],
            features=features
        shell:
            """
            echo "Running PySR on full-dataset noise regimes (no plots)."
            python src/pipelines/regimes/noise_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --no-plots
            echo "PySR full noise regime evaluation completed."
            """

    rule noise_regime_plots:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
        output:
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/error_landscape.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/feature_error_correlation_grid.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/feature_error_correlation_overall.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/model_error_correlation_grid.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/model_error_correlation_overall.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae/log_mae_horizontal_boxplot.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae/log_mae_horizontal_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae/log_mae_vertical_boxplot.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae/log_mae_vertical_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae/log_mae_noise_regime_lineplot.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae/log_mae_noise_regime_lineplot_template.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae/log_mae_error_distributions.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae/sqssa_vs_tqssa_baselines_log_mae.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/relative_mae/sqssa_vs_tqssa_baselines_relative_mae.png",
        conda:
            "../../envs/pysr.yaml"
        params:
            dataset_size=config["dataset_sizes"]["pysr"],
            features=features
        shell:
            """
            echo "Running PySR on full-dataset noise regimes (plots-only target)."
            python src/pipelines/regimes/noise_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --plots-only
            echo "PySR full noise regime plotting completed."
            """

    rule noise_regimes_full_tqssa:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv",
            sqssa=f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/results/pysr/all_pysr_formulas_sQSSA.txt"
        output:
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/results/pysr/all_pysr_formulas_tQSSA.txt",
        conda:
            "../../envs/pysr.yaml"
        params:
            dataset_size=config["dataset_sizes"]["pysr"],
            features=features,
            output_dir=f"data/{enzyme_model}/{data_type}/noise_regimes_full"
        shell:
            """
            echo "Refreshing tQSSA PySR on full-dataset noise regimes (no plots)."
            python src/pipelines/regimes/noise_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --variants tQSSA --output-dir {params.output_dir} --no-plots
            echo "tQSSA PySR full noise regime refresh completed."
            """

    rule noise_regime_plots_tqssa:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv",
            sqssa=f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/results/pysr/all_pysr_formulas_sQSSA.txt",
            trained=f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/results/pysr/all_pysr_formulas_tQSSA.txt"
        output:
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/tqssa_refresh.done",
        conda:
            "../../envs/pysr.yaml"
        params:
            dataset_size=config["dataset_sizes"]["pysr"],
            features=features,
            output_dir=f"data/{enzyme_model}/{data_type}/noise_regimes_full"
        shell:
            """
            echo "Refreshing shared full-dataset noise regime plots after tQSSA rerun."
            python src/pipelines/regimes/noise_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --output-dir {params.output_dir} --plots-only
            mkdir -p "$(dirname {output})"
            touch {output}
            echo "Shared full-dataset noise regime plotting refresh completed."
            """

    rule mm_deviation_regimes:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
        output:
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/results/pysr/all_pysr_formulas.txt",
        conda:
            "../../envs/pysr.yaml"
        params:
            dataset_size=config["dataset_sizes"]["pysr"],
            features=features
        shell:
            """
            echo "Running PySR on Michaelis-Menten deviation regimes (no plots)."
            python src/pipelines/regimes/mm_deviation_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --no-plots
            echo "PySR MM deviation regime evaluation completed."
            """

    rule mm_deviation_regime_plots:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
        output:
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/error_landscape.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/feature_error_correlation_grid.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/feature_error_correlation_overall.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/model_error_correlation_grid.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/model_error_correlation_overall.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae/log_mae_horizontal_boxplot.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae/log_mae_horizontal_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae/log_mae_vertical_boxplot.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae/log_mae_vertical_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae/log_mae_deviation_lineplot.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae/log_mae_deviation_lineplot_template.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae/log_mae_error_distributions.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae/sqssa_vs_tqssa_baselines_log_mae.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/relative_mae/sqssa_vs_tqssa_baselines_relative_mae.png",
        conda:
            "../../envs/pysr.yaml"
        params:
            dataset_size=config["dataset_sizes"]["pysr"],
            features=features
        shell:
            """
            echo "Running PySR on Michaelis-Menten deviation regimes (plots-only target)."
            python src/pipelines/regimes/mm_deviation_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --plots-only
            echo "PySR MM deviation regime plotting completed."
            """

    rule mm_deviation_regimes_tqssa:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv",
            sqssa=f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/results/pysr/all_pysr_formulas_sQSSA.txt"
        output:
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/results/pysr/all_pysr_formulas_tQSSA.txt",
        conda:
            "../../envs/pysr.yaml"
        params:
            dataset_size=config["dataset_sizes"]["pysr"],
            features=features,
            output_dir=f"data/{enzyme_model}/{data_type}/mm_deviation_regimes"
        shell:
            """
            echo "Refreshing tQSSA PySR on Michaelis-Menten deviation regimes (no plots)."
            python src/pipelines/regimes/mm_deviation_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --variants tQSSA --output-dir {params.output_dir} --no-plots
            echo "tQSSA PySR MM deviation regime refresh completed."
            """

    rule mm_deviation_regime_plots_tqssa:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv",
            sqssa=f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/results/pysr/all_pysr_formulas_sQSSA.txt",
            trained=f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/results/pysr/all_pysr_formulas_tQSSA.txt"
        output:
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/tqssa_refresh.done",
        conda:
            "../../envs/pysr.yaml"
        params:
            dataset_size=config["dataset_sizes"]["pysr"],
            features=features,
            output_dir=f"data/{enzyme_model}/{data_type}/mm_deviation_regimes"
        shell:
            """
            echo "Refreshing shared Michaelis-Menten deviation regime plots after tQSSA rerun."
            python src/pipelines/regimes/mm_deviation_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --output-dir {params.output_dir} --plots-only
            mkdir -p "$(dirname {output})"
            touch {output}
            echo "Shared Michaelis-Menten deviation regime plotting refresh completed."
            """

    rule dataset_size_regimes:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
        output:
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/results/pysr/all_pysr_formulas.txt"
        conda:
            "../../envs/pysr.yaml"
        params:
            features=features,
            pysr_size=config["dataset_sizes"]["pysr"],
            nn_size=config["dataset_sizes"].get("nn", 20000)
        shell:
            """
            echo "Running dataset-size regime evaluation (no plots)."
            python src/pipelines/regimes/dataset_size_regimes.py --dataset {input.dataset} --features {params.features} --pysr-base-size {params.pysr_size} --nn-base-size {params.nn_size} --no-plots
            echo "Dataset-size regime evaluation completed."
            """

    rule dataset_size_regime_plots:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
        output:
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/error_landscape.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/feature_error_correlation_grid.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/feature_error_correlation_overall.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/model_error_correlation_grid.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/model_error_correlation_overall.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/log_mae/log_mae_horizontal_boxplot.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/log_mae/log_mae_horizontal_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/log_mae/log_mae_vertical_boxplot.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/log_mae/log_mae_vertical_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/log_mae/log_mae_dataset_size_lineplot.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/log_mae/log_mae_dataset_size_lineplot_template.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/log_mae/log_mae_error_distributions.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/log_mae/sqssa_vs_tqssa_baselines_log_mae.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/relative_mae/sqssa_vs_tqssa_baselines_relative_mae.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/relative_mae/relative_mae_horizontal_boxplot.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/relative_mae/relative_mae_horizontal_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/relative_mae/relative_mae_vertical_boxplot.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/relative_mae/relative_mae_vertical_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/relative_mae/relative_mae_dataset_size_lineplot.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/relative_mae/relative_mae_dataset_size_lineplot_template.png",
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/relative_mae/relative_mae_error_distributions.png",
        conda:
            "../../envs/pysr.yaml"
        params:
            features=features,
            pysr_size=config["dataset_sizes"]["pysr"],
            nn_size=config["dataset_sizes"].get("nn", 20000)
        shell:
            """
            echo "Running dataset-size regime plotting."
            python src/pipelines/regimes/dataset_size_regimes.py --dataset {input.dataset} --features {params.features} --pysr-base-size {params.pysr_size} --nn-base-size {params.nn_size} --plots-only
            echo "Dataset-size regime plotting completed."
            """

    rule dataset_size_regimes_tqssa:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv",
            sqssa=f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/results/pysr/all_pysr_formulas_sQSSA.txt"
        output:
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/results/pysr/all_pysr_formulas_tQSSA.txt"
        conda:
            "../../envs/pysr.yaml"
        params:
            features=features,
            pysr_size=config["dataset_sizes"]["pysr"],
            nn_size=config["dataset_sizes"].get("nn", 20000),
            output_dir=f"data/{enzyme_model}/{data_type}/dataset_size_regimes"
        shell:
            """
            echo "Refreshing tQSSA dataset-size regime evaluation (no plots)."
            python src/pipelines/regimes/dataset_size_regimes.py --dataset {input.dataset} --features {params.features} --pysr-base-size {params.pysr_size} --nn-base-size {params.nn_size} --variants tQSSA --output-dir {params.output_dir} --no-plots
            echo "tQSSA dataset-size regime refresh completed."
            """

    rule dataset_size_regime_plots_tqssa:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv",
            sqssa=f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/results/pysr/all_pysr_formulas_sQSSA.txt",
            trained=f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/results/pysr/all_pysr_formulas_tQSSA.txt"
        output:
            f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/tqssa_refresh.done",
        conda:
            "../../envs/pysr.yaml"
        params:
            features=features,
            pysr_size=config["dataset_sizes"]["pysr"],
            nn_size=config["dataset_sizes"].get("nn", 20000),
            output_dir=f"data/{enzyme_model}/{data_type}/dataset_size_regimes"
        shell:
            """
            echo "Refreshing shared dataset-size regime plots after tQSSA rerun."
            python src/pipelines/regimes/dataset_size_regimes.py --dataset {input.dataset} --features {params.features} --pysr-base-size {params.pysr_size} --nn-base-size {params.nn_size} --output-dir {params.output_dir} --plots-only
            mkdir -p "$(dirname {output})"
            touch {output}
            echo "Shared dataset-size regime plotting refresh completed."
            """

    if data_type == "dynamic":

        rule timepoint_regimes:
            input:
                dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
            output:
                f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/results/pysr/all_pysr_formulas.txt",
            conda:
                "../../envs/pysr.yaml"
            params:
                dataset_size=config["dataset_sizes"]["pysr"],
                features=features
            shell:
                """
                echo "Running PySR across timepoint groups (no plots)."
                python src/pipelines/regimes/timepoint_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --no-plots
                echo "Timepoint benchmarking completed."
                """

        rule timepoint_regime_plots:
            input:
                dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
            output:
                f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/plots/log_mae/log_mae_timepoint_lineplot.png",
                f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/plots/log_mae/log_mae_timepoint_lineplot_template.png",
                f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/plots/log_mae/log_mae_horizontal_boxplot.png",
                f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/plots/log_mae/log_mae_horizontal_boxplot_no_outliers.png",
                f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/plots/log_mae/log_mae_vertical_boxplot.png",
                f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/plots/log_mae/log_mae_vertical_boxplot_no_outliers.png",
                f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/plots/log_mae/log_mae_error_distributions.png",
                f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/plots/error_landscape.png",
                f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/plots/feature_error_correlation_grid.png",
                f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/plots/feature_error_correlation_overall.png",
                f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/plots/model_error_correlation_grid.png",
                f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/plots/model_error_correlation_overall.png",
            conda:
                "../../envs/pysr.yaml"
            params:
                dataset_size=config["dataset_sizes"]["pysr"],
                features=features
            shell:
                """
                echo "Running PySR across timepoint groups (plots-only target)."
                python src/pipelines/regimes/timepoint_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --plots-only
                echo "Timepoint regime plotting completed."
                """

        rule timepoint_regimes_tqssa:
            input:
                dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv",
                sqssa=f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/results/pysr/all_pysr_formulas_sQSSA.txt"
            output:
                f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/results/pysr/all_pysr_formulas_tQSSA.txt",
            conda:
                "../../envs/pysr.yaml"
            params:
                dataset_size=config["dataset_sizes"]["pysr"],
                features=features,
                output_dir=f"data/{enzyme_model}/{data_type}/timepoint_regimes"
            shell:
                """
                echo "Refreshing tQSSA PySR across timepoint groups (no plots)."
                python src/pipelines/regimes/timepoint_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --variants tQSSA --output-dir {params.output_dir} --no-plots
                echo "tQSSA timepoint benchmarking refresh completed."
                """

        rule timepoint_regime_plots_tqssa:
            input:
                dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv",
                sqssa=f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/results/pysr/all_pysr_formulas_sQSSA.txt",
                trained=f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/results/pysr/all_pysr_formulas_tQSSA.txt"
            output:
                f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/plots/tqssa_refresh.done",
            conda:
                "../../envs/pysr.yaml"
            params:
                dataset_size=config["dataset_sizes"]["pysr"],
                features=features,
                output_dir=f"data/{enzyme_model}/{data_type}/timepoint_regimes"
            shell:
                """
                echo "Refreshing shared timepoint regime plots after tQSSA rerun."
                python src/pipelines/regimes/timepoint_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --output-dir {params.output_dir} --plots-only
                mkdir -p "$(dirname {output})"
                touch {output}
                echo "Shared timepoint regime plotting refresh completed."
                """
# New rule to perform grid search for the NN model
rule nn_grid_search:
    output:
        report=f"data/{enzyme_model}/{data_type}/sr_comparison/results/nn_grid_search/grid_search_results.txt"
    shell:
        """
        mkdir -p $(dirname {output.report})
        echo "Using previously determined NN hyperparameters." > {output.report}
        """

PAN_PLOT_VARIANTS = ["sqssa", "tqssa"]

rule pan_enzyme_model_plots:
    input:
        root_dir="data"
    output:
        "data/panmodel_plots/{variant}/symbolic_model_r2_scores_bar.png",
        "data/panmodel_plots/{variant}/symbolic_model_r2_scores_scatter.png",
        "data/panmodel_plots/{variant}/symbolic_model_relmae_scores_box.png",
        "data/panmodel_plots/{variant}/all_symbolic_formulas.txt"
    conda:
        "../../envs/base.yaml"
    params:
        discovery_scales="'" + config["discovery_scales"] + "'"
    wildcard_constraints:
        variant="|".join(PAN_PLOT_VARIANTS)
    shell:
        "python src/pan_enzyme_model_plots.py --root-dir {input.root_dir} --discovery-scales {params.discovery_scales} --variant {wildcards.variant}"
