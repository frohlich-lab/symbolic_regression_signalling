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

# Linear-regression reference model, trained and scored on the SAME top-GFP
# out-of-distribution split as PySR. The directory name records the split, because an
# in-distribution version of this baseline reverses the manuscript's conclusion: on
# random GFP bins linreg solves 10/32 contexts at k=10 and PySR holds no advantage
# (p = 0.455), against 3/32 and p = 0.035 under dose extrapolation. Any comparison
# against SR must use this one.
select_k_dir = f"{exp_runs_root}/linreg_ood"
select_k_split_policy = "top_gfp_bins"
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

# Sparse Neural ODE baseline (Draft SR-MM v5). A small MLP right-hand side is
# integrated by an adaptive Dopri5 solver and trained on a trajectory-level loss by
# backpropagation through the solver, with a group-sparsity (L21) penalty on the
# columns of the input Jacobian. The penalty is what makes the participation ratio
# readable as a count of the variables the learned field actually uses, which is what
# puts the network and PySR on comparable footing.
#
# All three variants are trained under the top_gfp_bins split, so their held-out score
# is dose extrapolation rather than dose interpolation. L21 at lambda_jac = 3.0 is the
# main-text "Neural ODE"; L1 and the path-regularised (C-NODE) variant are the SI
# robustness check (Fig. S3A, Table S8) showing held-out accuracy is insensitive to the
# choice of penalty.
sparse_node_dir = f"{exp_runs_root}/sparse_neural_ode"
sparse_node_l21_dir   = f"{sparse_node_dir}/l21_lam3"
sparse_node_l1_dir    = f"{sparse_node_dir}/l1"
sparse_node_cnode_dir = f"{sparse_node_dir}/pathreg"
sparse_node_seeds = [42, 43, 44]


def _sparse_node_metrics(variant_dir):
    """The real per-seed outputs of a variant, rather than a .done sentinel.

    A sentinel would let a completed run look unbuilt (and a half-finished one look
    complete). These runs were staged from the cluster, where no sentinel was ever
    written, so naming the actual metrics files is both more honest and what makes the
    downstream figure rules depend on something real.
    """
    return [
        f"{variant_dir}/seed_{seed}/neural_ode_diffrax_metrics.csv"
        for seed in sparse_node_seeds
    ]


sparse_node_l21_done   = _sparse_node_metrics(sparse_node_l21_dir)
sparse_node_l1_done    = _sparse_node_metrics(sparse_node_l1_dir)
sparse_node_cnode_done = _sparse_node_metrics(sparse_node_cnode_dir)

# ---------------------------------------------------------------------
# Draft SR-MM v5 experimental stages. The rules that build these live in
# workflow/rules/experimental.smk; the paths are here so that `rule all`
# below can reference them.
# ---------------------------------------------------------------------

pysr_sweep_dir      = f"{exp_runs_root}/pysr_config_sweep"
pysr_sweep_screen   = f"{pysr_sweep_dir}/metrics/screen_final.csv"
pysr_sweep_table    = f"{pysr_sweep_dir}/metrics/table_s10_config_sweep.csv"
pysr_sweep_config   = f"{pysr_sweep_dir}/selected_config.tsv"

# The six-context development set. Fixed here rather than derived, because the
# configuration was chosen on exactly these and Table S10 is only meaningful for
# them; a silently widened set would not be the reported sweep.
pysr_sweep_dev_contexts = ["AKT3", "ALPK2", "DYRK2", "ERBB2", "PIP5K3", "PTPN7"]

pysr_final_dir       = f"{exp_runs_root}/pysr_ood_final"
pysr_final_seeds     = [42, 43, 44]
pysr_final_formulas  = [
    f"{pysr_final_dir}/seeds/seed_{s}/formulas/all_per_minute.txt"
    for s in pysr_final_seeds
]
pysr_final_metrics   = [
    f"{pysr_final_dir}/seeds/seed_{s}/metrics/"
    f"marker_integration_metrics_per_minute.csv"
    for s in pysr_final_seeds
]
pysr_final_per_fit   = f"{pysr_final_dir}/metrics/integrated_r2_per_fit.csv"
pysr_final_summary   = f"{pysr_final_dir}/metrics/success_rate_summary.csv"
pysr_final_exemplars = f"{pysr_final_dir}/metrics/exemplar_ranking.csv"

# R2 >= 0.6 is the satisfactory-model criterion: trajectories up to this cutoff
# still reproduce the main qualitative features of the measured dynamics on
# integration (Fig. S2).
exp_r2_threshold = 0.6

# Control contexts carry no overexpression construct, so they have no GFP
# dose-response and "extrapolating to an unseen dose" reduces to predicting the same
# trajectory again. They score a median 0.85 and are excluded from the comparison
# panels; including them would lift the headline rate from 8/32 to 15/40 with no
# dose-response biology behind it. untransfected1 is already excluded by default.
exp_control_contexts = [
    "untransfected2", "untransfected3", "untransfected4",
    "FLAG-GFP1", "FLAG-GFP2", "FLAG-GFP3", "FLAG-GFP4",
]
exp_exclude_flags = " ".join(
    f"--exclude-marker {m}" for m in exp_control_contexts
)

paper_fig_dir = f"{exp_runs_root}/paper_figures"
paper_fig_parsimony_tradeoff = f"{paper_fig_dir}/scatter_parsimony_tradeoff.png"
paper_fig_cutoff_robustness  = f"{paper_fig_dir}/cutoff_robustness.png"
paper_fig_nn_appendix        = f"{paper_fig_dir}/nn_appendix_comparison.png"
paper_fig_sr_vs_linreg       = f"{paper_fig_dir}/sr_vs_linreg_ood_scatter.png"
paper_fig_sr_vs_linreg_k4    = f"{paper_fig_dir}/sr_vs_linreg_ood_scatter_k4.png"

# PySR's solved equations use a median of four variables, so k=4 is the
# complexity-matched baseline (Fig. S4) and k=10 the full-pool one (Fig. 4D). The
# match is on the NUMBER of inputs, not their identity: the baseline picks its k by
# univariate F-test while PySR picks its own subset by search.
linreg_k_full = 10
linreg_k_matched = 4

# Fig. S3B: the lambda_jac elbow sweep. One subdirectory per lambda, all markers.
sparse_node_lambda_sweep_dir = f"{sparse_node_dir}/lambda_sweep"
sparse_node_lambda_sweep_csv = f"{sparse_node_dir}/lambda_sweep_summary.csv"
sparse_node_lambda_values = [1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0]

# Table S9 / Fig. S3C: architecture + optimisation grid, scored on in-distribution
# validation R2 only, so the held-out highest-dose bins play no part in the choice.
sparse_node_arch_grid_csv = f"{sparse_node_dir}/arch_grid.csv"

# Aliases kept because the surviving rules and helper below were written against the
# older names.
neural_ode_diffrax_dir = sparse_node_dir
neural_ode_diffrax_l1_seeds_dir    = sparse_node_l1_dir
neural_ode_diffrax_l21_seeds_dir   = sparse_node_l21_dir
neural_ode_diffrax_cnode_seeds_dir = sparse_node_cnode_dir
neural_ode_diffrax_l1_done    = sparse_node_l1_done
neural_ode_diffrax_l21_done   = sparse_node_l21_done
neural_ode_diffrax_cnode_done = sparse_node_cnode_done


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
        # The `select_k_panel_*` artefacts used to be requested here too. They came from
        # `experimental_plot_pysr_vs_selectk`, an in-distribution PySR-vs-linreg
        # comparison that v5 does not use; the OOD comparison is
        # `experimental_paper_fig_sr_vs_linreg`. The rule is gone, so requesting its
        # outputs would leave the default target unbuildable.
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

    # The v5 manuscript outputs. Rules that build them are in experimental.smk; their
    # paths are defined above so this target can name them.
    experimental_rule_all_inputs.extend([
        pysr_sweep_table,
        pysr_sweep_config,
        pysr_final_per_fit,
        pysr_final_summary,
        pysr_final_exemplars,
        paper_fig_parsimony_tradeoff,
        paper_fig_sr_vs_linreg,
        paper_fig_sr_vs_linreg_k4,
        paper_fig_nn_appendix,
        paper_fig_cutoff_robustness,
    ])

    # Must stay the FIRST rule in the workflow: Snakemake takes the first rule it parses
    # as the default target, and the `default_target: True` directive is not honoured
    # from inside a conditional block. Moving this below `experimental_marker_inputs`
    # silently reduces a bare `snakemake` to building only the data-prep stage.
    #
    # Note this requests `marker_summary_output` and the snapshot/per-minute integration
    # metrics -- the in-distribution PySR run, whose outputs are archived as superseded,
    # so building `all` from scratch re-runs it. For the manuscript outputs alone use
    # `experimental_v5_all`; for just the numbers, `experimental_v5_tables`.
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
            split_policy=select_k_split_policy,
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
                --test-split-policy {params.split_policy} \
                --measured-timepoints {params.measured}
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
    # `experimental_marker_pysr_grid` used to live here. It ran a small ad hoc
    # hyperparameter grid off the marker summary; the manuscript's configuration now
    # comes from the 288-arm sweep plus `select_pysr_config.py` in experimental.smk.
    # The branch itself is kept because the `else:` below is what declares the
    # sr_comparison `rule all` -- collapsing the conditional would change which
    # default target the synthetic workflow gets.
    pass

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
        seed_flag = " --seed 42"  # seed every method (data subsample + framework RNG), not just pysr
        # Force the activated conda env's bin to the front of PATH. snakemake
        # --use-conda logs "Activating conda environment ..." and sets
        # CONDA_PREFIX, but on some nodes the loaded Anaconda *module* python
        # still shadows the env's python/pip (pip then does a user-site install
        # into the wrong interpreter and the build/import fails). Prepending
        # $CONDA_PREFIX/bin is a no-op when activation already worked and a fix
        # when it didn't.
        env_path_fix = '[ -n "$CONDA_PREFIX" ] && export PATH="$CONDA_PREFIX/bin:$PATH";'
        # AI-Feynman hardcodes cwd-relative output (results/, model/, mystery.dat,
        # args.dat, qaz.dat) with no option to redirect it, so running it from the repo
        # root dumped all of that at the repo root, un-attributable to a model or
        # variant. Give each run its own working directory next to its temp output.
        work_dir_flag = ""
        if model == "aifeynman":
            suffix = f"_{variant}" if variant else ""
            work_dir_flag = f" --work_dir $(dirname {temp_file})/aifeynman_work{suffix}"
        return f"""
            {env_path_fix} {install_command}{separator} timeout {config["timeout_duration"]} python src/sr_models/{model}_model.py --dataset {dataset} --dataset_size {dataset_size} --features {features} --temp_file {temp_file}{variant_flag}{seed_flag}{work_dir_flag} || test -s {temp_file}
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
            [ -n "$CONDA_PREFIX" ] && export PATH="$CONDA_PREFIX/bin:$PATH"; PYTHONPATH=src python src/pipelines/sr_comparison/get_best_formula.py --methods {params.methods} --hall_of_fame {input.temp_files} --save {output.formula_files} --features {params.features}
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
            [ -n "$CONDA_PREFIX" ] && export PATH="$CONDA_PREFIX/bin:$PATH"; PYTHONPATH=src python src/pipelines/sr_comparison/integrate_and_plot_methods.py --dataset {input.dataset} --formulas {input.formulas} --methods {params.methods}  --discovery-scales {params.discovery_scales} --data-proportion {params.data_proportion} --trajectory-column {params.trajectory_column} --output {output.csv} --plot {output.plot}
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
            [ -n "$CONDA_PREFIX" ] && export PATH="$CONDA_PREFIX/bin:$PATH"; PYTHONPATH=src python src/pipelines/sr_comparison/plot_methods.py --dataset {input.dataset} --formulas {input.formulas} --output {output} --discovery-scales {params.discovery_scales}
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
            [ -n "$CONDA_PREFIX" ] && export PATH="$CONDA_PREFIX/bin:$PATH"; PYTHONPATH=src python src/pipelines/sr_comparison/plot_sr_timepoint_lineplot.py \
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
