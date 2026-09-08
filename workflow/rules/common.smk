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
# The linear baseline is fitted on all ten inputs; the k sweep it used to run is not
# reported anywhere, so the per-K importance artefacts are no longer produced.
select_k_boxplot_dt = f"{select_k_dir}/boxplot_dt_r2.png"
select_k_boxplot_integ = f"{select_k_dir}/boxplot_integ_r2.png"
select_k_boxplot_ode = f"{select_k_dir}/boxplot_ode_r2.png"
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

# Four-layer L21 (lambda=3) neural ODEs -- the architecture and penalty strength the
# calibration sweep jointly selected, and the models every complexity number in the
# Results is computed from. Kept distinct from sparse_node_l21_dir (three layers), which
# the earlier drafts used.
node_hl4_dir          = f"{exp_runs_root}/reducibility/l21_hl4_local"
# Off-diagonal-normalised interaction counts: scoring cross-terms against the
# global Hessian maximum is not comparable across function classes (symbolic laws
# with divisions have diagonal curvature that swamps their interactions).
node_hl4_interactions = f"{exp_runs_root}/reducibility/interactions_offnorm.csv"
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
pysr_final_traj      = [
    f"{pysr_final_dir}/seeds/seed_{s}/metrics/"
    f"marker_integration_trajectories_per_minute.csv"
    for s in pysr_final_seeds
]
# The per-(context, seed) tree this run was recovered from: one directory per context,
# each holding the trajectory CSV that Fig. 5 and Fig. S2 draw individual bins from.
# scripts/migrate_v5_layout.py rescued it from a cluster scratchpad and derived seeds/
# by concatenation, so no rule produces it -- the figure rules below take it as an
# ancient() input for the same reason the sweep config is ancient(): it is a recorded
# run, and a timestamp must not re-trigger 70 CPU-hours.
pysr_final_fits_dir  = f"{pysr_final_dir}/fits"
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
# dose-response biology behind it.
#
# All eight are listed. This used to omit untransfected1 on the grounds that it was
# "already excluded by default", but that default was an argparse bug -- a non-empty
# default on an action="append" option, which the command line then could not override
# (docs/experimental_provenance.md, corrections of 6 Aug 2026). It is the origin of v4's
# "39 overexpression contexts". plot_parsimony_tradeoff.py has since been fixed to
# default to [], so leaning on it left the sparsity panel on 33 contexts rather than the
# 32 the denominator convention calls for.
exp_control_contexts = [
    "untransfected1", "untransfected2", "untransfected3", "untransfected4",
    "FLAG-GFP1", "FLAG-GFP2", "FLAG-GFP3", "FLAG-GFP4",
]
exp_exclude_flags = " ".join(
    f"--exclude-marker {m}" for m in exp_control_contexts
)

# Figures land with the question they answer, not in a shared bucket named after where
# they get printed. A cross-method result goes under comparisons/; anything that probes
# one model's internals goes inside that model's directory.
exp_comparisons_dir = f"{exp_runs_root}/comparisons"

# SR against the OLS baseline, at two feature counts.
sr_vs_linreg_dir  = f"{exp_comparisons_dir}/sr_vs_linreg"
sr_vs_linreg_k10  = f"{sr_vs_linreg_dir}/heldout_r2_k10.png"
sr_vs_linreg_k4   = f"{sr_vs_linreg_dir}/heldout_r2_k4.png"

# The one bucket named after where a figure is printed rather than what it asks, because
# the manuscript panels have to be collectable as a set. Everything here must also be a
# rule output: these were hand-run for the v5 draft, which is how the printed panel and
# the pipeline's own version drifted apart in the first place.
exp_paper_figures_dir = f"{exp_runs_root}/paper_figures"
exp_supplementary_dir = f"{exp_paper_figures_dir}/supplementary"
# Same comparison as sr_vs_linreg_k10, minus the control contexts -- see
# exp_control_contexts for why the printed panel drops them.
fig_4d_sr_vs_linreg = f"{exp_paper_figures_dir}/fig_4d_sr_vs_linreg.png"
# Fig. 4E/F as printed. Identical arguments to sr_vs_node_panel; the second output
# exists so the panel can be collected with the rest of the figure rather than copied
# out of comparisons/ by hand.
fig_4ef_parsimony = f"{exp_paper_figures_dir}/fig_4ef_accuracy_and_parsimony.png"

# Both baselines and the parsimony panel on one row: linear regression, the Neural
# ODE, and the driver counts. Same script and same numbers as fig_4ef_parsimony,
# with the linreg scatter added as a third panel -- a rule output rather than a
# hand-run command, so it cannot drift from the panels it is assembled next to.
fig_4_joint_baselines = f"{exp_paper_figures_dir}/fig_4_joint_linreg_parsimony.png"

# Closure cost: the dependency count and the interacting-pair count collapsed into one box
# column, C = k + rho*pairs, beside the curve of the contrast against rho. The two counts
# are not independent (pairs are bounded by C(k,2) and the networks sit at ~0.98 of that
# bound), so two boxes and two tests report close to one fact; the curve is what shows the
# result does not depend on the price assigned to an interaction.
fig_4_closure_cost = f"{exp_paper_figures_dir}/fig_4_closure_cost.png"

# Two variants of Fig. 4E kept as rule outputs because the draft chooses between them:
# _nointeractions drops the third box column, for the layout that reports the dependency
# count alone; _srquadrant restricts the SR box to the scatter's top-right quadrant, so it
# covers the same nine contexts as the neural-ODE box and the bracket between them is a
# paired comparison rather than one across two different context sets.
fig_4ef_parsimony_nointeractions = (
    f"{exp_paper_figures_dir}/fig_4ef_accuracy_and_parsimony_nointeractions.png")
fig_4ef_parsimony_srquadrant = (
    f"{exp_paper_figures_dir}/fig_4ef_accuracy_and_parsimony_srquadrant.png")

# The per-context panel values both Fig. 4E variants are drawn from, written by the same
# rule that draws them so the closure-cost figure cannot be computed from a different
# selection than the panel it replaces.
fig_4_panel_inputs = f"{exp_runs_root}/reducibility/panel_inputs_fig4.csv"

# Fig. 5 exemplars: per-GFP-bin trajectories for the two best perturbation contexts.
# Pinned here so the rule's outputs are static rather than depending on a ranking file's
# row order (see docs/experimental_provenance.md).
#
# The CONTEXTS are unchanged and not close: PIP5K3 (0.928) and PTPN7 (0.910) lead the
# perturbations, with ALPK2 third at 0.729.
#
# The SEEDS changed on 2026-08-12, from s43/s42 to s44/s43. Both were previously chosen by
# best held-out R2, which is selection on the test set; every other figure and Table S8 now
# retain the seed with the best TRAINING integrated R2, and these two must match or the
# manuscript shows one seed's equation beside another seed's statistics. The earlier comment
# here defended PTPN7 s42 on the grounds that s42 and s43 tie within noise (0.9071 vs
# 0.9097) and s42 was already printed -- true, but the tie is not the reason to choose: s43
# is what the training rule selects, and it happens to be best held-out as well.
# Accuracy is essentially unaffected (PIP5K3 0.937 -> 0.928, PTPN7 0.907 -> 0.910) and both
# laws keep the same variable sets, so only the displayed expressions change.
fig5_exemplars = [("PIP5K3", 44), ("PTPN7", 43)]
fig5_perbin = [
    f"{exp_paper_figures_dir}/fig_5_perbin_{marker}_s{seed}.png"
    for marker, seed in fig5_exemplars
]
# Fig. 5 centre panel: the fitted fan beside the integrated law, across all GFP bins. Built
# from the same per-fit trajectory table as fig5_perbin so the two panels of one figure
# cannot come from different fits -- the superseded fans under exemplar_fans/ were produced
# from a sweep directory that no longer exists.
fig5_fans = [
    f"{exp_paper_figures_dir}/fig_5_fan_{marker}_s{seed}.png"
    for marker, seed in fig5_exemplars
]

# Supplementary panels. The manuscript cites exactly Figures S1-S3 and Tables S8-S10;
# anything the draft dropped is still built, but into archive/ so a full run cannot
# repopulate the clean supplementary folder with figures no caption refers to.
exp_supplementary_archive_dir = f"{exp_supplementary_dir}/archive"

fig_s1_loss_ablation      = f"{exp_supplementary_dir}/fig_s1_loss_ablation.png"
fig_s2_threshold_examples = f"{exp_supplementary_dir}/fig_s2_threshold_examples.png"
fig_s3_node_selection     = f"{exp_supplementary_dir}/fig_s3_neural_ode_selection.png"

# Retired 2026-08-12 with the SI reorganisation: the threshold-calibration trio compared
# the dependency count against the participation ratio (a measure the draft no longer
# reports), the matched-k linreg panel and the all-40 sparsity panel lost their citations
# when Fig. 4 absorbed both comparisons. Kept reproducible, kept out of the way.
fig_s4_sr_vs_linreg       = f"{exp_supplementary_archive_dir}/fig_s4_sr_vs_linreg_matched.png"
fig_s5_parsimony_all40    = f"{exp_supplementary_archive_dir}/fig_s5_parsimony_all40.png"
fig_threshold_calibration = f"{exp_supplementary_archive_dir}/fig_threshold_calibration.png"
fig_threshold_error_tradeoff = f"{exp_supplementary_archive_dir}/fig_threshold_error_tradeoff.png"
fig_threshold_intuition = f"{exp_supplementary_archive_dir}/fig_threshold_intuition.png"

# Per-context symbolic-regression results: the table the Results text points at for the
# seed retained, its accuracy, the variables used and the recovered law. Its own rule,
# because it reads the integrated per-fit table and the Fig. 4 panel values rather than
# the neural-ODE sweeps the batch below summarises.
table_s8_sr_per_context = f"{exp_supplementary_dir}/table_s8_sr_per_context.csv"

# Tables S9-S10, written as one batch because they share the run artefacts they read.
# The two *_full.csv are the complete grids, deposited rather than typeset. S9 is the
# PySR configuration sweep and S10 the neural-ODE architecture search -- that order is
# fixed by the order the main text cites them, and the two were swapped until
# 2026-08-12; do not renumber them back without also fixing the captions.
supp_tables = [
    f"{exp_supplementary_dir}/table_s9_pysr_config_marginals.csv",
    f"{exp_supplementary_dir}/table_s10_neural_ode_arch_marginals.csv",
    f"{exp_supplementary_dir}/neural_ode_arch_grid_full.csv",
    f"{exp_supplementary_dir}/pysr_config_sweep_full.csv",
]

# Tables S12-S14: the threshold sensitivities of the two complexity contrasts, and the
# lambda calibration behind Fig. S3A. Wired 2026-08-17 -- all three were hand-built until
# then, which meant the numbers the SI quotes as robustness evidence were the only ones in
# the paper that a full run could not regenerate. S12/S13 come from one rule because a
# single pass over the checkpoints serves both sweeps; S14 is a deposit of the lambda
# sweep summary the calibration rule already writes.
table_s12_threshold_sensitivity = (
    f"{exp_supplementary_dir}/table_s12_threshold_sensitivity.csv")
table_s13_interaction_sensitivity = (
    f"{exp_supplementary_dir}/table_s13_interaction_threshold_sensitivity.csv")
table_s14_lambda_calibration = (
    f"{exp_supplementary_dir}/table_s14_lambda_calibration.csv")
# Per-(marker, seed) counts behind S12/S13, deposited so the two tables are auditable
# without rerunning the Jacobian and Hessian passes.
node_hl4_threshold_per_fit = (
    f"{exp_runs_root}/reducibility/threshold_sensitivity_per_fit.csv")

# Dropped from the manuscript, still built into archive/: the regulariser comparison went
# when the draft kept only the L21 lambda calibration sweep, and the readout key was never
# cited. They remain rule outputs so the numbers stay reproducible.
supp_tables_retired = [
    f"{exp_supplementary_archive_dir}/table_regulariser_comparison.csv",
    f"{exp_supplementary_archive_dir}/table_readout_key.csv",
]

# SR against the sparse Neural ODE: held-out accuracy plus effective driver counts.
sr_vs_node_dir    = f"{exp_comparisons_dir}/sr_vs_neural_ode"
sr_vs_node_panel  = f"{sr_vs_node_dir}/accuracy_and_drivers.png"

# Probes of the network itself, so they live with it.
node_driver_readout   = f"{sparse_node_dir}/driver_readout/cutoff_robustness.png"
node_regulariser_comp = f"{sparse_node_dir}/regulariser_comparison/l21_vs_l1_vs_cnode.png"

# k=10 uses the full input pool -- the strongest form of the baseline. k=4 matches the
# median number of variables in a solved SR equation, so the comparison is on equation
# size. The match is on the COUNT, not the identity: the baseline picks its k by
# univariate F-test while SR picks its own subset by search.
linreg_k_full = 10
linreg_k_matched = 4
# The select-k sweep must cover every k any figure asks for -- Fig. 4D reads k=10
# and Fig. S4 reads k=4 -- plus the smaller k the boxplots sweep over. Emitting only
# linreg_k_full silently produces a metrics file that the k=4 panels cannot be built
# from, and rebuilding the file overwrites the wider sweep.
linreg_k_min = 2
linreg_k_max = linreg_k_full

# Fig. S3B: the lambda_jac elbow sweep. One subdirectory per lambda, all markers.
sparse_node_lambda_sweep_dir = f"{sparse_node_dir}/lambda_sweep"
sparse_node_lambda_sweep_csv = f"{sparse_node_dir}/lambda_sweep_summary.csv"
sparse_node_lambda_values = [1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0]

# Table S9 / Fig. S3C: architecture + optimisation grid, scored on in-distribution
# validation R2 only, so the held-out highest-dose bins play no part in the choice.
# The 54 cells are the product below; naming follows the lambda sweep's `lam_3p0`
# convention, with `.` written as `p`.
sparse_node_arch_grid_dir = f"{sparse_node_dir}/arch_grid"
sparse_node_arch_grid_csv = f"{sparse_node_dir}/arch_grid.csv"
sparse_node_arch_widths      = [32, 64, 128]
sparse_node_arch_depths      = [2, 3, 4]
sparse_node_arch_lrs         = [0.001, 0.003, 0.01]
sparse_node_arch_activations = ["tanh", "softplus"]
# One seed only. The grid answers "which architecture", and the reported models are
# retrained at three seeds afterwards; a 3-seed grid would triple 54 GPU-hours to
# sharpen a choice the lambda sweep shows is insensitive at this resolution.
sparse_node_arch_grid_seed = 42
# CAUTION: re-running the grid rule will NOT reproduce the checked-in arch_grid.csv
# row for row. That file records 1 to 39 contexts per cell -- the cells were not all
# run over the same contexts -- so its cells are not strictly comparable to each other.
# The rule below runs every cell over every context, which is the comparison Table S9
# claims to report; expect the marginals to move when it is next run. The rule takes
# ancient() inputs so a timestamp cannot trigger 54 GPU-hours by accident.

# Table S8's dependency-count column: participation ratio of the input Jacobian with
# GFP excluded, read off the saved checkpoints of all three regulariser variants.
sparse_node_participation_csv = f"{sparse_node_dir}/participation_ratio_variants.csv"

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
        select_k_boxplot_dt,
        select_k_boxplot_integ,
        select_k_boxplot_ode,
        # The PySR-vs-linreg comparison is experimental_compare_sr_vs_linreg, on
        # the OOD split. There is no in-distribution equivalent by design.
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
        sr_vs_node_panel,
        sr_vs_linreg_k10,
        sr_vs_linreg_k4,
        node_regulariser_comp,
        node_driver_readout,
        # Printed panels and supplementary tables. Listed individually rather than by
        # pulling in `experimental_results`, because a rule cannot be an input.
        fig_4d_sr_vs_linreg,
        fig_4ef_parsimony,
        fig_s2_threshold_examples,
        fig_s3_node_selection,
        fig_s4_sr_vs_linreg,
        fig_s5_parsimony_all40,
        sparse_node_participation_csv,
        *fig5_perbin,
        *supp_tables,
    ])

    # Must stay the FIRST rule in the workflow: Snakemake takes the first rule it parses
    # as the default target, and the `default_target: True` directive is not honoured
    # from inside a conditional block. Moving this below `experimental_marker_inputs`
    # silently reduces a bare `snakemake` to building only the data-prep stage.
    #
    # Note this requests `marker_summary_output` and the snapshot/per-minute integration
    # metrics -- the in-distribution PySR run, whose outputs are archived as superseded,
    # so building `all` from scratch re-runs it. For the manuscript outputs alone use
    # `experimental_results`; for just the numbers, `experimental_metrics`.
    rule all:
        input:
            experimental_rule_all_inputs
