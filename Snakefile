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
exp_output_prefix = experimental_cfg.get("output_prefix", "functional_groups")
exp_gfp_bins = experimental_cfg.get("gfp_bins", 50)
exp_min_points = experimental_cfg.get("min_points", 5)
exp_extra_times = experimental_cfg.get("extra_times", [1, 3])
exp_extra_times_args = "--extra-times " + " ".join(str(t) for t in exp_extra_times) if exp_extra_times else ""
exp_force_rebin = bool(experimental_cfg.get("force_rebin", False))
exp_force_rebin_arg = "--force-rebin" if exp_force_rebin else ""

exp_group_definitions_csv = experimental_cfg.get(
    "group_definitions_csv",
    f"{exp_processed_dir}/functional_groups.csv",
)

functional_group_base_dir = exp_processed_dir

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

functional_group_output_root = f"{functional_group_base_dir}/{exp_output_prefix}"
functional_group_time_series_csv = f"{functional_group_output_root}/functional_groups_time_series.csv"
functional_group_feature_matrix_csv = f"{functional_group_output_root}/functional_groups_feature_matrix.csv"
functional_group_fit_snapshot_csv = f"{functional_group_output_root}/functional_groups_fit_snapshot.csv"
functional_group_per_minute_csv = f"{functional_group_output_root}/functional_groups_per_minute_fit.csv"
reports_summary_output = f"{exp_reports_root}/summary/functional_group_summary.csv"
reports_summary_seed_output = f"{exp_reports_root}/summary/functional_group_summary_seed.csv"
# Per-minute outputs still live under a subdir for plots convenience
functional_group_formula_files = [
    f"{exp_reports_root}/formulas/{mode}_{suffix}.txt"
    for mode in exp_feature_modes
    for suffix in ("snapshot", "per_minute")
]
functional_group_summary_output = reports_summary_output
functional_group_summary_seed_output = reports_summary_seed_output

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
        functional_group_time_series_csv,
        functional_group_feature_matrix_csv,
        functional_group_fit_snapshot_csv,
        functional_group_per_minute_csv,
        functional_group_summary_output,
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

    experimental_rule_all_inputs.extend(functional_group_formula_files)
    experimental_rule_all_inputs.append(fit_sanity_done)

    rule all:
        input:
            experimental_rule_all_inputs

    rule experimental_functional_group_inputs:
        input:
            raw=exp_raw_time_course
        output:
            time_series=functional_group_time_series_csv,
            feature_matrix=functional_group_feature_matrix_csv,
            fit_snapshot=functional_group_fit_snapshot_csv,
            per_minute=functional_group_per_minute_csv
        conda:
            "envs/pysr.yaml"
        params:
            output_dir=functional_group_base_dir,
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
                python src/experimental/data_prep/functional_groups_inputs.py \
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
            time=functional_group_time_series_csv,
            dense=functional_group_per_minute_csv
        output:
            done=fit_sanity_done
        conda:
            "envs/pysr.yaml"
        params:
            out_dir=fit_sanity_dir,
            protein=exp_fit_sanity_protein,
            marker_flag=(f"--markers {exp_fit_sanity_markers_args}" if exp_fit_sanity_markers_args else ""),
            max_cols=exp_fit_sanity_max_columns,
        shell:
            """
            mkdir -p {params.out_dir}
            python src/experimental/data_prep/fit_sanity_plots.py \
                --time-trajectories {input.time} \
                --fit-trajectories {input.dense} \
                --output-dir {params.out_dir} \
                --protein "{params.protein}" \
                --max-columns {params.max_cols} \
                {params.marker_flag}
            touch {output.done}
            """

    rule experimental_functional_group_sr:
        input:
            dataset=functional_group_fit_snapshot_csv,
            per_minute=functional_group_per_minute_csv
        output:
            summary=functional_group_summary_output,
            seed_summary=functional_group_summary_seed_output,
            formulas=functional_group_formula_files
        conda:
            "envs/pysr.yaml"
        params:
            output_dir=exp_sr_output_dir,
            group_definitions_csv=exp_group_definitions_csv,
            models=exp_models_args,
            measured=" ".join(str(t) for t in exp_measured_timepoints)
        shell:
            """
            mkdir -p {params.output_dir}
            python src/experimental/sr_pipeline/run_functional_groups.py \
                --dataset {input.dataset} \
                --per-minute-dataset {input.per_minute} \
                --output-dir {params.output_dir} \
                --group-definitions-csv {params.group_definitions_csv} \
                --models {params.models} \
                --measured-timepoints {params.measured}
            """

    rule experimental_select_k_linreg:
        input:
            per_minute=functional_group_per_minute_csv
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
            "envs/pysr.yaml"
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
            python src/experimental/sr_pipeline/select_k_linreg_per_minute.py \
                --dataset {input.per_minute} \
                --output-dir {params.out_dir} \
                --per-minute-max-time {params.max_time} \
                --per-minute-sampling-strategy {params.strategy} \
                --late-sample-window {params.late_window_start} {params.late_window_end} \
                --late-sample-points {params.late_points} \
                --seeds {params.seeds} \
                --measured-timepoints {params.measured}
            """

    rule experimental_marker_integration:
        input:
            summary_mean=functional_group_summary_output,
            summary_seed=functional_group_summary_seed_output,
            snapshot=functional_group_fit_snapshot_csv,
            per_minute=functional_group_per_minute_csv
        output:
            integration_snapshot=integration_snapshot_metrics,
            integration_per_minute=integration_per_minute_metrics,
            integration_snapshot_traj=integration_snapshot_traj,
            integration_per_minute_traj=integration_per_minute_traj,
            predicted_snapshot=predicted_snapshot_traj,
            predicted_per_minute=predicted_per_minute_traj,
        conda:
            "envs/pysr.yaml"
        params:
            metrics_dir=exp_metrics_root,
            seeds_dir=exp_runs_seeds,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            trajectories_dir=exp_trajectories_root,
        shell:
            """
            mkdir -p {params.metrics_dir} {params.trajectories_dir}
            seed_summaries=($(ls {params.seeds_dir}/seed_*/summary/functional_group_summary.csv 2>/dev/null || true))

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
                    python src/experimental/sr_pipeline/compute_marker_integration.py \
                        --dataset {input.snapshot} \
                        --summary "$summary_path" \
                        --sr-trajectories "{params.seeds_dir}/seed_${{seed_id}}/metrics/predicted_trajectories_snapshot.csv" \
                        --dataset-mode snapshot \
                        --output "$seed_snapshot" \
                        --trajectories-output "$seed_snapshot_traj" \
                        --aggregate-output {integration_snapshot_metrics_all_seeds} \
                        --measured-timepoints {params.measured} \
                        --seed "$seed_id"

                    python src/experimental/sr_pipeline/compute_marker_integration.py \
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
                python src/experimental/sr_pipeline/compute_marker_integration.py \
                    --dataset {input.snapshot} \
                    --summary {input.summary_seed} \
                    --sr-trajectories "$seed_metrics_dir/predicted_trajectories_snapshot.csv" \
                    --dataset-mode snapshot \
                    --output "$seed_metrics_dir/marker_integration_metrics_snapshot.csv" \
                    --trajectories-output "$seed_metrics_dir/marker_integration_trajectories_snapshot.csv" \
                    --aggregate-output {integration_snapshot_metrics_all_seeds} \
                    --measured-timepoints {params.measured}
                python src/experimental/sr_pipeline/compute_marker_integration.py \
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
            """

    rule experimental_plots:
        input:
            summary_mean=functional_group_summary_output,
            summary_seed=functional_group_summary_seed_output,
            snapshot=functional_group_fit_snapshot_csv,
            per_minute=functional_group_per_minute_csv,
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
            feature_importance_svg=feature_importance_plots_svg
        conda:
            "envs/pysr.yaml"
        params:
            overlays_dir=exp_runs_seeds,
            metrics_plots_dir=plots_metrics_dir,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            metrics_dir=exp_metrics_root,
            seeds_dir=exp_runs_seeds,
        shell:
            """
            mkdir -p {params.overlays_dir} {params.metrics_plots_dir} {params.metrics_dir}
            seed_summaries=($(ls {params.seeds_dir}/seed_*/summary/functional_group_summary.csv 2>/dev/null || true))

            if [ "${{#seed_summaries[@]}}" -gt 0 ]; then
                # Per-seed overlays only
                for summary_path in "${{seed_summaries[@]}}"; do
                    seed_id="$(basename "$(dirname "$(dirname "$summary_path")")" | sed 's/seed_//')"
                    seed_metrics_dir="{params.seeds_dir}/seed_${{seed_id}}/metrics"
                    seed_overlay_dir="{params.overlays_dir}/seed_${{seed_id}}"
                    mkdir -p "$seed_overlay_dir" "$seed_metrics_dir"

                    python src/experimental/sr_pipeline/plot_marker_overlays.py \
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
                    python src/experimental/sr_pipeline/plot_marker_overlays.py \
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

                    python src/experimental/sr_pipeline/plot_marker_overlays.py \
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
                    python src/experimental/sr_pipeline/plot_marker_overlays.py \
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
                python src/experimental/sr_pipeline/plot_marker_overlays.py \
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
                python src/experimental/sr_pipeline/plot_marker_overlays.py \
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
                python src/experimental/sr_pipeline/plot_marker_overlays.py \
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
                python src/experimental/sr_pipeline/plot_marker_overlays.py \
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
            python src/experimental/sr_pipeline/plot_metrics_summary.py \
                --summary {input.summary_mean} \
                --output-dir {params.metrics_plots_dir} \
                --integration-metrics-dir {params.metrics_dir} \
                --snapshot-dataset {input.snapshot} \
                --per-minute-dataset {input.per_minute}
            """

if run_pysr:
        rule experimental_functional_group_pysr_grid:
            input:
                summary=functional_group_summary_output,
                dataset=functional_group_fit_snapshot_csv,
                groups=exp_group_definitions_csv
            output:
                results=pysr_grid_results_csv
            conda:
                "envs/pysr.yaml"
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
                python src/experimental/experiments/pysr_hyperparam_grid.py \
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
            "envs/base.yaml"
        params:
            target_feature=config["target_feature"]
        shell:
            """
            python src/preprocessing.py --train {input.train} --test {input.test} --valid {input.valid} --output {output.merged} --train-output {output.train_split} --test-output {output.test_split} --target_feature {params.target_feature}
            """

if "pysindy" in sweep_best_configs:
    rule sweep_pysindy:
        input:
            train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv"
        output:
            config_file=sweep_best_configs["pysindy"]
        conda:
            "envs/pysindy.yaml"
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
            "envs/aifeynman.yaml"
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
            "envs/dso.yaml"
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
            "envs/kan.yaml"
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
            "envs/pysr.yaml"
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
        "aifeynman": "pip install --no-deps aifeynman",
        "dso": "pip install absl-py==0.7.0 && pip install numpy==1.18 && pip install -e src/dso",
        "pysindy": "pip install -e src/pysindy && pip install cvxpy && pip install tensorflow"
    }
    install_command = install_cmds.get(model, "")
    separator = ";" if install_command else ""
    variant_flag = f" --variant {variant}" if variant else ""
    return f"""
        {install_command}{separator} timeout {config["timeout_duration"]} python src/sr_models/{model}_model.py --dataset {dataset} --dataset_size {dataset_size} --features {features} --temp_file {temp_file}{variant_flag} || test -s {temp_file}
    """

# Rules for symbolic regression for each model
rule pysindy:
    input:
        train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv",
        sweep=lambda wildcards: sweep_input("pysindy")
    output:
        temp_file=temporary(f"data/{enzyme_model}/{data_type}/sr_comparison/temp/temp_formula_pysindy.txt")
    conda:
        "envs/pysindy.yaml"
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
        "envs/aifeynman.yaml"
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
        "envs/dso.yaml"
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
        "envs/kan.yaml"
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
        "envs/pysr.yaml"
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
        "envs/nn.yaml"
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
        "envs/base.yaml"
    params:
        methods=" ".join(models),
        features=features
    shell:
        """
        echo "Extracting best formulas for methods: {params.methods}"
        python src/get_best_formula.py --methods {params.methods} --hall_of_fame {input.temp_files} --save {output.formula_files} --features {params.features}
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
        "envs/base.yaml"
    params:
        methods=" ".join(models),
        data_proportion=config["data_proportion"],
        features=features,
        discovery_scales="'" + config["discovery_scales"] + "'",
        trajectory_column="condition_id"
    shell:
        """
        echo "Integrating and plotting results for methods: {params.methods}"
        strace -o output_int.log -T -f python src/integrate_and_plot_methods.py --dataset {input.dataset} --formulas {input.formulas} --methods {params.methods}  --discovery-scales {params.discovery_scales} --data-proportion {params.data_proportion} --trajectory-column {params.trajectory_column} --output {output.csv} --plot {output.plot}
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
        "envs/base.yaml"
    params:
        discovery_scales="'" + config["discovery_scales"] + "'"
    shell:
        """
        echo "Generating comparison plot for all methods."
        strace -o output_plot.log -T -f python src/plot_methods.py --dataset {input.dataset} --formulas {input.formulas} --output {output} --discovery-scales {params.discovery_scales}
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
        "envs/base.yaml"
    params:
        methods=" ".join(models),
        features=features,
        discovery_scales="'" + config["discovery_scales"] + "'",
        nn_model_args=nn_model_arg_str,
        nn_dataset_size_arg=nn_dataset_size_arg,
        seed=42
    shell:
        """
        python src/plot_sr_timepoint_lineplot.py \
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
    conda:
        "envs/pysr.yaml"
    params:
        dataset_size=config["dataset_sizes"]["pysr"],
        features=features
    shell:
        """
        echo "Running PySR on different kinetic regimes."
        python src/kinetic_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features}
        echo "PySR biochemical regime evaluation completed."
        """

rule noise_regimes_full:
    input:
        dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
    output:
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
    conda:
        "envs/pysr.yaml"
    params:
        dataset_size=config["dataset_sizes"]["pysr"],
        features=features
    shell:
        """
        echo "Running PySR on full-dataset noise regimes."
        python src/noise_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features}
        echo "PySR full noise regime evaluation completed."
        """

rule mm_deviation_regimes:
    input:
        dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
    output:
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
    conda:
        "envs/pysr.yaml"
    params:
        dataset_size=config["dataset_sizes"]["pysr"],
        features=features
    shell:
        """
        echo "Running PySR on Michaelis-Menten deviation regimes."
        python src/mm_deviation_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features}
        echo "PySR MM deviation regime evaluation completed."
        """

rule dataset_size_regimes:
    input:
        dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
    output:
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
        f"data/{enzyme_model}/{data_type}/dataset_size_regimes/shared/plots/relative_mae/relative_mae_error_distributions.png"
    conda:
        "envs/pysr.yaml"
    params:
        features=features,
        pysr_size=config["dataset_sizes"]["pysr"],
        nn_size=config["dataset_sizes"].get("nn", 20000)
    shell:
        """
        echo "Running dataset-size regime evaluation."
        python src/dataset_size_regimes.py --dataset {input.dataset} --features {params.features} --pysr-base-size {params.pysr_size} --nn-base-size {params.nn_size}
        echo "Dataset-size regime evaluation completed."
        """

if data_type == "dynamic":

    rule timepoint_regimes:
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
            f"data/{enzyme_model}/{data_type}/timepoint_regimes/shared/results/pysr/all_pysr_formulas.txt",
        conda:
            "envs/pysr.yaml"
        params:
            dataset_size=config["dataset_sizes"]["pysr"],
            features=features
        shell:
            """
            echo "Running PySR across timepoint groups."
            python src/timepoint_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features}
            echo "Timepoint benchmarking completed."
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

rule pan_enzyme_model_plots:
    input:
        formulas=formula_files,
        root_dir="data",
        dataset=f"data/{enzyme_model}/{data_type}/processed/data_test.csv"
    output:
        "data/panmodel_plots/symbolic_model_r2_scores_bar.png",
        "data/panmodel_plots/symbolic_model_r2_scores_scatter.png",
        "data/panmodel_plots/all_symbolic_formulas.txt"
    conda:
        "envs/base.yaml"
    params:
        discovery_scales="'" + config["discovery_scales"] + "'"
    shell:
        "python src/pan_enzyme_model_plots.py --root-dir {input.root_dir} --dataset {input.dataset} --discovery-scales {params.discovery_scales}"
