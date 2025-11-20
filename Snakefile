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

exp_sr_output_dir_cfg = experimental_cfg.get("sr_output_dir")
if exp_sr_output_dir_cfg:
    exp_sr_output_dir = exp_sr_output_dir_cfg.rstrip("/")
else:
    exp_sr_output_dir = f"{functional_group_base_dir}/outputs"

exp_feature_modes = experimental_cfg.get("feature_modes", ["all", "gfp"])
exp_feature_modes_args = " ".join(exp_feature_modes)
exp_gfp_columns = experimental_cfg.get("gfp_columns", ["GFP", 'p-ERK1-2', 'p-MEK1-2', 'p-ERK1-2_min', 'p-MEK1-2_min'])
exp_gfp_columns_args = " ".join(exp_gfp_columns)
exp_confidence_threshold = experimental_cfg.get("confidence_threshold", 4.0)
exp_models = [model.lower() for model in experimental_cfg.get("models", ["pysr"])]
exp_run_per_minute_sr = bool(experimental_cfg.get("run_per_minute_sr", False))
exp_per_minute_max_time = float(experimental_cfg.get("per_minute_max_time", 30.0))
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
    f"{exp_sr_output_dir}/tradeoff_runs",
)
exp_tradeoff_skip_existing = bool(experimental_cfg.get("tradeoff_skip_existing", True))
exp_tradeoff_uncapped_max = int(experimental_cfg.get("tradeoff_uncapped_max", 1_000_000))
exp_tradeoff_seed = int(experimental_cfg.get("tradeoff_seed", 42))
exp_tradeoff_min_bin_samples = int(experimental_cfg.get("tradeoff_min_bin_samples", 500))
exp_tradeoff_eval_feature_mode = experimental_cfg.get("tradeoff_feature_mode_eval", "gfp")
exp_tradeoff_eval_model_name = experimental_cfg.get("tradeoff_model_name", "pysr")
exp_tradeoff_include_linreg = bool(experimental_cfg.get("tradeoff_include_linreg", True))

weighting_cfg = experimental_cfg.get("weighting", {})
weighting_groups = weighting_cfg.get("groups", exp_tradeoff_groups)
weighting_groups_args = " ".join(shlex.quote(group) for group in weighting_groups)
weighting_strategies = weighting_cfg.get(
    "strategies",
    ["none", "inverse_oom", "sqrt_inverse_oom", "marker_equal"],
)
weighting_strategies_args = " ".join(weighting_strategies)
weighting_fit_modes = weighting_cfg.get("fit_modes", ["weighted"])
weighting_fit_modes_args = " ".join(weighting_fit_modes)
weighting_output = weighting_cfg.get(
    "output",
    f"{exp_sr_output_dir}/reports/diagnostics/pysr_weighting_study.txt",
)
weighting_feature_mode = weighting_cfg.get("feature_mode", "gfp")
weighting_test_size = float(weighting_cfg.get("test_size", 0.2))
weighting_random_state = int(weighting_cfg.get("random_state", exp_tradeoff_seed))
weighting_min_group_size = int(weighting_cfg.get("min_group_size", 200))
weighting_niterations = int(weighting_cfg.get("niterations", 300))
weighting_population_size = int(weighting_cfg.get("population_size", 30))
weighting_populations = int(weighting_cfg.get("populations", 30))
weighting_max_size = int(weighting_cfg.get("max_size", 20))
weighting_parsimony = float(weighting_cfg.get("parsimony", 0.8))
weighting_binary_ops = weighting_cfg.get("binary_operators", ["+", "-", "*", "/"])
weighting_unary_ops = weighting_cfg.get("unary_operators", [])

# Optional: reuse weighting knobs for the main SR run, too
exp_sr_weighting_flags = ""
if weighting_strategies:
    exp_sr_weighting_flags += f"--weighting {weighting_strategies_args} "
if weighting_fit_modes:
    exp_sr_weighting_flags += f"--fit-modes {weighting_fit_modes_args}"
exp_sr_weighting_flags = exp_sr_weighting_flags.strip()

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
reports_summary_output = f"{exp_sr_output_dir}/reports/summary/functional_group_summary.csv"
functional_group_formula_files = [
    f"{exp_sr_output_dir}/reports/formulas/{mode}.txt" for mode in exp_feature_modes
]
functional_group_summary_output = reports_summary_output
exp_sr_per_minute_output_dir = f"{exp_sr_output_dir}/per_minute"
functional_group_summary_per_minute = f"{exp_sr_per_minute_output_dir}/reports/summary/functional_group_summary.csv"
functional_group_formula_files_per_minute = [
    f"{exp_sr_per_minute_output_dir}/reports/formulas/{mode}.txt" for mode in exp_feature_modes
]

tradeoff_diagnostics_output = f"{exp_sr_output_dir}/reports/diagnostics/balancing_tradeoff.txt"
weighting_study_output = weighting_output

plots_root = f"{exp_sr_output_dir}/plots"

functional_group_png_output = f"{plots_root}/pysr/metrics/functional_group_log_r2.png"
functional_group_svg_output = f"{plots_root}/pysr/metrics/functional_group_log_r2.svg"
functional_group_relative_png_output = f"{plots_root}/pysr/metrics/functional_group_relative_mae.png"
functional_group_relative_svg_output = f"{plots_root}/pysr/metrics/functional_group_relative_mae.svg"
functional_group_r2_scatter_png_output = f"{plots_root}/pysr/metrics/functional_group_log_r2_scatter.png"
functional_group_r2_scatter_svg_output = f"{plots_root}/pysr/metrics/functional_group_log_r2_scatter.svg"
functional_group_r2_confidence_png_output = f"{plots_root}/pysr/metrics/functional_group_log_r2_confidence.png"
functional_group_r2_confidence_svg_output = f"{plots_root}/pysr/metrics/functional_group_log_r2_confidence.svg"
functional_group_r2_confidence_high_png_output = (
    f"{plots_root}/pysr/metrics/functional_group_log_r2_confidence_high.png"
)
functional_group_r2_confidence_high_svg_output = (
    f"{plots_root}/pysr/metrics/functional_group_log_r2_confidence_high.svg"
)
functional_group_r2_direction_png_output = f"{plots_root}/pysr/metrics/functional_group_log_r2_direction.png"
functional_group_r2_direction_svg_output = f"{plots_root}/pysr/metrics/functional_group_log_r2_direction.svg"

functional_group_model_comparison_gfp_png_output = (
    f"{plots_root}/comparisons/core/functional_group_model_comparison_gfp.png"
)
functional_group_model_comparison_gfp_svg_output = (
    f"{plots_root}/comparisons/core/functional_group_model_comparison_gfp.svg"
)
functional_group_model_comparison_all_png_output = (
    f"{plots_root}/comparisons/core/functional_group_model_comparison_all.png"
)
functional_group_model_comparison_all_svg_output = (
    f"{plots_root}/comparisons/core/functional_group_model_comparison_all.svg"
)
functional_group_model_comparison_gfp_high_png_output = (
    f"{plots_root}/comparisons/core/high/functional_group_model_comparison_gfp_high.png"
)
functional_group_model_comparison_gfp_high_svg_output = (
    f"{plots_root}/comparisons/core/high/functional_group_model_comparison_gfp_high.svg"
)
functional_group_model_comparison_all_high_png_output = (
    f"{plots_root}/comparisons/core/high/functional_group_model_comparison_all_high.png"
)
functional_group_model_comparison_all_high_svg_output = (
    f"{plots_root}/comparisons/core/high/functional_group_model_comparison_all_high.svg"
)

functional_group_upset_gfp_png_output = f"{plots_root}/pysr/upset/upset_gfp.png"
functional_group_upset_gfp_svg_output = f"{plots_root}/pysr/upset/upset_gfp.svg"
functional_group_upset_all_png_output = f"{plots_root}/pysr/upset/upset_all.png"
functional_group_upset_all_svg_output = f"{plots_root}/pysr/upset/upset_all.svg"

fits_dir = f"{plots_root}/pysr/fits"
fits_done = f"{fits_dir}/.done"
per_minute_plots_root = f"{exp_sr_per_minute_output_dir}/plots"
per_minute_fits_dir = f"{per_minute_plots_root}/pysr/fits"
per_minute_pysr_done = f"{per_minute_plots_root}/pysr/.done"
per_minute_linreg_done = f"{per_minute_plots_root}/linear_regression/.done"
fit_sanity_dir = f"{plots_root}/fit_sanity"
fit_sanity_done = f"{fit_sanity_dir}/.done"

pysr_grid_output_dir = f"{plots_root}/pysr_grid"
pysr_grid_results_csv = f"{pysr_grid_output_dir}/pysr_hyperparam_grid_results.csv"

pysr_plot_outputs = [
    functional_group_png_output,
    functional_group_svg_output,
    functional_group_relative_png_output,
    functional_group_relative_svg_output,
    functional_group_r2_scatter_png_output,
    functional_group_r2_scatter_svg_output,
    functional_group_r2_confidence_png_output,
    functional_group_r2_confidence_svg_output,
    functional_group_r2_confidence_high_png_output,
    functional_group_r2_confidence_high_svg_output,
    functional_group_r2_direction_png_output,
    functional_group_r2_direction_svg_output,
    functional_group_model_comparison_gfp_png_output,
    functional_group_model_comparison_gfp_svg_output,
    functional_group_model_comparison_all_png_output,
    functional_group_model_comparison_all_svg_output,
    functional_group_model_comparison_gfp_high_png_output,
    functional_group_model_comparison_gfp_high_svg_output,
    functional_group_model_comparison_all_high_png_output,
    functional_group_model_comparison_all_high_svg_output,
    functional_group_upset_gfp_png_output,
    functional_group_upset_gfp_svg_output,
    functional_group_upset_all_png_output,
    functional_group_upset_all_svg_output,
]

pysr_grid_outputs = [pysr_grid_results_csv]

functional_group_png_output_linear_regression = (
    f"{plots_root}/linear_regression/metrics/functional_group_log_r2.png"
)
functional_group_svg_output_linear_regression = (
    f"{plots_root}/linear_regression/metrics/functional_group_log_r2.svg"
)
functional_group_relative_png_output_linear_regression = (
    f"{plots_root}/linear_regression/metrics/functional_group_relative_mae.png"
)
functional_group_relative_svg_output_linear_regression = (
    f"{plots_root}/linear_regression/metrics/functional_group_relative_mae.svg"
)
functional_group_r2_scatter_png_output_linear_regression = (
    f"{plots_root}/linear_regression/metrics/functional_group_log_r2_scatter.png"
)
functional_group_r2_scatter_svg_output_linear_regression = (
    f"{plots_root}/linear_regression/metrics/functional_group_log_r2_scatter.svg"
)
functional_group_r2_confidence_png_output_linear_regression = (
    f"{plots_root}/linear_regression/metrics/functional_group_log_r2_confidence.png"
)
functional_group_r2_confidence_svg_output_linear_regression = (
    f"{plots_root}/linear_regression/metrics/functional_group_log_r2_confidence.svg"
)
functional_group_r2_confidence_high_png_output_linear_regression = (
    f"{plots_root}/linear_regression/metrics/functional_group_log_r2_confidence_high.png"
)
functional_group_r2_confidence_high_svg_output_linear_regression = (
    f"{plots_root}/linear_regression/metrics/functional_group_log_r2_confidence_high.svg"
)
functional_group_r2_direction_png_output_linear_regression = (
    f"{plots_root}/linear_regression/metrics/functional_group_log_r2_direction.png"
)
functional_group_r2_direction_svg_output_linear_regression = (
    f"{plots_root}/linear_regression/metrics/functional_group_log_r2_direction.svg"
)

functional_group_model_comparison_gfp_png_output_linear_regression = (
    f"{plots_root}/comparisons/linear_regression/functional_group_model_comparison_gfp.png"
)
functional_group_model_comparison_gfp_svg_output_linear_regression = (
    f"{plots_root}/comparisons/linear_regression/functional_group_model_comparison_gfp.svg"
)
functional_group_model_comparison_all_png_output_linear_regression = (
    f"{plots_root}/comparisons/linear_regression/functional_group_model_comparison_all.png"
)
functional_group_model_comparison_all_svg_output_linear_regression = (
    f"{plots_root}/comparisons/linear_regression/functional_group_model_comparison_all.svg"
)
functional_group_model_comparison_gfp_high_png_output_linear_regression = (
    f"{plots_root}/comparisons/linear_regression/high/functional_group_model_comparison_gfp_high.png"
)
functional_group_model_comparison_gfp_high_svg_output_linear_regression = (
    f"{plots_root}/comparisons/linear_regression/high/functional_group_model_comparison_gfp_high.svg"
)
functional_group_model_comparison_all_high_png_output_linear_regression = (
    f"{plots_root}/comparisons/linear_regression/high/functional_group_model_comparison_all_high.png"
)
functional_group_model_comparison_all_high_svg_output_linear_regression = (
    f"{plots_root}/comparisons/linear_regression/high/functional_group_model_comparison_all_high.svg"
)

functional_group_upset_gfp_png_output_linear_regression = (
    f"{plots_root}/linear_regression/upset/upset_gfp.png"
)
functional_group_upset_gfp_svg_output_linear_regression = (
    f"{plots_root}/linear_regression/upset/upset_gfp.svg"
)
functional_group_upset_all_png_output_linear_regression = (
    f"{plots_root}/linear_regression/upset/upset_all.png"
)
functional_group_upset_all_svg_output_linear_regression = (
    f"{plots_root}/linear_regression/upset/upset_all.svg"
)

linreg_plot_outputs = [
    functional_group_png_output_linear_regression,
    functional_group_svg_output_linear_regression,
    functional_group_relative_png_output_linear_regression,
    functional_group_relative_svg_output_linear_regression,
    functional_group_r2_scatter_png_output_linear_regression,
    functional_group_r2_scatter_svg_output_linear_regression,
    functional_group_r2_confidence_png_output_linear_regression,
    functional_group_r2_confidence_svg_output_linear_regression,
    functional_group_r2_confidence_high_png_output_linear_regression,
    functional_group_r2_confidence_high_svg_output_linear_regression,
    functional_group_r2_direction_png_output_linear_regression,
    functional_group_r2_direction_svg_output_linear_regression,
    functional_group_model_comparison_gfp_png_output_linear_regression,
    functional_group_model_comparison_gfp_svg_output_linear_regression,
    functional_group_model_comparison_all_png_output_linear_regression,
    functional_group_model_comparison_all_svg_output_linear_regression,
    functional_group_model_comparison_gfp_high_png_output_linear_regression,
    functional_group_model_comparison_gfp_high_svg_output_linear_regression,
    functional_group_model_comparison_all_high_png_output_linear_regression,
    functional_group_model_comparison_all_high_svg_output_linear_regression,
    functional_group_upset_gfp_png_output_linear_regression,
    functional_group_upset_gfp_svg_output_linear_regression,
    functional_group_upset_all_png_output_linear_regression,
    functional_group_upset_all_svg_output_linear_regression,
]

pysr_outputs = (pysr_plot_outputs + pysr_grid_outputs) if run_pysr else []
linreg_outputs = linreg_plot_outputs if run_linreg else []

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
        tradeoff_diagnostics_output,
        weighting_study_output,
    ]

    experimental_rule_all_inputs.extend(pysr_outputs)
    experimental_rule_all_inputs.extend(linreg_outputs)

    experimental_rule_all_inputs.extend(functional_group_formula_files)
    if exp_run_per_minute_sr:
        experimental_rule_all_inputs.append(functional_group_summary_per_minute)
        experimental_rule_all_inputs.extend(functional_group_formula_files_per_minute)
        experimental_rule_all_inputs.append(per_minute_pysr_done)
        if run_linreg:
            experimental_rule_all_inputs.append(per_minute_linreg_done)
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
            dataset=functional_group_fit_snapshot_csv
        output:
            summary=functional_group_summary_output,
            formulas=functional_group_formula_files
        conda:
            "envs/pysr.yaml"
        params:
            output_dir=exp_sr_output_dir,
            feature_modes=exp_feature_modes_args,
            gfp_columns=exp_gfp_columns_args,
            group_definitions_csv=exp_group_definitions_csv,
            models=exp_models_args
        shell:
            """
            mkdir -p {params.output_dir}
            python src/experimental/sr_pipeline/run_functional_groups.py \
                --dataset {input.dataset} \
                --output-dir {params.output_dir} \
                --feature-modes {params.feature_modes} \
                --gfp-columns {params.gfp_columns} \
                --group-definitions-csv {params.group_definitions_csv} \
                --models {params.models}
            """

    if exp_run_per_minute_sr:
        rule experimental_functional_group_sr_per_minute:
            input:
                dataset=functional_group_per_minute_csv
            output:
                summary=functional_group_summary_per_minute,
                formulas=functional_group_formula_files_per_minute
            conda:
                "envs/pysr.yaml"
            params:
                output_dir=exp_sr_per_minute_output_dir,
                feature_modes=exp_feature_modes_args,
                gfp_columns=exp_gfp_columns_args,
                group_definitions_csv=exp_group_definitions_csv,
                models=exp_models_args,
                max_time=exp_per_minute_max_time
            shell:
                """
                mkdir -p {params.output_dir}
                python src/experimental/sr_pipeline/run_functional_groups.py \
                    --dataset {input.dataset} \
                    --dataset-mode per_minute \
                    --per-minute-max-time {params.max_time} \
                    --disable-balancing \
                    --output-dir {params.output_dir} \
                    --feature-modes {params.feature_modes} \
                    --gfp-columns {params.gfp_columns} \
                    --group-definitions-csv {params.group_definitions_csv} \
                    --models {params.models}
                """

        rule experimental_functional_group_plots_per_minute:
            input:
                summary=functional_group_summary_per_minute
            output:
                done=per_minute_pysr_done
            conda:
                "envs/pysr.yaml"
            params:
                output_dir=per_minute_plots_root,
                basename="pysr/metrics/functional_group_log_r2",
                relative_basename="pysr/metrics/functional_group_relative_mae",
                r2_basename="pysr/metrics/functional_group_log_r2_scatter",
                r2_confidence_basename="pysr/metrics/functional_group_log_r2_confidence",
                r2_confidence_high_basename="pysr/metrics/functional_group_log_r2_confidence_high",
                r2_direction_basename="pysr/metrics/functional_group_log_r2_direction",
                comparison_gfp_basename="comparisons/core/functional_group_model_comparison_gfp",
                comparison_all_basename="comparisons/core/functional_group_model_comparison_all",
                comparison_gfp_high_basename="comparisons/core/high/functional_group_model_comparison_gfp_high",
                comparison_all_high_basename="comparisons/core/high/functional_group_model_comparison_all_high",
                upset_gfp_basename="pysr/upset/upset_gfp",
                upset_all_basename="pysr/upset/upset_all",
                group_definitions_csv=exp_group_definitions_csv,
                confidence_threshold=exp_confidence_threshold,
                dataset=functional_group_per_minute_csv,
                log10_cutoff=exp_log10_cutoff,
                fit_grid_columns=exp_fit_grid_columns,
                fit_output_subdir="pysr/fits"
            shell:
                """
                mkdir -p {params.output_dir} "{params.output_dir}/pysr/fits"
                python src/experimental/sr_pipeline/summarize_functional_groups.py \
                    --summary {input.summary} \
                    --output-dir {params.output_dir} \
                    --group-definitions-csv {params.group_definitions_csv} \
                    --basename {params.basename} \
                    --model "PySR" \
                    --relative-basename {params.relative_basename} \
                    --r2-basename {params.r2_basename} \
                    --r2-confidence-basename {params.r2_confidence_basename} \
                    --r2-confidence-high-basename {params.r2_confidence_high_basename} \
                    --confidence-threshold {params.confidence_threshold} \
                    --r2-direction-basename {params.r2_direction_basename} \
                    --comparison-gfp-basename {params.comparison_gfp_basename} \
                    --comparison-all-basename {params.comparison_all_basename} \
                    --comparison-gfp-high-basename {params.comparison_gfp_high_basename} \
                    --comparison-all-high-basename {params.comparison_all_high_basename} \
                    --upset-gfp-basename {params.upset_gfp_basename} \
                    --upset-all-basename {params.upset_all_basename} \
                    --variant per_minute \
                    --dataset {params.dataset} \
                    --log10-cutoff {params.log10_cutoff} \
                    --fit-grid-columns {params.fit_grid_columns} \
                    --fit-output-subdir {params.fit_output_subdir}

                echo ok > "{output.done}.tmp"
                mv "{output.done}.tmp" "{output.done}"
                """

        if run_linreg:
            rule experimental_functional_group_plots_per_minute_linreg:
                input:
                    summary=functional_group_summary_per_minute
                output:
                    done=per_minute_linreg_done
                conda:
                    "envs/pysr.yaml"
                params:
                    output_dir=per_minute_plots_root,
                    basename="linear_regression/metrics/functional_group_log_r2",
                    relative_basename="linear_regression/metrics/functional_group_relative_mae",
                    r2_basename="linear_regression/metrics/functional_group_log_r2_scatter",
                    r2_confidence_basename="linear_regression/metrics/functional_group_log_r2_confidence",
                    r2_confidence_high_basename="linear_regression/metrics/functional_group_log_r2_confidence_high",
                    r2_direction_basename="linear_regression/metrics/functional_group_log_r2_direction",
                    comparison_gfp_basename="comparisons/linear_regression/functional_group_model_comparison_gfp",
                    comparison_all_basename="comparisons/linear_regression/functional_group_model_comparison_all",
                    comparison_gfp_high_basename="comparisons/linear_regression/high/functional_group_model_comparison_gfp_high",
                    comparison_all_high_basename="comparisons/linear_regression/high/functional_group_model_comparison_all_high",
                    upset_gfp_basename="linear_regression/upset/upset_gfp",
                    upset_all_basename="linear_regression/upset/upset_all",
                    group_definitions_csv=exp_group_definitions_csv,
                    confidence_threshold=exp_confidence_threshold
                shell:
                    """
                    mkdir -p {params.output_dir} "{params.output_dir}/linear_regression"
                    python src/experimental/sr_pipeline/summarize_functional_groups.py \
                        --summary {input.summary} \
                        --output-dir {params.output_dir} \
                        --group-definitions-csv {params.group_definitions_csv} \
                        --basename {params.basename} \
                        --model "Linear Regression" \
                        --relative-basename {params.relative_basename} \
                        --r2-basename {params.r2_basename} \
                        --r2-confidence-basename {params.r2_confidence_basename} \
                        --r2-confidence-high-basename {params.r2_confidence_high_basename} \
                        --confidence-threshold {params.confidence_threshold} \
                        --r2-direction-basename {params.r2_direction_basename} \
                        --comparison-gfp-basename {params.comparison_gfp_basename} \
                        --comparison-all-basename {params.comparison_all_basename} \
                        --comparison-gfp-high-basename {params.comparison_gfp_high_basename} \
                        --comparison-all-high-basename {params.comparison_all_high_basename} \
                        --upset-gfp-basename {params.upset_gfp_basename} \
                        --upset-all-basename {params.upset_all_basename} \
                        --variant per_minute

                    echo ok > "{output.done}.tmp"
                    mv "{output.done}.tmp" "{output.done}"
                    """

    rule experimental_balancing_tradeoff:
        input:
            dataset=functional_group_fit_snapshot_csv,
            group_defs=exp_group_definitions_csv
        output:
            report=tradeoff_diagnostics_output
        conda:
            "envs/pysr.yaml"
        params:
            log10_cutoff=exp_log10_cutoff,
            caps=exp_tradeoff_caps_args,
            groups_flag=f"--groups {exp_tradeoff_groups_args}" if exp_tradeoff_groups_args else "",
            include_uncapped="--include-uncapped" if exp_tradeoff_include_uncapped else "",
            run_flag="--run-pysr" if exp_tradeoff_run_pysr else "",
            pysr_output_root=f"--pysr-output-root {shlex.quote(exp_tradeoff_output_root)}",
            pysr_feature_modes=f"--pysr-feature-modes {exp_feature_modes_args}" if exp_feature_modes_args else "",
            pysr_gfp_columns=f"--pysr-gfp-columns {exp_gfp_columns_args}" if exp_gfp_columns_args else "",
            pysr_models=f"--pysr-models {exp_models_args}" if exp_models_args else "",
            skip_existing="--skip-existing" if exp_tradeoff_skip_existing else "",
            uncapped_max=f"--uncapped-max-value {exp_tradeoff_uncapped_max}",
            seed=f"--seed {exp_tradeoff_seed}",
            pysr_random_state=f"--pysr-random-state {exp_tradeoff_seed}",
            min_bin_samples=f"--min-bin-samples {exp_tradeoff_min_bin_samples}",
            feature_mode_eval=f"--pysr-feature-mode-eval {exp_tradeoff_eval_feature_mode}",
            model_name_eval=f"--pysr-model-name {exp_tradeoff_eval_model_name}",
            linreg_flag="--include-linreg" if exp_tradeoff_include_linreg else ""
        shell:
            """
            mkdir -p $(dirname {output.report})
            python src/experimental/experiments/evaluate_balancing_tradeoff.py \
                --dataset {input.dataset} \
                --group-definitions {input.group_defs} \
                --log10-cutoff {params.log10_cutoff} \
                --max-samples {params.caps} \
                {params.groups_flag} \
                {params.include_uncapped} \
                {params.seed} \
                {params.run_flag} \
                {params.pysr_output_root} \
                {params.pysr_feature_modes} \
                {params.pysr_gfp_columns} \
                {params.pysr_models} \
                {params.pysr_random_state} \
                {params.min_bin_samples} \
                {params.uncapped_max} \
                {params.feature_mode_eval} \
                {params.model_name_eval} \
                {params.skip_existing} \
                {params.linreg_flag} \
            --output {output.report}
        """

    rule experimental_pysr_weighting_study:
        input:
            dataset=functional_group_fit_snapshot_csv,
            group_defs=exp_group_definitions_csv
        output:
            report=weighting_study_output
        conda:
            "envs/pysr.yaml"
        params:
            groups_str=" ".join(weighting_groups),
            gfp_columns_str=" ".join(exp_gfp_columns),
            weighting_str=" ".join(weighting_strategies),
            fit_modes_str=["weighted", "duplicate"],
            log10_cutoff=exp_log10_cutoff,
            test_size=weighting_test_size,
            random_state=weighting_random_state,
            min_group_size=weighting_min_group_size,
            max_iterations=weighting_niterations,
            population_size=weighting_population_size,
            populations=weighting_populations,
            max_size=weighting_max_size,
            parsimony=weighting_parsimony
        shell:
            r"""
            set -euo pipefail
            mkdir -p "$(dirname {output.report})"

            python src/experimental/experiments/pysr_weighting_study.py \
            --dataset {input.dataset} \
            --group-definitions {input.group_defs} \
            --groups {params.groups_str} \
            --gfp-columns {params.gfp_columns_str} \
            --weighting {params.weighting_str} \
            --fit-modes {params.fit_modes_str} \
            --log10-cutoff {params.log10_cutoff} \
            --test-size {params.test_size} \
            --random-state {params.random_state} \
            --min-group-size {params.min_group_size} \
            --max-iterations {params.max_iterations} \
            --population-size {params.population_size} \
            --populations {params.populations} \
            --max-size {params.max_size} \
            --parsimony {params.parsimony} \
            --output {output.report}
            """

    rule experimental_functional_group_plots:
        input:
            summary=functional_group_summary_output
        output:
            png=functional_group_png_output,
            svg=functional_group_svg_output,
            png_relative=functional_group_relative_png_output,
            svg_relative=functional_group_relative_svg_output,
            png_r2=functional_group_r2_scatter_png_output,
            svg_r2=functional_group_r2_scatter_svg_output,
            png_r2_conf=functional_group_r2_confidence_png_output,
            svg_r2_conf=functional_group_r2_confidence_svg_output,
            png_r2_conf_high=functional_group_r2_confidence_high_png_output,
            svg_r2_conf_high=functional_group_r2_confidence_high_svg_output,
            png_r2_dir=functional_group_r2_direction_png_output,
            svg_r2_dir=functional_group_r2_direction_svg_output,
            png_model_comp_gfp=functional_group_model_comparison_gfp_png_output,
            svg_model_comp_gfp=functional_group_model_comparison_gfp_svg_output,
            png_model_comp_all=functional_group_model_comparison_all_png_output,
            svg_model_comp_all=functional_group_model_comparison_all_svg_output,
            png_model_comp_gfp_high=functional_group_model_comparison_gfp_high_png_output,
            svg_model_comp_gfp_high=functional_group_model_comparison_gfp_high_svg_output,
            png_model_comp_all_high=functional_group_model_comparison_all_high_png_output,
            svg_model_comp_all_high=functional_group_model_comparison_all_high_svg_output,
            png_upset_gfp=functional_group_upset_gfp_png_output,
            svg_upset_gfp=functional_group_upset_gfp_svg_output,
            png_upset_all=functional_group_upset_all_png_output,
            svg_upset_all=functional_group_upset_all_svg_output,
            # NEW: track the fits folder (which contains group__marker.* files) via a sentinel
            fits_dir=directory(fits_dir),
            fits_done=fits_done
        conda:
            "envs/pysr.yaml"
        params:
            output_dir=plots_root,
            basename=_relative_basename(functional_group_png_output, plots_root),
            model=exp_model_name_map["pysr"],
            variant="legacy",
            relative_basename=_relative_basename(functional_group_relative_png_output, plots_root),
            r2_basename=_relative_basename(functional_group_r2_scatter_png_output, plots_root),
            r2_confidence_basename=_relative_basename(functional_group_r2_confidence_png_output, plots_root),
            r2_confidence_high_basename=_relative_basename(functional_group_r2_confidence_high_png_output, plots_root),
            confidence_threshold=exp_confidence_threshold,
            r2_direction_basename=_relative_basename(functional_group_r2_direction_png_output, plots_root),
            group_definitions_csv=exp_group_definitions_csv,
            comparison_gfp_basename=_relative_basename(functional_group_model_comparison_gfp_png_output, plots_root),
            comparison_all_basename=_relative_basename(functional_group_model_comparison_all_png_output, plots_root),
            comparison_gfp_high_basename=_relative_basename(functional_group_model_comparison_gfp_high_png_output, plots_root),
            comparison_all_high_basename=_relative_basename(functional_group_model_comparison_all_high_png_output, plots_root),
            upset_gfp_basename=_relative_basename(functional_group_upset_gfp_png_output, plots_root),
            upset_all_basename=_relative_basename(functional_group_upset_all_png_output, plots_root),
            dataset=functional_group_fit_snapshot_csv,
            log10_cutoff=exp_log10_cutoff,
            fit_groups_arg=(" --fit-groups " + exp_fit_groups_args) if exp_fit_groups_args else "",
            fit_grid_columns=exp_fit_grid_columns,
            fit_output_subdir=exp_fit_output_subdir
        shell:
            r"""
            mkdir -p {params.output_dir} "{output.fits_dir}"
            python src/experimental/sr_pipeline/summarize_functional_groups.py \
                --summary {input.summary} \
                --output-dir {params.output_dir} \
                --group-definitions-csv {params.group_definitions_csv} \
                --basename {params.basename} \
                --model "{params.model}" \
                --relative-basename {params.relative_basename} \
                --r2-basename {params.r2_basename} \
                --r2-confidence-basename {params.r2_confidence_basename} \
                --r2-confidence-high-basename {params.r2_confidence_high_basename} \
                --confidence-threshold {params.confidence_threshold} \
                --r2-direction-basename {params.r2_direction_basename} \
                --comparison-gfp-basename {params.comparison_gfp_basename} \
                --comparison-all-basename {params.comparison_all_basename} \
                --comparison-gfp-high-basename {params.comparison_gfp_high_basename} \
                --comparison-all-high-basename {params.comparison_all_high_basename} \
                --upset-gfp-basename {params.upset_gfp_basename} \
                --upset-all-basename {params.upset_all_basename} \
                --variant {params.variant} \
                --dataset {params.dataset} \
                --log10-cutoff {params.log10_cutoff}{params.fit_groups_arg} \
                --fit-grid-columns {params.fit_grid_columns} \
                --fit-output-subdir {params.fit_output_subdir}

            # Mark the fits dir complete (avoids listing every group__marker file)
            echo ok > "{output.fits_done}.tmp"
            mv "{output.fits_done}.tmp" "{output.fits_done}"
            """

    if run_linreg:
        rule experimental_functional_group_plots_linreg:
            input:
                summary=functional_group_summary_output
            output:
                png=functional_group_png_output_linear_regression,
                svg=functional_group_svg_output_linear_regression,
                png_relative=functional_group_relative_png_output_linear_regression,
                svg_relative=functional_group_relative_svg_output_linear_regression,
                png_r2=functional_group_r2_scatter_png_output_linear_regression,
                svg_r2=functional_group_r2_scatter_svg_output_linear_regression,
                png_r2_conf=functional_group_r2_confidence_png_output_linear_regression,
                svg_r2_conf=functional_group_r2_confidence_svg_output_linear_regression,
                png_r2_conf_high=functional_group_r2_confidence_high_png_output_linear_regression,
                svg_r2_conf_high=functional_group_r2_confidence_high_svg_output_linear_regression,
                png_r2_dir=functional_group_r2_direction_png_output_linear_regression,
                svg_r2_dir=functional_group_r2_direction_svg_output_linear_regression,
                png_model_comp_gfp=functional_group_model_comparison_gfp_png_output_linear_regression,
                svg_model_comp_gfp=functional_group_model_comparison_gfp_svg_output_linear_regression,
                png_model_comp_all=functional_group_model_comparison_all_png_output_linear_regression,
                svg_model_comp_all=functional_group_model_comparison_all_svg_output_linear_regression,
                png_model_comp_gfp_high=functional_group_model_comparison_gfp_high_png_output_linear_regression,
                svg_model_comp_gfp_high=functional_group_model_comparison_gfp_high_svg_output_linear_regression,
                png_model_comp_all_high=functional_group_model_comparison_all_high_png_output_linear_regression,
                svg_model_comp_all_high=functional_group_model_comparison_all_high_svg_output_linear_regression,
                png_upset_gfp=functional_group_upset_gfp_png_output_linear_regression,
                svg_upset_gfp=functional_group_upset_gfp_svg_output_linear_regression,
                png_upset_all=functional_group_upset_all_png_output_linear_regression,
                svg_upset_all=functional_group_upset_all_svg_output_linear_regression
            conda:
                "envs/pysr.yaml"
            params:
                output_dir=plots_root,
                basename=_relative_basename(functional_group_png_output_linear_regression, plots_root),
                model=exp_model_name_map["linreg"],
                variant="legacy",
                relative_basename=_relative_basename(functional_group_relative_png_output_linear_regression, plots_root),
                r2_basename=_relative_basename(functional_group_r2_scatter_png_output_linear_regression, plots_root),
                r2_confidence_basename=_relative_basename(functional_group_r2_confidence_png_output_linear_regression, plots_root),
                r2_confidence_high_basename=_relative_basename(functional_group_r2_confidence_high_png_output_linear_regression, plots_root),
                confidence_threshold=exp_confidence_threshold,
                r2_direction_basename=_relative_basename(functional_group_r2_direction_png_output_linear_regression, plots_root),
                group_definitions_csv=exp_group_definitions_csv,
                upset_gfp_basename=_relative_basename(functional_group_upset_gfp_png_output_linear_regression, plots_root),
                upset_all_basename=_relative_basename(functional_group_upset_all_png_output_linear_regression, plots_root),
                comparison_gfp_basename=_relative_basename(functional_group_model_comparison_gfp_png_output_linear_regression, plots_root),
                comparison_all_basename=_relative_basename(functional_group_model_comparison_all_png_output_linear_regression, plots_root),
                comparison_gfp_high_basename=_relative_basename(functional_group_model_comparison_gfp_high_png_output_linear_regression, plots_root),
                comparison_all_high_basename=_relative_basename(functional_group_model_comparison_all_high_png_output_linear_regression, plots_root)
            shell:
                """
                mkdir -p {params.output_dir}
                python src/experimental/sr_pipeline/summarize_functional_groups.py \
                    --summary {input.summary} \
                    --output-dir {params.output_dir} \
                    --group-definitions-csv {params.group_definitions_csv} \
                    --basename {params.basename} \
                    --model "{params.model}" \
                    --relative-basename {params.relative_basename} \
                    --r2-basename {params.r2_basename} \
                    --r2-confidence-basename {params.r2_confidence_basename} \
                    --r2-confidence-high-basename {params.r2_confidence_high_basename} \
                    --confidence-threshold {params.confidence_threshold} \
                    --r2-direction-basename {params.r2_direction_basename} \
                --comparison-gfp-basename {params.comparison_gfp_basename} \
                --comparison-all-basename {params.comparison_all_basename} \
                --comparison-gfp-high-basename {params.comparison_gfp_high_basename} \
                --comparison-all-high-basename {params.comparison_all_high_basename} \
                --upset-gfp-basename {params.upset_gfp_basename} \
                --upset-all-basename {params.upset_all_basename} \
                --variant {params.variant}
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
                log10_cutoff=exp_log10_cutoff,
                gfp_columns=exp_gfp_columns_args
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
                    --log10-cutoff {params.log10_cutoff} \
                    --gfp-columns {params.gfp_columns}
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
