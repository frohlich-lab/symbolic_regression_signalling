# Global configuration for feature selection, data type, dataset sizes, and timeout duration
configfile: "config.yaml"

# Define data type and parameters from configuration
data_type = config["data_type"]  # Example: 'dynamic'
enzyme_model = config["enzyme_model"]
models = config["models"]  # Example: ['pysindy', 'aifeynman', 'dso', 'kan', 'pysr']
output_extension = config["output_extension"]
features = config["features"][data_type]

# Experimental configuration for functional marker groups
experimental_cfg = config.get("experimental", {})
exp_raw_time_course = experimental_cfg.get("raw_time_course", "data/experimental/raw/lun_2019.csv")
exp_processed_dir = experimental_cfg.get("processed_dir", "data/experimental/processed").rstrip("/")
exp_output_prefix = experimental_cfg.get("output_prefix", "functional_groups")
exp_gfp_bins = experimental_cfg.get("gfp_bins", 50)
exp_min_points = experimental_cfg.get("min_points", 5)
exp_extra_times = experimental_cfg.get("extra_times", [1, 3])
exp_extra_times_args = "--extra-times " + " ".join(str(t) for t in exp_extra_times) if exp_extra_times else ""

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
exp_include_fresh = experimental_cfg.get("include_fresh_groups", False)
exp_include_fresh_flag = "--include-fresh-groups" if exp_include_fresh else ""

functional_group_time_output = f"{functional_group_base_dir}/time_trajectories.csv"
functional_group_filtered_output = f"{functional_group_base_dir}/filtered_features.csv"
functional_group_extra_output = f"{functional_group_base_dir}/extra_fit_trajectories.csv"
functional_group_summary_output = f"{exp_sr_output_dir}/functional_group_sr_summary.csv"
functional_group_png_output = f"{exp_sr_output_dir}/functional_group_log_r2.png"
functional_group_svg_output = f"{exp_sr_output_dir}/functional_group_log_r2.svg"
functional_group_png_combined_output = (
    f"{exp_sr_output_dir}/functional_group_log_r2_combined.png"
)
functional_group_svg_combined_output = (
    f"{exp_sr_output_dir}/functional_group_log_r2_combined.svg"
)
functional_group_formula_files = [
    f"{exp_sr_output_dir}/functional_group_formulas_{mode}.txt" for mode in exp_feature_modes
]

# Precompute paths for temporary and final formula files
temp_files = [
    f"data/{enzyme_model}/{data_type}/sr_comparison/temp/hall_of_fame.csv" if model == "pysr" 
    else f"data/{enzyme_model}/{data_type}/sr_comparison/temp/temp_formula_{model}.{output_extension[model]}" 
    for model in models
]
formula_files = [f"data/{enzyme_model}/{data_type}/sr_comparison/results/formula_{model}.txt" for model in models]


# Define output files for the entire workflow (Snakemake will skip existing ones)
if enzyme_model == "experimental":

    experimental_rule_all_inputs = [
        functional_group_time_output,
        functional_group_filtered_output,
        functional_group_extra_output,
        functional_group_summary_output,
        functional_group_png_output,
        functional_group_svg_output,
        functional_group_png_combined_output,
        functional_group_svg_combined_output,
        *functional_group_formula_files,
    ]

    rule all:
        input:
            experimental_rule_all_inputs

    rule experimental_functional_group_inputs:
        input:
            raw=exp_raw_time_course
        output:
            time=functional_group_time_output,
            filtered=functional_group_filtered_output,
            extra=functional_group_extra_output
        conda:
            "envs/pysr.yaml"
        params:
            output_dir=functional_group_base_dir,
            prefix=exp_output_prefix,
            gfp_bins=exp_gfp_bins,
            min_points=exp_min_points,
            extra_times=exp_extra_times_args
        shell:
            """
            mkdir -p {params.output_dir}
            python src/generate_functional_group_inputs.py \
                --time-course {input.raw} \
                --output-dir {params.output_dir} \
                --output-prefix {params.prefix} \
                --gfp-bins {params.gfp_bins} \
                --min-points {params.min_points} {params.extra_times}
            """

    rule experimental_functional_group_sr:
        input:
            dataset=functional_group_extra_output
        output:
            summary=functional_group_summary_output,
            formulas=functional_group_formula_files
        conda:
            "envs/pysr.yaml"
        params:
            output_dir=exp_sr_output_dir,
            feature_modes=exp_feature_modes_args,
            gfp_columns=exp_gfp_columns_args,
            include_fresh=exp_include_fresh_flag
        shell:
            """
            mkdir -p {params.output_dir}
            python src/sr_functional_groups.py \
                --dataset {input.dataset} \
                --output-dir {params.output_dir} \
                --feature-modes {params.feature_modes} \
                --gfp-columns {params.gfp_columns} \
                {params.include_fresh}
            """

    rule experimental_functional_group_plots:
        input:
            summary=functional_group_summary_output
        output:
            png=functional_group_png_output,
            svg=functional_group_svg_output
        conda:
            "envs/pysr.yaml"
        params:
            output_dir=exp_sr_output_dir,
            basename="functional_group_log_r2",
            variant="legacy"
        shell:
            """
            mkdir -p {params.output_dir}
            python src/experimental/plot_functional_group_results.py \
                --summary {input.summary} \
                --output-dir {params.output_dir} \
                --basename {params.basename} \
                --variant {params.variant}
            """

    rule experimental_functional_group_plots_combined:
        input:
            summary=functional_group_summary_output
        output:
            png=functional_group_png_combined_output,
            svg=functional_group_svg_combined_output
        conda:
            "envs/pysr.yaml"
        params:
            output_dir=exp_sr_output_dir,
            basename="functional_group_log_r2_combined",
            variant="combined"
        shell:
            """
            mkdir -p {params.output_dir}
            python src/experimental/plot_functional_group_results.py \
                --summary {input.summary} \
                --output-dir {params.output_dir} \
                --basename {params.basename} \
                --variant {params.variant}
            """

    rule parameter_inference:
        input:
            "data/experimental/static/raw/experimental_data_alexandrov.xlsx"
        output:
            "src/experimental/parameter_inference/constants.py"
        shell:
            """
            source amici_env/bin/activate 
            python src/experimental/parameter_inference/train.py 
            deactivate
            """

    rule experimental_preprocessing:
        input:
            excel="data/experimental/static/raw/experimental_data_alexandrov.xlsx",
            cst="src/experimental/parameter_inference/constants.py"
        output:
            "data/experimental/static/processed/data_merged.csv"
        conda:
            "envs/base.yaml"
        shell:
            "python src/experimental/preprocessing.py --input {input.excel} --output {output}"

else:

    rule all:
        input:
            f"data/{enzyme_model}/{data_type}/sr_comparison/plots/results_plot.png",
            f"data/{enzyme_model}/{data_type}/sr_comparison/plots/integrated_results_plot.png",
            f"data/{enzyme_model}/{data_type}/sr_comparison/results/loss_comparison.csv",
            f"data/{enzyme_model}/{data_type}/sr_comparison/models/nn/model_nn.pth",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/regime_loss_comparison.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/error_landscape.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/results/pysr/all_pysr_formulas.txt",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae_horizontal_boxplot.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae_vertical_boxplot.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae_horizontal_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae_vertical_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/sanity_scatter_mm_vs_pu_over_tk.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/sanity_scatter_mm_vs_km_over_pu.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/sanity_scatter_first_order_vs_km_over_pu.png",
            f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/sanity_scatter_zero_order_vs_km_over_pu.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/regime_loss_comparison.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/error_landscape.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/results/pysr/all_pysr_formulas.txt",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae_horizontal_boxplot.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae_vertical_boxplot.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae_horizontal_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae_vertical_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae_deviation_lineplot.png",
            f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae_deviation_lineplot_template.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/plots/regime_loss_comparison.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/plots/error_landscape.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/results/pysr/all_pysr_formulas.txt",
            f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/plots/log_mae_horizontal_boxplot.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/plots/log_mae_vertical_boxplot.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/plots/log_mae_horizontal_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/plots/log_mae_vertical_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/plots/log_mae_noise_regime_lineplot.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/plots/log_mae_noise_regime_lineplot_template.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/regime_loss_comparison.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/error_landscape.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/results/pysr/all_pysr_formulas.txt",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae_horizontal_boxplot.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae_vertical_boxplot.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae_horizontal_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae_vertical_boxplot_no_outliers.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae_noise_regime_lineplot.png",
            f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae_noise_regime_lineplot_template.png",
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

# Function to generate shell commands for symbolic regression with timeout and optional installs
def symbolic_regression_rule(model, dataset, dataset_size, features, temp_file):
    install_cmds = {
        "aifeynman": "pip install --no-deps aifeynman",
        "dso": "pip install absl-py==0.7.0 && pip install numpy==1.18 && pip install -e src/dso",
        "pysindy": "pip install -e src/pysindy && pip install cvxpy && pip install tensorflow"
    }
    install_command = install_cmds.get(model, "")
    separator = ";" if install_command else ""
    return f"""
        {install_command}{separator} timeout {config["timeout_duration"]} python src/sr_models/{model}_model.py --dataset {dataset} --dataset_size {dataset_size} --features {features} --temp_file {temp_file} || test -s {temp_file}
    """

# Rules for symbolic regression for each model
rule pysindy:
    input:
        train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv"
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
        train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv"
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
        train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv"
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
        train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv"
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
        train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv"
    output:
        temp_file=temporary(f"data/{enzyme_model}/{data_type}/sr_comparison/temp/hall_of_fame.csv")
    conda:
        "envs/pysr.yaml"
    params:
        dataset_size=config["dataset_sizes"].get("nn", config["dataset_sizes"]["pysr"]),
        features=features
    shell:
        symbolic_regression_rule("pysr", "{input.train}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

rule nn:
    input:
        train=f"data/{enzyme_model}/{data_type}/processed/data_train.csv"
    output:
        output=f"data/{enzyme_model}/{data_type}/sr_comparison/models/nn/model_nn.pth"
    conda:
        "envs/nn.yaml"
    params:
        dataset_size=config["dataset_sizes"]["nn"],
        features=features
    shell:
        """
        python src/nn_model.py --dataset {input.train} --dataset_size {params.dataset_size} --features {params.features} --output {output.output} || true
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
        methods=models,
        features=features
    shell:
        """
        echo "Extracting best formulas for methods: {params.methods}"
        python src/get_best_formula.py --method {params.methods} --hall_of_fame {input.temp_files} --save {output.formula_files} --features {params.features}
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
        methods=models,
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

rule kinetic_regimes:
    input:
        dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
    output:
        f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/regime_loss_comparison.png",
        f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/error_landscape.png",
        f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/results/pysr/all_pysr_formulas.txt",
        f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae_horizontal_boxplot.png",
        f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae_vertical_boxplot.png",
        f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae_horizontal_boxplot_no_outliers.png",
        f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae_vertical_boxplot_no_outliers.png"
        , f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/sanity_scatter_mm_vs_pu_over_tk.png"
        , f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/sanity_scatter_mm_vs_km_over_pu.png"
        , f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/sanity_scatter_first_order_vs_km_over_pu.png"
        , f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/sanity_scatter_zero_order_vs_km_over_pu.png"
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

rule noise_regimes:
    input:
        dataset=f"data/{enzyme_model}/{data_type}/processed/data_test.csv"
    output:
        f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/plots/regime_loss_comparison.png",
        f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/plots/error_landscape.png",
        f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/results/pysr/all_pysr_formulas.txt",
        f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/plots/log_mae_horizontal_boxplot.png",
        f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/plots/log_mae_vertical_boxplot.png",
        f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/plots/log_mae_horizontal_boxplot_no_outliers.png",
        f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/plots/log_mae_vertical_boxplot_no_outliers.png",
        f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/plots/log_mae_noise_regime_lineplot.png",
        f"data/{enzyme_model}/{data_type}/noise_regimes_filtered/shared/plots/log_mae_noise_regime_lineplot_template.png"
    conda:
        "envs/pysr.yaml"
    params:
        dataset_size=config["dataset_sizes"]["pysr"],
        features=features
    shell:
        """
        echo "Running PySR on filtered noise regimes."
        python src/noise_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --mode filtered
        echo "PySR filtered noise regime evaluation completed."
        """

rule noise_regimes_full:
    input:
        dataset=f"data/{enzyme_model}/{data_type}/processed/data_test.csv"
    output:
        f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/regime_loss_comparison.png",
        f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/error_landscape.png",
        f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/results/pysr/all_pysr_formulas.txt",
        f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae_horizontal_boxplot.png",
        f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae_vertical_boxplot.png",
        f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae_horizontal_boxplot_no_outliers.png",
        f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae_vertical_boxplot_no_outliers.png",
        f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae_noise_regime_lineplot.png",
        f"data/{enzyme_model}/{data_type}/noise_regimes_full/shared/plots/log_mae_noise_regime_lineplot_template.png"
    conda:
        "envs/pysr.yaml"
    params:
        dataset_size=config["dataset_sizes"]["pysr"],
        features=features
    shell:
        """
        echo "Running PySR on full-dataset noise regimes."
        python src/noise_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features} --mode full
        echo "PySR full noise regime evaluation completed."
        """

rule mm_deviation_regimes:
    input:
        dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
    output:
        f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/regime_loss_comparison.png",
        f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/error_landscape.png",
        f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/results/pysr/all_pysr_formulas.txt",
        f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae_horizontal_boxplot.png",
        f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae_vertical_boxplot.png",
        f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae_horizontal_boxplot_no_outliers.png",
        f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae_vertical_boxplot_no_outliers.png",
        f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae_deviation_lineplot.png",
        f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae_deviation_lineplot_template.png"
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
