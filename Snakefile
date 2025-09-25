# Global configuration for feature selection, data type, dataset sizes, and timeout duration
configfile: "config.yaml"

# Define data type and parameters from configuration
data_type = config["data_type"]  # Example: 'dynamic'
enzyme_model=config["enzyme_model"]
models = config["models"]  # Example: ['pysindy', 'aifeynman', 'dso', 'kan', 'pysr']
output_extension = config["output_extension"]
features = config["features"][data_type]

# Precompute paths for temporary and final formula files
temp_files = [
    f"data/{enzyme_model}/{data_type}/sr_comparison/temp/hall_of_fame.csv" if model == "pysr" 
    else f"data/{enzyme_model}/{data_type}/sr_comparison/temp/temp_formula_{model}.{output_extension[model]}" 
    for model in models
]
formula_files = [f"data/{enzyme_model}/{data_type}/sr_comparison/results/formula_{model}.txt" for model in models]

common_inputs = [
    temp_files,
    f"data/{enzyme_model}/{data_type}/sr_comparison/plots/results_plot.png",
    f"data/{enzyme_model}/{data_type}/sr_comparison/models/nn/model_nn.pth",
    f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/regime_loss_comparison.png",
    f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/error_landscape.png",
    f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/results/pysr/all_pysr_formulas.txt",
    f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae_horizontal_boxplot.png",
    f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/regime_loss_comparison.png",
    f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/error_landscape.png",
    f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/results/pysr/all_pysr_formulas.txt",
    f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae_horizontal_boxplot.png",
    f"data/{enzyme_model}/{data_type}/noise_regimes/shared/plots/regime_loss_comparison.png",
    f"data/{enzyme_model}/{data_type}/noise_regimes/shared/plots/error_landscape.png",
    f"data/{enzyme_model}/{data_type}/noise_regimes/shared/plots/log_mae_horizontal_boxplot.png",
    f"data/{enzyme_model}/{data_type}/noise_regimes/shared/results/pysr/all_pysr_formulas.txt",
    "data/panmodel_plots/symbolic_model_r2_scores_bar.png",
    "data/panmodel_plots/symbolic_model_r2_scores_scatter.png",
    "data/panmodel_plots/all_symbolic_formulas.txt"
]

# Add integrate-and-plot outputs only for static data
if data_type == "dynamic":
    common_inputs += [
        f"data/{enzyme_model}/{data_type}/sr_comparison/plots/integrated_results_plot.png",
        f"data/{enzyme_model}/{data_type}/sr_comparison/results/loss_comparison.csv"
    ]

rule all:
    input: common_inputs

if enzyme_model == "experimental":
    
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
            merged=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
        conda:
            "envs/base.yaml"
        params:
            target_feature=config["target_feature"]
        shell:
            """
            python src/preprocessing.py --train {input.train} --test {input.test} --valid {input.valid} --output {output.merged} --target_feature {params.target_feature}
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
        {install_command}{separator} python src/sr_models/{model}_model.py --dataset {dataset} --dataset_size {dataset_size} --features {features} --temp_file {temp_file} || true
    """

# Rules for symbolic regression for each model
rule pysindy:
    input:
        merged=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
    output:
        temp_file=temporary(f"data/{enzyme_model}/{data_type}/sr_comparison/temp/temp_formula_pysindy.txt")
    conda:
        "envs/pysindy.yaml"
    params:
        dataset_size=config["dataset_sizes"]["pysindy"],
        features=features
    shell:
        symbolic_regression_rule("pysindy", "{input.merged}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

rule aifeynman:
    input:
        merged=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
    output:
        temp_file=temporary(f"data/{enzyme_model}/{data_type}/sr_comparison/temp/temp_formula_aifeynman.txt")
    conda:
        "envs/aifeynman.yaml"
    params:
        dataset_size=config["dataset_sizes"]["aifeynman"],
        features=features
    shell:
        symbolic_regression_rule("aifeynman", "{input.merged}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

rule dso:
    input:
        merged=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
    output:
        temp_file=temporary(f"data/{enzyme_model}/{data_type}/sr_comparison/temp/temp_formula_dso.txt")
    conda:
        "envs/dso.yaml"
    params:
        dataset_size=config["dataset_sizes"]["dso"],
        features=features
    shell:
        symbolic_regression_rule("dso", "{input.merged}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

rule kan:
    input:
        merged=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
    output:
        temp_file=temporary(f"data/{enzyme_model}/{data_type}/sr_comparison/temp/temp_formula_kan.txt")
    conda:
        "envs/kan.yaml"
    params:
        dataset_size=config["dataset_sizes"]["kan"],
        features=features
    shell:
        symbolic_regression_rule("kan", "{input.merged}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

rule pysr:
    input:
        merged=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
    output:
        temp_file=temporary(f"data/{enzyme_model}/{data_type}/sr_comparison/temp/hall_of_fame.csv")
    conda:
        "envs/pysr.yaml"
    params:
        dataset_size=config["dataset_sizes"]["pysr"],
        features=features
    shell:
        symbolic_regression_rule("pysr", "{input.merged}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

rule nn:
    input:
        merged=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
    output:
        output=f"data/{enzyme_model}/{data_type}/sr_comparison/models/nn/model_nn.pth"
    conda:
        "envs/nn.yaml"
    params:
        dataset_size=config["dataset_sizes"]["nn"],
        features=features
    shell:
        """
        python src/nn_model.py --dataset {input.merged} --dataset_size {params.dataset_size} --features {params.features} --output {output.output} || true
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
if data_type == "dynamic":
    rule integrate_and_plot_results:
        input:
            dataset=lambda wildcards: f"data/{enzyme_model}/{data_type}/processed/data_merged.csv",
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
            python src/integrate_and_plot_methods.py --dataset {input.dataset} --formulas {input.formulas} --methods {params.methods}  --discovery-scales {params.discovery_scales} --data-proportion {params.data_proportion} --trajectory-column {params.trajectory_column} --output {output.csv} --plot {output.plot}
            echo "Integrated results saved to {output.csv}, plot saved to {output.plot}"
            """

# Rule to generate a comparison plot of all methods
rule plot_methods:
    input:
        dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv",
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
        python src/plot_methods.py --dataset {input.dataset} --formulas {input.formulas} --output {output} --discovery-scales {params.discovery_scales}
        echo "Comparison plot saved to {output}"
        """

rule kinetic_regimes:
    input:
        dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
    output:
        f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/regime_loss_comparison.png",
        f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/error_landscape.png",
        f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/results/pysr/all_pysr_formulas.txt",
        f"data/{enzyme_model}/{data_type}/kinetic_regimes/shared/plots/log_mae_horizontal_boxplot.png"
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
        dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
    output:
        f"data/{enzyme_model}/{data_type}/noise_regimes/shared/plots/regime_loss_comparison.png",
        f"data/{enzyme_model}/{data_type}/noise_regimes/shared/plots/error_landscape.png",
        f"data/{enzyme_model}/{data_type}/noise_regimes/shared/results/pysr/all_pysr_formulas.txt",
        f"data/{enzyme_model}/{data_type}/noise_regimes/shared/plots/log_mae_horizontal_boxplot.png"
    conda:
        "envs/pysr.yaml"
    params:
        dataset_size=config["dataset_sizes"]["pysr"],
        features=features
    shell:
        """
        echo "Running PySR on different noise regimes."
        python src/noise_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features}
        echo "PySR noise regime evaluation completed."
        """

rule mm_deviation_regimes:
    input:
        dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
    output:
        f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/regime_loss_comparison.png",
        f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/error_landscape.png",
        f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/results/pysr/all_pysr_formulas.txt",
        f"data/{enzyme_model}/{data_type}/mm_deviation_regimes/shared/plots/log_mae_horizontal_boxplot.png"
    conda:
        "envs/pysr.yaml"
    params:
        dataset_size=config["dataset_sizes"]["pysr"],
        features=features
    shell:
        """
        echo "Running PySR on different MM deviation regimes."
        python src/mm_deviation_regimes.py --dataset {input.dataset} --dataset_size {params.dataset_size} --features {params.features}
        echo "PySR noise regime evaluation completed."
        """

# New rule to perform grid search for the NN model
rule nn_grid_search:
    input:
        merged=f"data/processed/{data_type}_merged.csv"
    output:
        report=f"data/{enzyme_model}/{data_type}/sr_comparison/results/nn_grid_search/grid_search_results.txt"
    conda:
        "envs/nn.yaml"
    params:
        dataset_size=config["dataset_sizes"]["nn"],
        features=features,
        output_dir=f"data/{enzyme_model}/{data_type}/sr_comparison/results/nn_grid_search/"
    shell:
        """
        echo "Running grid search over NN hyperparameters."
        python src/nn_grid_search.py \
            --dataset {input.merged} \
            --dataset_size {params.dataset_size} \
            --features {params.features} \
            --output_dir {params.output_dir}
        echo "Grid search completed and saved to {output.report}"
        """

rule pan_enzyme_model_plots:
    input:
        formulas=formula_files,
        root_dir="data",
        dataset=f"data/{enzyme_model}/{data_type}/processed/data_merged.csv"
    output:
        "data/panmodel_plots/symbolic_model_r2_scores_bar.png",
        "data/panmodel_plots/symbolic_model_r2_scores_scatter.png",
        "data/panmodel_plots/all_symbolic_formulas.txt"
    params:
        discovery_scales="'" + config["discovery_scales"] + "'"
    conda:
        "envs/base.yaml"
    shell:
        "python src/pan_enzyme_model_plots.py --root-dir {input.root_dir} --dataset {input.dataset} --discovery-scales {params.discovery_scales}"
