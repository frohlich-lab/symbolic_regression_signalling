# Global configuration for feature selection, data type, dataset sizes, and timeout duration
configfile: "config.yaml"

# Define data type from configuration
data_type = config["data_type"]  # Example: data_type: 'dynamic'
models = config["models"] # Example: models: ['pysindy', 'aifeynman', 'dso', 'kan', 'pysr']
output_extension = config["output_extension"]  # Example: output_extension: {'pysindy': 'txt', 'aifeynman': 'txt', 'dso': 'txt', 'kan': 'txt', 'pysr': 'csv'}
features = config["features"][data_type]  

# Precompute input and output paths with the correct file extensions
temp_files = [f"data/temp_results/temp_formula_{model}_{data_type}.{output_extension[model]}" for model in models]
formula_files = [f"data/results/formula_{model}_{data_type}.txt" for model in models]

# Define the output files expected by the workflow
rule all:
    input:
        temp_files,
        "data/plots/results_plot.png",
        "data/plots/integrated_results_plot.png",
        "data/results/loss_comparison.csv"

# Preprocessing rule to merge datasets
rule preprocessing:
    input:
        train=f"../base_models/two_step_enzyme/output_simu_data/train_{data_type}_data.csv",
        test=f"../base_models/two_step_enzyme/output_simu_data/test_{data_type}_data.csv",
        valid=f"../base_models/two_step_enzyme/output_simu_data/val_{data_type}_data.csv"
    output:
        f"data/datasets/{data_type}_merged.csv"
    conda:
        "envs/preprocessing.yaml"
    params:
        target_feature=config["target_feature"]
    shell:
        "python src/preprocessing.py --train {input.train} --test {input.test} --valid {input.valid} --output {output} --target_feature {params.target_feature}"

# Function to generate shell command with timeout and additional installations
def symbolic_regression_rule(model, dataset, dataset_size, features, temp_file):
    install_cmds = {
        "aifeynman": "pip install --no-deps aifeynman",
        "dso": (
            "pip install absl-py==0.7.0 && "
            "pip install numpy==1.18 && "
            "pip install --no-deps 'git+https://github.com/dso-org/deep-symbolic-optimization.git@master#egg=dso&subdirectory=dso/'"
        ),
    }
    install_command = install_cmds.get(model, "")
    separator = ";" if install_command else ""
    return f"""
        {install_command}{separator} gtimeout {config["timeout_duration"]} python src/sr_models/{model}_model.py --dataset {dataset} --dataset_size {dataset_size} --features {features} --temp_file {temp_file} || true
    """

# PySINDy rule for symbolic regression
rule pysindy:
    input:
        f"data/datasets/{data_type}_merged.csv"
    output:
        temp_file=temporary(f"data/temp_results/temp_formula_pysindy_{data_type}.txt")
    conda:
        "envs/pysindy.yaml"
    params:
        dataset_size=config["dataset_sizes"]["pysindy"],
        features=features
    shell:
        symbolic_regression_rule("pysindy", "{input}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

# Repeat similar rule structure for other models
rule aifeynman:
    input:
        f"data/datasets/{data_type}_merged.csv"
    output:
        temp_file=temporary(f"data/temp_results/temp_formula_aifeynman_{data_type}.txt")
    conda:
        "envs/aifeynman.yaml"
    params:
        dataset_size=config["dataset_sizes"]["aifeynman"],
        features=features
    shell:
        symbolic_regression_rule("aifeynman", "{input}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

rule dso:
    input:
        f"data/datasets/{data_type}_merged.csv"
    output:
        temp_file=temporary(f"data/temp_results/temp_formula_dso_{data_type}.txt")
    conda:
        "envs/dso.yaml"
    params:
        dataset_size=config["dataset_sizes"]["dso"],
        features=features
    shell:
        symbolic_regression_rule("dso", "{input}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

rule kan:
    input:
        f"data/datasets/{data_type}_merged.csv"
    output:
        temp_file=temporary(f"data/temp_results/temp_formula_kan_{data_type}.txt")
    conda:
        "envs/kan.yaml"
    params:
        dataset_size=config["dataset_sizes"]["kan"],
        features=features
    shell:
        symbolic_regression_rule("kan", "{input}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

rule pysr:
    input:
        f"data/datasets/{data_type}_merged.csv"
    output:
        temp_file=temporary(f"data/temp_results/temp_formula_pysr_{data_type}.csv")
    conda:
        "envs/pysr.yaml"
    params:
        dataset_size=config["dataset_sizes"]["pysr"],
        features=features
    shell:
        symbolic_regression_rule("pysr", "{input}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

# Rule to get the best formula from the temporary files for all models
rule get_best_formula:
    input:
        temp_files=temp_files
    output:
        formula_files=formula_files
    conda:
        "envs/base.yaml"
    shell:
        """
        python src/get_best_formula.py --method {models} --hall_of_fame {input} --save {output} --features {features}
        """

# Rule to integrate and plot results from all models
rule integrate_and_plot_results:
    input:
        dataset=lambda wildcards: f"data/datasets/{data_type}_merged.csv" if data_type == "dynamic" else None,
        formulas=formula_files if data_type == "dynamic" else None
    output:
        csv="data/results/loss_comparison.csv",
        plot="data/plots/integrated_results_plot.png"
    conda:
        "envs/base.yaml"
    params:
        methods=["pysindy", "aifeynman", "dso", "kan", "pysr"],
        data_proportion=config["data_proportion"],
        features=features,
        trajectory_column="trajectory_id"
    shell:
        """
        python src/integrate_and_plot_methods.py --dataset {input.dataset} --formulas {input.formulas} --methods {params.methods} --data_proportion {params.data_proportion} --trajectory_column {params.trajectory_column} --output {output.csv} --plot {output.plot}
        """

# Rule to generate a comparison plot of all methods
rule plot_methods:
    input:
        dataset=f"data/datasets/{data_type}_merged.csv",
        formulas=formula_files
    output:
        "data/plots/results_plot.png"
    conda:
        "envs/base.yaml"
    shell:
        "python src/plot_methods.py --dataset {input.dataset} --formulas {input.formulas} --output {output}"