# Global configuration for feature selection, data type, dataset sizes, and timeout duration
configfile: "config.yaml"

# Define data type and parameters from configuration
data_type = config["data_type"]  # Example: 'dynamic'
models = config["models"]  # Example: ['pysindy', 'aifeynman', 'dso', 'kan', 'pysr']
output_extension = config["output_extension"]
features = config["features"][data_type]

# Precompute paths for temporary and final formula files
temp_files = [f"data/temp_results/temp_formula_{model}_{data_type}.{output_extension[model]}" for model in models]
formula_files = [f"data/results/formula_{model}_{data_type}.txt" for model in models]

# Define output files for the entire workflow
rule all:
    input:
        temp_files,
        "data/plots/results_plot.png",
        "data/plots/integrated_results_plot.png",
        "data/results/loss_comparison.csv"

# Preprocessing rule to merge and process raw data
rule preprocessing:
    input:
        train=f"data/raw/train_{data_type}_data.csv",
        test=f"data/raw/test_{data_type}_data.csv",
        valid=f"data/raw/val_{data_type}_data.csv"
    output:
        merged=f"data/processed/{data_type}_merged.csv"
    conda:
        "envs/base.yaml"
    params:
        target_feature=config["target_feature"]
    shell:
        """
        echo "Starting preprocessing with train, test, and validation datasets"
        python src/preprocessing.py --train {input.train} --test {input.test} --valid {input.valid} --output {output.merged} --target_feature {params.target_feature}
        echo "Preprocessing completed: Merged dataset saved to {output.merged}"
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
        {install_command}{separator} timeout {config["timeout_duration"]} python src/sr_models/{model}_model.py --dataset {dataset} --dataset_size {dataset_size} --features {features} --temp_file {temp_file} || true
    """

# Rules for symbolic regression for each model
rule pysindy:
    input:
        merged=f"data/processed/{data_type}_merged.csv"
    output:
        temp_file=temporary(f"data/temp_results/temp_formula_pysindy_{data_type}.txt")
    conda:
        "envs/pysindy.yaml"
    params:
        dataset_size=config["dataset_sizes"]["pysindy"],
        features=features
    shell:
        symbolic_regression_rule("pysindy", "{input.merged}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

rule aifeynman:
    input:
        merged=f"data/processed/{data_type}_merged.csv"
    output:
        temp_file=temporary(f"data/temp_results/temp_formula_aifeynman_{data_type}.txt")
    conda:
        "envs/aifeynman.yaml"
    params:
        dataset_size=config["dataset_sizes"]["aifeynman"],
        features=features
    shell:
        symbolic_regression_rule("aifeynman", "{input.merged}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

rule dso:
    input:
        merged=f"data/processed/{data_type}_merged.csv"
    output:
        temp_file=temporary(f"data/temp_results/temp_formula_dso_{data_type}.txt")
    conda:
        "envs/dso.yaml"
    params:
        dataset_size=config["dataset_sizes"]["dso"],
        features=features
    shell:
        symbolic_regression_rule("dso", "{input.merged}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

rule kan:
    input:
        merged=f"data/processed/{data_type}_merged.csv"
    output:
        temp_file=temporary(f"data/temp_results/temp_formula_kan_{data_type}.txt")
    conda:
        "envs/kan.yaml"
    params:
        dataset_size=config["dataset_sizes"]["kan"],
        features=features
    shell:
        symbolic_regression_rule("kan", "{input.merged}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

rule pysr:
    input:
        merged=f"data/processed/{data_type}_merged.csv"
    output:
        temp_file=temporary(f"data/temp_results/temp_formula_pysr_{data_type}.csv")
    conda:
        "envs/pysr.yaml"
    params:
        dataset_size=config["dataset_sizes"]["pysr"],
        features=features
    shell:
        symbolic_regression_rule("pysr", "{input.merged}", "{params.dataset_size}", "{params.features}", "{output.temp_file}")

rule nn:
    input:
        merged=f"data/processed/{data_type}_merged.csv"
    output:
        temp_file=temporary(f"data/temp_results/temp_model_nn_{data_type}.pth")
    conda:
        "envs/nn.yaml"
    params:
        dataset_size=config["dataset_sizes"]["nn"],
        features=features
    shell:
        """
        timeout {config["timeout_duration"]} python src/nn_model.py --dataset {input.merged} --dataset_size {params.dataset_size} --features {params.features} --temp_file {output.temp_file} || true
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
        dataset=lambda wildcards: f"data/processed/{data_type}_merged.csv",
        formulas=formula_files
    output:
        csv="data/results/loss_comparison.csv",
        plot="data/plots/integrated_results_plot.png"
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
        dataset=f"data/processed/{data_type}_merged.csv",
        formulas=formula_files
    output:
        "data/plots/results_plot.png"
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