# Global configuration for feature selection, data type, dataset sizes, and timeout duration
configfile: "config.yaml"

# get the specific data type from the configuration file
data_type = config["data_type"]  # Example: data_type: 'static'

rule all:
    input:
        f"data/results/formula_pysindy_{data_type}.txt",
        f"data/results/formula_aifeynman_{data_type}.txt",
        f"data/results/formula_dso_{data_type}.txt",
        f"data/results/formula_kan_{data_type}.txt",
        f"data/results/formula_pysr_{data_type}.txt",
        "data/plots/results_plot.png",
        "data/plots/integrated_results_plot.png",
        "data/results/loss_comparison.csv"

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

rule pysindy:
    input:
        f"data/datasets/{data_type}_merged.csv"
    output:
        save=f"data/results/formula_pysindy_{data_type}.txt",
        temp_file=temporary(f"data/temp_results/temp_formula_pysindy_{data_type}.txt")
    conda:
        "envs/pysindy.yaml"
    params:
        dataset_size=config["dataset_sizes"]["pysindy"],
        features=config["features"][data_type]
    resources:
        time=config["timeout_duration"]
    shell:
        """
        # Run the PySINDy model with specified parameters and save the best formula to a temporary file
        python src/sr_models/pysindy_model.py --dataset {input} --dataset_size {params.dataset_size} --features {params.features} --temp_file {output.temp_file}

        # get the best formula specific to PySINDy from the temporary file
        python src/get_best_formula.py --hall_of_fame {output.temp_file} --save {output.save} --method pysindy
        """

rule aifeynman:
    input:
        f"data/datasets/{data_type}_merged.csv"
    output:
        save=f"data/results/formula_aifeynman_{data_type}.txt",
        temp_file=temporary(f"data/temp_results/temp_formula_aifeynman_{data_type}.txt")
    conda:
        "envs/aifeynman.yaml"
    params:
        dataset_size=config["dataset_sizes"]["aifeynman"],
        features=config["features"][data_type],
        timeout=config["timeout_duration"]
    shell:
        """
        # Install the AI Feynman model from PyPI without dependencies
        pip install --no-deps aifeynman

        # Run the AI Feynman model with a timeout using the 'timeout' command
        timeout {params.timeout} python src/sr_models/aifeynman_model.py --dataset {input} --dataset_size {params.dataset_size} --features {params.features} --temp_file {output.temp_file} || true

        # get the best formula specific to AI Feynman from the temporary file
        python src/get_best_formula.py --hall_of_fame {output.temp_file} --save {output.save} --method aifeynman
        """

rule dso:
    input:
        f"data/datasets/{data_type}_merged.csv"
    output:
        save=f"data/results/formula_dso_{data_type}.txt",
        temp_file=temporary(f"data/temp_results/dso_ExperimnetName_0_hof.csv")
    conda:
        "envs/dso.yaml"
    params:
        dataset_size=config["dataset_sizes"]["dso"],
        features=config["features"][data_type],
        timeout=config["timeout_duration"]
    shell:
        """
        # Install the DSO model from the GitHub repository 
        pip install absl-py==0.7.0
        pip install numpy==1.18
        pip install --no-deps "git+https://github.com/dso-org/deep-symbolic-optimization.git@master#egg=dso&subdirectory=dso/"
        python -c "from dso.task import set_task"
        # Run the DSO model with a timeout using the 'timeout' command
        timeout {params.timeout} python src/sr_models/dso_model.py --dataset {input} --dataset_size {params.dataset_size} --features {params.features} --temp_file {output.temp_file} || true

        # get the best formula specific to DSO from the temporary file
        python src/get_best_formula.py --hall_of_fame {output.temp_file} --save {output.save} --method dso
        """

rule kan:
    input:
        f"data/datasets/{data_type}_merged.csv"
    output:
        save=f"data/results/formula_kan_{data_type}.txt",
        temp_file=temporary(f"data/temp_results/temp_formula_kan_{data_type}.txt")
    conda:
        "envs/kan.yaml"
    params:
        dataset_size=config["dataset_sizes"]["kan"],
        features=config["features"][data_type],
        timeout=config["timeout_duration"]
    shell:
        """
        # Run the KAN model with a timeout using the 'timeout' command
        timeout {params.timeout} python src/sr_models/kan_model.py --dataset {input} --dataset_size {params.dataset_size} --features {params.features} --temp_file {output.temp_file} || true

        # get the best formula specific to KAN from the temporary file
        python src/get_best_formula.py --hall_of_fame {output.temp_file} --save {output.save} --method kan
        """
        
rule pysr:
    input:
        f"data/datasets/{data_type}_merged.csv"
    output:
        save=f"data/results/formula_pysr_{data_type}.txt",
        temp_file=temporary(f"data/temp_results/temp_formula_pysr_{data_type}.txt")
    conda:
        "envs/pysr.yaml"
    params:
        dataset_size=config["dataset_sizes"]["pysr"],
        features=config["features"][data_type],
        timeout=config["timeout_duration"]
    shell:
        """
        # Run the PySR model with a timeout using the 'timeout' command
        timeout {params.timeout} python src/sr_models/pysr_model.py --dataset {input} --dataset_size {params.dataset_size} --features {params.features} --temp_file {output.temp_file} || true

        # get the best formula specific to PySR from the temporary file
        python src/get_best_formula.py --hall_of_fame {output.temp_file} --save {output.save} --method pysr
        """

rule integrate_and_plot_results:
    input:
        dataset=f"data/datasets/{data_type}_merged.csv",
        formulas=[
            f"data/results/formula_pysindy_{data_type}.txt",
            f"data/results/formula_aifeynman_{data_type}.txt",
            f"data/results/formula_dso_{data_type}.txt",
            f"data/results/formula_kan_{data_type}.txt",
            f"data/results/formula_pysr_{data_type}.txt"
        ]
    output:
        csv="data/results/loss_comparison.csv",
        plot="data/plots/integrated_results_plot.png"
    params:
        methods=["pysindy", "aifeynman", "dso", "kan", "pysr"],
        data_proportion=config["data_proportion"],
        features=config["features"][config["data_type"]],
        trajectory_column="trajectory_id",
    run:
        shell(
            "python src/integrate_and_plot_methods.py "
            "--dataset {input.dataset} "
            "--formulas {input.formulas} "
            "--methods {params.methods} "
            "--data_proportion {params.data_proportion} "
            "--trajectory_column {params.trajectory_column} "
            "--output {output.csv} "
            "--plot {output.plot}"
        )

rule plot_methods:
    input:
        dataset=f"data/datasets/{data_type}_merged.csv",
        formulas=[
            f"data/results/formula_pysindy_{data_type}.txt",
            f"data/results/formula_aifeynman_{data_type}.txt",
            f"data/results/formula_dso_{data_type}.txt",
            f"data/results/formula_kan_{data_type}.txt",
            f"data/results/formula_pysr_{data_type}.txt"
        ]
    conda:
        "envs/base.yaml"
    output:
        "data/plots/results_plot.png"
    shell:
        "python src/plot_methods.py --dataset {input.dataset} --formulas {input.formulas} --output {output}"