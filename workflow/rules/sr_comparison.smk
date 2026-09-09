# =============================================================================
# Synthetic-benchmark SR comparison
# =============================================================================
#
#   generate_data_model ──► preprocessing_model ──┬─► pysindy ────┐
#     mass-action ODE          train/val/test     ├─► aifeynman ──┤
#     simulation               splits             ├─► dso ────────┼─► get_best_formula
#                                                 ├─► kan ────────┤        │
#                                                 ├─► pysr ───────┤        ▼
#                                                 └─► nn ─────────┘  integrate_and_plot
#                                                    (MLP baseline)   plot_methods
#                                                                     sr_timepoint_lineplot
#
# SECTIONS
#   1  Default target and data     rule all, generate_data_model, preprocessing_model
#   2  Sweep arms                  per-method best-config lookups, when sweeps are on
#   3  SR method wrappers          the five methods plus the MLP baseline
#   4  Selection and plots         best formula, integration, comparison figures
#   5  Cross-model plots           nn_grid_search, pan_enzyme_model_plots
#
# Each method runs through `symbolic_regression_rule` (section 3), which builds one
# shell command: optional env install, then src/sr_models/<method>_model.py under a
# wall-clock timeout, seeded with 42. PySR additionally runs once per measurement-lens
# variant (sQSSA, tQSSA); the others run once.
#
# Configuration and paths live in common.smk. Regime-degradation rules are in
# regimes.smk and share this file's `enzyme_model != "experimental"` guard.
# =============================================================================

# --- 1. Default target and data ----------------------------------------------
#
# This block must stay FIRST in this file. On the synthetic branch neither common.smk
# nor experimental.smk declares any rule, so the default target is whichever rule is
# parsed first here.
#
# The guard is required, not decorative: the experimental branch declares its own
# `rule all` in common.smk, and a second unconditional one is a duplicate-rule error.
if enzyme_model != "experimental":

    rule all:
        input:
            f"data/{enzyme_model}/{data_type}/sr_comparison/plots/results_plot.png",
            f"data/{enzyme_model}/{data_type}/sr_comparison/plots/integrated_results_plot.png",
            f"data/{enzyme_model}/{data_type}/sr_comparison/results/loss_comparison.csv",
            f"data/{enzyme_model}/{data_type}/sr_comparison/results/rhs_loss_comparison.csv",
            f"data/{enzyme_model}/{data_type}/sr_comparison/plots/rhs_comparison.png",
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
            # pan_enzyme_model_plots writes under a {variant} directory, so these
            # must name the variant or they can never be satisfied.
            *[
                f"data/panmodel_plots/{variant}/{name}"
                for variant in variant_keys
                for name in (
                    "symbolic_model_r2_scores_bar.png",
                    "symbolic_model_r2_scores_scatter.png",
                    "all_symbolic_formulas.txt",
                )
            ]

    rule generate_data_model:
        output:
            train=f"data/{enzyme_model}/{data_type}/raw/train_data.csv",
            test=f"data/{enzyme_model}/{data_type}/raw/test_data.csv",
            valid=f"data/{enzyme_model}/{data_type}/raw/val_data.csv"
        shell:
            """
            # amici_env/ is a local virtualenv and is not in the repository.
            # Use it when present; otherwise rely on the ambient interpreter,
            # which needs simulation_requirements.txt installed (see README).
            if [ -f amici_env/bin/activate ]; then . amici_env/bin/activate; fi
            python src/synthetic/generate_data.py {enzyme_model} {data_type}
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
            python src/pipelines/sr_comparison/preprocessing.py --train {input.train} --test {input.test} --valid {input.valid} --output {output.merged} --train-output {output.train_split} --test-output {output.test_split} --target_feature {params.target_feature}
            """

if enzyme_model != "experimental":

    # --- 2. Sweep arms -------------------------------------------------------
    #
    # Only declared when a per-method best-config file exists, so a repo without
    # sweep results still builds.
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
    # --- 3. SR method wrappers -----------------------------------------------
    #
    # One shell command per method. `install_cmds` covers the three that cannot be
    # declared in a conda env: AI-Feynman needs f2py under setuptools<60, DSO pins
    # numpy 1.18, PySINDy is installed from the vendored tree.
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
        # Guarantee the temp file exists even if the method crashes or produces
        # nothing (e.g. AI-Feynman finding an empty Pareto front): run the method
        # tolerantly, then ensure a (possibly empty) temp file so the DAG proceeds
        # to get_best_formula instead of aborting with MissingOutputException. An
        # empty temp -> get_best_formula reports "no formula" and that method is
        # simply absent from the comparison.
        return f"""
            {env_path_fix} {install_command}{separator} timeout {config["timeout_duration"]} python src/sr_models/{model}_model.py --dataset {dataset} --dataset_size {dataset_size} --features {features} --temp_file {temp_file}{variant_flag}{seed_flag}{work_dir_flag} || true
            test -s {temp_file} || : > {temp_file}
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
            python src/shared/nn_model.py --dataset {input.train} --dataset_size {params.dataset_size} --features {params.features} --variant {params.variant} --output {output} || true
            """

    # Rule to extract the best formula from each temporary result file
    # --- 4. Selection and plots ----------------------------------------------
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

    # Direct (non-integrated) RHS fit: how well each formula predicts kcat_cg on
    # the test set, no ODE. Produces a CSV + bar-chart PNG.
    rule rhs_comparison:
        input:
            dataset=f"data/{enzyme_model}/{data_type}/processed/data_test.csv",
            formulas=formula_files
        output:
            csv=f"data/{enzyme_model}/{data_type}/sr_comparison/results/rhs_loss_comparison.csv",
            plot=f"data/{enzyme_model}/{data_type}/sr_comparison/plots/rhs_comparison.png"
        conda:
            "../../envs/base.yaml"
        params:
            methods=" ".join(models),
            features=features,
            discovery_scales="'" + config["discovery_scales"] + "'"
        shell:
            """
            echo "Computing direct RHS fit for methods: {params.methods}"
            [ -n "$CONDA_PREFIX" ] && export PATH="$CONDA_PREFIX/bin:$PATH"; PYTHONPATH=src python src/pipelines/sr_comparison/plot_rhs_comparison.py --dataset {input.dataset} --formulas {input.formulas} --methods {params.methods} --features {params.features} --discovery-scales {params.discovery_scales} --output {output.csv} --plot {output.plot}
            echo "RHS comparison saved to {output.csv}, plot {output.plot}"
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


# New rule to perform grid search for the NN model
# --- 5. Cross-model plots ----------------------------------------------------
#
# Module level, not inside the guard: these span enzyme models rather than
# belonging to one.
rule nn_grid_search:
    output:
        report=f"data/{enzyme_model}/{data_type}/sr_comparison/results/nn_grid_search/grid_search_results.txt"
    shell:
        """
        mkdir -p $(dirname {output.report})
        echo "Using previously determined NN hyperparameters." > {output.report}
        """

# Must match variant_keys (from SR_VARIANTS), which the default targets use.
PAN_PLOT_VARIANTS = list(variant_keys)

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
        "python src/pipelines/sr_comparison/pan_enzyme_model_plots.py --root-dir {input.root_dir} --discovery-scales {params.discovery_scales} --variant {wildcards.variant}"
