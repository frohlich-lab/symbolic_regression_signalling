# =============================================================================
# Data-degradation regimes
# =============================================================================
#
# How well SR recovers a known law as the data get worse. One axis is degraded at a
# time, and each axis is crossed with the tQSSA measurement lens.
#
#   data_merged.csv ──┬─► <axis>_regimes ──────► all_pysr_formulas.txt
#                     │     (search only)
#                     └─► <axis>_regime_plots ─► error landscape, correlation grids,
#                           (search + render)     log-MAE boxplots, lineplots
#
# AXES               DIRECTORY                  VARIED
#   kinetic          kinetic_regimes/           position in kcat/Km space
#   noise            noise_regimes_full/        measurement noise level
#   mm_deviation     mm_deviation_regimes/      departure from MM assumptions
#   dataset_size     dataset_size_regimes/      number of training samples
#   timepoint        timepoint_regimes/         timepoint coverage (dynamic only)
#
# Each axis has four rules: search and plots, then `_tqssa` copies of both that rerun
# under the total-substrate lens. Search and plots are separate because the search is
# the expensive half and the figures get re-cut far more often -- the plots rules pass
# `--plots-only` and reuse the formulas already on disk.
#
# Outputs land under data/<enzyme_model>/<data_type>/<axis>/shared/. Configuration and
# paths live in common.smk. Shares the `enzyme_model != "experimental"` guard with
# sr_comparison.smk, and depends on that file's preprocessing_model for its input.
# =============================================================================

if enzyme_model != "experimental":

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
