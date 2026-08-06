# =============================================================================
# Draft SR-MM v5 — experimental ERK pipeline
# =============================================================================
#
# The chain the manuscript's experimental results actually come from, in order:
#
#   markers_per_minute_fit.csv            (experimental_marker_inputs, below)
#        |
#        +-- pysr_config_sweep            288 arms x 6 development contexts x 3 seeds
#        |        |                       -> screen.csv                     [expensive]
#        |        +-- select_pysr_config   train-ranked; 72 candidate arms
#        |                 |               -> Table S10 + selected_config.tsv
#        |                 v
#        +-- pysr_ood_final               selected config, all 40 contexts x 3 seeds
#        |        |                       -> per-seed formulas + metrics     [expensive]
#        |        +-- summarise_pysr_ood   -> success rates, Fig. 5 exemplars
#        |
#        +-- linreg_ood                   OLS on all ten inputs, same split
#        |
#        +-- sparse_neural_ode            L21 / L1 / C-NODE variants        [expensive]
#                 |
#                 +-- paper figures
#
# Every stage is scored on ONE split (top 20% of GFP bins held out) and ONE metric (R2
# of the ODE-integrated p-ERK trajectory, best of seeds 42/43/44). Mixing in the
# derivative-fit R2 or the in-distribution split changes the conclusions, so neither is
# used anywhere below. See docs/experimental_provenance.md for the figure-by-figure map.
#
# The two stages marked [expensive] are ~3,000 and ~70 CPU-hours and were run on NEMO
# via the sharding helpers in src/pipelines/experimental/sweeps/. Their outputs are
# committed under data/experimental/runs/, so Snakemake treats them as up to date and
# only the cheap downstream stages re-run. Deleting an output triggers a full re-run;
# do that deliberately, and on a cluster.

if enzyme_model == "experimental":


    # ------------------------------------------------- 1. PySR configuration sweep

    rule experimental_pysr_config_sweep:
        # EXPENSIVE (~3,000 CPU-hours: 288 arms x 6 contexts x 3 seeds, ~34 min/fit).
        # Run on a cluster. The shipped screen CSV is this rule's recorded output, so
        # Snakemake will not rebuild it unless you delete it.
        #
        # 288 arms are run; 72 are selection candidates (see the next rule). The wider
        # grid also varies the unary-operator and no-division axes, which is how we know
        # the candidate family is the right one to restrict to rather than an assumption.
        input:
            # ancient(): these outputs were produced out of band on the cluster, so their
            # mtimes bear no relation to the inputs' and a plain dependency would offer
            # to spend 3,000 CPU-hours rebuilding a committed artefact on any touch.
            # Rebuild deliberately by deleting the output.
            per_minute=ancient(marker_per_minute_csv),
            snapshot=ancient(marker_fit_snapshot_csv)
        output:
            screen=pysr_sweep_screen
        conda:
            "../../envs/pysr.yaml"
        params:
            sweep_dir=pysr_sweep_dir,
            contexts=" ".join(pysr_sweep_dev_contexts),
            seeds=" ".join(str(s) for s in pysr_final_seeds),
        shell:
            """
            set -euo pipefail
            mkdir -p {params.sweep_dir}/arms {params.sweep_dir}/metrics

            # Arm definitions. Stability penalties stay at their defaults in every arm:
            # they encode the dynamical-admissibility prior (d[p-ERK]/dt decreasing in
            # p-ERK), so relaxing them to gain R2 would be tuning away the prior.
            python src/pipelines/experimental/sweeps/make_grid_arms.py \
                {params.sweep_dir}/arms/arms.tsv

            # One shard per (context, seed) so the grid can be run as a job array.
            python src/pipelines/experimental/sweeps/make_shards.py \
                {params.sweep_dir} --markers {params.contexts}

            # Shape-aware screen: collects held-out AND training integrated R2, decay
            # error and fan ordering for every fit into one table.
            python src/pipelines/experimental/sweeps/screen_sweep.py \
                {params.sweep_dir} {output.screen}
            """

    rule experimental_pysr_select_config:
        # Cheap and deterministic. Ranks the 72 candidate arms on median integrated R2
        # over the TRAINING GFP bins only, so the held-out highest-dose bins play no
        # part in the choice. Emits Table S10 and the flag string every later PySR run
        # is driven by, rather than leaving that string duplicated across scripts.
        input:
            screen=pysr_sweep_screen
        output:
            table=pysr_sweep_table,
            config=pysr_sweep_config
        conda:
            "../../envs/pysr.yaml"
        shell:
            """
            python src/pipelines/experimental/sr_pipeline/select_pysr_config.py \
                --screen {input.screen} \
                --output-table {output.table} \
                --output-config {output.config}
            """

    # ---------------------------------------------- 2. frozen-config run, all 40

    rule experimental_pysr_ood_final:
        # EXPENSIVE (~70 CPU-hours: 40 contexts x 3 seeds, ~35 min/fit). Applies the
        # selected configuration UNCHANGED to all 40 contexts -- including the 34 the
        # sweep never saw. This separation is the point: an earlier iteration reported
        # sweep-derived numbers as general and they did not survive contact with the
        # unseen contexts.
        input:
            # ancient() for the same reason as the sweep above: this is a recorded
            # 70-CPU-hour cluster run, not something to rebuild because a timestamp
            # moved. A genuine configuration change therefore does NOT re-trigger it --
            # delete the per-seed outputs when you mean to re-run.
            config=ancient(pysr_sweep_config),
            per_minute=ancient(marker_per_minute_csv),
            snapshot=ancient(marker_fit_snapshot_csv)
        output:
            formulas=pysr_final_formulas,
            metrics=pysr_final_metrics
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=pysr_final_dir,
            group_definitions_csv=exp_group_definitions_csv,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            seeds=" ".join(str(s) for s in pysr_final_seeds),
        shell:
            """
            set -euo pipefail
            FLAGS=$(tail -n1 {input.config} | cut -f3)
            echo "frozen PySR configuration: $FLAGS"

            # run_markers.py picks its output layout from whether seeds/ and aggregated/
            # already exist under --output-dir. Without them it writes a reports/ layout
            # that omits the predicted-trajectory file the integration stage needs.
            mkdir -p {params.out_dir}/aggregated
            for seed in {params.seeds}; do
                mkdir -p {params.out_dir}/seeds/seed_${{seed}}
            done

            for seed in {params.seeds}; do
                # `set -f` because arm flags contain `*`, which bash would otherwise
                # expand into repo filenames and hand to PySR as operator names.
                set -f
                python src/pipelines/experimental/sr_pipeline/run_markers.py \
                    --dataset {input.snapshot} \
                    --per-minute-dataset {input.per_minute} \
                    --output-dir {params.out_dir} \
                    --group-definitions-csv {params.group_definitions_csv} \
                    --models pysr \
                    --measured-timepoints {params.measured} \
                    --seeds ${{seed}} \
                    --test-split-policy top_gfp_bins \
                    $FLAGS
                set +f

                # run_markers.py does not integrate; the trajectory metrics that every
                # reported number is based on come from this separate stage.
                python src/pipelines/experimental/sr_pipeline/compute_marker_integration.py \
                    --dataset {input.per_minute} \
                    --sr-trajectories {params.out_dir}/seeds/seed_${{seed}}/trajectories/predicted_trajectories_per_minute.csv \
                    --output-dir {params.out_dir}/seeds/seed_${{seed}}/metrics \
                    --measured-timepoints {params.measured}
            done
            """

    rule experimental_pysr_ood_summary:
        # The reported success rates and the Fig. 5 exemplar ranking. A context counts
        # as solved only if the same fit clears the threshold on both the training doses
        # and the held-out ones -- see the module docstring for why the "both" clause
        # does real work.
        input:
            metrics=pysr_final_metrics
        output:
            per_fit=pysr_final_per_fit,
            summary=pysr_final_summary,
            exemplars=pysr_final_exemplars
        conda:
            "../../envs/pysr.yaml"
        params:
            run_dir=pysr_final_dir,
            threshold=exp_r2_threshold,
        shell:
            """
            python src/pipelines/experimental/sr_pipeline/summarise_pysr_ood.py \
                --run-dir {params.run_dir} \
                --output-metrics {output.per_fit} \
                --output-summary {output.summary} \
                --output-exemplars {output.exemplars} \
                --r2-threshold {params.threshold}
            """

    # The Neural ODE sweep helper, shared by the three regulariser variants and the
    # lambda sweep below. Defined here rather than beside the variants because Python
    # executes this file top to bottom: a helper used by the first rule cannot be
    # declared after it.

    # ---------------------------------------------------------------------
    # Paper figures: three OOD regulariser sweeps + plotting rules.
    # Each sweep trains a model per (marker, seed) under top_gfp_bins split
    # and saves the model checkpoints + per-seed metrics CSVs needed by the
    # paper figure scripts.
    # ---------------------------------------------------------------------
    def _diffrax_sweep_shell(seeds_dir, tag, extra_flags):
        return f"""
        mkdir -p {{params.out_dir}}
        for seed in {{params.seeds}}; do
            mkdir -p {{params.out_dir}}/seed_${{{{seed}}}}
            python src/pipelines/experimental/sr_pipeline/neural_ode_diffrax_baseline.py \\
                --dataset {{input.per_minute}} \\
                --output-dir {{params.out_dir}}/seed_${{{{seed}}}} \\
                --seeds ${{{{seed}}}} \\
                --epochs 200 --patience 20 \\
                --per-minute-max-time {{params.max_time}} \\
                --per-minute-sampling-strategy {{params.strategy}} \\
                --late-sample-window {{params.late_window_start}} {{params.late_window_end}} \\
                --late-sample-points {{params.late_points}} \\
                --measured-timepoints {{params.measured}} \\
                --save-models \\
                --test-split-policy top_gfp_bins \\
                {extra_flags} \\
                --tag {tag}_seed${{{{seed}}}}
        done
        """

    _diffrax_sweep_params = dict(
        runs_root=exp_runs_root,
        seeds="42 43 44",
        measured=" ".join(str(t) for t in exp_measured_timepoints),
        max_time=exp_per_minute_max_time,
        strategy=exp_per_minute_sampling_strategy if 'exp_per_minute_sampling_strategy' in globals() else "early_plus_sparse_late",
        late_window_start=exp_late_sample_window[0] if 'exp_late_sample_window' in globals() else 30.0,
        late_window_end=exp_late_sample_window[1] if 'exp_late_sample_window' in globals() else 60.0,
        late_points=exp_late_sample_points if 'exp_late_sample_points' in globals() else 15,
    )

    # ---------------------------------------------------- 3. sparse Neural ODE SI

    rule experimental_sparse_node_lambda_sweep:
        # Fig. S3B. Sweeps lambda_jac and picks by an elbow rule -- the largest
        # lambda_jac within 0.05 of the best mean IN-DISTRIBUTION validation R2, giving
        # lambda_jac = 3.0. Because a stronger penalty yields a sparser model, this rule
        # favours the sparsest network that is not measurably worse, so the reported
        # dependency counts are conservative with respect to sparsity.
        input:
            per_minute=marker_per_minute_csv
        output:
            summary=sparse_node_lambda_sweep_csv
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=sparse_node_lambda_sweep_dir,
            lambdas=" ".join(f"{v:g}" for v in sparse_node_lambda_values),
            # _diffrax_sweep_params already carries `seeds`; do not re-declare it here.
            **_diffrax_sweep_params,
        shell:
            """
            set -euo pipefail
            mkdir -p {params.out_dir}
            for lam in {params.lambdas}; do
                tag=$(echo "lam_${{lam}}" | tr '.' 'p')
                for seed in {params.seeds}; do
                    mkdir -p {params.out_dir}/${{tag}}/seed_${{seed}}
                    python src/pipelines/experimental/sr_pipeline/neural_ode_diffrax_baseline.py \
                        --dataset {input.per_minute} \
                        --output-dir {params.out_dir}/${{tag}}/seed_${{seed}} \
                        --seeds ${{seed}} \
                        --epochs 200 --patience 20 \
                        --per-minute-max-time {params.max_time} \
                        --per-minute-sampling-strategy {params.strategy} \
                        --late-sample-window {params.late_window_start} {params.late_window_end} \
                        --late-sample-points {params.late_points} \
                        --measured-timepoints {params.measured} \
                        --save-models \
                        --test-split-policy top_gfp_bins \
                        --jac-reg ${{lam}} --jac-reg-mode l21 --hess-reg 0.0 \
                        --tag ${{tag}}_seed${{seed}}
                done
            done
            python src/pipelines/experimental/sweeps/summarise_lambda_sweep.py \
                --sweep-dir {params.out_dir} \
                --output {output.summary}
            """

    # ------------------------------------------------------------ 4. paper figures

    rule experimental_paper_fig_parsimony_tradeoff:
        # Fig. 4E + the sparsity panel. Both axes are held-out ODE-integrated R2:
        # --pysr-integ-csv is what makes PySR's axis comparable to the network's
        # trajectory R2. Without it PySR is plotted on its derivative-fit R2, which is a
        # different quantity (the two correlate ~0.5) and was how the earlier version of
        # this panel came to mix metrics.
        input:
            nn_done=sparse_node_l21_done,
            pysr_formulas=pysr_final_formulas,
            pysr_integ=pysr_final_per_fit,
            dataset=marker_per_minute_csv
        output:
            fig=paper_fig_parsimony_tradeoff
        conda:
            "../../envs/pysr.yaml"
        params:
            nn_dir=sparse_node_l21_dir,
            pysr_dir=f"{pysr_final_dir}/seeds",
            excludes=exp_exclude_flags,
            threshold=exp_r2_threshold,
        shell:
            """
            mkdir -p $(dirname {output.fig})
            python src/pipelines/experimental/sr_pipeline/paper_figures/plot_parsimony_tradeoff.py \
                --nn-dir {params.nn_dir} \
                --pysr-dir {params.pysr_dir} \
                --pysr-integ-csv {input.pysr_integ} \
                --dataset {input.dataset} \
                --r2-threshold {params.threshold} \
                {params.excludes} \
                --output {output.fig}
            """

    rule experimental_paper_fig_cutoff_robustness:
        # Seed robustness, between-method agreement and driver-set size against the
        # top-X% Jacobian-mass cutoff used to read dependencies off the network.
        input:
            nn_done=sparse_node_l21_done,
            pysr_formulas=pysr_final_formulas,
            dataset=marker_per_minute_csv
        output:
            fig=paper_fig_cutoff_robustness
        conda:
            "../../envs/pysr.yaml"
        params:
            nn_dir=sparse_node_l21_dir,
            pysr_dir=f"{pysr_final_dir}/seeds",
        shell:
            """
            mkdir -p $(dirname {output.fig})
            python src/pipelines/experimental/sr_pipeline/paper_figures/plot_cutoff_robustness.py \
                --nn-dir {params.nn_dir} \
                --pysr-dir {params.pysr_dir} \
                --dataset {input.dataset} \
                --output {output.fig}
            """

    rule experimental_paper_fig_nn_appendix:
        # Fig. S3A / Table S8: L21 vs L1 vs C-NODE. The point is a null -- median
        # held-out R2 0.617 / 0.629 / 0.610, indistinguishable (Wilcoxon p = 0.19 and
        # 0.29 against L21) -- so the reported dependency counts are not an artefact of
        # the particular penalty. L21 is preferred on structural grounds, not empirical
        # ones: group sparsity drives whole inputs to zero, which is what makes the
        # participation ratio readable as a count of variables used.
        input:
            l1=sparse_node_l1_done,
            l21=sparse_node_l21_done,
            cnode=sparse_node_cnode_done
        output:
            fig=paper_fig_nn_appendix
        conda:
            "../../envs/pysr.yaml"
        params:
            l1_dir=sparse_node_l1_dir,
            l21_dir=sparse_node_l21_dir,
            cnode_dir=sparse_node_cnode_dir,
        shell:
            """
            mkdir -p $(dirname {output.fig})
            python src/pipelines/experimental/sr_pipeline/paper_figures/plot_nn_appendix_comparison.py \
                --l1-dir {params.l1_dir} \
                --l21-dir {params.l21_dir} \
                --cnode-dir {params.cnode_dir} \
                --output {output.fig}
            """

    rule experimental_paper_fig_sr_vs_linreg:
        # Fig. 4D (k=10, the full ten-input pool -- the toughest baseline) and Fig. S4
        # (k=4, complexity-matched to PySR's median equation). Controls are included
        # here, unlike in the sparsity panel; PySR wins at every k tested.
        input:
            linreg=select_k_metrics,
            pysr_formulas=pysr_final_formulas,
            pysr_integ=pysr_final_per_fit
        output:
            full=paper_fig_sr_vs_linreg,
            matched=paper_fig_sr_vs_linreg_k4
        conda:
            "../../envs/pysr.yaml"
        params:
            pysr_dir=f"{pysr_final_dir}/seeds",
            k_full=linreg_k_full,
            k_matched=linreg_k_matched,
        shell:
            """
            set -euo pipefail
            mkdir -p $(dirname {output.full})
            for spec in "{params.k_full}:{output.full}" "{params.k_matched}:{output.matched}"; do
                k=${{spec%%:*}}
                out=${{spec#*:}}
                python src/pipelines/experimental/sr_pipeline/paper_figures/plot_sr_vs_linreg_ood.py \
                    --linreg-metrics {input.linreg} \
                    --pysr-dir {params.pysr_dir} \
                    --pysr-integ-csv {input.pysr_integ} \
                    --linreg-k ${{k}} \
                    --output ${{out}}
            done
            """

    # ------------------------------------------------------------- 5. aggregators

    rule experimental_v5_all:
        # Everything the manuscript's experimental sections depend on.
        #   snakemake --use-conda -j1 --config enzyme_model=experimental experimental_v5_all
        #
        # Editing envs/pysr.yaml makes Snakemake offer to rebuild every conda-backed
        # stage, including the expensive PySR ones -- `ancient()` guards mtime, not
        # environment hashes. Add `--rerun-triggers mtime` to rebuild only what actually
        # changed on disk.
        input:
            # configuration provenance
            pysr_sweep_table,
            pysr_sweep_config,
            # frozen-config results
            pysr_final_per_fit,
            pysr_final_summary,
            pysr_final_exemplars,
            # main text
            paper_fig_parsimony_tradeoff,
            paper_fig_sr_vs_linreg,
            # SI
            paper_fig_sr_vs_linreg_k4,
            paper_fig_nn_appendix,
            paper_fig_cutoff_robustness,
            custom_loss_ablation_metrics,

    rule experimental_v5_tables:
        # Just the cheap numeric stages, for when you want to re-derive the reported
        # numbers without touching a figure or a GPU.
        input:
            pysr_sweep_table,
            pysr_sweep_config,
            pysr_final_per_fit,
            pysr_final_summary,
            pysr_final_exemplars,

    # ---------------------------------------------------------------------
    # Data preparation, the in-distribution reference run, and the shared
    # experimental stages. Relocated verbatim from common.smk, which had kept
    # rules for all three domains while sr_comparison.smk and regimes.smk sat
    # empty. `rule all` stays in common.smk: it has to be the first rule the
    # workflow parses to remain the default target.
    # ---------------------------------------------------------------------

    rule experimental_marker_inputs:
        input:
            raw=exp_raw_time_course
        output:
            time_series=marker_time_series_csv,
            feature_matrix=marker_feature_matrix_csv,
            fit_snapshot=marker_fit_snapshot_csv,
            per_minute=marker_per_minute_csv
        conda:
            "../../envs/pysr.yaml"
        params:
            output_dir=marker_base_dir,
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
                python src/pipelines/experimental/data_prep/marker_inputs.py \
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
            time=marker_time_series_csv,
            dense=marker_per_minute_csv
        output:
            done=fit_sanity_done
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=fit_sanity_dir,
            protein=exp_fit_sanity_protein,
            marker_flag=(f"--markers {exp_fit_sanity_markers_args}" if exp_fit_sanity_markers_args else ""),
            max_cols=exp_fit_sanity_max_columns,
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/data_prep/fit_sanity_plots.py \
                --time-trajectories {input.time} \
                --fit-trajectories {input.dense} \
                --output-dir {params.out_dir} \
                --protein "{params.protein}" \
                --max-columns {params.max_cols} \
                {params.marker_flag}
            touch {output.done}
            """

    rule experimental_marker_sr:
        input:
            dataset=marker_fit_snapshot_csv,
            per_minute=marker_per_minute_csv
        output:
            summary=marker_summary_output,
            seed_summary=marker_summary_seed_output,
            formulas=marker_formula_files
        conda:
            "../../envs/pysr.yaml"
        params:
            output_dir=exp_sr_output_dir,
            group_definitions_csv=exp_group_definitions_csv,
            models=exp_models_args,
            measured=" ".join(str(t) for t in exp_measured_timepoints)
        shell:
            """
            mkdir -p {params.output_dir}
            python src/pipelines/experimental/sr_pipeline/run_markers.py \
                --dataset {input.dataset} \
                --per-minute-dataset {input.per_minute} \
                --output-dir {params.output_dir} \
                --group-definitions-csv {params.group_definitions_csv} \
                --models {params.models} \
                --measured-timepoints {params.measured}
            """

    rule experimental_custom_loss_ablation:
        input:
            dataset=marker_fit_snapshot_csv,
            per_minute=marker_per_minute_csv
        output:
            metrics=custom_loss_ablation_metrics,
            boxplot=custom_loss_ablation_boxplot,
            boxplot_svg=custom_loss_ablation_boxplot_svg
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=custom_loss_ablation_dir,
            group_definitions_csv=exp_group_definitions_csv,
            seed=exp_custom_loss_ablation_seed,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            max_time=exp_per_minute_max_time,
            strategy=exp_per_minute_sampling_strategy,
            late_window_start=exp_late_sample_window[0],
            late_window_end=exp_late_sample_window[1],
            late_points=exp_late_sample_points,
            metric_column=exp_custom_loss_ablation_metric,
            metric_dataset_mode=exp_custom_loss_ablation_dataset_mode,
            metric_feature_mode=exp_custom_loss_ablation_feature_mode,
            metric_source=exp_custom_loss_ablation_metric_source,
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/experiments/pysr_custom_loss_ablation.py \
                --dataset {input.dataset} \
                --per-minute-dataset {input.per_minute} \
                --group-definitions-csv {params.group_definitions_csv} \
                --output-dir {params.out_dir} \
                --seed {params.seed} \
                --measured-timepoints {params.measured} \
                --per-minute-max-time {params.max_time} \
                --per-minute-sampling-strategy {params.strategy} \
                --late-sample-window {params.late_window_start} {params.late_window_end} \
                --late-sample-points {params.late_points} \
                --metric-column {params.metric_column} \
                --metric-dataset-mode {params.metric_dataset_mode} \
                --metric-feature-mode {params.metric_feature_mode} \
                --metric-source {params.metric_source} \
                --output-csv {output.metrics} \
                --output-plot {output.boxplot} \
                --output-plot-svg {output.boxplot_svg}
            """

    rule experimental_select_k_linreg:
        input:
            per_minute=marker_per_minute_csv
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
            "../../envs/pysr.yaml"
        params:
            out_dir=select_k_dir,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            max_time=exp_per_minute_max_time,
            strategy=exp_per_minute_sampling_strategy if 'exp_per_minute_sampling_strategy' in globals() else "early_plus_sparse_late",
            late_window_start=exp_late_sample_window[0] if 'exp_late_sample_window' in globals() else 30.0,
            late_window_end=exp_late_sample_window[1] if 'exp_late_sample_window' in globals() else 60.0,
            late_points=exp_late_sample_points if 'exp_late_sample_points' in globals() else 15,
            seeds="42 43 44",
            split_policy=select_k_split_policy,
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/sr_pipeline/select_k_linreg_per_minute.py \
                --dataset {input.per_minute} \
                --output-dir {params.out_dir} \
                --per-minute-max-time {params.max_time} \
                --per-minute-sampling-strategy {params.strategy} \
                --late-sample-window {params.late_window_start} {params.late_window_end} \
                --late-sample-points {params.late_points} \
                --seeds {params.seeds} \
                --test-split-policy {params.split_policy} \
                --measured-timepoints {params.measured}
            """



    rule experimental_neural_ode_diffrax_ood_l1:
        # L1 element-wise Jacobian sparsity, λ_jac = 1.0 (appendix variant).
        input:
            per_minute=marker_per_minute_csv
        output:
            done=neural_ode_diffrax_l1_done
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=neural_ode_diffrax_l1_seeds_dir,
            **_diffrax_sweep_params,
        shell:
            _diffrax_sweep_shell(
                neural_ode_diffrax_l1_seeds_dir,
                "diffrax_ood_l1_lam1",
                "--jac-reg 1.0 --jac-reg-mode l1 --hess-reg 0.0",
            )

    rule experimental_neural_ode_diffrax_ood_l21:
        # L21 group-sparse Jacobian, λ_jac = 3.0 — main-text "Neural ODE".
        # Calibrated on VAL R²; see scripts/calib_l21jac_then_decide.sh.
        input:
            per_minute=marker_per_minute_csv
        output:
            done=neural_ode_diffrax_l21_done
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=neural_ode_diffrax_l21_seeds_dir,
            **_diffrax_sweep_params,
        shell:
            _diffrax_sweep_shell(
                neural_ode_diffrax_l21_seeds_dir,
                "diffrax_ood_l21j3_valcalib",
                "--jac-reg 3.0 --jac-reg-mode l21 --hess-reg 0.0",
            )

    rule experimental_neural_ode_diffrax_ood_cnode:
        # C-NODE path-product regulariser (Aliee/Theis/Kilbertus 2022), λ = 0.01.
        # Calibrated on VAL R²; see scripts/calib_pathreg_then_sweep.sh.
        input:
            per_minute=marker_per_minute_csv
        output:
            done=neural_ode_diffrax_cnode_done
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=neural_ode_diffrax_cnode_seeds_dir,
            **_diffrax_sweep_params,
        shell:
            _diffrax_sweep_shell(
                neural_ode_diffrax_cnode_seeds_dir,
                "diffrax_ood_cnode_lam0p01",
                "--path-reg 0.01",
            )


    rule experimental_marker_integration:
        input:
            summary_mean=marker_summary_output,
            summary_seed=marker_summary_seed_output,
            snapshot=marker_fit_snapshot_csv,
            per_minute=marker_per_minute_csv
        output:
            integration_snapshot=integration_snapshot_metrics,
            integration_per_minute=integration_per_minute_metrics,
            integration_snapshot_traj=integration_snapshot_traj,
            integration_per_minute_traj=integration_per_minute_traj,
            predicted_snapshot=predicted_snapshot_traj,
            predicted_per_minute=predicted_per_minute_traj,
        conda:
            "../../envs/pysr.yaml"
        params:
            metrics_dir=exp_metrics_root,
            seeds_dir=exp_runs_seeds,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            trajectories_dir=exp_trajectories_root,
        shell:
            """
            mkdir -p {params.metrics_dir} {params.trajectories_dir}
            seed_summaries=($(ls {params.seeds_dir}/seed_*/summary/marker_summary.csv 2>/dev/null || true))

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
                    python src/pipelines/experimental/sr_pipeline/compute_marker_integration.py \
                        --dataset {input.snapshot} \
                        --summary "$summary_path" \
                        --sr-trajectories "{params.seeds_dir}/seed_${{seed_id}}/metrics/predicted_trajectories_snapshot.csv" \
                        --dataset-mode snapshot \
                        --output "$seed_snapshot" \
                        --trajectories-output "$seed_snapshot_traj" \
                        --aggregate-output {integration_snapshot_metrics_all_seeds} \
                        --measured-timepoints {params.measured} \
                        --seed "$seed_id"

                    python src/pipelines/experimental/sr_pipeline/compute_marker_integration.py \
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
                python src/pipelines/experimental/sr_pipeline/compute_marker_integration.py \
                    --dataset {input.snapshot} \
                    --summary {input.summary_seed} \
                    --sr-trajectories "$seed_metrics_dir/predicted_trajectories_snapshot.csv" \
                    --dataset-mode snapshot \
                    --output "$seed_metrics_dir/marker_integration_metrics_snapshot.csv" \
                    --trajectories-output "$seed_metrics_dir/marker_integration_trajectories_snapshot.csv" \
                    --aggregate-output {integration_snapshot_metrics_all_seeds} \
                    --measured-timepoints {params.measured}
                python src/pipelines/experimental/sr_pipeline/compute_marker_integration.py \
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

            # Ensure predicted trajectory outputs exist even when snapshot predictions are absent.
            if [ ! -f {output.predicted_snapshot} ]; then
                touch {output.predicted_snapshot}
            fi
            if [ ! -f {output.predicted_per_minute} ]; then
                touch {output.predicted_per_minute}
            fi
            """

    rule experimental_plots:
        input:
            summary_mean=marker_summary_output,
            summary_seed=marker_summary_seed_output,
            snapshot=marker_fit_snapshot_csv,
            per_minute=marker_per_minute_csv,
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
            feature_importance_svg=feature_importance_plots_svg,
            overlay_snapshot=overlay_snapshot_plot,
            overlay_snapshot_svg=overlay_snapshot_plot_svg,
            overlay_snapshot_metrics=overlay_snapshot_metrics,
            overlay_per_minute=overlay_per_minute_plot,
            overlay_per_minute_svg=overlay_per_minute_plot_svg,
            overlay_per_minute_metrics=overlay_per_minute_metrics
        conda:
            "../../envs/pysr.yaml"
        params:
            overlays_dir=exp_runs_seeds,
            metrics_plots_dir=plots_metrics_dir,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            metrics_dir=exp_metrics_root,
            trajectories_dir=exp_trajectories_root,
            seeds_dir=exp_runs_seeds,
        shell:
            """
            mkdir -p {params.overlays_dir} {params.metrics_plots_dir} {params.metrics_dir} {params.trajectories_dir}
            seed_summaries=($(ls {params.seeds_dir}/seed_*/summary/marker_summary.csv 2>/dev/null || true))

            if [ "${{#seed_summaries[@]}}" -gt 0 ]; then
                # Per-seed overlays only
                for summary_path in "${{seed_summaries[@]}}"; do
                    seed_id="$(basename "$(dirname "$(dirname "$summary_path")")" | sed 's/seed_//')"
                    seed_metrics_dir="{params.seeds_dir}/seed_${{seed_id}}/metrics"
                    seed_overlay_dir="{params.overlays_dir}/seed_${{seed_id}}"
                    mkdir -p "$seed_overlay_dir" "$seed_metrics_dir"

                    python src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
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
                    python src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
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

                    python src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
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
                    python src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
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
                python src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
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
                python src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
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
                python src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
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
                python src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
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
            python src/pipelines/experimental/sr_pipeline/plot_metrics_summary.py \
                --summary {input.summary_mean} \
                --output-dir {params.metrics_plots_dir} \
                --integration-metrics-dir {params.metrics_dir} \
                --trajectories-dir {params.trajectories_dir} \
                --snapshot-dataset {input.snapshot} \
                --per-minute-dataset {input.per_minute}

            # Copy one set of overlay plots/metrics into the aggregated outputs Snakemake expects.
            mkdir -p "$(dirname {output.overlay_snapshot})" "$(dirname {output.overlay_snapshot_svg})" \
                     "$(dirname {output.overlay_per_minute})" "$(dirname {output.overlay_per_minute_svg})" \
                     "$(dirname {output.overlay_snapshot_metrics})" "$(dirname {output.overlay_per_minute_metrics})"
            if [ "${{#seed_summaries[@]}}" -gt 0 ]; then
                first_seed="$(basename "$(dirname "$(dirname "${{seed_summaries[0]}}")")" | sed 's/seed_//')"
                src_overlay_dir="{params.overlays_dir}/seed_${{first_seed}}"
                src_metrics_dir="{params.seeds_dir}/seed_${{first_seed}}/metrics"
            else
                src_overlay_dir="{params.overlays_dir}"
                src_metrics_dir="{params.metrics_dir}"
            fi
            for mode in snapshot per_minute; do
                png_out="{output.overlay_snapshot}" ; svg_out="{output.overlay_snapshot_svg}" ; csv_out="{output.overlay_snapshot_metrics}"
                if [ "$mode" = "per_minute" ]; then
                    png_out="{output.overlay_per_minute}" ; svg_out="{output.overlay_per_minute_svg}" ; csv_out="{output.overlay_per_minute_metrics}"
                fi
                if [ -f "$src_overlay_dir/marker_overlay_${{mode}}.png" ]; then
                    cp "$src_overlay_dir/marker_overlay_${{mode}}.png" "$png_out"
                else
                    touch "$png_out"
                fi
                if [ -f "$src_overlay_dir/marker_overlay_${{mode}}.svg" ]; then
                    cp "$src_overlay_dir/marker_overlay_${{mode}}.svg" "$svg_out"
                else
                    touch "$svg_out"
                fi
                if [ -f "$src_metrics_dir/marker_overlay_metrics_${{mode}}.csv" ]; then
                    cp "$src_metrics_dir/marker_overlay_metrics_${{mode}}.csv" "$csv_out"
                else
                    touch "$csv_out"
                fi
            done
            """

    rule experimental_pysr_r2_threshold_sanity:
        output:
            plot=metrics_pysr_r2_threshold_sanity,
            plot_svg=metrics_pysr_r2_threshold_sanity_svg,
            examples_csv=metrics_pysr_r2_threshold_sanity_csv
        conda:
            "../../envs/pysr.yaml"
        params:
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            trajectories=integration_per_minute_traj,
            raw_dataset=marker_time_series_csv,
            override_065="0.65,DUSP10 (P2),22,44",
            override_070="0.70,DYRK3,5,43"
        shell:
            """
            python src/pipelines/experimental/sr_pipeline/plot_pysr_r2_threshold_sanity.py \
                --trajectories {params.trajectories} \
                --raw-dataset {params.raw_dataset} \
                --output-plot {output.plot} \
                --output-csv {output.examples_csv} \
                --override-example {params.override_065} \
                --override-example {params.override_070} \
                --examples-per-target 3 \
                --measured-timepoints {params.measured}
            """
