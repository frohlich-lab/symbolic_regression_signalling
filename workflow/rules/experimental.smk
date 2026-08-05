# =============================================================================
# Draft SR-MM v5 — experimental ERK pipeline
# =============================================================================
#
# The chain the manuscript's experimental results actually come from, in order:
#
#   markers_per_minute_fit.csv            (data prep — common.smk)
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
#        +-- linreg_ood                   OLS on all ten inputs, same split (common.smk)
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
