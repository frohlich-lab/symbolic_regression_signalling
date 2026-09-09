# =============================================================================
# Experimental ERK pipeline
# =============================================================================
#
#   marker inputs ──┬─► pysr_config_sweep ──► select_pysr_config ──┐   $$$
#     (data prep)   │      288 arms            72 candidates,      │
#                   │      6 contexts          ranked on train     │
#                   │                                             ▼
#                   ├─────────────────────────► pysr_ood_final          $$$
#                   │                             40 contexts x 3 seeds
#                   │                                    │
#                   │                                    ▼
#                   │                            summarise_pysr_ood
#                   │                             rates + exemplars
#                   │                                    │
#                   ├─► linreg_ood ─────────────────┐    │
#                   │     OLS, 10 inputs            │    │
#                   │                               ▼    ▼
#                   └─► sparse_neural_ode ──────►  paper figures
#                         L21 / L1 / C-NODE   $$$
#
#   $$$ = cluster-scale (~3,000 and ~70 CPU-hours for the two PySR stages)
#
# SECTIONS
#   1  PySR configuration sweep   the 288-arm grid, and the rule that picks from it
#   2  Frozen-config run          that configuration applied to all 40 contexts
#   3  Sparse Neural ODE          lambda sweep, then the L21 / L1 / C-NODE variants
#   4  Paper figures
#   5  Aggregators                experimental_results, experimental_metrics
#   6  Data prep and shared       marker inputs, ODE integration, summary plots
#
# CONVENTIONS  every stage below obeys all three; see also common.smk for the paths
#   split    top 20% of GFP bins held out, so scores measure dose extrapolation
#   metric   R2 of the ODE-integrated p-ERK trajectory (not the derivative fit)
#   seeds    42/43/44, reported as best-of
#
# The $$$ stages ran on NEMO via src/pipelines/experimental/sweeps/. Their outputs are
# committed, so Snakemake sees them as up to date and only the cheap stages re-run;
# their inputs are ancient() so a touched timestamp cannot trigger a rebuild. Delete an
# output to re-run deliberately, on a cluster.
#
# Figure- and table-by-artefact map: docs/experimental_provenance.md

if enzyme_model == "experimental":


    # --- 1. PySR configuration sweep -----------------------------------------

    rule experimental_pysr_config_sweep:
        # $$$ ~3,000 CPU-hours (288 arms x 6 contexts x 3 seeds, ~34 min/fit).
        #
        # Runs 288 arms; 72 of them are selection candidates -- those keeping division
        # with no unary operators. The wider grid varies the unary-operator and
        # no-division axes so that restriction is measured rather than assumed.
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

            # Arm definitions. Stability penalties are held at their defaults in every
            # arm -- they encode the prior that d[p-ERK]/dt decreases in p-ERK, so they
            # are not a tunable axis.
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
        # Ranks the 72 candidates on median integrated R2 over the TRAINING bins only,
        # so the held-out doses play no part in the choice. Deterministic.
        #
        # Emits two things: the ranked table, and the flag string that drives every
        # later PySR run -- so the configuration is defined in one place.
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

    # --- 2. Frozen-config run, all 40 contexts --------------------------------

    rule experimental_pysr_ood_final:
        # $$$ ~70 CPU-hours (40 contexts x 3 seeds, ~35 min/fit).
        #
        # Applies the selected configuration UNCHANGED to all 40 contexts, including the
        # 34 the sweep never saw. Keeping selection and application separate is what
        # makes the reported rates generalise beyond the development set.
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
            metrics=pysr_final_metrics,
            # Fig. 5 and Fig. S2 draw individual held-out bins, so the trajectories are
            # a reported output rather than a by-product; declaring them is also what
            # would have caught the broken integration call below.
            trajectories=pysr_final_traj
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=pysr_final_dir,
            group_definitions_csv=exp_group_definitions_csv,
            raw_measurements=exp_raw_time_course,
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
                #
                # --summary, --output and --trajectories-output are all required and
                # there is no --output-dir. This call used to pass --output-dir and no
                # --summary, so it exited on argparse; nothing noticed because the rule
                # takes ancient() inputs and its outputs were already on disk from the
                # recovered cluster run. Naming every path explicitly is what keeps that
                # from recurring.
                seed_dir={params.out_dir}/seeds/seed_${{seed}}
                python src/pipelines/experimental/sr_pipeline/compute_marker_integration.py \
                    --dataset {input.per_minute} \
                    --raw-dataset {params.raw_measurements} \
                    --summary "$seed_dir/summary/marker_summary.csv" \
                    --sr-trajectories "$seed_dir/trajectories/predicted_trajectories_per_minute.csv" \
                    --dataset-mode per_minute \
                    --output "$seed_dir/metrics/marker_integration_metrics_per_minute.csv" \
                    --trajectories-output "$seed_dir/metrics/marker_integration_trajectories_per_minute.csv" \
                    --measured-timepoints {params.measured} \
                    --seed ${{seed}}
            done
            """

    rule experimental_pysr_ood_summary:
        # The reported success rates and the Fig. 5 exemplar ranking. A context counts
        # as solved when the retained fit clears R2 >= 0.6 on the held-out GFP bins;
        # the training bins are not part of the criterion. The seed is chosen on
        # training R2, so seed selection never touches held-out data. This matches
        # Table S8 and the manuscript, which score the median across test bins.
        input:
            metrics=pysr_final_metrics
        output:
            per_fit=pysr_final_per_fit,
            summary=pysr_final_summary,
            exemplars=pysr_final_exemplars
        conda:
            "../../envs/pysr.yaml"
        params:
            raw_measurements=exp_raw_time_course,
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
                --raw-dataset {{params.raw_measurements}} \\
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

    # --- 3. Sparse Neural ODE -------------------------------------------------
    #
    # L21 at lambda_jac = 3 is the reference model; L1 and the path-regularised
    # (C-NODE) variant exist to show the dependency counts are not an artefact of
    # the penalty. The lambda sweep is what fixes lambda_jac = 3.

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
            raw_measurements=exp_raw_time_course,
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
                        --raw-dataset {params.raw_measurements} \
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

    rule experimental_sparse_node_arch_grid:
        # $$$ Fig. S3C / Table S9, and the most expensive rule here after the PySR run:
        # 54 cells (3 widths x 3 depths x 3 learning rates x 2 activations) x every
        # context, at one seed. Scored on IN-DISTRIBUTION validation R2 only, so the
        # held-out highest-dose bins take no part in choosing an architecture.
        #
        # ancient() for the same reason as the PySR sweep: this is a recorded run, and a
        # moved timestamp must not spend 54 GPU-hours. See common.smk for why re-running
        # will not reproduce the checked-in arch_grid.csv row for row.
        input:
            per_minute=ancient(marker_per_minute_csv)
        output:
            summary=sparse_node_arch_grid_csv
        conda:
            "../../envs/pysr.yaml"
        params:
            raw_measurements=exp_raw_time_course,
            out_dir=sparse_node_arch_grid_dir,
            widths=" ".join(str(v) for v in sparse_node_arch_widths),
            depths=" ".join(str(v) for v in sparse_node_arch_depths),
            lrs=" ".join(f"{v:g}" for v in sparse_node_arch_lrs),
            activations=" ".join(sparse_node_arch_activations),
            seed=sparse_node_arch_grid_seed,
            # _diffrax_sweep_params already carries `seeds`, which this rule does not use
            # (the grid is single-seed); take the shared per-minute sampling settings only.
            **{k: v for k, v in _diffrax_sweep_params.items() if k != "seeds"},
        shell:
            """
            set -euo pipefail
            mkdir -p {params.out_dir}
            for hd in {params.widths}; do
              for hl in {params.depths}; do
                for lr in {params.lrs}; do
                  for act in {params.activations}; do
                    # `.` -> `p` so the cell name survives as a directory and parses back;
                    # summarise_arch_grid.py reads the hyperparameters out of this name.
                    tag="hd${{hd}}_hl${{hl}}_lr$(echo "${{lr}}" | tr '.' 'p')_${{act}}"
                    mkdir -p {params.out_dir}/${{tag}}
                    python src/pipelines/experimental/sr_pipeline/neural_ode_diffrax_baseline.py \
                        --dataset {input.per_minute} \
                        --raw-dataset {params.raw_measurements} \
                        --output-dir {params.out_dir}/${{tag}} \
                        --seeds {params.seed} \
                        --epochs 200 --patience 20 \
                        --per-minute-max-time {params.max_time} \
                        --per-minute-sampling-strategy {params.strategy} \
                        --late-sample-window {params.late_window_start} {params.late_window_end} \
                        --late-sample-points {params.late_points} \
                        --measured-timepoints {params.measured} \
                        --test-split-policy top_gfp_bins \
                        --hidden-dim ${{hd}} --hidden-layers ${{hl}} \
                        --lr ${{lr}} --activation ${{act}} \
                        --jac-reg 3.0 --jac-reg-mode l21 --hess-reg 0.0 \
                        --tag ${{tag}}
                  done
                done
              done
            done
            python src/pipelines/experimental/sweeps/summarise_arch_grid.py \
                --grid-dir {params.out_dir} \
                --output {output.summary}
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

    # --- 4. Paper figures -----------------------------------------------------

    rule experimental_compare_sr_vs_neural_ode:
        # Fig. 4E + the sparsity panel. Both axes are held-out ODE-integrated R2:
        # --pysr-integ-csv is what makes PySR's axis comparable to the network's
        # trajectory R2. Without it PySR is plotted on its derivative-fit R2, which is a
        # different quantity (the two correlate ~0.5) and was how the earlier version of
        # this panel came to mix metrics.
        #
        # Four outputs, one script. `fig` and `paper` are the same 32-perturbation
        # panel -- the second is the printed copy, an output rather than a hand `cp`.
        # `all40` keeps the controls in and is Fig. S5, the robustness check on the
        # same comparison. `joint` adds the linear-regression scatter as a third
        # panel, so both baselines and the driver counts read as one row.
        #
        # --linreg-metrics takes the PER-SEED file for the same reason Fig. S4 does:
        # both axes are best-of-three-seeds, and select_k_metrics_agg is already
        # averaged over seeds, so it would put a seed-mean baseline against a
        # best-of-seeds SR.
        input:
            nn_done=sparse_node_l21_done,
            pysr_formulas=pysr_final_formulas,
            pysr_integ=pysr_final_per_fit,
            linreg=select_k_metrics,
            dataset=marker_per_minute_csv
        output:
            fig=sr_vs_node_panel,
            fig_pdf=sr_vs_node_panel.replace(".png", ".pdf"),
            paper=fig_4ef_parsimony,
            paper_pdf=fig_4ef_parsimony.replace(".png", ".pdf"),
            all40=fig_s5_parsimony_all40,
            all40_pdf=fig_s5_parsimony_all40.replace(".png", ".pdf"),
            joint=fig_4_joint_baselines,
            joint_pdf=fig_4_joint_baselines.replace(".png", ".pdf"),
            closure=fig_4_closure_cost,
            closure_pdf=fig_4_closure_cost.replace(".png", ".pdf"),
            closure_curve=fig_4_closure_cost.replace(".png", "_rho_curve.csv"),
            panel_inputs=fig_4_panel_inputs,
            nointer=fig_4ef_parsimony_nointeractions,
            nointer_pdf=fig_4ef_parsimony_nointeractions.replace(".png", ".pdf"),
            srquad=fig_4ef_parsimony_srquadrant,
            srquad_pdf=fig_4ef_parsimony_srquadrant.replace(".png", ".pdf")
        conda:
            "../../envs/pysr.yaml"
        params:
            nn_dir=node_hl4_dir,
            pysr_dir=f"{pysr_final_dir}/seeds",
            excludes=exp_exclude_flags,
            threshold=exp_r2_threshold,
            linreg_k=linreg_k_full,
            # Specification shared with the Results text and Table S8: dependency count
            # (not the participation ratio) averaged over all three seeds, neural-ODE seed
            # chosen on validation and its test score reported, PySR seed chosen by
            # training R2, and both methods scored on the same ODE-integrated R2. Both
            # selection rules read training/validation data only.
            # --sr-parsimony was used until 2026-08-12; it partitions 14/26 rather than the
            # 13/27 of Table S8, because AKT3's highest-training seed does not generalise.
            # Split so the no-interaction variant can drop the third box column without
            # restating the selection rules, which must be identical across variants.
            spec_base=("--dependency-count --select-on-val --sr-best-train "
                       "--node-integrated-r2"),
            spec=("--dependency-count --select-on-val --sr-best-train "
                  "--node-integrated-r2 --symbolic-interaction-box"),
            interactions=f"--interaction-csv {node_hl4_interactions}",
            # Callouts only, not data: the control dots stay in the scatter and every
            # panel number is unchanged. The joint row draws this axes at ~1/3 of an
            # 11-in figure that is placed at ~7 in, so the axes is height-limited and can
            # be neither widened nor set in smaller type (9.6 pt is already ~6 pt on the
            # page). Dropping the eight control names is the only lever that removes the
            # crossing leaders, and they are the longest strings for the least biology.
            label_skips=" ".join(f"--label-skip {m}" for m in exp_control_contexts),
            table_s8=f"{exp_supplementary_dir}/table_s8_sr_per_context.csv",
        shell:
            """
            set -euo pipefail
            mkdir -p $(dirname {output.fig}) $(dirname {output.paper}) \
                     $(dirname {output.all40}) $(dirname {output.joint})
            # No {params.excludes} here: the Results text reports all 40 contexts, and 7 of
            # the 13 SR successes are controls, so excluding them changes the panel's claim.
            for out in "{output.fig}" "{output.paper}"; do
                python src/pipelines/experimental/sr_pipeline/figures/plot_parsimony_tradeoff.py \
                    --nn-dir {params.nn_dir} \
                    --pysr-dir {params.pysr_dir} \
                    --pysr-integ-csv {input.pysr_integ} \
                    --dataset {input.dataset} \
                    --r2-threshold {params.threshold} \
                    {params.spec} {params.interactions} \
                    --dump-inputs {output.panel_inputs} \
                    --output "$out"
            done
            # Same panel values, collapsed to one box column. Built from the dump above
            # rather than recomputed, so it cannot disagree with the panel it replaces.
            python src/pipelines/experimental/sr_pipeline/figures/plot_fig4_composite_complexity.py \
                --panel-inputs {output.panel_inputs} \
                --table-s8 {params.table_s8} \
                --rho 1.0 \
                --threshold {params.threshold} \
                --output {output.closure}
            # Layout variants the draft chooses between: no interaction column, and the
            # SR box restricted to the top-right quadrant so its bracket is paired.
            python src/pipelines/experimental/sr_pipeline/figures/plot_parsimony_tradeoff.py \
                --nn-dir {params.nn_dir} \
                --pysr-dir {params.pysr_dir} \
                --pysr-integ-csv {input.pysr_integ} \
                --dataset {input.dataset} \
                --r2-threshold {params.threshold} \
                {params.spec_base} \
                --output {output.nointer}
            python src/pipelines/experimental/sr_pipeline/figures/plot_parsimony_tradeoff.py \
                --nn-dir {params.nn_dir} \
                --pysr-dir {params.pysr_dir} \
                --pysr-integ-csv {input.pysr_integ} \
                --dataset {input.dataset} \
                --r2-threshold {params.threshold} \
                {params.spec} {params.interactions} --sr-box-quadrant \
                --output {output.srquad}
            python src/pipelines/experimental/sr_pipeline/figures/plot_parsimony_tradeoff.py \
                --nn-dir {params.nn_dir} \
                --pysr-dir {params.pysr_dir} \
                --pysr-integ-csv {input.pysr_integ} \
                --dataset {input.dataset} \
                --r2-threshold {params.threshold} \
                {params.spec} {params.interactions} \
                --output {output.all40}
            python src/pipelines/experimental/sr_pipeline/figures/plot_parsimony_tradeoff.py \
                --nn-dir {params.nn_dir} \
                --pysr-dir {params.pysr_dir} \
                --pysr-integ-csv {input.pysr_integ} \
                --dataset {input.dataset} \
                --linreg-metrics {input.linreg} \
                --linreg-k {params.linreg_k} \
                --r2-threshold {params.threshold} \
                {params.spec} {params.interactions} \
                {params.excludes} {params.label_skips} \
                --no-panel-titles --no-tick-legend \
                --output {output.joint}
            """

    rule experimental_node_driver_readout:
        # Seed robustness, between-method agreement and driver-set size against the
        # top-X% Jacobian-mass cutoff used to read dependencies off the network.
        input:
            nn_done=sparse_node_l21_done,
            pysr_formulas=pysr_final_formulas,
            dataset=marker_per_minute_csv
        output:
            fig=node_driver_readout,
            fig_pdf=node_driver_readout.replace(".png", ".pdf")
        conda:
            "../../envs/pysr.yaml"
        params:
            nn_dir=sparse_node_l21_dir,
            pysr_dir=f"{pysr_final_dir}/seeds",
        shell:
            """
            mkdir -p $(dirname {output.fig})
            python src/pipelines/experimental/sr_pipeline/figures/plot_cutoff_robustness.py \
                --nn-dir {params.nn_dir} \
                --pysr-dir {params.pysr_dir} \
                --dataset {input.dataset} \
                --output {output.fig}
            """

    rule experimental_node_regulariser_comparison:
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
            fig=node_regulariser_comp,
            fig_pdf=node_regulariser_comp.replace(".png", ".pdf"),
            # The per-variant summary the figure is annotated from; Table S8 quotes it,
            # so it is a result rather than a by-product.
            summary=node_regulariser_comp.replace(".png", ".csv")
        conda:
            "../../envs/pysr.yaml"
        params:
            l1_dir=sparse_node_l1_dir,
            l21_dir=sparse_node_l21_dir,
            cnode_dir=sparse_node_cnode_dir,
        shell:
            """
            mkdir -p $(dirname {output.fig})
            python src/pipelines/experimental/sr_pipeline/figures/plot_nn_appendix_comparison.py \
                --l1-dir {params.l1_dir} \
                --l21-dir {params.l21_dir} \
                --cnode-dir {params.cnode_dir} \
                --output {output.fig}
            """

    rule experimental_compare_sr_vs_linreg:
        # Diagnostic sweep at k=10 (the full ten-input pool, the toughest baseline) and
        # k=4 (complexity-matched to PySR's median equation), both over all 40 contexts:
        # PySR wins at every k tested, and keeping the controls in is what shows that.
        #
        # The printed panel is the third output and drops the controls, because there
        # "extrapolating to an unseen dose" is a context with no dose-response at all.
        # It is generated here rather than by hand so that the figure in the manuscript
        # and the figure in the pipeline cannot disagree.
        input:
            linreg=select_k_metrics,
            pysr_formulas=pysr_final_formulas,
            pysr_integ=pysr_final_per_fit
        output:
            full=sr_vs_linreg_k10,
            full_pdf=sr_vs_linreg_k10.replace(".png", ".pdf"),
            matched=sr_vs_linreg_k4,
            matched_pdf=sr_vs_linreg_k4.replace(".png", ".pdf"),
            paper=fig_4d_sr_vs_linreg,
            paper_pdf=fig_4d_sr_vs_linreg.replace(".png", ".pdf")
        conda:
            "../../envs/pysr.yaml"
        params:
            pysr_dir=f"{pysr_final_dir}/seeds",
            k_full=linreg_k_full,
            k_matched=linreg_k_matched,
            exclude=exp_exclude_flags,
        shell:
            """
            set -euo pipefail
            mkdir -p $(dirname {output.full}) $(dirname {output.paper})
            for spec in "{params.k_full}:{output.full}" "{params.k_matched}:{output.matched}"; do
                k=${{spec%%:*}}
                out=${{spec#*:}}
                python src/pipelines/experimental/sr_pipeline/figures/plot_sr_vs_linreg_ood.py \
                    --linreg-metrics {input.linreg} \
                    --pysr-dir {params.pysr_dir} \
                    --pysr-integ-csv {input.pysr_integ} \
                    --linreg-k ${{k}} \
                    --output ${{out}}
            done
            # --sr-best-train: Fig. 4D must partition contexts the same way Fig. 4E and
            # Table S8 do. Without it the panel selected each marker's best held-out seed
            # and showed 15 contexts above the cutoff where the text reports 13.
            python src/pipelines/experimental/sr_pipeline/figures/plot_sr_vs_linreg_ood.py \
                --linreg-metrics {input.linreg} \
                --pysr-dir {params.pysr_dir} \
                --pysr-integ-csv {input.pysr_integ} \
                --linreg-k {params.k_full} \
                --sr-best-train \
                {params.exclude} \
                --output {output.paper}
            """

    rule experimental_fig_s4_sr_vs_linreg_matched:
        # Fig. S4: the k=4 comparison as printed. Same numbers as sr_vs_linreg_k4 above,
        # drawn on the supplementary style sheet and annotated with the 32-perturbation
        # Wilcoxon test.
        #
        # --linreg-metrics takes the PER-SEED file, not select_k_metrics_agg: both axes
        # are best-of-three-seeds, and the agg file is already averaged over seeds, so
        # feeding it would silently put a seed-mean baseline against a best-of-seeds SR.
        input:
            linreg=select_k_metrics,
            pysr_integ=pysr_final_per_fit
        output:
            png=fig_s4_sr_vs_linreg,
            pdf=fig_s4_sr_vs_linreg.replace(".png", ".pdf")
        conda:
            "../../envs/pysr.yaml"
        params:
            k_matched=linreg_k_matched,
            threshold=exp_r2_threshold,
        shell:
            """
            mkdir -p $(dirname {output.png})
            python src/pipelines/experimental/sr_pipeline/figures/plot_fig_s4_sr_vs_linreg_matched.py \
                --linreg-metrics {input.linreg} \
                --pysr-integ-csv {input.pysr_integ} \
                --linreg-k {params.k_matched} \
                --threshold {params.threshold} \
                --output {output.png}
            """

    rule experimental_fig_s2_threshold_examples:
        # Fig. S2: what the R2 = 0.6 cutoff buys. Draws the six perturbation fits either
        # side of it, ordered by the marker-level integrated R2 the cutoff acts on, one
        # held-out bin each on a normalised axis. The six-a-side window is contiguous, so
        # every perturbation fit inside it is shown (the script asserts this). Controls
        # are excluded: their R2 cluster at 0.60-0.73, so including them made "cleared the
        # cutoff" and "is a trivially easy control" the same visual category. Read it as
        # showing the transition is gradual and 0.6 sits inside it -- it does NOT show a
        # discontinuity at 0.6, and the caption must not claim one.
        input:
            fits=ancient(pysr_final_fits_dir)
        output:
            png=fig_s2_threshold_examples,
            pdf=fig_s2_threshold_examples.replace(".png", ".pdf")
        conda:
            "../../envs/pysr.yaml"
        shell:
            """
            mkdir -p $(dirname {output.png})
            python src/pipelines/experimental/sr_pipeline/figures/plot_fig_s2_threshold_examples.py \
                --fits-dir {input.fits} \
                --output {output.png}
            """

    rule experimental_threshold_calibration:
        # Why R2 = 0.6. One panel: fraction of contexts clearing each candidate cutoff,
        # real vs a wrong-context null -- integrated R2 earned by a real SR trajectory
        # belonging to a DIFFERENT context, built with the same median-over-bins and
        # best-of-three-seeds selection the reported statistic uses, so the comparison is
        # like-for-like. The claim is the vertical GAP at 0.6 (~9,000x); where the null
        # curve ends is the Monte Carlo resolution limit and must not be captioned. 0.6
        # is safe and inconsequential, NOT uniquely optimal -- read
        # docs/experimental_provenance.md before writing a caption for it.
        input:
            fits=ancient(pysr_final_fits_dir),
            per_fit=pysr_final_per_fit,
            # No rule writes this one; it is a recorded artefact like fits/, so it
            # is ancient() for the same reason.
            panel=ancient(f"{pysr_final_dir}/metrics/panel_inputs_32perturbations.csv")
        output:
            png=fig_threshold_calibration,
            pdf=fig_threshold_calibration.replace(".png", ".pdf")
        conda:
            "../../envs/pysr.yaml"
        shell:
            """
            mkdir -p $(dirname {output.png})
            python src/pipelines/experimental/sr_pipeline/figures/plot_threshold_calibration.py \
                --fits-dir {input.fits} \
                --per-fit-csv {input.per_fit} \
                --panel-csv {input.panel} \
                --output {output.png}
            """

    rule experimental_threshold_error_tradeoff:
        # What each candidate cutoff buys: held-out prediction error of everything it
        # would accept, as a % of each bin's measured dynamic range, against contexts
        # kept. Summarised CUMULATIVELY and bootstrapped over fits -- marginal bands of
        # +/-0.05 hold only 3-5 fits and produced a spurious "plateau" whose mean and
        # median disagreed by 11 points. The curve is a smooth monotone tradeoff with
        # overlapping intervals: there is NO knee at 0.6 and it must not be drawn as one.
        # It supports the accuracy claim (~12% of range at 0.6), not the cutoff choice.
        input:
            fits=ancient(pysr_final_fits_dir),
            per_fit=pysr_final_per_fit
        output:
            png=fig_threshold_error_tradeoff,
            pdf=fig_threshold_error_tradeoff.replace(".png", ".pdf")
        conda:
            "../../envs/pysr.yaml"
        shell:
            """
            mkdir -p $(dirname {output.png})
            python src/pipelines/experimental/sr_pipeline/figures/plot_threshold_error_tradeoff.py \
                --fits-dir {input.fits} \
                --per-fit-csv {input.per_fit} \
                --output {output.png}
            """

    rule experimental_threshold_intuition:
        # What a held-out curve at R2 ~ 0.5 / 0.6 / 0.7 actually looks like: the 8 curves
        # nearest each value, measurement and prediction together. For the reviewer who
        # assumes 0.7 is the standard bar -- the 0.6 and 0.7 blocks are hard to tell
        # apart (18% vs 15% of range missed), so 0.7 buys ~3 points of accuracy.
        # It does NOT show 0.6 beating 0.7 and must not be captioned that way.
        # Selection is load-bearing here: three other framings were tried and one of them
        # (quantiles of the accepted population) was rejected as misleading. Read the
        # docs/experimental_provenance.md entry before changing the rule.
        input:
            fits=ancient(pysr_final_fits_dir)
        output:
            png=fig_threshold_intuition,
            pdf=fig_threshold_intuition.replace(".png", ".pdf")
        conda:
            "../../envs/pysr.yaml"
        shell:
            """
            mkdir -p $(dirname {output.png})
            python src/pipelines/experimental/sr_pipeline/figures/plot_threshold_intuition.py \
                --fits-dir {input.fits} \
                --output {output.png}
            """

    rule experimental_fig5_perbin_trajectories:
        # Fig. 5: one square panel per held-out GFP bin for a single exemplar fit, which
        # is what shows the dose gradient is reproduced bin by bin rather than only on
        # average. The exemplar pair is pinned in common.smk; the fan overlays in
        # pysr_ood_final/exemplar_fans/ are the diagnostic version of the same fits.
        input:
            traj=ancient(f"{pysr_final_fits_dir}/{{marker}}/seed_{{seed}}/metrics/"
                         f"marker_integration_trajectories_per_minute.csv")
        output:
            png=f"{exp_paper_figures_dir}/fig_5_perbin_{{marker}}_s{{seed}}.png",
            pdf=f"{exp_paper_figures_dir}/fig_5_perbin_{{marker}}_s{{seed}}.pdf",
            fan=f"{exp_paper_figures_dir}/fig_5_fan_{{marker}}_s{{seed}}.png",
            fan_pdf=f"{exp_paper_figures_dir}/fig_5_fan_{{marker}}_s{{seed}}.pdf"
        wildcard_constraints:
            # Only the pinned exemplars, so a marker name containing "_" cannot be
            # mis-split against the "_s<seed>" suffix.
            marker="|".join(m for m, _ in fig5_exemplars),
            seed=r"\d+",
        conda:
            "../../envs/pysr.yaml"
        params:
            measured=" ".join(str(t) for t in exp_measured_timepoints),
        shell:
            """
            mkdir -p $(dirname {output.png})
            python src/pipelines/experimental/sr_pipeline/figures/plot_perbin_trajectories.py \
                --traj {input.traj} \
                --measured {params.measured} \
                --title "{wildcards.marker} (seed {wildcards.seed})" \
                --output {output.png}
            # Centre panel, from the same table, so both panels describe one fit.
            python src/pipelines/experimental/sr_pipeline/figures/plot_exemplar_fans.py \
                --traj {input.traj} \
                --title "{wildcards.marker} | seed {wildcards.seed}" \
                --output {output.fan}
            """

    rule experimental_node_participation_ratios:
        # Table S8's dependency count: participation ratio of the input Jacobian with GFP
        # excluded. Reads the saved checkpoints of all three regulariser variants, so it
        # depends on the sweeps having run with --save-models rather than on any metrics
        # file.
        input:
            l1=sparse_node_l1_done,
            l21=sparse_node_l21_done,
            cnode=sparse_node_cnode_done
        output:
            csv=sparse_node_participation_csv
        conda:
            "../../envs/pysr.yaml"
        params:
            nde_root=sparse_node_dir,
        shell:
            """
            python src/pipelines/experimental/sr_pipeline/figures/compute_participation_ratios.py \
                --nde-root {params.nde_root} \
                --output {output.csv}
            """

    rule experimental_fig_s3_node_selection:
        # Fig. S3: the three panels behind the Neural ODE's settings -- regulariser
        # variants (A), the lambda ladder and its elbow (B), and whether validation R2
        # predicts held-out R2 at all (C, it barely does).
        input:
            regulariser=node_regulariser_comp,
            lambda_sweep=sparse_node_lambda_sweep_csv,
            arch_grid=sparse_node_arch_grid_csv,
            table_s10=pysr_sweep_table
        output:
            png=fig_s3_node_selection,
            pdf=fig_s3_node_selection.replace(".png", ".pdf")
        conda:
            "../../envs/pysr.yaml"
        params:
            nde_root=sparse_node_dir,
        shell:
            """
            mkdir -p $(dirname {output.png})
            python src/pipelines/experimental/sr_pipeline/figures/plot_fig_s3_neural_ode_selection.py \
                --nde-root {params.nde_root} \
                --table-s10 {input.table_s10} \
                --output {output.png}
            """

    rule experimental_table_s8_sr_per_context:
        # Table S8: one row per context, the seed retained by training R2 and what it
        # recovered. Separate from the batch below because it reads the PySR per-fit table
        # and the Fig. 4 panel dump, so it is guaranteed to agree with the printed panel
        # on which contexts count as successes -- the hand-maintained version did not.
        input:
            pysr_integ=pysr_final_per_fit,
            panel_inputs=fig_4_panel_inputs
        output:
            table_s8_sr_per_context
        conda:
            "../../envs/pysr.yaml"
        params:
            threshold=exp_r2_threshold,
        shell:
            """
            mkdir -p $(dirname {output})
            python src/pipelines/experimental/sr_pipeline/figures/make_table_s8_sr_per_context.py \
                --pysr-integ-csv {input.pysr_integ} \
                --panel-inputs {input.panel_inputs} \
                --threshold {params.threshold} \
                --output {output}
            """

    rule experimental_fig_s1_loss_ablation:
        # Fig. S1: the custom-loss ablation. Seed 42 across all four loss configurations by
        # design, so this panel is independent of the seed-selection rule the rest of the
        # figures use and does not need rebuilding when that rule changes.
        input:
            fits=ancient(pysr_final_fits_dir)
        output:
            png=fig_s1_loss_ablation,
            pdf=fig_s1_loss_ablation.replace(".png", ".pdf"),
            metrics=f"{exp_supplementary_dir}/fig_s1_loss_ablation_metrics.csv"
        conda:
            "../../envs/pysr.yaml"
        params:
            threshold=exp_r2_threshold,
        shell:
            """
            mkdir -p $(dirname {output.png})
            python src/pipelines/experimental/sr_pipeline/figures/plot_fig_s1_loss_ablation.py \
                --fits-dir {input.fits} \
                --metrics-out {output.metrics} \
                --threshold {params.threshold} \
                --output {output.png}
            """

    rule experimental_supplementary_tables:
        # Tables S9-S10 in one rule because they read one set of run artefacts: S9 is the
        # PySR configuration sweep, S10 the Neural ODE architecture search. That order is
        # the order the main text cites them; they were numbered the other way until
        # 2026-08-12. The two *_full.csv are the complete 72- and 54-row grids, deposited
        # with the code rather than typeset. The regulariser comparison and readout key are
        # still computed but land in archive/, having lost their citations.
        input:
            participation=sparse_node_participation_csv,
            arch_grid=sparse_node_arch_grid_csv,
            l1=sparse_node_l1_done,
            l21=sparse_node_l21_done,
            cnode=sparse_node_cnode_done,
            table_s10=pysr_sweep_table
        output:
            *supp_tables,
            *supp_tables_retired
        conda:
            "../../envs/pysr.yaml"
        params:
            nde_root=sparse_node_dir,
            out_dir=exp_supplementary_dir,
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/sr_pipeline/figures/make_supplementary_tables.py \
                --nde-root {params.nde_root} \
                --table-s10-src {input.table_s10} \
                --out-dir {params.out_dir}
            """

    # --- 5. Aggregators -------------------------------------------------------

    rule experimental_results:
        # Everything the manuscript's experimental sections depend on.
        #   snakemake --use-conda -j1 --config enzyme_model=experimental experimental_results
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
            # main text -- every printed panel, so `paper_figures/` is collectable as a
            # set without a single hand-run command
            sr_vs_node_panel,
            sr_vs_linreg_k10,
            fig_4d_sr_vs_linreg,
            fig_4ef_parsimony,
            fig_4_joint_baselines,
            fig_4_closure_cost,
            fig_4ef_parsimony_nointeractions,
            fig_4ef_parsimony_srquadrant,
            fig5_perbin,
            fig5_fans,
            # SI
            fig_s1_loss_ablation,
            table_s8_sr_per_context,
            sr_vs_linreg_k4,
            node_regulariser_comp,
            node_driver_readout,
            custom_loss_ablation_metrics,
            fig_s2_threshold_examples,
            fig_threshold_calibration,
            fig_threshold_error_tradeoff,
            fig_threshold_intuition,
            fig_s3_node_selection,
            fig_s4_sr_vs_linreg,
            fig_s5_parsimony_all40,
            supp_tables,
            table_s12_threshold_sensitivity,
            table_s13_interaction_sensitivity,
            table_s14_lambda_calibration,

    rule experimental_metrics:
        # Just the cheap numeric stages, for when you want to re-derive the reported
        # numbers without touching a figure or a GPU.
        input:
            pysr_sweep_table,
            pysr_sweep_config,
            pysr_final_per_fit,
            pysr_final_summary,
            pysr_final_exemplars,
            sparse_node_participation_csv,
            table_s8_sr_per_context,
            supp_tables,

    # --- 6. Data preparation and shared stages -------------------------------
    #
    # Marker inputs, fit diagnostics, the in-distribution reference run, ODE
    # integration and the summary plots. Everything above depends on these.

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
            boxplot_dt=select_k_boxplot_dt,
            boxplot_integ=select_k_boxplot_integ,
            boxplot_ode=select_k_boxplot_ode
        conda:
            "../../envs/pysr.yaml"
        params:
            out_dir=select_k_dir,
            measured=" ".join(str(t) for t in exp_measured_timepoints),
            raw_measurements=exp_raw_time_course,
            max_time=exp_per_minute_max_time,
            strategy=exp_per_minute_sampling_strategy if 'exp_per_minute_sampling_strategy' in globals() else "early_plus_sparse_late",
            late_window_start=exp_late_sample_window[0] if 'exp_late_sample_window' in globals() else 30.0,
            late_window_end=exp_late_sample_window[1] if 'exp_late_sample_window' in globals() else 60.0,
            late_points=exp_late_sample_points if 'exp_late_sample_points' in globals() else 15,
            seeds="42 43 44",
            split_policy=select_k_split_policy,
            k_min=linreg_k_min,
            k_max=linreg_k_max,
        shell:
            """
            mkdir -p {params.out_dir}
            python src/pipelines/experimental/sr_pipeline/select_k_linreg_per_minute.py \
                --dataset {input.per_minute} \
                --raw-dataset {params.raw_measurements} \
                --output-dir {params.out_dir} \
                --per-minute-max-time {params.max_time} \
                --per-minute-sampling-strategy {params.strategy} \
                --late-sample-window {params.late_window_start} {params.late_window_end} \
                --late-sample-points {params.late_points} \
                --seeds {params.seeds} \
                --test-split-policy {params.split_policy} \
                --min-k {params.k_min} --max-k {params.k_max} \
                --measured-timepoints {params.measured}
            """


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
            raw_measurements=exp_raw_time_course,
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
                        --raw-dataset {params.raw_measurements} \
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
                        --raw-dataset {params.raw_measurements} \
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
                    --raw-dataset {params.raw_measurements} \
                    --summary {input.summary_seed} \
                    --sr-trajectories "$seed_metrics_dir/predicted_trajectories_snapshot.csv" \
                    --dataset-mode snapshot \
                    --output "$seed_metrics_dir/marker_integration_metrics_snapshot.csv" \
                    --trajectories-output "$seed_metrics_dir/marker_integration_trajectories_snapshot.csv" \
                    --aggregate-output {integration_snapshot_metrics_all_seeds} \
                    --measured-timepoints {params.measured}
                python src/pipelines/experimental/sr_pipeline/compute_marker_integration.py \
                    --dataset {input.per_minute} \
                    --raw-dataset {params.raw_measurements} \
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

# ---------------------------------------------------------------------------
# Four-layer L21 neural ODEs: the models every complexity number in the Results
# is computed from. Assembled from the lambda sweep (which trained the selected
# architecture) plus the two markers that sweep was missing, so the set is all 40.
# ---------------------------------------------------------------------------
rule assemble_node_hl4:
    output:
        marker=touch(f"{node_hl4_dir}/.assembled")
    params:
        src=f"{sparse_node_dir}/lambda_sweep/lam_3p0",
        fill=f"{exp_runs_root}/reducibility/l21_hl4_fill",
        dst=node_hl4_dir
    shell:
        """
        set -euo pipefail
        for S in 42 43 44; do
            mkdir -p "{params.dst}/seed_$S/models"
            cp -n {params.src}/seed_$S/models/* "{params.dst}/seed_$S/models/" 2>/dev/null || true
            cp -n {params.src}/seed_$S/neural_ode_diffrax_metrics.csv                   "{params.dst}/seed_$S/" 2>/dev/null || true
        done
        for d in {params.fill}/*_*/; do
            [ -d "$d/models" ] || continue
            slug=$(basename "$d"); S=${{slug##*_}}
            cp -n "$d"/models/* "{params.dst}/seed_$S/models/" 2>/dev/null || true
            tail -n +2 "$d/neural_ode_diffrax_metrics.csv"                 >> "{params.dst}/seed_$S/neural_ode_diffrax_metrics.csv" 2>/dev/null || true
        done
        """


rule tables_s12_s13_threshold_sensitivity:
    """Tables S12/S13: how both complexity contrasts move with their thresholds.

    Keyed off the Fig. 4 panel dump rather than recomputing the partition, so the tables
    cannot classify a context differently from the panel they support -- the same guarantee
    Table S8 relies on. One rule for both because a single Jacobian/Hessian pass over the
    120 checkpoints serves the dependency sweep and the interacting-pair sweep, and the
    Hessians are the slow part.
    """
    input:
        assembled=f"{node_hl4_dir}/.assembled",
        panel_inputs=fig_4_panel_inputs
    output:
        s12=table_s12_threshold_sensitivity,
        s13=table_s13_interaction_sensitivity,
        per_fit=node_hl4_threshold_per_fit
    conda:
        "../../envs/pysr.yaml"
    params:
        nn_dir=node_hl4_dir,
        threshold=exp_r2_threshold,
    shell:
        """
        set -euo pipefail
        mkdir -p $(dirname {output.s12})
        JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        python src/pipelines/experimental/reducibility/make_tables_s12_s13.py \
            --nn-dir {params.nn_dir} \
            --panel-inputs {input.panel_inputs} \
            --r2-threshold {params.threshold} \
            --per-fit-csv {output.per_fit} \
            --table-s12 {output.s12} \
            --table-s13 {output.s13}
        """


rule table_s14_lambda_calibration:
    """Table S14: the lambda ladder Fig. S3A plots, deposited as the caption promises.

    A copy rather than a recomputation: the calibration rule already writes exactly these
    columns, and recomputing them here would create a second path to the same numbers that
    could drift from the one the figure reads.
    """
    input:
        summary=sparse_node_lambda_sweep_csv
    output:
        csv=table_s14_lambda_calibration
    shell:
        """
        set -euo pipefail
        mkdir -p $(dirname {output.csv})
        cp {input.summary} {output.csv}
        """


rule node_hl4_interaction_counts:
    """Interacting input pairs per marker, scored against each law's own off-diagonal
    scale so symbolic and network laws are measured on a common footing."""
    input:
        assembled=f"{node_hl4_dir}/.assembled"
    output:
        csv=node_hl4_interactions
    conda:
        "../../envs/pysr.yaml"
    shell:
        """
        set -euo pipefail
        JAX_PLATFORMS=cpu JAX_ENABLE_X64=1         python src/pipelines/experimental/reducibility/compute_interaction_counts.py
        """


