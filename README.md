# symbolic_regression_signalling

Code accompanying:

> **Symbolic regression enables coarse-grained model discovery of intracellular
> signalling dynamics.**
> Theodore de Pomereu, Fabian Fröhlich.
> Preprint: https://doi.org/10.64898/2026.08.20.745973

Symbolic regression (SR) is used here in two ways: as a method for discovering
coarse-grained rate laws, and as a data-driven test of whether the measured
variables support a compact description at all. The repository covers the
synthetic enzyme-kinetics benchmarks, the data-degradation experiments, and the
symbolic-regression and sparse neural ODE models of ERK phosphorylation across
40 cancer-relevant overexpression contexts.

## Layout

| Path | Contents |
|---|---|
| `Snakefile`, `workflow/rules/` | Snakemake workflow; one rule file per analysis strand (`experimental`, `sr_comparison`, `regimes`) |
| `config.yaml` | Enzyme model, data type, feature sets, per-method dataset sizes and time budgets |
| `src/synthetic/base_models/` | PySB mass-action models: two-, three- and four-step catalysis, closed (`base_model.py`) and open (`dynamic_model.py`) |
| `src/pipelines/synthetic/` | Synthetic data generation and preprocessing |
| `src/pipelines/regimes/` | The five data-degradation experiments: noise, dataset size, Michaelis-Menten deviation, kinetic limits, measurement lens |
| `src/pipelines/sr_comparison/` | Cross-method benchmark (PySR, AI-Feynman, DSO, KAN) and the MLP baseline |
| `src/pipelines/experimental/` | ERK analysis: trajectory fitting, symbolic regression, sparse neural ODE |
| `src/sr_models/` | Thin wrappers giving each SR method a common interface |
| `envs/` | Conda environments, one per method (methods have conflicting dependencies) |
| `results/published/` | Hyperparameter grids and per-context metrics reported in the paper — see the README there |
| `sweeps/` | Search-space definitions for the W&B hyperparameter sweeps |
| `simulation_requirements.txt` | pip requirements for the AMICI simulation stack used to generate the synthetic data; not installed by the workflow |
| `tests/` | Unit tests |
| `docs/` | Figure-and-table map, preregistration, cluster runbook — see `docs/README.md` |

## Data

The experimental measurements are **not redistributed here**. They are the mass
cytometry data of Lun et al. (2019), available from Mendeley Data at
https://doi.org/10.17632/3kh7ypz232.1 under CC BY 4.0. Place the download at the
path given by `experimental.raw_time_course` in `config.yaml`. The file used
here is 7,070,048 bytes with

    sha256  06d53807ddf20d8a886f5c182038b7b7f95e1812df19fa18c745a3ac0b6000e7

The synthetic enzyme-kinetics datasets are not stored either; they are
regenerated deterministically by the workflow. The `data/` tree the workflow
produces is roughly 20 GB and is not version controlled.

## Installation

Requires `conda` (or `mamba`) and Snakemake. Everything else is installed by the
workflow itself, one environment per method, because several of the symbolic
regression backends have mutually incompatible dependencies:

```bash
conda install -n base -c conda-forge mamba snakemake-minimal
git clone https://github.com/frohlich-lab/symbolic_regression_signalling
cd symbolic_regression_signalling
```

Generating the synthetic enzyme datasets additionally needs AMICI, which the
workflow does not install. Create that environment separately:

```bash
pip install -r simulation_requirements.txt
```

Julia is not installed separately — PySR provisions it through `juliapkg`. The
exact Julia stack used for the published results is pinned in
`envs/julia/Manifest.toml` (115 packages, SymbolicRegression.jl 1.11.3 on Julia
1.12.1).

## Running

The workflow is driven by Snakemake, with a separate conda environment per SR
method:

```bash
snakemake --use-conda --cores 8
```

Which enzyme model and data type are analysed is set in `config.yaml`
(`enzyme_model`, `data_type`). The synthetic strands regenerate their own data;
the experimental strand needs the Lun et al. download in place first.

## Tests

```bash
pytest tests
```

The suite covers marker metadata loading, regime variant handling, the
train/test split policies and shared constants. Two tests skip without the
experimental dataset in place.

## Reproducibility

Analyses use fixed seed 42 unless stated otherwise; the degradation and
experimental runs use seeds 42–44. Full search settings, the 72-configuration
PySR sweep and the 54-configuration neural ODE sweep are described in the paper's
SI Appendix, and their outputs are in `results/published/`.

Software versions matter here: the symbolic-regression search is driven by
SymbolicRegression.jl, pinned exactly by `envs/julia/Manifest.toml`. The conda
environment files specify Python dependencies loosely, so
`envs/RECORDED_VERSIONS.md` records the versions actually installed when the
published results were produced, and how to export exact locks.

## Which code made which figure

`docs/FIGURE_MAP.md` maps every figure and table in the paper to the rule or
script that produces it, and notes which panels are drawn by hand rather than
generated.

## Citing

See `CITATION.cff`. Please cite both the paper and the archived software release.

## Licence

MIT — see `LICENSE`.
