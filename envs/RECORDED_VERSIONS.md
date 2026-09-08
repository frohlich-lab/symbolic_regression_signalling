# Recorded software versions

The environment files in this directory (`*.yaml`) specify dependencies loosely,
so a fresh `conda env create` will not necessarily reproduce the stack the
published results were computed on. This file records the versions that were
actually installed, recovered from the solved conda environments and the Julia
depot at the time of the release.

## Julia (drives the symbolic-regression search)

Pinned exactly by `envs/julia/Manifest.toml`, which records 115 packages.

| | Version |
|---|---|
| SymbolicRegression.jl | 1.11.3 |
| Julia | 1.12.1 |
| PythonCall.jl | 0.9.26 (pinned in `Project.toml`) |

The three Julia environments the workflow created (`dataset_size_regimes`,
`mm_deviation_regimes`, `noise_regimes`) resolved to byte-identical manifests,
so the symbolic-regression backend was consistent across all runs.

## Python

Twelve conda environments were solved over the life of the project. Because
`pysr` was unpinned, they did not all resolve to the same version:

| Package | Versions observed | Notes |
|---|---|---|
| pysr | 1.5.6, 1.5.8, 1.5.9, 1.5.10 | 1.5.9 in 9 of 12 environments |
| numpy | 1.26.4, 2.0.2, 2.2.5 | `numpy<2.0` constrains the pysr environment only |
| torch | 2.4.1, 2.5.1, 2.7.0, 2.7.1 | |
| jax | 0.4.28, 0.4.30 | |
| diffrax | 0.6.1 | |
| sympy | 1.13.3, 1.14.0 | |
| python | 3.9 (pysr env) | |

`envs/pysr.yaml` now pins `pysr=1.5.9`, the version used for the majority of
runs. The Julia manifest above fixes the component that determines search
behaviour, so the PySR patch-version spread affects the Python wrapper rather
than the underlying algorithm.

## Exact locks

`envs/locks/` holds fully solved exports taken from the conda environments
Snakemake actually built (`.snakemake/conda/<hash>`), not from named
environments — Snakemake does not create those, so `conda env export -n <name>`
returns nothing.

| Environment | Locked | Covers |
|---|---|---|
| `pysr` | yes, 113 packages | all symbolic-regression results |
| `nn` | yes, 116 packages | MLP baseline and sparse neural ODE |
| `base` | yes, 129 packages | general analysis and plotting |
| `aifeynman`, `kan` | no | prefixes on disk are empty shells |
| `dso`, `pysindy` | no | export failed against the stored prefix |

The four unlocked environments are the alternative symbolic-regression backends
used only for the four-method benchmark; their observed versions are in the
table above.

To regenerate a lock, export from the prefix rather than by name:

```bash
conda env export -p .snakemake/conda/<hash> --no-builds \
  | grep -v '^prefix:' > envs/locks/<env>.lock.yaml
```
