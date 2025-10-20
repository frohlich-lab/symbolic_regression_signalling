#!/usr/bin/env python3
"""Script to generate synthetic datasets for a specific model."""

import os
import platform
import random
import sys
from pathlib import Path

import amici.pysb_import
import bionetgen as bng
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src' / 'utils'))
from seeding import seed_everything

from data_conf import N_SAMPLES, N_T, PAR_LB, PAR_UB
from common import get_model_dir
from generate_data_utils import (
    compile_model,
    find_good_time_interval,
    generate_dynamic_conditions,
    setup_dynamic_simulation,
    setup_static_simulation,
)


def _get_platform():
    plat = platform.system().lower()
    if 'darwin' in plat:
        return 'mac'
    if 'linux' in plat:
        return 'linux'
    if 'windows' in plat:
        return 'win'
    return 'unknown'


os.environ['BNGPATH'] = str(
    Path(bng.__file__).parent / f'bng-{_get_platform()}'
)

STATIC_TRAJ_VARIANTS = {
    "static_traj": ["substrate", "product"],
    "static_traj_substrate": ["substrate"],
    "static_traj_product": ["product"],
}

DATASET_SEED_BASE = {'train': 10_000, 'val': 20_000, 'test': 30_000}

if len(sys.argv) < 3:
    raise SystemExit(
        "Usage: generate_data.py <model_name> <static|dynamic|static_traj|"
        "static_traj_substrate|static_traj_product> "
        "[--static-perturb=...] [--static-perturb-count=...] [--static-perturb-range=...]"
    )

model_name = sys.argv[1]
var = sys.argv[2]

static_perturb = None
static_perturb_count = 22
static_perturb_range = 100.0

valid_modes = {"static", "dynamic", *STATIC_TRAJ_VARIANTS}

if var not in valid_modes:
    raise SystemExit(f"Unknown simulation mode '{var}'.")

mode_key = "dynamic" if var == "dynamic" else "static"

if var in STATIC_TRAJ_VARIANTS:
    # Default to generating both perturbations unless the caller pins one via CLI.
    static_perturb = "both"

for arg in sys.argv[3:]:
    if arg.startswith("--static-perturb="):
        value = arg.split("=", 1)[1].strip().lower()
        if value not in {"none", "substrate", "product", "both"}:
            raise SystemExit(f"Unknown value for --static-perturb: {value}")
        if value != "none":
            static_perturb = value
    elif arg.startswith("--static-perturb-count="):
        static_perturb_count = int(arg.split("=", 1)[1])
    elif arg.startswith("--static-perturb-range="):
        static_perturb_range = float(arg.split("=", 1)[1])

if static_perturb_count < 1:
    static_perturb_count = 1
if static_perturb_range <= 0:
    static_perturb_range = 1.0
seed_everything(0)

# import pysb model
outdir = get_model_dir(model_name)
sys.path.insert(0, outdir)

model_specs = {
    'static': {
        'name': 'base_model',
        'model': amici.pysb_import.pysb_model_from_path(
            outdir / 'base_model.py'
        ),
        'observables': ['kcat_cg'],
        'model_dir': outdir / 'amici_models' / 'base_model',
    },
    'dynamic': {
        'name': 'dynamic_model',
        'model': amici.pysb_import.pysb_model_from_path(
            outdir / 'dynamic_model.py'
        ),
        'observables': [
            'P_p',
            'tK',
            'kcat_cg',
            'P_p_d',
            'P_u_d',
            'K_d',
        ],
        'model_dir': outdir / 'amici_models' / 'dynamic_model',
     },
}

# Compile only the specified model
compile_model(model_specs['static'], PAR_LB, PAR_UB, do_compile=True)
compile_model(model_specs['dynamic'], PAR_LB, PAR_UB, do_compile=True)

# Generate training/test set for the specified model
if var in STATIC_TRAJ_VARIANTS:
    if static_perturb in (None, "both"):
        perturb_options = STATIC_TRAJ_VARIANTS[var]
    elif static_perturb in STATIC_TRAJ_VARIANTS[var]:
        perturb_options = [static_perturb]
    else:
        raise SystemExit(
            f"Static trajectory mode '{var}' incompatible with perturb '{static_perturb}'."
        )
else:
    perturb_options = [static_perturb]

is_static_traj_mode = var in STATIC_TRAJ_VARIANTS

for perturb_option in perturb_options:
    if is_static_traj_mode:
        current_suffix = perturb_option or "mixed"
        variant_root = Path('data') / model_name / f"static_traj_{current_suffix}"
        current_dir = variant_root / 'raw'
        current_perturb = perturb_option
    else:
        variant_root = Path('data') / model_name / var
        current_dir = variant_root / 'raw'
        current_perturb = static_perturb
    os.makedirs(current_dir, exist_ok=True)

    # Clean up legacy filenames like rawtrain_data.csv that may live next to the
    # raw/ directory from older runs.
    for stray in variant_root.glob("raw*_data.csv"):
        try:
            stray.unlink()
        except OSError:
            pass

    for dataset in ['train', 'val', 'test']:
        datas = {mode_key: []}
        n_samples = N_SAMPLES[dataset]
        sample_index = 0

        for _ in tqdm(range(n_samples), desc=f"{dataset} ({var})", leave=False):
            base_seed = None
            if is_static_traj_mode:
                base_seed = DATASET_SEED_BASE[dataset] + sample_index
            retry = 0
            while True:
                if base_seed is not None:
                    seed_val = (base_seed + retry) % (2**32 - 1)
                    if seed_val == 0:
                        seed_val = 1
                    np.random.seed(seed_val)
                    random.seed(seed_val)
                edatas = {}
                models = {}
                solvers = {}
                if mode_key == 'static':
                    # Static part
                    (
                        models['static'],
                        solvers['static'],
                        edata_base,
                        edatas['static'],
                    ) = setup_static_simulation(
                        specs=model_specs,
                        ub=PAR_UB,
                        lb=PAR_LB,
                        n_samples=N_T + 2,
                        perturb=current_perturb if mode_key == 'static' else None,
                        perturb_count=static_perturb_count,
                        perturb_range=static_perturb_range,
                    )
                    for ie, edata in enumerate(edatas['static']):
                        edata.id = f"sample{sample_index}_{ie}"
                elif mode_key == 'dynamic':
                    (
                        _,
                        _,
                        edata_base,
                        _,
                    ) = setup_static_simulation(
                        specs=model_specs,
                        ub=PAR_UB,
                        lb=PAR_LB,
                        n_samples=N_T + 2,
                        perturb=None,
                    )

                    # Dynamic part
                    (
                        models['dynamic'],
                        solvers['dynamic'],
                        edata_dyn,
                        lbs,
                        ubs,
                    ) = setup_dynamic_simulation(model_specs, edata_base)
                    edatas['dynamic'] = []
                    tmin = 0.1
                    tmax = 10.0

                    ec = generate_dynamic_conditions(
                        model=models['dynamic'],
                        edata_dyn=edata_dyn,
                        t_min=tmin,
                        t_max=tmax,
                        ubs=ubs,
                        lbs=lbs,
                        nt=N_T,
                    )
                    ec.id = f"sample{sample_index}_{len(edatas['dynamic'])}"
                    rdata = amici.runAmiciSimulation(
                        models['dynamic'], solvers['dynamic'], ec
                    )
                    if rdata.status != amici.AMICI_SUCCESS:
                        continue

                    simulation_failed, ec, rdata = find_good_time_interval(
                        ec,
                        tmin,
                        tmax,
                        rdata,
                        models['dynamic'],
                        solvers['dynamic'],
                    )
                    if simulation_failed:
                        continue

                    for field in ['x', 'y']:
                        if not np.isfinite(rdata[field]).all():
                            continue
                    if not np.isfinite(np.log(rdata['x'])).all():
                        continue

                    edatas['dynamic'].append(ec)

                rdatas = {
                    k: amici.runAmiciSimulations(models[k], solvers[k], edatas[k])
                    for k in edatas
                }

                if any(
                    r.status != amici.AMICI_SUCCESS or np.isnan(r.y[0, :]).any()
                    for rdata in rdatas.values()
                    for r in rdata
                ):
                    if base_seed is not None:
                        retry += 1
                    continue

                break

            for (
                key,
                rdata,
            ) in rdatas.items():
                df_obs = amici.getSimulationObservablesAsDataFrame(
                    models[key], edatas[key], rdata
                ).dropna(
                    axis=1,
                    how='all',
                )
                df_sim = amici.getSimulationStatesAsDataFrame(
                    models[key], edatas[key], rdata
                ).dropna(axis=1, how='all')
                df = pd.concat(
                    (df_obs, df_sim[list(models[key].getStateNames())]),
                    axis=1,
                )
                # make sure everything is in log-scale
                rescale_cols = (
                    list(models[key].getStateNames())
                    + list(models[key].getObservableNames())
                    + list(models[key].getFixedParameterIds())
                    + [
                        fp + '_preeq'
                        for fp in models[key].getFixedParameterIds()
                        if fp + '_preeq' in df.columns
                    ]
                )
                df[rescale_cols] = df[rescale_cols].apply(np.log)
                df.drop(
                    columns=['datatype', 't_presim']
                    + [c for c in df.columns if c.endswith(('_scale', '_std'))],
                    inplace=True,
                )
                datas.setdefault(key, []).append(df)

            sample_index += 1

    if len(datas[mode_key]) != 0:
        df = pd.concat(datas[mode_key], ignore_index=True)

        if is_static_traj_mode and "condition_id" in df.columns:
            ids = df["condition_id"].astype(str)
            suffix = ids.str.extract(r"_(\d+)$")[0]
            df["condition_id"] = ids.str.replace(r"_(\d+)$", "", regex=True)
            suffix = pd.to_numeric(suffix, errors="coerce")
            if "time" in df.columns:
                df.drop(columns=["time"], inplace=True)
            df["time"] = 0
            mask = suffix.notna()
            df.loc[mask, "time"] = suffix[mask].astype(int)
            df.sort_values(["condition_id", "time"], inplace=True)
            df["time"] = df.groupby("condition_id").cumcount()
            df.reset_index(drop=True, inplace=True)

        df.to_csv(str(current_dir / f'{dataset}_data.csv'), index=False)
