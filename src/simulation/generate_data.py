#!/usr/bin/env python3
"""Script to generate synthetic datasets for a specific model."""

import os
import platform
import sys
from pathlib import Path

import amici.pysb_import
import bionetgen as bng
import numpy as np
import pandas as pd
from tqdm import tqdm

from simulation.data_conf import N_SAMPLES, N_T, PAR_LB, PAR_UB
from simulation.common import get_model_dir
from simulation.generate_data import (
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

model_name = sys.argv[1]

np.random.seed(0)

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
    # 'dynamic': {
    #     'name': 'dynamic_model',
    #     'model': amici.pysb_import.pysb_model_from_path(
    #         outdir / 'dynamic_model.py'
    #     ),
    #     'observables': [
    #         'P_p',
    #         'tK',
    #         'kcat_cg',
    #         'P_p_d',
    #         'P_u_d',
    #         'K_d',
    #     ],
    #     'model_dir': outdir / 'amici_models' / 'dynamic_model',
    # },
}

for model_spec in model_specs.values():
    # compile amici models
    compile_model(model_spec, PAR_LB, PAR_UB, do_compile=True)

# generate training/test set
for dataset in ['train', 'val', 'test']:
    datas = {
        'static': [],
        # 'dynamic': [],
    }
    n_samples = N_SAMPLES[dataset]
    for _ in tqdm(range(n_samples)):
        while True:
            edatas = {}
            models = {}
            solvers = {}
            # base part
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
            )
            for ie, edata in enumerate(edatas['static']):
                edata.id = f"sample{len(datas['static'])}_{ie}"

            # # dynamic part
            # (
            #     models['dynamic'],
            #     solvers['dynamic'],
            #     edata_dyn,
            #     lbs,
            #     ubs,
            # ) = setup_dynamic_simulation(model_specs, edata_base)
            # edatas['dynamic'] = []
            # tmin = 0.1
            # tmax = 10.0

            # ec = generate_dynamic_conditions(
            #     model=models['dynamic'],
            #     edata_dyn=edata_dyn,
            #     t_min=tmin,
            #     t_max=tmax,
            #     ubs=ubs,
            #     lbs=lbs,
            #     nt=N_T,
            # )
            # ec.id = f"sample{len(datas['static'])}_{len(edatas['dynamic'])}"
            # rdata = amici.runAmiciSimulation(
            #     models['dynamic'], solvers['dynamic'], ec
            # )
            # if rdata.status != amici.AMICI_SUCCESS:
            #     continue

            # simulation_failed, ec, rdata = find_good_time_interval(
            #     ec,
            #     tmin,
            #     tmax,
            #     rdata,
            #     models['dynamic'],
            #     solvers['dynamic'],
            # )
            # if simulation_failed:
            #     continue

            # for field in ['x', 'y']:
            #     if not np.isfinite(rdata[field]).all():
            #         continue
            # if not np.isfinite(np.log(rdata['x'])).all():
            #     continue

            # edatas['dynamic'].append(ec)

            rdatas = {
                k: amici.runAmiciSimulations(models[k], solvers[k], edatas[k])
                for k in edatas
            }

            if any(
                r.status != amici.AMICI_SUCCESS or np.isnan(r.y[0, :]).any()
                for rdata in rdatas.values()
                for r in rdata
            ):
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
            datas[key].append(df)

    for key, data in datas.items():
        if len(data) != 0:
            df = pd.concat(data)
            df.to_csv(outdir / f'{dataset}_{key}_data.csv', index=False)
