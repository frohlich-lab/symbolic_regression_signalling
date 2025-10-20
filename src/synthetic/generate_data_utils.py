"""Useful functions for generating synthetic data using amici."""

from collections.abc import Sequence

import amici
import numpy as np
import numpy.typing as npt
from amici.pysb_import import pysb2amici

TOL = 1e-12


def unscale(
    pars: Sequence[np.float64], scales: amici.ParameterScalingVector
) -> list[np.float64]:
    """
    Unscale parameters.

    :param pars: scaled parameters
    :param scales: parameter scales
    """
    return [
        par
        if scale == amici.ParameterScaling.none
        else np.log10(par)
        if scale == amici.ParameterScaling.log10
        else np.log(par)
        for par, scale in zip(pars, scales, strict=False)
    ]


def scale(
    pars: Sequence[np.float64], scales: amici.ParameterScalingVector
) -> list[np.float64]:
    """
    Scale parameters.

    :param pars: unscaled parameters
    :param scales: parameter scales
    """
    return [
        par
        if scale == amici.ParameterScaling.none
        else 10**par
        if scale == amici.ParameterScaling.log10
        else np.exp(par)
        for par, scale in zip(pars, scales, strict=False)
    ]


def compile_model(model_spec, lb, ub, *, do_compile=True):
    """
    Compile the model and generate model specification.

    :param model_spec: model specification, will be populated in-place
    :param lb: lower bound for parameter sampling
    :param ub: upper bound for parameter sampling
    :param do_compile: whether to compile the model
    """
    if model_spec['name'] == 'base_model':
        constant_parameters = [
            p.name
            for p in model_spec['model'].parameters
            if not p.name.endswith('0')
        ]
    elif model_spec['name'] == 'dynamic_model':
        constant_parameters = [
            p.name
            for p in model_spec['model'].parameters
            if p.name.endswith('0')
        ] + [
            'kinact',
            'krev',
        ]
    else:
        raise ValueError(f"Unknown model name {model_spec['name']}")

    if do_compile:
        pysb2amici(
            model_spec['model'],
            observables=model_spec['observables'],
            constant_parameters=constant_parameters,
            output_dir=model_spec['model_dir'],
            compute_conservation_laws=True,
        )

    # import amici models
    model_spec['module'] = amici.import_model_module(
        model_spec['model'].name, model_spec['model_dir']
    )
    amici_model = model_spec['module'].getModel()
    amici_solver = amici_model.getSolver()
    amici_solver.setAbsoluteTolerance(TOL)
    amici_solver.setRelativeTolerance(TOL)
    amici_solver.setAbsoluteToleranceSteadyState(0.0)
    amici_solver.setRelativeToleranceSteadyState(TOL * 1e4)
    amici_solver.setNewtonStepSteadyStateCheck(True)
    model_spec['amici_solver'] = amici_solver

    # set parameter scales and timepoints
    fpar_names = amici_model.getFixedParameterNames()
    amici_model.setParameterScale(
        amici.parameterScalingFromIntVector([
            amici.ParameterScaling.ln
            if not name.startswith(('phi_', 'ddG_'))
            else amici.ParameterScaling.none
            for name in amici_model.getParameterNames()
        ]),
    )
    model_spec['amici_model'] = amici_model
    # set sampling bounds
    model_spec['lbs'] = np.asarray([
        0.0 if name.startswith('phi_') else lb * np.log(10)
        for name in fpar_names
    ])
    # LB = log10(lb) = ln(lb) * ln(10)
    model_spec['ubs'] = np.asarray([
        1.0 if name.startswith('phi_') else ub * np.log(10)
        for name in fpar_names
    ])


def setup_static_simulation(
    specs: dict[str, dict],
    ub: float,
    lb: float,
    n_samples: int,
    perturb: str | None = None,
    perturb_count: int = 22,
    perturb_range: float = 100.0,
) -> tuple[amici.Model, amici.Solver, amici.ExpData, list[amici.ExpData]]:
    """
    Set up the static simulation.

    This function sets up the static simulation by generating random
    fixed and regular parameter values for the model. This routine generates
    a single set of fixed parameters and multiple sets of regular parameters,
    one for each condition.

    :param specs: model spec
    :param ub: upper bound for parameter sampling
    :param lb: lower bound for parameter sampling
    :param n_samples: number of conditions to sample
    :returns model: amici model
    :returns solver: amici solver
    :returns edata_base: reference simulation conditions for static model
    :returns edatas_base: sets of simulation conditions for static model
    """
    model = specs['static']['amici_model']
    solver = specs['static']['amici_solver']
    lbs = specs['static']['lbs']
    ubs = specs['static']['ubs']

    edata_base = amici.ExpData(model)
    fps = np.random.random(lbs.shape) * (ubs - lbs) + lbs
    edata_base.fixedParameters = tuple(
        np.exp(fp) if not name.startswith(('phi_', 'ddG_')) else fp
        for fp, name in zip(fps, model.getFixedParameterNames(), strict=False)
    )
    edata_base.setTimepoints([np.inf])

    edatas_base = []

    param_names = model.getParameterNames()
    param_scales = model.getParameterScale()

    if perturb is None:
        init = np.random.random((n_samples, len(param_names))) * (
            ub * np.log(10) - lb * np.log(10)
        ) + lb * np.log(10)
        for init_cond in init:
            ec = amici.ExpData(edata_base)
            ec.parameters = tuple(init_cond)
            edatas_base.append(ec)
    else:
        # Sample a single base parameter draw (in scaled space)
        base_scaled = np.random.random(len(param_names)) * (
            ub * np.log(10) - lb * np.log(10)
        ) + lb * np.log(10)
        base_actual = np.array(scale(base_scaled, param_scales), dtype=float)

        def add_variants(param: str, count: int) -> None:
            if param not in param_names:
                return
            idx = param_names.index(param)
            if count <= 0:
                return
            log_range = np.log(perturb_range)
            exponents = np.linspace(-log_range, log_range, count)
            exponents[count // 2] = 0.0
            factors = np.exp(exponents)
            for factor in factors:
                actual = base_actual.copy()
                actual[idx] = max(actual[idx] * factor, 1e-12)
                scaled = np.array(unscale(actual, param_scales), dtype=float)
                ec = amici.ExpData(edata_base)
                ec.parameters = tuple(scaled)
                edatas_base.append(ec)

        if perturb in {"substrate", "both"}:
            add_variants("uP0", perturb_count)
        if perturb in {"product", "both"}:
            add_variants("pP0", perturb_count)
        if perturb not in {"substrate", "product", "both"}:
            # Fallback to the base draw if an unknown option is supplied
            ec = amici.ExpData(edata_base)
            ec.parameters = tuple(base_scaled)
            edatas_base.append(ec)

    return model, solver, edata_base, edatas_base


def setup_dynamic_simulation(
    specs: dict[str, dict],
    edata_base: amici.ExpData,
) -> tuple[
    amici.AmiciModel,
    amici.AmiciSolver,
    amici.ExpData,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Set up base experimentation condition for dynamic model simulation.

    :param specs: model spec
    :param edata_base: reference simulation conditions for dynamic model
    :returns model: amici model
    :returns solver: amici solver
    :returns edata_dyn: dynamic simulation conditions
    :returns lbs: lower boundary for parameter sampling
    :returns ubs: upper boundary for parameter sampling
    """
    model = specs['dynamic']['amici_model']
    solver = specs['dynamic']['amici_solver']
    lbs = specs['dynamic']['lbs']
    ubs = specs['dynamic']['ubs']

    edata_dyn = amici.ExpData(model)
    edata_dyn.parameters = [
        *unscale(edata_base.fixedParameters, model.getParameterScale()),
    ]
    return model, solver, edata_dyn, lbs, ubs


def generate_dynamic_conditions(
    model: amici.AmiciModel,
    edata_dyn: amici.ExpData,
    t_min: float,
    t_max: float,
    ubs: npt.NDArray[np.float64],
    lbs: npt.NDArray[np.float64],
    nt: int = 20,
) -> amici.ExpData:
    """
    Generate dynamic experimentation conditions.

    This function generates random fixed parameters for the dynamic model.
    The timescale separation between the inactivation of the upstream activator
    and the reverse reaction is limited by resampling the inactivation
    rate. Similarly, the target concentration for the upstream activator
    is sampled in a narrow range around the pre-equilibration target value.

    :param model: amici Model
    :param edata_dyn: reference simulation conditions for dynamic model
    :param t_min: first time point for dynamic simulation
    :param t_max: last time point for dynamic simulation
    :param ubs: upper boundary for parameter sampling
    :param lbs: lower boundary for parameter sampling
    :returns ec: dynamic simulation conditions
    """
    ec = amici.ExpData(edata_dyn)

    # generate random fixed parameters
    fps = dict(
        zip(
            model.getFixedParameterNames(),
            np.exp(np.random.random(lbs.shape) * (ubs - lbs) + lbs),
            strict=False,
        ),
    )
    # resample kinact parameter based on krev parameter to avoid timescale
    # separation
    rrange = 10
    fps['kinact'] = fps['krev'] * np.exp(
        np.random.random() * np.log(rrange) * 2 - np.log(rrange)
    )

    ec.fixedParameters = tuple(fps.values())

    # resample K0 in narrower rage
    rrange = 100
    fps['K0'] *= np.exp(
        np.random.random() * np.log(rrange) * 2 - np.log(rrange)
    )
    ec.fixedParametersPreequilibration = tuple(fps.values())

    # set timepoints
    ec.setTimepoints([
        0,
        *list(np.logspace(np.log10(t_min), np.log10(t_max), nt)),
        np.inf,
    ])
    return ec


def find_good_time_interval(
    ec: amici.ExpData,
    tmin: float,
    tmax: float,
    rdata: amici.ReturnData,
    model: amici.AmiciModel,
    solver: amici.AmiciSolver,
    rtol: float = 1e-2,
) -> tuple[bool, amici.ExpData, amici.ReturnData]:
    """
    Refines the time interval for dynamic simulation.

    Refinement ensures that
    - simulation succeeds
    - the states at the first timepoint are close to preequilbration
    - the states at the final timepoint are close to steady-state

    :param ec: simulation condition
    :param tmin: initial guess for minimal timepoint
    :param tmax: initial guess for maximal timepoint
    :param rdata: simulation results for initial timepoint guesses
    :param model: amici model
    :param solver: amici solver
    :param rtol: relative tolerance for state comparison
    :returns simulation_failed: whether the simulation failed
    :returns ec: refined simulation condition
    :returns rdata: refined simulation results
    """
    simulation_failed = False
    while not simulation_failed and np.logical_or(
        (minbad := not np.allclose(rdata.x[0, :], rdata.x[1, :], rtol=rtol)),
        (maxbad := not np.allclose(rdata.x[-1, :], rdata.x[-2, :], rtol=rtol)),
    ):
        if minbad:
            tmin /= 10.0
        if maxbad:
            tmax *= 10.0
        nt = len(ec.getTimepoints()) - 2
        ec.setTimepoints([
            0,
            *list(np.logspace(np.log10(tmin), np.log10(tmax), nt)),
            np.inf,
        ])
        rdata = amici.runAmiciSimulation(model, solver, ec)
        if rdata.status != amici.AMICI_SUCCESS:
            simulation_failed = True
            continue

    return simulation_failed, ec, rdata
