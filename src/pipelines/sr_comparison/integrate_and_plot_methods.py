"""This module provides methods for integrating ODE systems and calculating loss for symbolic regression in signaling pathways."""

import numpy as np
import jax.numpy as jnp
import pandas as pd
import json
import argparse
import sympy
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from pathlib import Path
from diffrax import ODETerm, diffeqsolve, Kvaerno3, SaveAt, SteadyStateEvent, PIDController, ImplicitAdjoint
from typing import Optional
from plot_style import apply_cell_systems_style

from mm_models import TARGET_COLUMN as MM_TARGET_COLUMN
from regime_variants import MODEL_COLORS, VARIANTS

# Solver parameters
STEADY_STATE_ATOL = 1e-14
STEADY_STATE_RTOL = 0
STEADY_STATE_PCOEFF = 0.4
STEADY_STATE_ICOEFF = 1
STEADY_STATE_DCOEFF = 0
STEADY_STATE_DT0 = 1e-8
STEADY_STATE_MAX_STEPS = 2**20

SIMULATION_ATOL = 1e-4
SIMULATION_RTOL = 1e-4
SIMULATION_DT0 = 1e-8
SIMULATION_MAX_STEPS = 2**23

N_TIME_STEPS = 22

solver = Kvaerno3()
apply_cell_systems_style()

BASE_METHOD_LABELS = {
    'pysr': 'PySR',
    'aifeynman': 'AI Feynman',
    'dso': 'DSO',
    'kan': 'KAN',
    'pysindy': 'PySINDy',
    'nn': 'Neural Network',
    'mm': 'Michaelis-Menten',
    'unknown': 'Unknown'
}

BASE_METHOD_COLORS = {
    'pysr': '#1f77b4',
    'aifeynman': '#ff7f0e',
    'dso': '#2ca02c',
    'kan': '#d62728',
    'pysindy': '#9467bd',
    'nn': '#17becf',
    'mm': '#bcbd22',
    'unknown': '#7f7f7f'
}


def split_method_variant(method: str):
    for variant_key in VARIANTS.keys():
        suffix = f"_{variant_key}"
        if method.endswith(suffix):
            return method[: -len(suffix)], variant_key
    return method, None


def method_display_name(method: str) -> str:
    base_method, variant_key = split_method_variant(method)
    base_label = BASE_METHOD_LABELS.get(
        base_method,
        base_method.replace('_', ' ').title(),
    )
    if variant_key:
        variant_label = VARIANTS.get(variant_key, variant_key)
        return f"{base_label} ({variant_label})"
    return base_label


def method_color(base_method: str, variant_key: Optional[str]) -> str:
    if variant_key and (base_method, variant_key) in MODEL_COLORS:
        return MODEL_COLORS[(base_method, variant_key)]
    return BASE_METHOD_COLORS.get(base_method, '#7f7f7f')


def adjust_pu_for_variant(p_u_value: float, target_value: float, variant_key: Optional[str]) -> float:
    if variant_key == "tQSSA":
        return p_u_value + target_value
    return p_u_value

def load_dataset(file_path, features=None, trajectory_column=None, data_proportion=1.0):
    """
    Load dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        features (str, optional): Comma-separated list of features to use. Defaults to None.
        trajectory_column (str, optional): Column name that identifies different trajectories. Defaults to None.
        data_proportion (float, optional): Proportion of the dataset to use. Defaults to 1.0.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    print("Loading dataset...")
    data = pd.read_csv(file_path)

    # Filtering columns if specified
    if features and features != "all":
        print("Filtering dataset to include only specified features and trajectory column.")
        feature_list = features.split(',')
        columns_to_select = feature_list + ([trajectory_column] if trajectory_column else [])
        data = data[columns_to_select]

    # Sampling data if a proportion less than 1.0 is specified
    if data_proportion < 1.0:
        print(f"Sampling {data_proportion*100}% of the data for analysis.")
        unique_conditions = data[trajectory_column].unique()
        sampled_conditions = np.random.choice(unique_conditions, size=int(len(unique_conditions) * data_proportion), replace=False)
        data = data[data[trajectory_column].isin(sampled_conditions)].reset_index(drop=True)

        columns_to_exclude = ['condition_id', 'time']
        columns_to_transform = [col for col in data.columns if col not in columns_to_exclude]
        data[columns_to_transform] = data[columns_to_transform].map(lambda x: np.exp(x))

    print("Dataset loaded and preprocessed successfully.")
    return data

def integrate_steady_state(ode_term, initial_values, params, solver):
    """
    Integrate the ODE system to steady state.

    Args:
        ode_term (ODETerm): ODE term representing the system.
        initial_values (jnp.array): Initial values for the ODE system.
        params (tuple): Parameters for the ODE system.
        solver (Kvaerno3): Solver to use for integration.

    Returns:
        Solution or None: Solution object if successful, None otherwise.
    """
    print("Integrating steady state...")
    try:
        solution_ss = diffeqsolve(
            ode_term,
            solver=solver,
            t0=0.0,
            t1=jnp.inf,
            dt0=STEADY_STATE_DT0,
            y0=initial_values,
            args=params,
            discrete_terminating_event=SteadyStateEvent(atol=STEADY_STATE_ATOL, rtol=STEADY_STATE_RTOL),
            stepsize_controller=PIDController(
                atol=STEADY_STATE_ATOL, rtol=STEADY_STATE_RTOL,
                pcoeff=STEADY_STATE_PCOEFF, icoeff=STEADY_STATE_ICOEFF, dcoeff=STEADY_STATE_DCOEFF
            ),
            adjoint=ImplicitAdjoint(),
            max_steps=STEADY_STATE_MAX_STEPS,
            throw=False
        )
        print("Steady state integration completed.")
        return solution_ss
    except Exception as e:
        print(f"Error during steady state integration: {e}")
        return None

def integrate_simulation(ode_term, initial_values, params, solver, ts):
    """
    Integrate the ODE system for simulation.

    Args:
        ode_term (ODETerm): ODE term representing the system.
        initial_values (jnp.array): Initial values for the ODE system.
        params (tuple): Parameters for the ODE system.
        solver (Kvaerno3): Solver to use for integration.
        ts (jnp.array): Time points for the simulation.

    Returns:
        Solution or None: Solution object if successful, None otherwise.
    """
    print("Integrating simulation...")
    try:
        solution_simu = diffeqsolve(
            terms=ode_term,
            solver=solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=SIMULATION_DT0,
            y0=initial_values,
            max_steps=SIMULATION_MAX_STEPS,
            stepsize_controller=PIDController(atol=SIMULATION_ATOL, rtol=SIMULATION_RTOL),
            args=params,
            saveat=SaveAt(ts=ts),
            throw=True
        )
        print("Forward simulation completed.")
        return solution_simu
    except Exception as e:
        print(f"Error during forward simulation: {e}")
        return None

EPS = 1e-20

def calculate_loss(groundtruth, simulation_output):
    print("Calculating log MAE loss between ground truth and simulation output.")
    log_groundtruth = np.log(np.clip(groundtruth, a_min=EPS, a_max=None))
    log_simulation_output = np.log(np.clip(simulation_output, a_min=EPS, a_max=None))
    loss = np.mean(np.abs(log_groundtruth - log_simulation_output), axis=0)
    print("Log MAE loss calculated:", loss)
    return loss

def calculate_complexity(formula):
    return len(formula.atoms(sympy.Symbol, sympy.Number)) + len(
        formula.atoms(sympy.Add, sympy.Mul, sympy.Pow, sympy.Function)
    )


def _build_dataset_context(dataset_path):
    path = Path(dataset_path)
    parts = [part.lower() for part in path.parts]
    enzyme = next((p for p in parts if p.endswith('enzyme')), None)
    regime = 'dynamic' if 'dynamic' in parts else 'static' if 'static' in parts else None

    labels = []
    if enzyme:
        labels.append(enzyme.replace('_', ' ').title())
    if regime:
        labels.append(regime.capitalize())

    return ' · '.join(labels) if labels else None


def plot_log_MAE_scatter(entries, plot_path, plot_context=None, dataset_context=None):
    valid_entries = [e for e in entries if e['loss'] is not None and np.isfinite(e['loss'])]

    if not valid_entries:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No valid integrated results to plot', ha='center', va='center')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        return

    grouped = {}
    for entry in valid_entries:
        label = entry['display']
        bucket = grouped.setdefault(
            label,
            {
                'complexity': [],
                'loss': [],
                'base': entry['base'],
                'variant': entry['variant'],
            },
        )
        bucket['complexity'].append(entry['complexity'])
        bucket['loss'].append(entry['loss'])

    cmap = get_cmap('tab10')

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for idx, (label, payload) in enumerate(grouped.items()):
        color = method_color(payload['base'], payload['variant']) or cmap(idx % cmap.N)
        complexities = payload['complexity']
        losses = payload['loss']

        ax.scatter(
            complexities,
            losses,
            label=label,
            color=color,
            edgecolor='k',
            linewidth=0.4,
            s=70,
            alpha=0.85,
        )

        for x, y in zip(complexities, losses):
            ax.annotate(
                label,
                (x, y),
                textcoords='offset points',
                xytext=(0, 6),
                ha='center',
                fontsize=9,
                color=color,
            )

    ax.set_xlabel('Symbolic Formula Complexity')
    ax.set_ylabel('Log-space MAE: Mean |ln(y_hat + ε) − ln(y + ε)| (ε = 1e−20)')

    context_bits = []
    if plot_context:
        context_bits.append(plot_context)
    if dataset_context:
        context_bits.append(dataset_context)
    context_suffix = f" ({'; '.join(context_bits)})" if context_bits else ''

    ax.set_title(f'Symbolic Regression Complexity vs Log-MAE{context_suffix}')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(title='Symbolic Regression Method', frameon=False, loc='best')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

def integrate_and_calculate_loss(data, formulas, discovery_scales, output_path, plot_path, dataset_path):
    """
    Integrate ODE systems and calculate loss for multiple methods.

    Args:
        data (pd.DataFrame): Dataset containing the conditions and initial values.
        formulas (dict): Dictionary of method names and their corresponding formula file paths.
        output_path (str): Path to save the loss results.
        plot_path (str): Path to save the generated plot.
    """
    print("Starting integration and loss calculation for all methods...")
    loss_results = {}
    plot_entries = []
    condition_ids = data['condition_id'].unique()
    total_conditions = len(condition_ids)

    for method, formula_path in formulas.items():
        base_method, variant_key = split_method_variant(method)
        base_key = base_method.lower()
        display_name = method_display_name(method)

        print(f"\nProcessing method: {display_name}")
        print("Loading formula from:", formula_path)

        with open(formula_path, 'r') as f:
            formula_str = f.read().strip()
        expression = sympy.sympify(formula_str)
        complexity = calculate_complexity(expression)
        arguments = ['k_off', 'k_D', 'k_cat', 'k_inact', 'tK', 'P_u']

        scale = discovery_scales.get(base_key, discovery_scales.get(method, 'linear'))
        if scale == 'log':
            # The log-scale methods (aifeynman/kan/dso) are trained on LINEAR
            # inputs (their load_dataset exps the inputs) but predict the LOG
            # target. So evaluate the formula on the linear ODE inputs directly
            # and exp() the result to recover kcat_cg. (Previously this also
            # substituted arg -> log(arg), re-logging already-linear inputs,
            # which produced nested logs -> complex/NaN -> diffrax rejected the
            # ODE term.)
            transformed_expression = sympy.exp(expression)
            ode_function = sympy.lambdify(args=arguments, expr=transformed_expression)
        elif scale == 'linear':
            ode_function = sympy.lambdify(args=arguments, expr=expression)
        else:
            raise ValueError(f"Unsupported discovery scale for {display_name}: {scale}")

        def system_ode(t, y, args):
            tK_val, P_u_val, Pp_val = y
            k_off_val, k_D_val, k_cat_val, k_inact_val, K0_val, krev_val, target_val = args
            tK_dot = k_inact_val * (K0_val - tK_val)
            pu_arg = adjust_pu_for_variant(P_u_val, target_val, variant_key)
            kfw = ode_function(k_off_val, k_D_val, k_cat_val, k_inact_val, tK_val, pu_arg)
            P_u_dot = -k_cat_val * kfw + krev_val * Pp_val
            Pp_dot = k_cat_val * kfw - krev_val * Pp_val
            return jnp.array([tK_dot, P_u_dot, Pp_dot])

        ode_term = ODETerm(system_ode)
        loss_df = []
        fails_indices = []

        for i, sample in enumerate(condition_ids):
            print(f"\nProcessing sample: {i}/{total_conditions}")
            data_sample = data[data['condition_id'] == sample]
            if data_sample.empty:
                continue
            target_value = float(data_sample.iloc[0][MM_TARGET_COLUMN])
            if not np.isfinite(target_value):
                print(
                    f"Skipping sample {sample}: non-finite {MM_TARGET_COLUMN} for {display_name}."
                )
                continue
            sublists = np.array_split(data_sample, 3)
            for j, data_subset in enumerate(sublists):  # Assuming three trajectories per condition
                if data_subset.empty:
                    continue
                initial_values_ss = jnp.array([
                    data_subset.iloc[0]['K0_preeq'],
                    data_subset.iloc[0]["uP0_preeq"],
                    data_subset.iloc[0]["pP0_preeq"],
                ])
                params_ss = (
                    data_subset.iloc[0]["k_off"],
                    data_subset.iloc[0]["k_D"],
                    data_subset.iloc[0]["k_cat"],
                    data_subset.iloc[0]['k_inact'],
                    data_subset.iloc[0]['K0_preeq'],
                    data_subset.iloc[0]['krev'],
                    target_value,
                )

                print("Integrating to steady state...")
                solution_ss = integrate_steady_state(ode_term, initial_values_ss, params_ss, solver)
                if solution_ss is None:
                    print(f"Steady state integration failed for sample {i}, trajectory {j}. Skipping to next trajectory.")
                    fails_indices.append((i, j))
                    loss_df.append([None, None, None])
                    continue

                initial_values_sim = solution_ss.ys[-1, :]
                params_simu = (
                    data_subset.iloc[0]["k_off"],
                    data_subset.iloc[0]["k_D"],
                    data_subset.iloc[0]["k_cat"],
                    data_subset.iloc[0]['k_inact'],
                    data_subset.iloc[0]['K0'],
                    data_subset.iloc[0]['krev'],
                    target_value,
                )
                ts = jnp.array(data_subset['time'].to_numpy()[:-1])

                print("Starting simulation integration...")
                solution_simu = integrate_simulation(ode_term, initial_values_sim, params_simu, solver, ts)
                if solution_simu is None:
                    print(f"Simulation integration failed for sample {i}, trajectory {j}. Skipping to next trajectory.")
                    fails_indices.append((i, j))
                    loss_df.append([None, None, None])
                    continue

                groundtruth = np.vstack([
                    data_subset['tK'][:-1].to_numpy(),
                    data_subset['P_u'][:-1].to_numpy(),
                    data_subset['P_p'][:-1].to_numpy(),
                ]).transpose()
                print("Calculating loss for sample:", i, "trajectory:", j)
                loss_ls = calculate_loss(groundtruth, solution_simu.ys)
                loss_df.append(loss_ls)

        print("Loss DataFrame contents", loss_df)
        valid_losses = [loss[2] for loss in loss_df if loss is not None and loss[2] is not None]
        if valid_losses:
            average_log_MAE = float(np.nanmean(valid_losses))
        else:
            print(f"No valid simulation results for {display_name}. Skipping.")
            average_log_MAE = np.nan
        print(f"Average Log MAE for {display_name}: {average_log_MAE}")
        loss_results[display_name] = average_log_MAE
        plot_entries.append({
            'method': method,
            'display': display_name,
            'base': base_method,
            'variant': variant_key,
            'loss': average_log_MAE,
            'complexity': complexity,
        })

    print("Saving loss results to:", output_path)
    with open(output_path, 'w') as f:
        for method, MAE in loss_results.items():
            f.write(f"{method}: {MAE}\n")
    print("Loss results saved.")

    print("Plotting log MAE results...")
    dataset_context = _build_dataset_context(dataset_path)
    plot_log_MAE_scatter(
        plot_entries,
        plot_path,
        plot_context='Integrated Error',
        dataset_context=dataset_context
    )
    print("Log MAE plot saved to:", plot_path)
    print("Integration and loss calculation completed for all methods.")

def main():
    """
    Main function to parse arguments and run the integration and loss calculation.
    """
    print('STARTED integrate ')
    parser = argparse.ArgumentParser(description='Integrate ODE system, calculate loss, and plot results using multiple formulas.')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use from the dataset')
    parser.add_argument('--discovery-scales', type=str, help='JSON of method names and their corresponding discovery scales')
    parser.add_argument('--trajectory-column', type=str, help='Column name that identifies different trajectories in the dataset')
    parser.add_argument('--data-proportion', type=float, help='Proportion of the dataset to use for analysis')
    parser.add_argument('--formulas', nargs='+', required=True, help='Paths to the formula files for each method')
    parser.add_argument('--methods', nargs='+', required=True, help='List of method names corresponding to the formula files')
    parser.add_argument('--output', required=True, help='Path to the file to save the loss results')
    parser.add_argument('--plot', required=True, help='Path to save the generated plot')
    args = parser.parse_args()

    formulas = dict(zip(args.methods, args.formulas))
    data = load_dataset(args.dataset, args.features, args.trajectory_column, args.data_proportion)
    discovery_scales = json.loads(args.discovery_scales)
    integrate_and_calculate_loss(
        data,
        formulas,
        discovery_scales,
        args.output,
        args.plot,
        dataset_path=args.dataset
    )

if __name__ == '__main__':
    main()
