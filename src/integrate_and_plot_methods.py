"""This module provides methods for integrating ODE systems and calculating loss for symbolic regression in signaling pathways."""

import numpy as np
import jax.numpy as jnp
import pandas as pd
import argparse
import sympy
import matplotlib.pyplot as plt
from diffrax import ODETerm, diffeqsolve, Kvaerno3, SaveAt, SteadyStateEvent, PIDController, ImplicitAdjoint

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

def calculate_loss(groundtruth, simulation_output):
    print("Calculating log MSE loss between ground truth and simulation output.")
    log_groundtruth = np.log10(np.clip(groundtruth, a_min=1e-20, a_max=None))
    log_simulation_output = np.log10(np.clip(simulation_output, a_min=1e-20, a_max=None))
    loss = np.mean(log_groundtruth - log_simulation_output, axis=0)
    print("Log MSE loss calculated:", loss)
    return loss

def plot_log_mse(loss_results, plot_path):
    """Plot the log MSE results for each method and save to the specified file."""
    methods = list(loss_results.keys())
    losses = list(loss_results.values())

    plt.figure()
    plt.bar(methods, losses)
    plt.xlabel('Method')
    plt.ylabel('Average Log MSE')
    plt.title('Log MSE by Method')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def integrate_and_calculate_loss(data, formulas, output_path, plot_path):
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

    for method, formula_path in formulas.items():
        print(f"\nProcessing method: {method}")
        print("Loading formula from:", formula_path)

        # Load and parse the formula
        with open(formula_path, 'r') as f:
            expression = sympy.sympify(f.read().strip())
        arguments = ['k_off', 'k_D', 'k_cat', 'k_inact', 'tK', 'P_u']
        ode_function = sympy.lambdify(args=arguments, expr=expression)

        def system_ode(t, y, args):
            tK, P_u, Pp = y
            k_off, k_D, k_cat, k_inact, K0, krev = args
            tK_dot = k_inact * (K0 - tK)
            kfw = ode_function(k_off, k_D, k_cat, k_inact, tK, P_u)
            P_u_dot = -k_cat * kfw + krev * Pp
            Pp_dot = k_cat * kfw - krev * Pp
            return jnp.array([tK_dot, P_u_dot, Pp_dot])

        ode_term = ODETerm(system_ode)
        loss_df = []
        fails_indices = []

        for i, sample in enumerate(data['condition_id'].unique()):
            print(f"\nProcessing sample: {i}/{len(data['condition_id'].unique())}")
            data_sample = data[data['condition_id'] == sample]
            sublists = np.array_split(data_sample, 3)
            for j, data_sample in enumerate(sublists):  # Assuming there are 3 trajectories per condition
                initial_values_ss = jnp.array([
                    data_sample.iloc[0]['K0_preeq'],
                    data_sample.iloc[0]["uP0_preeq"],
                    data_sample.iloc[0]["pP0_preeq"]
                ])
                params_ss = (
                    data_sample.iloc[0]["k_off"],
                    data_sample.iloc[0]["k_D"],
                    data_sample.iloc[0]["k_cat"],
                    data_sample.iloc[0]['k_inact'],
                    data_sample.iloc[0]['K0_preeq'],
                    data_sample.iloc[0]['krev']
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
                    data_sample.iloc[0]["k_off"],
                    data_sample.iloc[0]["k_D"],
                    data_sample.iloc[0]["k_cat"],
                    data_sample.iloc[0]['k_inact'],
                    data_sample.iloc[0]['K0'],
                    data_sample.iloc[0]['krev']
                )
                ts = jnp.array(data_sample['time'].to_numpy()[:-1])

                print("Starting simulation integration...")
                solution_simu = integrate_simulation(ode_term, initial_values_sim, params_simu, solver, ts)
                if solution_simu is None:
                    print(f"Simulation integration failed for sample {i}, trajectory {j}. Skipping to next trajectory.")
                    fails_indices.append((i, j))
                    loss_df.append([None, None, None])
                    continue
                print(data_sample.columns)
                groundtruth = np.vstack([
                    data_sample['tK'][:-1].to_numpy(),
                    data_sample['P_u'][:-1].to_numpy(),
                    data_sample['P_p'][:-1].to_numpy()
                ]).T
                print("Calculating loss for sample:", i, "trajectory:", j)
                loss_ls = calculate_loss(groundtruth, solution_simu.ys)
                loss_df.append(loss_ls)

        average_log_mse = np.nanmean([np.mean(loss) for loss in loss_df if loss is not None])
        print(f"Average Log MSE for {method}: {average_log_mse}")
        loss_results[method] = average_log_mse

    print("Saving loss results to:", output_path)
    with open(output_path, 'w') as f:
        for method, mse in loss_results.items():
            f.write(f"{method}: {mse}\n")
    print("Loss results saved.")

    print("Plotting log MSE results...")
    plot_log_mse(loss_results, plot_path)
    print("Log MSE plot saved to:", plot_path)
    print("Integration and loss calculation completed for all methods.")

def main():
    """
    Main function to parse arguments and run the integration and loss calculation.
    """
    parser = argparse.ArgumentParser(description='Integrate ODE system, calculate loss, and plot results using multiple formulas.')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use from the dataset')
    parser.add_argument('--trajectory_column', type=str, help='Column name that identifies different trajectories in the dataset')
    parser.add_argument('--data_proportion', type=float, help='Proportion of the dataset to use for analysis')
    parser.add_argument('--formulas', nargs='+', required=True, help='Paths to the formula files for each method')
    parser.add_argument('--methods', nargs='+', required=True, help='List of method names corresponding to the formula files')
    parser.add_argument('--output', required=True, help='Path to the file to save the loss results')
    parser.add_argument('--plot', required=True, help='Path to save the generated plot')
    args = parser.parse_args()

    formulas = dict(zip(args.methods, args.formulas))
    data = load_dataset(args.dataset, args.features, args.trajectory_column, args.data_proportion)
    integrate_and_calculate_loss(data, formulas, args.output, args.plot)

if __name__ == '__main__':
    main()