import numpy as np
import jax.numpy as jnp
import pandas as pd
import argparse
import sympy
import matplotlib.pyplot as plt
from diffrax import ODETerm, diffeqsolve, Kvaerno3, SaveAt, SteadyStateEvent, PIDController, ImplicitAdjoint

# Solver parameters (constants) - easily adjustable at the top of the script
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
SIMULATION_MAX_STEPS = 2**22

# Choose the solvers for steady-state and simulation integration
solver = Kvaerno3()
solver2 = Kvaerno3()

def load_dataset(file_path, features=None, trajectory_column=None, data_proportion=1.0):
    """
    Load the dataset from a CSV file, filter by features, and split into trajectories if specified.

    Parameters:
    - file_path: Path to the CSV file containing the dataset.
    - features: List of features to include from the dataset (None means include all).
    - trajectory_column: Column name that identifies different trajectories in the dataset (optional).
    - data_proportion: The proportion of the dataset to use for the analysis (default is 1.0, i.e., all data).

    Returns:
    - data: The filtered dataset with relevant features.
    """
    data = pd.read_csv(file_path)

    # Filter dataset columns based on features
    if features and features != "all":
        feature_list = features.split(',')
        data = data[feature_list + [trajectory_column]] if trajectory_column else data[feature_list]

    # Sample a proportion of the dataset if specified
    if data_proportion < 1.0:
        data = data.sample(frac=data_proportion, random_state=42).reset_index(drop=True)

    return data

def integrate_steady_state(ode_term, initial_values, params, solver):
    """
    Integrates the steady-state ODE system.
    """
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
                atol=SIMULATION_ATOL, rtol=SIMULATION_RTOL,
                pcoeff=STEADY_STATE_PCOEFF, icoeff=STEADY_STATE_ICOEFF, dcoeff=STEADY_STATE_DCOEFF
            ),
            adjoint=ImplicitAdjoint(),
            max_steps=STEADY_STATE_MAX_STEPS,
            throw=False
        )
        return solution_ss
    except:
        return None

def integrate_simulation(ode_term, initial_values, params, solver, ts):
    """
    Integrates the ODE system for simulation.
    """
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
        return solution_simu
    except:
        return None

def calculate_loss(groundtruth, predictions):
    """
    Calculates the mean squared error in log space between the ground truth and predictions.
    """
    loss_ls = []
    for j in range(3):
        loss = np.mean(
            (np.log(np.maximum(groundtruth[j], 0) + 1e-100) -
             np.log(np.maximum(predictions[:, j], 0) + 1e-100))**2
        )
        loss_ls.append(loss)
    return loss_ls

def plot_log_mse(loss_results, plot_path):
    """
    Generates a bar plot for the log MSE of each method and saves it to a file.

    Parameters:
    - loss_results: Dictionary containing the methods and their respective log MSE values.
    - plot_path: File path to save the generated plot.
    """
    methods = list(loss_results.keys())
    log_mse_values = list(loss_results.values())

    plt.figure(figsize=(10, 6))
    plt.bar(methods, log_mse_values, color='skyblue')
    plt.xlabel('Methods')
    plt.ylabel('Log MSE')
    plt.title('Log Mean Squared Error (MSE) for Different Methods')
    plt.grid(axis='y')

    # Save the plot to the specified file
    plt.savefig(plot_path)
    plt.close()

def integrate_and_calculate_loss(data, formulas, output_path, plot_path):
    """
    Integrates the ODE system using multiple formulas and calculates the log MSE for each method.
    Also generates a plot of the log MSE for each method.

    Parameters:
    - data: DataFrame containing the sample data and ground truth.
    - formulas: Dictionary of method names and their corresponding formula file paths.
    - output_path: File path to save the calculated loss results.
    - plot_path: File path to save the generated plot.
    """
    loss_results = {}

    for method, formula_path in formulas.items():
        # Load the symbolic formula for the current method
        with open(formula_path, 'r') as f:
            expression = sympy.sympify(f.read().strip())

        # Create a lambdified version of the expression for fast computation
        arguments = ['k_off', 'k_D', 'k_cat', 'kinact', 'tK', 'P_u']
        pp_ode_base = sympy.lambdify(args=arguments, expr=expression)

        def ode_function(t, y, args):
            tK, P_u, Pp = y
            k_off, k_D, k_cat, kinact, K0, krev = args

            tK_dot = kinact * (K0 - tK)
            kfw = pp_ode_base(k_off, k_D, k_cat, kinact, tK, P_u)
            P_u_dot = -k_cat * kfw + krev * Pp
            Pp_dot = k_cat * kfw - krev * Pp

            derivatives = jnp.array([tK_dot, P_u_dot, Pp_dot])
            return derivatives

        ode_term = ODETerm(ode_function)

        loss_df = []
        fails_indices = []

        for i, sample in enumerate(data['condition_id'].unique()):
            data_sample = data[data['condition_id'] == sample]

            initial_values_ss = jnp.array([
                data_sample.iloc[0]['K0_preeq'],
                data_sample.iloc[0]["uP0_preeq"],
                data_sample.iloc[0]["pP0_preeq"]
            ])

            params_ss = (
                data_sample.iloc[0]["k_off"],
                data_sample.iloc[0]["k_D"],
                data_sample.iloc[0]["k_cat"],
                data_sample.iloc[0]['kinact'],
                data_sample.iloc[0]['K0_preeq'],
                data_sample.iloc[0]['krev']
            )

            solution_ss = integrate_steady_state(ode_term, initial_values_ss, params_ss, solver)
            if solution_ss is None:
                fails_indices.append(i)
                loss_df.append([None, None, None])
                continue

            initial_values_sim = solution_ss.ys[-1, :]
            params_simu = (
                data_sample.iloc[0]["k_off"],
                data_sample.iloc[0]["k_D"],
                data_sample.iloc[0]["k_cat"],
                data_sample.iloc[0]['kinact'],
                data_sample.iloc[0]['K0'],
                data_sample.iloc[0]['krev']
            )

            ts = jnp.array(data_sample['time'].to_numpy()[:-1])
            solution_simu = integrate_simulation(ode_term, initial_values_sim, params_simu, solver2, ts)

            if solution_simu is None:
                fails_indices.append(i)
                loss_df.append([None, None, None])
                continue

            groundtruth = [
                data_sample['tK'][:-1].to_numpy(),
                data_sample['P_u'][:-1].to_numpy(),
                data_sample['Pp'][:-1].to_numpy()
            ]

            loss_ls = calculate_loss(groundtruth, solution_simu.ys)
            loss_df.append(loss_ls)

        # Calculate the average log MSE for the current method
        average_log_mse = np.nanmean([np.mean(loss) for loss in loss_df if loss is not None])
        loss_results[method] = average_log_mse

    # Save the loss results for each method to the output file
    with open(output_path, 'w') as f:
        for method, mse in loss_results.items():
            f.write(f"{method}: {mse}\n")

    # Generate the plot of log MSE values for each method
    plot_log_mse(loss_results, plot_path)

def main():
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

    # Create a dictionary of methods and their corresponding formula paths
    formulas = dict(zip(args.methods, args.formulas))

    data = load_dataset(args.dataset, args.features, args.trajectory_column, args.data_proportion)
    integrate_and_calculate_loss(data, formulas, args.output, args.plot)

if __name__ == '__main__':
    main()