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

solver = Kvaerno3()
solver2 = Kvaerno3()

def load_dataset(file_path, features=None, trajectory_column=None, data_proportion=1.0):
    print("Loading dataset...")
    data = pd.read_csv(file_path)

    if features and features != "all":
        feature_list = features.split(',')
        data = data[feature_list + [trajectory_column]] if trajectory_column else data[feature_list]

    if data_proportion < 1.0:
        data = data.sample(frac=data_proportion, random_state=42).reset_index(drop=True)
    print("Dataset loaded successfully.")
    return data

def integrate_steady_state(ode_term, initial_values, params, solver):
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
                atol=SIMULATION_ATOL, rtol=SIMULATION_RTOL,
                pcoeff=STEADY_STATE_PCOEFF, icoeff=STEADY_STATE_ICOEFF, dcoeff=STEADY_STATE_DCOEFF
            ),
            adjoint=ImplicitAdjoint(),
            max_steps=STEADY_STATE_MAX_STEPS,
            throw=False
        )
        print("Steady state integration complete.")
        return solution_ss
    except:
        print("Steady state integration failed.")
        return None

def integrate_simulation(ode_term, initial_values, params, solver, ts):
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
        print("Simulation integration complete.")
        return solution_simu
    except:
        print("Simulation integration failed.")
        return None

def integrate_and_calculate_loss(data, formulas, output_path, plot_path):
    print("Starting integration and loss calculation for all methods...")
    loss_results = {}

    for method, formula_path in formulas.items():
        print(f"Processing method: {method}")

        with open(formula_path, 'r') as f:
            expression = sympy.sympify(f.read().strip())
        arguments = ['k_off', 'k_D', 'k_cat', 'k_inact', 'tK', 'P_u']
        pp_ode_base = sympy.lambdify(args=arguments, expr=expression)

        def ode_function(t, y, args):
            tK, P_u, Pp = y
            k_off, k_D, k_cat, k_inact, K0, krev = args
            tK_dot = k_inact * (K0 - tK)
            kfw = pp_ode_base(k_off, k_D, k_cat, k_inact, tK, P_u)
            P_u_dot = -k_cat * kfw + krev * Pp
            Pp_dot = k_cat * kfw - krev * Pp
            return jnp.array([tK_dot, P_u_dot, Pp_dot])

        ode_term = ODETerm(ode_function)
        loss_df = []
        fails_indices = []

        for i, sample in enumerate(data['condition_id'].unique()):
            print(f"Sample {i+1}/{len(data['condition_id'].unique())} for method {method}")

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
                data_sample.iloc[0]['k_inact'],
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
                data_sample.iloc[0]['k_inact'],
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

        average_log_mse = np.nanmean([np.mean(loss) for loss in loss_df if loss is not None])
        print(f"Average Log MSE for {method}: {average_log_mse}")
        loss_results[method] = average_log_mse

    with open(output_path, 'w') as f:
        for method, mse in loss_results.items():
            f.write(f"{method}: {mse}\n")
    print(f"Loss results saved to {output_path}")

    plot_log_mse(loss_results, plot_path)
    print(f"Log MSE plot saved to {plot_path}")
    print("Integration and loss calculation completed for all methods.")

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

    formulas = dict(zip(args.methods, args.formulas))
    data = load_dataset(args.dataset, args.features, args.trajectory_column, args.data_proportion)
    integrate_and_calculate_loss(data, formulas, args.output, args.plot)

if __name__ == '__main__':
    main()