import copy
import os
from pathlib import Path

import amici.swig_wrappers
import bionetgen as bng
import matplotlib.pyplot as plt
import petab.v1.C
import pypesto.petab
import seaborn as sns
from model import model
from pypesto.optimize import minimize
from pypesto.optimize.optimizer import FidesOptimizer
from pypesto.visualize import parameters, waterfall

from data import INCUBATION_TIME, MTX, SUBSTRATE, load_and_process_data

os.environ["BNGPATH"] = bng.defaults.bng_path
amici.swig_wrappers.logger.setLevel(0)

# Inputs
file_path = "data/experimental/static/raw/experimental_data_alexandrov.xlsx"
output_dir = Path("data/experimental/static/inference_results")
output_dir.mkdir(exist_ok=True, parents=True)

# Load and process data
problem = load_and_process_data(file_path, model)

problem.to_files(
    condition_file="conditions.tsv",
    observable_file="observables.tsv",
    parameter_file="parameters.tsv",
    measurement_file="measurements.tsv",
    model_file="model.py",
    prefix_path=output_dir,
    yaml_file="problem.yaml",
)

problem_clean = copy.deepcopy(problem)
problem_clean.condition_df = problem_clean.condition_df.drop(columns=[INCUBATION_TIME])

importer = pypesto.petab.PetabImporter(petab_problem=problem_clean)
model = importer.create_model(compute_conservation_laws=False, force_compile=True)
solver = importer.create_solver(model)
solver.setMaxSteps(int(1e4))
solver.setAbsoluteTolerance(1e-3)
solver.setRelativeTolerance(1e-6)

edatas = importer.create_objective().edatas

# Implement pre-incubation (not supported in PEtab)
for edata in edatas:
    condition = problem.condition_df.loc[edata.id]
    it = condition[INCUBATION_TIME]
    if it > 0:
        fp_presim = list(edata.fixedParameters)
        fp_presim[model.getFixedParameterIds().index(SUBSTRATE)] = 0.0
        edata.fixedParametersPresimulation = tuple(fp_presim)
        edata.reinitialization_state_idxs_sim = [
            model.getStateNames().index("glucose(switch=None, H='H1')")
        ]
        edata.reinitialization_state_idxs_presim = [
            model.getStateNames().index("glucose(switch=None, H='H1')")
        ]
        edata.t_presim = it

objective = importer.create_objective(edatas=edatas, model=model, solver=solver)
ppp = importer.create_problem(objective=objective)

r = minimize(
    problem=ppp,
    optimizer=FidesOptimizer(),
    n_starts=10,
)

waterfall(r)
parameters(r)
plt.show()

rdatas = ppp.objective(
    x=ppp.get_reduced_vector(r.optimize_result.list[0]["x"]),
    return_dict=True,
)[pypesto.C.RDATAS]
simulation_df = importer.rdatas_to_simulation_df(rdatas, model)
simulation_df[petab.v1.C.MEASUREMENT] = problem.measurement_df[petab.v1.C.MEASUREMENT]

simulation_df["dataset"] = (
    simulation_df[petab.v1.C.SIMULATION_CONDITION_ID].str.split("_").str[0]
)
simulation_df["incubation_time"] = simulation_df[
    petab.v1.C.SIMULATION_CONDITION_ID
].map(lambda x: problem.condition_df.loc[x, INCUBATION_TIME])
simulation_df["MTX"] = simulation_df[petab.v1.C.SIMULATION_CONDITION_ID].map(
    lambda x: problem.condition_df.loc[x, MTX]
)
df = simulation_df.melt(
    value_vars=[petab.v1.C.MEASUREMENT, petab.v1.C.SIMULATION],
    id_vars=[
        "dataset",
        "incubation_time",
        "MTX",
        petab.v1.C.TIME,
        petab.v1.C.SIMULATION_CONDITION_ID,
    ],
)

g = sns.FacetGrid(
    df,
    col=MTX,
    row="incubation_time",
)
g.map_dataframe(sns.lineplot, x=petab.v1.C.TIME, y="value", hue=MTX, style="variable")
plt.show()

# Extracting optimized parameters
parameter_names = [
    "bind_mtx_Ea",
    "bind_mtx_dG",
    "bind_mtx_phi",
    "bind_s_Ea",
    "bind_s_dG",
    "bind_s_H_ddG",
    "bind_s_phi",
    "k_cat_off",
    "k_cat_on",
    "k_on",
    "scale_product_c0",
]

print("Proposed values for each parameter across all runs:\n")
with open(output_dir / "all_optimized_parameters.txt", "w") as f:
    for name in parameter_names:
        print(f"Values for {name}:")
        f.write(f"Values for {name}:\n")
        for run in r.optimize_result.list:
            value = run["x"][parameter_names.index(name)]
            print(f"{value}")
            f.write(f"{value}\n")
        print()  # New line after each parameter's values
        f.write("\n")  # New line in the file after each parameter's values

# Approach 1: Extract the parameters from the best run
best_run = min(r.optimize_result.list, key=lambda run: run["fval"])

print(f"Best fval: {best_run['fval']}")
print("Best parameters:")
for name, value in zip(parameter_names, best_run["x"]):
    print(f"{name}: {value}")

with open(output_dir / "best_optimized_parameters.txt", "w") as f:
    f.write(f"Best fval: {best_run['fval']}\n")
    f.write("Best parameters:\n")
    for name, value in zip(parameter_names, best_run["x"]):
        f.write(f"{name}: {value}\n")

# Output constants file
constants_file = Path("./src/experimental/parameter_inference") / "constants.py"

with open(constants_file, "w") as f:
    f.write("# constants.py\n\n")
    f.write("# Found with parameter inference script train.py\n")
    f.write("# Extracted from petab/average_optimized_parameters.txt\n\n")

    f.write("# Parameters in linear scale (Best parameters)\n")
    f.write(f"BIND_MTX_EA = {best_run['x'][0]}   # Linear scale\n")
    f.write(f"BIND_MTX_DG = {best_run['x'][1]}    # Linear scale\n")
    f.write(f"BIND_MTX_PHI = {best_run['x'][2]}   # Linear scale\n\n")

    f.write(f"BIND_S_EA = {best_run['x'][3]}     # Linear scale\n")
    f.write(f"BIND_S_DG = {best_run['x'][4]}     # Linear scale\n")
    f.write(f"BIND_S_H_DDG = {best_run['x'][5]}  # Linear scale\n")
    f.write(f"BIND_S_PHI = {best_run['x'][6]}    # Linear scale\n\n")

    f.write("# Parameters in log10 scale (converted to linear values)\n")
    f.write(f"K_CAT_OFF = 10**({best_run['x'][7]})   # Log10 scale -> Linear value\n")
    f.write(f"K_CAT_ON = 10**({best_run['x'][8]})   # Log10 scale -> Linear value\n")
    f.write(f"K_ON = 10**({best_run['x'][9]})    # Log10 scale -> Linear value\n")
    f.write(
        f"SCALE_PRODUCT_C0 = 10**({best_run['x'][10]})  # Log10 scale -> Linear value\n\n"
    )

    f.write("# Notes:\n")
    f.write(
        "# - Parameters with log10 scales have been converted to their linear values.\n"
    )
    f.write("# - Ensure to use these linear values directly in computations.\n\n")

    f.write("# Initial values\n")
    f.write("INITIAL_ACTIVATOR_CONC = 1000\n")
    f.write("ENZYME_CONC = 10\n")
    f.write("SUBSTRATE_CONC = 50000\n\n")

    f.write("LATER_ACTIVATORS_CONC = 0.0\n\n")

    f.write("FINAL_TIME_EXPERIMENT = 44  # minutes\n")

print(f"Optimized parameters saved in {constants_file}")
