"""This module provides functionality for loading and processing experimental data for inference.
Extracts Excel file and prepares it for parameter inference using the PEtab format.

Functions:
----------
- load_and_process_data(file_path, model): Load and process experimental data from an Excel file and prepare it for parameter inference.

Constants:
----------
- TIME_COL: Column name for time in the experimental data.
- MTX: Column name for MTX concentration in the experimental data.
- NC: Column name for NC concentration in the experimental data.
- SWITCH: Column name for switch concentration in the experimental data.
- SUBSTRATE: Column name for substrate concentration in the experimental data.
- PRODUCT: Column name for product concentration in the experimental data.
- PRODUCT_VARIABLE: Variable name for product in the experimental data.
- INCUBATION_TIME: Column name for incubation time in the experimental data.
"""

import re

import pandas as pd
import petab

TIME_COL = "Time (min)"
MTX = "MTX"
NC = "NC"
SWITCH = "SWITCH"
SUBSTRATE = "substrate"
PRODUCT = "product"
PRODUCT_VARIABLE = "prod"
INCUBATION_TIME = "incubation_time"


def load_and_process_data(file_path, model):
    """Load and process experimental data from an Excel file and prepare it for parameter inference.

    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing the experimental data.
    model : pysb.Model
        The PySB model object for which the data is being processed.
    Returns:
    --------
    petab.v1.Problem
        A PEtab problem instance containing the processed measurement, condition, observable, and parameter data.
    Notes:
    ------
    - The function reads specific sheets from the Excel file and extracts experimental conditions and measurements.
    - The conditions and measurements are processed and formatted according to PEtab standards.
    - The function handles different experimental setups, such as titration with and without incubation, and sensor response to MTX.
    - The resulting PEtab problem instance can be used for parameter estimation and model fitting.
    """
    # Load the Excel file
    excel_file = pd.ExcelFile(file_path)

    measurements = {}
    conditions = {}

    # Load the sheets
    for ic, (sheet, endcol) in enumerate(
        [
            # ("titration with 15min incubation", "250nM"),
            ("titration without incubation", "1uM"),
            # ("sensor response to MTX", "30min"),
        ]
    ):
        experiment = pd.read_excel(excel_file, sheet_name=sheet)
        conditions[f"c{ic}"] = {
            SWITCH: float(
                re.search(
                    "([0-9]*)nM BLA-2CaM-VHH", experiment.loc[0, "conditions"]
                ).group(1)
            ),
            NC: float(
                re.search("([0-9]*)nM NC5-BP", experiment.loc[1, "conditions"]).group(1)
            ),
            MTX: float(
                re.search("([0-9]*)nM MTX", experiment.loc[2, "conditions"]).group(1)
            ),
            "incubation_time": float(
                (
                    m.group(1)
                    if (
                        m := re.search(" ([0-9]*)min ", experiment.loc[3, "conditions"])
                    )
                    else 0
                )
            ),
            SUBSTRATE: float(
                re.search("([0-9]*)uM", experiment.loc[5, "conditions"]).group(1)
            )
            * 1e3,
        }
        experiment = experiment.loc[:, TIME_COL:endcol]
        measurements[sheet] = experiment.melt(
            id_vars=[TIME_COL],
            var_name=petab.v1.C.SIMULATION_CONDITION_ID,
            value_name="measurement",
        )
        measurements[sheet][petab.v1.C.OBSERVABLE_ID] = "product"
        measurements[sheet].rename(columns={TIME_COL: petab.v1.C.TIME}, inplace=True)

        if sheet in ("titration with 15min incubation", "titration without incubation"):
            measurements[sheet][petab.v1.C.SIMULATION_CONDITION_ID] = measurements[
                sheet
            ][petab.v1.C.SIMULATION_CONDITION_ID].apply(
                lambda x: f"c{ic}_{conditions[f'c{ic}']['incubation_time']}min_{x}"
            )
        if sheet == "sensor response to MTX":
            measurements[sheet][petab.v1.C.SIMULATION_CONDITION_ID] = measurements[
                sheet
            ][petab.v1.C.SIMULATION_CONDITION_ID].apply(
                lambda x: f"c{ic}_{x}_{conditions[f'c{ic}']['MTX']}nM"
            )
        measurements[sheet][petab.v1.C.OBSERVABLE_PARAMETERS] = f"scale_{PRODUCT}_c{ic}"

    measurements_df = pd.concat(measurements.values(), ignore_index=True)

    # conditions_df
    conditions_df = pd.DataFrame(
        index=measurements_df[petab.v1.C.SIMULATION_CONDITION_ID].unique()
    )
    for value in [SWITCH, NC, SUBSTRATE]:
        if value == NC and NC not in [m.name for m in model.monomers]:
            continue
        conditions_df[value] = conditions_df.index.map(
            lambda x: conditions[x.split("_")[0]][value]
        )

    conditions_df[INCUBATION_TIME] = conditions_df.index.map(
        lambda x: float(x.split("_")[1].replace("min", ""))
    )

    def convert_to_nm(x):
        m = re.search("([0-9\\.]*)([un])M", x)
        c = float(m.group(1))
        if m.group(2) == "u":
            c *= 1e3
        return c

    conditions_df[MTX] = conditions_df.index.map(
        lambda x: convert_to_nm(x.split("_")[2])
    )

    conditions_df.index.name = petab.v1.C.CONDITION_ID

    # sanitize conditions ids
    conditions_df.index = conditions_df.index.str.replace(".", "_")
    measurements_df[petab.v1.C.SIMULATION_CONDITION_ID] = measurements_df[
        petab.v1.C.SIMULATION_CONDITION_ID
    ].str.replace(".", "_")
    # observables_df
    observables_df = pd.DataFrame(
        [
            {
                petab.v1.C.OBSERVABLE_ID: PRODUCT,
                petab.v1.C.OBSERVABLE_FORMULA: f"observableParameter1_{PRODUCT} * {PRODUCT_VARIABLE}",
                petab.v1.C.NOISE_FORMULA: 1.0,
            }
        ]
    ).set_index(petab.v1.C.OBSERVABLE_ID)

    # parameters_df
    parameters_df = pd.DataFrame(
        [
            {
                petab.v1.C.PARAMETER_ID: par.name,
                petab.v1.C.ESTIMATE: int(not par.name.endswith("_phi")),
                petab.v1.C.PARAMETER_SCALE: (
                    petab.C.LIN
                    if par.name.endswith(("_Ea", "_dG", "_ddG", "_phi"))
                    else petab.C.LOG10
                ),
                petab.v1.C.NOMINAL_VALUE: par.value,
                petab.v1.C.LOWER_BOUND: (
                    0
                    if par.name.endswith("_phi")
                    else -15 if par.name.endswith(("_Ea", "_dG", "_ddG")) else 1e-3
                ),
                petab.v1.C.UPPER_BOUND: (
                    1
                    if par.name.endswith("_phi")
                    else 15 if par.name.endswith(("_Ea", "_dG", "_ddG")) else 1e8
                ),
            }
            for par in model.parameters
            if par.name not in conditions_df.columns
        ]
        + [
            {
                petab.v1.C.PARAMETER_ID: par,
                petab.v1.C.ESTIMATE: 1,
                petab.v1.C.PARAMETER_SCALE: petab.C.LOG10,
                petab.v1.C.NOMINAL_VALUE: 1e-5,
                petab.v1.C.LOWER_BOUND: 1e-7,
                petab.v1.C.UPPER_BOUND: 1e-2,
            }
            for par in measurements_df[petab.v1.C.OBSERVABLE_PARAMETERS].unique()
        ]
    ).set_index(petab.v1.C.PARAMETER_ID)

    problem = petab.v1.Problem(
        measurement_df=measurements_df,
        condition_df=conditions_df,
        observable_df=observables_df,
        parameter_df=parameters_df,
        model=petab.v1.models.pysb_model.PySBModel(model, model_id=model.name),
    )

    petab.v1.lint.lint_problem(problem)

    return problem
