import argparse
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

from parameter_inference import model
from pysb.simulator import ScipyOdeSimulator
import re

# --- Import constants from external module ---
from parameter_inference.constants import (
    BIND_S_EA,
    BIND_S_DG,
    BIND_S_PHI,
    BIND_S_H_DDG,
    BIND_MTX_EA,
    BIND_MTX_DG,
    BIND_MTX_PHI,
    K_CAT_OFF,
    K_CAT_ON,
    K_ON,
    ENZYME_CONC,
    SUBSTRATE_CONC,
    INITIAL_ACTIVATOR_CONC,
)

from parameter_inference.data import (
    MTX,
    SUBSTRATE,
    SWITCH,
)

# Set model parameters
model.model.parameters['bind_s_Ea'].value = BIND_S_EA
model.model.parameters['bind_s_dG'].value = BIND_S_DG
model.model.parameters['bind_s_phi'].value = BIND_S_PHI
model.model.parameters['bind_s_H_ddG'].value = BIND_S_H_DDG
model.model.parameters['bind_mtx_Ea'].value = BIND_MTX_EA
model.model.parameters['bind_mtx_dG'].value = BIND_MTX_DG
model.model.parameters['bind_mtx_phi'].value = BIND_MTX_PHI
model.model.parameters['k_cat_off'].value = K_CAT_OFF  # Example value, adjust as needed
model.model.parameters['k_cat_on'].value = K_CAT_ON  # Example value, adjust as needed
model.model.parameters['k_on'].value = K_ON
model.model.parameters[SUBSTRATE].value = SUBSTRATE_CONC
model.model.parameters[SWITCH].value = ENZYME_CONC  # Example value, adjust as needed
model.model.parameters[MTX].value = INITIAL_ACTIVATOR_CONC  # Example value, adjust as needed

SHEET_NAME = 'titration with 15min incubation'

def load_and_preprocess_data(filepath):
    data = pd.read_excel(filepath, sheet_name=SHEET_NAME)
    data = data.dropna(axis=1)
    data = data.rename(columns={'1uM': '1000nM'})
    data = data.set_index('Time (min)')
    return data

def compute_kinetic_constants():
    k_on_sub = np.exp(-BIND_S_EA)
    k_off_sub = k_on_sub * np.exp(BIND_S_DG)
    k_D = k_on_sub / k_off_sub
    k_cat = k_on_sub
    tK = ENZYME_CONC
    return k_on_sub, k_off_sub, k_D, k_cat, tK

def compute_features(time, signal, mtx, P_u_simulated, P_p_simulated, k_on_sub, k_off_sub, k_D, k_cat, tK):
    kcat_cg = np.gradient(signal, time) / k_cat
    # Interpolate the signal to smooth the R-shaped curve
    interpolation_function = interp1d(time, signal, kind='cubic', fill_value="extrapolate")
    interpolated_signal = interpolation_function(time)
    kcat_cg = np.gradient(interpolated_signal, time) / k_cat
    P_p = signal 
    P_u = SUBSTRATE_CONC - P_p

    return pd.DataFrame({
        'k_off': k_off_sub,
        'k_D': k_D,
        'k_cat': k_cat,
        'tK': tK,
        'P_u': P_u,
        'P_p': P_p,
        'mtx': mtx,
        'P_u_simulated': P_u_simulated,
        'P_p_simulated': P_p_simulated,
        'kcat_cg': kcat_cg,
    })

def build_dataset(data):
    time = data.index.to_numpy()
    k_on_sub, k_off_sub, k_D, k_cat, tK = compute_kinetic_constants()
    all_dfs = []

    for conc in data.columns:
        # Extract numeric value from the concentration column name
        match = re.search(r"([\d.]+)nM", conc)
        if match:
            conc_value = float(match.group(1))
        else:
            raise ValueError(f"Unexpected concentration format: {conc}")
        signal = data[conc].to_numpy()

        print("mtx_conc", conc_value)
        model.model.parameters[MTX].value = conc_value
        t = data.index.values  # Example time span, adjust as needed
        simres = ScipyOdeSimulator(model.model, tspan=t).run()
        dataset = simres.dataframe
        mtx = dataset['activated_switch'].values
        P_u_simulated = dataset['substrate_unbound'].values
        P_p_simulated = dataset['product'].values

        df = compute_features(time, signal, mtx, P_u_simulated, P_p_simulated, k_on_sub, k_off_sub, k_D, k_cat, tK)
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)

def save_dataset(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Saved processed experimental dataset to: {output_path}")

def logify_dataset(df):
    for column in df.columns:
        df[column] = df[column].apply(lambda x: np.log(x) if x > 0 else np.nan)
        df[column] = df[column].fillna(method='ffill')
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input Excel file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    args = parser.parse_args()

    data = load_and_preprocess_data(args.input)
    dataset = build_dataset(data)
    dataset = logify_dataset(dataset)
    save_dataset(dataset, args.output)

if __name__ == '__main__':
    main()