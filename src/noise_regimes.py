"""
This module provides functionality for symbolic regression using the PySR algorithm.
It evaluates different **biochemical regimes** by filtering the dataset based on feature ratios,
running PySR separately for each regime, and comparing the results against a Michaelis-Menten fit.

Usage:
    python pysr_regimes.py --dataset <path_to_dataset> --dataset_size <size> --features <feature_list> --temp_dir <path_to_temp_dir>
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import torch

from pysr import PySRRegressor
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from nn_model import load_dataset as nn_load_dataset, train_model, evaluate_model
import matplotlib.markers as mmarkers
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

from constants import PYSR_CONFIG

# Suppress all warnings
warnings.filterwarnings("ignore")

np.random.seed(42)  # Set random seed for reproducibility

# PySR configuration

def file_exists(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0

def michaelis_menten(P_u, k_off, k_D, k_cat, tK, k_inact=None):
    if k_inact is not None:
        denom = P_u + ((k_cat + k_off + k_inact) / (k_off * k_D))
    else:
        denom = P_u + ((k_cat + k_off) / k_off * k_D)
    return (tK * P_u) / denom

def load_dataset(file_path, dataset_size=None, features=None):
    data = pd.read_csv(file_path)
    if features and features != "all":
        data = data[features.split(',')]
    return data.map(np.exp)

def run_pysr(X, y, model_path, output_file):
    if os.path.exists(model_path):
        print(f"Loading existing PySR model from {model_path}")
        model = PySRRegressor.from_file(run_directory=os.path.dirname(model_path))
        return model
    else:
        print(f"Training new PySR model and saving to {model_path}")
        model = PySRRegressor(**PYSR_CONFIG, output_directory=os.path.dirname(os.path.dirname(model_path)), run_id="pysr")
        model.fit(X, y)
        return model
    
def save_pysr_formulas(model_dict, features, output_path):
    with open(output_path, "w") as f:
        for regime, models in model_dict.items():
            model = models['pysr']
            if model is None or model.equations_ is None or model.equations_.empty:
                continue

            # Get best formula (highest score)
            best_row = model.equations_.sort_values(by="score", ascending=False).iloc[0]
            formula = best_row["equation"]

            # Replace x0, x1, ... with actual variable names
            feature_names = features.split(',')
            for i, name in enumerate(feature_names):
                formula = formula.replace(f"x{i}", name)

            f.write(f"{regime}: {formula}    [score={best_row['score']:.4f}, loss={best_row['loss']:.4f}]\n")

def apply_log_space_gaussian_noise(df, std_multiplier):
    noisy_df = df.copy()
    target_col = noisy_df.columns[-1]
    
    # Apply log transform first (assume positive input)
    if (noisy_df[target_col] <= 0).any():
        raise ValueError("Log transform undefined for zero or negative values")
    log_vals = np.log(noisy_df[target_col])

    # Add Gaussian noise in log space
    std = log_vals.std()
    noise = np.random.normal(loc=0, scale=std_multiplier * std, size=len(noisy_df))
    noisy_df[target_col] = log_vals + noise

    # Reverse log transform
    noisy_df[target_col] = np.exp(noisy_df[target_col])


    return noisy_df 

def define_regimes_from_low_noise(base_df):
    return {
        "very_low_noise": base_df,
        "low_noise": apply_log_space_gaussian_noise(base_df, 1e-2),
        "medium_noise": apply_log_space_gaussian_noise(base_df, 1e-1),
        "high_noise": apply_log_space_gaussian_noise(base_df, 1.0),
    }

def preprocess_data(X):
    pipeline = Pipeline([
        ("log_transform", FunctionTransformer(np.log, validate=True)),
        ("scaler", StandardScaler())
    ])
    return pipeline.fit_transform(X), pipeline

def evaluate_models(data, features, output_dir, dataset_size):
        """
        Evaluate models for each noise regime using PySR, Michaelis-Menten, and Neural Network approaches.
        Generate plots and save results for further analysis.

        Args:
            data (pd.DataFrame): Input dataset.
            features (str): Comma-separated list of features.
            temp_dir (str): Directory to store temporary files and results.
            dataset_size (int): Maximum number of samples to use per regime.
        """
        os.makedirs(output_dir, exist_ok=True)
        losses, model_dict = {}, {}

        # Step 1: Filter base dataset using tight MM threshold
        base_df = data[
            (np.abs(np.log(data.iloc[:, -1]) - np.log(michaelis_menten(
            *data.iloc[:, :-1].values.T
            ))) < 0.01)
        ].reset_index(drop=True)

        # Step 2: Generate noise-based regimes
        regimes = define_regimes_from_low_noise(base_df)

        for regime, filtered in regimes.items():
            # Filter data for the current regime
            filtered = filtered.reset_index(drop=True)

            if filtered.empty:
                print(f"No data for {regime} regime.")
                continue

            os.makedirs(os.path.join(output_dir, f"{regime}/processed"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, f"{regime}/plots"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, f"{regime}/models/pysr"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, f"{regime}/models/nn"), exist_ok=True)

            filtered.to_csv(os.path.join(output_dir, f"{regime}/processed/filtered_data.csv"), index=False)

            # Sample data for training
            sample = filtered.sample(n=min(dataset_size, len(filtered)))
            sample.to_csv(os.path.join(output_dir, f"{regime}/processed/filtered_data_train_sample.csv"), index=False)
            X_sample, y_sample = sample.iloc[:, :-1].values, sample.iloc[:, -1].values
            X, y = filtered.iloc[:, :-1].values, filtered.iloc[:, -1].values

            print(f"Evaluating {regime} with {len(sample)} samples")

            # PySR Model
            temp_file = os.path.join(output_dir, f"{regime}/models/pysr/hall_of_fame.csv")
            model_path = os.path.join(output_dir, f"{regime}/models/pysr/hall_of_fame.pkl")
            pysr_model = run_pysr(X_sample, y_sample, model_path, temp_file)
            y_pred_pysr = pysr_model.predict(X)
            log_mae = np.mean(np.abs(np.log(np.maximum(y_pred_pysr, 1e-25)) - np.log(np.maximum(y, 1e-25))))
            losses[regime] = log_mae
            print(f"PySR Loss: {log_mae}")

            # Michaelis-Menten Model
            y_pred_mm = michaelis_menten(*filtered.iloc[:, :-1].values.T)
            mm_loss = np.mean(np.abs(np.log(np.maximum(y_pred_mm, 1e-25)) - np.log(np.maximum(y, 1e-25))))
            losses[f"{regime}_mm"] = mm_loss
            print(f"Michaelis-Menten Loss: {mm_loss}")

            # Neural Network Model
            X_pre, pipeline = preprocess_data(X)
            X_sample_pre = pipeline.transform(X_sample)
            full_data_pre = pd.DataFrame(np.column_stack([X_pre, np.log(y)]), columns=list(filtered.columns))
            sample_pre = pd.DataFrame(np.column_stack([X_sample_pre, np.log(y_sample)]), columns=list(filtered.columns))
            nn_path = os.path.join(output_dir, f"{regime}/models/nn/model.pkl")
            if os.path.exists(nn_path):
                print(f"Loading existing NN model for {regime}")
                nn_model = train_model(sample_pre, nn_path, verbose=False, retrain=False)
            else:
                print(f"Training new NN model for {regime}")
                nn_model = train_model(sample_pre, nn_path, verbose=False)
            losses[f"{regime}_nn"], _ = evaluate_model(nn_model, full_data_pre)
            print(f"NN Loss: {losses[f'{regime}_nn']}")

            # Store models and data for the current regime
            model_dict[regime] = {
                'data': filtered,
                'pysr': pysr_model,
                'mm': y_pred_mm,
                'nn': nn_model,
                'preprocessed': full_data_pre
            }
            print()

        # Generate plots and save results
        plot_model_subregimes(model_dict, output_dir)
        plot_error_distributions(model_dict, output_dir)
        save_pysr_formulas(model_dict, features, os.path.join(output_dir, "shared/results/pysr/all_pysr_formulas.txt"))
        plot_input_error_correlation(model_dict, output_dir)
        plot_model_error_correlation(model_dict, output_dir)
        plot_nn_vs_mm_response_curves_linear(model_dict, output_dir)
        plot_horizontal_boxplot_noise_regimes(model_dict, output_dir)

def plot_error_distributions(model_dict, output_dir):
    """
    For each regime, plot a violin plot comparing Log-MAE errors for PySR, MM, and NN.
    Output is a grid of subplots, one per regime.
    """
    error_records = []

    for regime, models in model_dict.items():
        data = models['data']
        data_pre = models['preprocessed']
        y_true = data.iloc[:, -1].values
        X = data.iloc[:, :-1].values

        # PySR
        y_pred_pysr = models['pysr'].predict(X)
        err_pysr = np.abs(np.log(np.maximum(y_pred_pysr, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        error_records.extend([{'Model': 'PySR', 'Regime': regime, 'Log-MAE': e} for e in err_pysr])

        # MM
        y_pred_mm = michaelis_menten(*data.iloc[:, :-1].values.T)
        err_mm = np.abs(np.log(np.maximum(y_pred_mm, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        error_records.extend([{'Model': 'Michaelis-Menten', 'Regime': regime, 'Log-MAE': e} for e in err_mm])

        # NN
        y_pred_nn = evaluate_model(models['nn'], data_pre)[1].detach().cpu().numpy().flatten()
        err_nn = np.abs(np.log(np.maximum(y_pred_nn, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        error_records.extend([{'Model': 'Neural Network', 'Regime': regime, 'Log-MAE': e} for e in err_nn])

    error_df = pd.DataFrame(error_records)

    regimes = model_dict.keys()
    ncols = 4
    nrows = 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), sharey=True)
    axes = axes.flatten()

    for i, regime in enumerate(regimes):
        ax = axes[i]
        subset = error_df[error_df['Regime'] == regime]
        sns.violinplot(data=subset, x='Model', y='Log-MAE', ax=ax, inner='box', palette='Set2')
        ax.set_title(regime)
        ax.set_xlabel('')
        if i % ncols == 0:
            ax.set_ylabel('Log-MAE')
        else:
            ax.set_ylabel('')

    # Remove empty axes if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Log-MAE Error Distributions by Regime", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "shared/plots/regime_loss_comparison.png"))

def plot_model_subregimes(model_dict, output_dir):
    regimes = list(model_dict.keys())
    models = ['pysr', 'mm', 'nn']
    model_labels = {'pysr': 'PySR', 'mm': 'Michaelis-Menten', 'nn': 'Neural Network'}
    model_colors = {'pysr': 'green', 'mm': 'red', 'nn': 'blue'}

    n_rows, n_cols = len(regimes), len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), sharey=True)

    indices = np.random.choice(np.arange(1, 3001), size=600, replace=False)

    for row_idx, regime_name in enumerate(regimes):
        data = model_dict[regime_name]['data'].loc[indices].copy()
        data_pre = model_dict[regime_name]['preprocessed'].loc[indices]
        X = data.iloc[:, :-1]
        y_true = data.iloc[:, -1]
        y_mm = michaelis_menten(*data.iloc[:, :-1].values.T)

        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]

            if model == 'pysr':
                y_pred = model_dict[regime_name]['pysr'].predict(X.values)
            elif model == 'mm':
                y_pred = y_mm
            elif model == 'nn':
                data_pre_subset = data_pre.loc[data.index]
                y_pred = evaluate_model(model_dict[regime_name]['nn'], data_pre_subset)[1].detach().cpu().numpy().flatten()

            error = np.abs(np.log(np.maximum(y_pred, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))

            ax.scatter(
                y_true, error, alpha=0.4, s=20,
                color=model_colors[model], label=model_labels[model],
                edgecolors='k', linewidths=0.2
            )

            if row_idx == len(regimes) - 1:
                ax.set_xlabel("Groundtruth kcat_cg", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel("Log-MAE", fontsize=11)
            if row_idx == 0:
                ax.set_title(model_labels[model], fontsize=13)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, loc='upper left')

            ax.set_xscale('log')

    fig.suptitle("Model Error Landscape by Regime and Model", fontsize=16, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "shared/plots/error_landscape.png")
    plt.savefig(output_path)

def plot_input_error_correlation(model_dict, output_dir):
    """
    Plot feature-error correlations for each regime in a grid.
    Also save an overall correlation across all regimes.
    """
    n = len(model_dict)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    all_error_df = []

    for idx, (regime, models) in enumerate(model_dict.items()):
        data = models['data']
        data_pre = models['preprocessed']
        y_true = data.iloc[:, -1].values
        X = data.iloc[:, :-1]

        error_df = pd.DataFrame()
        error_df['pysr_error'] = np.abs(np.log(np.maximum(models['pysr'].predict(X.values), 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        y_pred_mm = michaelis_menten(*data.iloc[:, :-1].values.T)
        error_df['mm_error'] = np.abs(np.log(np.maximum(y_pred_mm, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        y_pred_nn = evaluate_model(models['nn'], data_pre)[1].detach().cpu().numpy().flatten()
        error_df['nn_error'] = np.abs(np.log(np.maximum(y_pred_nn, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))

        for col in X.columns:
            error_df[col] = X[col].values

        corr = error_df.corr()[['pysr_error', 'mm_error', 'nn_error']].drop(['pysr_error', 'mm_error', 'nn_error'], axis=0)
        all_error_df.append(error_df)

        ax = axes[idx]
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title(f"{regime}")

    # Remove extra subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Feature-Error Correlations by Regime", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "shared/plots/feature_error_correlation_grid.png"))
    plt.close()

    # Overall correlation
    full_df = pd.concat(all_error_df, ignore_index=True)
    overall_corr = full_df.corr()[['pysr_error', 'mm_error', 'nn_error']].drop(['pysr_error', 'mm_error', 'nn_error'], axis=0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(overall_corr, annot=True, cmap='coolwarm', center=0)
    plt.title("Overall Feature-Error Correlation (All Regimes)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shared/plots/feature_error_correlation_overall.png"))
    plt.close()

def plot_model_error_correlation(model_dict, output_dir):
    """
    Plot model-model error correlations for each regime in a grid.
    Also save an overall correlation across all regimes.
    """
    n = len(model_dict)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    all_model_error_df = []

    for idx, (regime, models) in enumerate(model_dict.items()):
        data = models['data']
        data_pre = models['preprocessed']
        y_true = data.iloc[:, -1].values
        X = data.iloc[:, :-1]

        err_pysr = np.abs(np.log(np.maximum(models['pysr'].predict(X.values), 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        err_mm = np.abs(np.log(np.maximum(michaelis_menten(*data.iloc[:, :-1].values.T), 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        err_nn = np.abs(np.log(np.maximum(evaluate_model(models['nn'], data_pre)[1].detach().cpu().numpy().flatten(), 1e-25)) - np.log(np.maximum(y_true, 1e-25)))

        df = pd.DataFrame({'PySR': err_pysr, 'MM': err_mm, 'NN': err_nn})
        all_model_error_df.append(df)

        corr = df.corr()
        ax = axes[idx]
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title(f"{regime}")

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Model Error Correlations by Regime", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "shared/plots/model_error_correlation_grid.png"))
    plt.close()

    # Overall correlation
    df_all = pd.concat(all_model_error_df, ignore_index=True)
    overall_corr = df_all.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(overall_corr, annot=True, cmap='coolwarm', center=0)
    plt.title("Overall Model Error Correlation (All Regimes)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shared/plots/model_error_correlation_overall.png"))
    plt.close()

def plot_nn_vs_mm_response_curves_linear(model_dict, output_dir, n_samples=10, n_pu_points=200):
    """
    For each regime:
    - Select n_samples covering low to high log-MAE.
    - Fix all inputs except P_u.
    - Sweep P_u linearly from 0.01*K_M to 100*K_M (per sample).
    - Compare NN predictions to Michaelis-Menten curve.
    - Plot in linear space with adaptive x/y limits.
    - Add vertical line at the original P_u value from the sample.
    """
    for regime, models in model_dict.items():
        print(f"Generating scaled response curves for {regime}...")

        data = models['data'].copy()
        preprocessed_data = models['preprocessed']
        model = models['nn']

        y_true = data.iloc[:, -1].values
        X = data.iloc[:, :-1]
        pu_index = X.columns.get_loc("P_u")

        _, pipeline = preprocess_data(X)

        y_pred_mm = michaelis_menten(*data.iloc[:, :-1].values.T)

        # Full NN prediction and error
        y_pred_nn = evaluate_model(model, preprocessed_data)[1].detach().cpu().numpy().flatten()
        log_mae = np.abs(np.log(np.maximum(y_pred_nn, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        log_mae_mm = np.abs(np.log(np.maximum(y_pred_mm, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))

        # Select samples from across log-MAE distribution
        percentiles = np.linspace(0, 100, n_samples + 2)[1:-1]
        thresholds = np.percentile(log_mae, percentiles)
        chosen_indices = []
        for t in thresholds:
            idx = np.argmin(np.abs(log_mae - t))
            chosen_indices.append(idx)

        fig, axs = plt.subplots(2, n_samples // 2, figsize=(20, 10), sharex=False, sharey=False)
        axs = axs.flatten()

        for i, idx in enumerate(chosen_indices):
            fixed_sample = X.iloc[idx].copy()
            original_pu = fixed_sample["P_u"]

            pu_vals = np.linspace(0.01 * original_pu, 5 * original_pu, n_pu_points)

            # Sweep P_u
            varied_inputs = np.tile(fixed_sample.values, (n_pu_points, 1))
            varied_inputs[:, pu_index] = pu_vals

            # Preprocess inputs
            X_varied_inputs_pre = pipeline.transform(varied_inputs)
            X_varied_inputs_pre = torch.tensor(X_varied_inputs_pre, dtype=torch.float32)
            
            # NN prediction
            model.eval()
            with torch.no_grad():
                y_nn = np.exp(model(X_varied_inputs_pre))
            # MM prediction
            y_mm = michaelis_menten(pu_vals, *data.iloc[idx, 1:-1].values.T)

            # Plot
            ax = axs[i]
            ax.plot(pu_vals, y_nn, label="Neural Network", color='tab:blue')
            ax.plot(pu_vals, y_mm, label="Michaelis-Menten", color='tab:red', linestyle='--')
            ax.scatter(original_pu, y_true[idx], color='tab:blue', marker='o', s=100, label="Groundtruth kcat_cg")
            ax.axvline(original_pu, color='gray', linestyle=':', linewidth=1.5, label='Sampled P_u')  
            ax.set_xlim(0.01 * original_pu, 5 * original_pu)
            ax.set_xlabel("P_u")
            if i % (n_samples // 2) == 0:
                ax.set_ylabel("kcat_cg")
            ax.set_title(f"Sample {i+1} | Log-MAE: NN = {log_mae[idx]:.3f} - MM {log_mae_mm[idx]:.3f}")

        # Add suptitle slightly higher than default
        fig.suptitle(f"NN vs MM — Response Curves — {regime}", fontsize=16, y=1.07)

        # Add legend below title, but above subplots
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

        # Adjust tight layout to allow room for both title and legend
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # bottom, left, right, top

        # Save cleanly
        plt.savefig(os.path.join(output_dir, f"{regime}/plots/nn_vs_mm_response_curves_scaled.png"), bbox_inches="tight")
        plt.close()

def plot_horizontal_boxplot_noise_regimes(model_dict, output_dir):
    """
    Create a vertical stack of horizontal boxplots (1 per noise regime) showing
    Log-MAE for PySR, Michaelis-Menten, and Neural Network models.
    Styled for Cell Systems standards.
    """
    sns.set(style="whitegrid", font_scale=1.2, rc={"axes.edgecolor": "black", "axes.linewidth": 1.0})

    regimes = ['very_low_noise', 'low_noise', 'medium_noise', 'high_noise']
    regime_labels = {
        'very_low_noise': 'Very Low Noise',
        'low_noise': 'Low Noise',
        'medium_noise': 'Medium Noise',
        'high_noise': 'High Noise'
    }

    fig, axes = plt.subplots(nrows=len(regimes), ncols=1, figsize=(10, 2.8 * len(regimes)), sharex=True)

    for ax, regime in zip(axes, regimes):
        if regime not in model_dict:
            continue

        models = model_dict[regime]
        data = models['data']
        data_pre = models['preprocessed']
        y_true = data.iloc[:, -1].values
        X = data.iloc[:, :-1]

        error_records = []

        # PySR
        y_pred_pysr = models['pysr'].predict(X.values)
        err_pysr = np.abs(np.log(np.maximum(y_pred_pysr, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        error_records.extend([{'Model': 'PySR', 'Log-MAE': e} for e in err_pysr])

        # Michaelis-Menten
        y_pred_mm = michaelis_menten(*X.values.T)
        err_mm = np.abs(np.log(np.maximum(y_pred_mm, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        error_records.extend([{'Model': 'Michaelis-Menten', 'Log-MAE': e} for e in err_mm])

        # Neural Network
        y_pred_nn = evaluate_model(models['nn'], data_pre)[1].detach().cpu().numpy().flatten()
        err_nn = np.abs(np.log(np.maximum(y_pred_nn, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        error_records.extend([{'Model': 'Neural Network', 'Log-MAE': e} for e in err_nn])

        # Convert and plot
        error_df = pd.DataFrame(error_records)
        sns.boxplot(
            data=error_df,
            x="Log-MAE",
            y="Model",
            palette="Set2",
            orient="h",
            ax=ax
        )
        ax.set_title(regime_labels[regime], fontsize=14, weight='bold')
        ax.set_ylabel("")  # Avoid repeating "Model" label
        ax.tick_params(axis='both', labelsize=11)

    # Label only the bottom x-axis
    axes[-1].set_xlabel("Log-MAE", fontsize=12)
    for ax in axes[:-1]:
        ax.set_xlabel("")

    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, "shared/plots"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "shared/plots/log_mae_horizontal_boxplot.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate noise regimes using symbolic regression and compare with MM and NN.')
    parser.add_argument('--dataset', required=True, help='CSV dataset path')
    parser.add_argument('--dataset_size', type=int, help='Max samples to use')
    parser.add_argument('--features', type=str, help='Comma-separated list of features or "all"')
    args = parser.parse_args()

    data = load_dataset(args.dataset, args.dataset_size, args.features)
    dir_path = '/'.join(args.dataset.split('/')[:3])
    evaluate_models(data, args.features, output_dir=dir_path+'/noise_regimes/', dataset_size=args.dataset_size)

if __name__ == '__main__':
    main()
