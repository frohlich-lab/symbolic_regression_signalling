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

# Suppress all warnings
warnings.filterwarnings("ignore")

# Define biochemical regimes
REGIMES = {
    "low_noise": lambda df: df[(np.abs(np.log(df["kcat_cg"]) - np.log(michaelis_menten(
        df["P_u"], df["tK"], df["k_cat"], df["k_off"], df["k_inact"], df["k_D"]))) < 0.01)],
    "medium_noise": lambda df: df[(np.abs(np.log(df["kcat_cg"]) - np.log(michaelis_menten(
        df["P_u"], df["tK"], df["k_cat"], df["k_off"], df["k_inact"], df["k_D"]))) >= 0.01) & 
        (np.abs(np.log(df["kcat_cg"]) - np.log(michaelis_menten(
        df["P_u"], df["tK"], df["k_cat"], df["k_off"], df["k_inact"], df["k_D"]))) < 0.1)],
    "high_noise": lambda df: df[(np.abs(np.log(df["kcat_cg"]) - np.log(michaelis_menten(
        df["P_u"], df["tK"], df["k_cat"], df["k_off"], df["k_inact"], df["k_D"]))) >= 0.1) & 
        (np.abs(np.log(df["kcat_cg"]) - np.log(michaelis_menten(
        df["P_u"], df["tK"], df["k_cat"], df["k_off"], df["k_inact"], df["k_D"]))) < 1.0)],
    "very_high_noise": lambda df: df[(np.abs(np.log(df["kcat_cg"]) - np.log(michaelis_menten(
        df["P_u"], df["tK"], df["k_cat"], df["k_off"], df["k_inact"], df["k_D"]))) >= 1.0)],
}

# PySR configuration
PYSR_CONFIG = {
    'niterations': 300,
    'population_size': 30,
    'populations': 15,
    'binary_operators': ["+", "*", "/", "-"],
    'unary_operators': ["exp", "log"],
    'maxsize': 20,
    'parsimony': 1,
    'verbosity': 0,
    'batching': True,
    'annealing': True,
    'elementwise_loss': "my_loss(x,y)=(log(max(x,0)+1e-25)-log(max(y,0)+1e-25))^2"
}
def file_exists(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0

def michaelis_menten(P_u, tK, k_cat, k_off, k_inact, k_D):
    denom = P_u + ((k_cat + k_off + k_inact) / (k_off * k_D))
    return (tK * P_u) / denom

def load_dataset(file_path, dataset_size=None, features=None):
    data = pd.read_csv(file_path)
    if features and features != "all":
        data = data[features.split(',')]
    return data.map(np.exp)

def run_pysr(X, y, model_path, output_file):
    if os.path.exists(model_path):
        print(f"Loading existing PySR model from {model_path}")
        model = PySRRegressor.from_file(model_path)
        return model
    else:
        print(f"Training new PySR model and saving to {model_path}")
        model = PySRRegressor(**PYSR_CONFIG, equation_file=output_file)
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

def preprocess_data(X):
    pipeline = Pipeline([
        ("log_transform", FunctionTransformer(np.log, validate=True)),
        ("scaler", StandardScaler())
    ])
    return pipeline.fit_transform(X), pipeline

def evaluate_models(data, features, temp_dir, dataset_size):
        """
        Evaluate models for each noise regime using PySR, Michaelis-Menten, and Neural Network approaches.
        Generate plots and save results for further analysis.

        Args:
            data (pd.DataFrame): Input dataset.
            features (str): Comma-separated list of features.
            temp_dir (str): Directory to store temporary files and results.
            dataset_size (int): Maximum number of samples to use per regime.
        """
        os.makedirs(temp_dir, exist_ok=True)
        losses, model_dict = {}, {}

        for regime, filter_func in REGIMES.items():
            # Filter data for the current regime
            filtered = filter_func(data).reset_index(drop=True)
            if filtered.empty:
                print(f"No data for {regime} regime.")
                continue

            # Sample data for training
            sample = filtered.sample(n=min(dataset_size, len(filtered)))
            X_sample, y_sample = sample.iloc[:, :-1].values, sample.iloc[:, -1].values
            X, y = filtered.iloc[:, :-1].values, filtered.iloc[:, -1].values

            print(f"Evaluating {regime} with {len(sample)} samples")

            # PySR Model
            temp_file = os.path.join(temp_dir, f"pysr_{regime}.csv")
            model_path = os.path.join(temp_dir, f"pysr_{regime}.pkl")
            pysr_model = run_pysr(X_sample, y_sample, model_path, temp_file)
            y_pred_pysr = pysr_model.predict(X)
            log_mae = np.mean(np.abs(np.log(np.maximum(y_pred_pysr, 1e-25)) - np.log(np.maximum(y, 1e-25))))
            losses[regime] = log_mae
            print(f"PySR Loss: {log_mae}")

            # Michaelis-Menten Model
            P_u, tK = filtered["P_u"].values, filtered["tK"].values
            k_cat, k_off = filtered["k_cat"].values, filtered["k_off"].values
            k_inact, k_D = filtered["k_inact"].values, filtered["k_D"].values
            y_pred_mm = michaelis_menten(P_u, tK, k_cat, k_off, k_inact, k_D)
            mm_loss = np.mean(np.abs(np.log(np.maximum(y_pred_mm, 1e-25)) - np.log(np.maximum(y, 1e-25))))
            losses[f"{regime}_mm"] = mm_loss
            print(f"Michaelis-Menten Loss: {mm_loss}")

            # Neural Network Model
            X_pre, pipeline = preprocess_data(X)
            X_sample_pre = pipeline.transform(X_sample)
            full_data_pre = pd.DataFrame(np.column_stack([X_pre, np.log(y)]), columns=list(filtered.columns))
            sample_pre = pd.DataFrame(np.column_stack([X_sample_pre, np.log(y_sample)]), columns=list(filtered.columns))
            nn_path = os.path.join(temp_dir, f"nn_{regime}.pth")
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
        plot_model_subregimes(model_dict, temp_dir)
        plot_error_distributions(model_dict, temp_dir)
        save_pysr_formulas(model_dict, features, os.path.join(temp_dir, "all_pysr_formulas.txt"))
        plot_input_error_correlation(model_dict, temp_dir)
        plot_model_error_correlation(model_dict, temp_dir)
        plot_nn_vs_mm_response_curves_linear(model_dict, temp_dir)

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
        y_pred_mm = michaelis_menten(
            data["P_u"], data["tK"],
            data["k_cat"], data["k_off"],
            data["k_inact"], data["k_D"]
        )
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
    plt.savefig(os.path.join(output_dir, "regime_loss_comparison.png"))
    plt.show()

def plot_model_subregimes(model_dict, output_dir):
    models = ['pysr', 'mm', 'nn']

    regime_colors = {
        "low_noise": "green",
        "medium_noise": "orange",
        "high_noise": "red",
        "very_high_noise": "purple"
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    indices = np.random.choice(np.arange(1, 3001), size=600, replace=False)

    for i, model in enumerate(models):
        ax = axes[i]

        for regime_name, _ in REGIMES.items():
            data = model_dict[regime_name]['data'].loc[indices].copy()
            data_pre = model_dict[regime_name]['preprocessed'].loc[indices]
            X = data.iloc[:, :-1]
            data["y_true"] = data.iloc[:, -1]
            data["y_mm"] = michaelis_menten(
                data["P_u"], data["tK"],
                data["k_cat"], data["k_off"],
                data["k_inact"], data["k_D"]
            )

            if model == 'pysr':
                y_pred = model_dict[regime_name]['pysr'].predict(X.values)
            elif model == 'mm':
                y_pred = data["y_mm"].values
            elif model == 'nn':
                data_pre_subset = data_pre.loc[data.index]
                y_pred = evaluate_model(model_dict[regime_name]['nn'], data_pre_subset)[1].detach().cpu().numpy().flatten()

            y_true = data["y_true"].values
            error = np.abs(np.log(np.maximum(y_pred, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))

            ax.scatter(
                data["y_true"], error, alpha=0.6, s=20,
                label=regime_name if regime_name not in ax.get_legend_handles_labels()[1] else None,
                color=regime_colors[regime_name], edgecolors='k', linewidths=0.2
            )

        ax.set_xscale('log')
        ax.set_xlabel("y_{true}", fontsize=11)
        if i == 0:
            ax.set_ylabel("Log Error (|log(pred) - log(true)|)", fontsize=11)
        ax.set_title(f"{model.upper()} Error by Noise Regime", fontsize=13)
        ax.legend(frameon=False)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "error_landscape_all_regimes.png")
    plt.savefig(output_path)
    plt.show()

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
        y_pred_mm = michaelis_menten(
            data["P_u"], data["tK"],
            data["k_cat"], data["k_off"],
            data["k_inact"], data["k_D"]
        )
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
    plt.savefig(os.path.join(output_dir, "feature_error_correlation_grid.png"))
    plt.close()

    # Overall correlation
    full_df = pd.concat(all_error_df, ignore_index=True)
    overall_corr = full_df.corr()[['pysr_error', 'mm_error', 'nn_error']].drop(['pysr_error', 'mm_error', 'nn_error'], axis=0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(overall_corr, annot=True, cmap='coolwarm', center=0)
    plt.title("Overall Feature-Error Correlation (All Regimes)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_error_correlation_overall.png"))
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
        err_mm = np.abs(np.log(np.maximum(michaelis_menten(
            data["P_u"], data["tK"],
            data["k_cat"], data["k_off"],
            data["k_inact"], data["k_D"]
        ), 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
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
    plt.savefig(os.path.join(output_dir, "model_error_correlation_grid.png"))
    plt.close()

    # Overall correlation
    df_all = pd.concat(all_model_error_df, ignore_index=True)
    overall_corr = df_all.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(overall_corr, annot=True, cmap='coolwarm', center=0)
    plt.title("Overall Model Error Correlation (All Regimes)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_error_correlation_overall.png"))
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

        y_pred_mm = michaelis_menten(
            data["P_u"], data["tK"],
            data["k_cat"], data["k_off"],
            data["k_inact"], data["k_D"]
        )

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

        fig, axs = plt.subplots(2, n_samples // 2, figsize=(20, 8), sharex=False, sharey=False)
        axs = axs.flatten()

        for i, idx in enumerate(chosen_indices):
            fixed_sample = X.iloc[idx].copy()
            original_pu = fixed_sample["P_u"]
            tK = fixed_sample["tK"]
            k_cat = fixed_sample["k_cat"]
            k_off = fixed_sample["k_off"]
            k_inact = fixed_sample["k_inact"]
            k_D = fixed_sample["k_D"]

            K_M = (k_cat + k_off + k_inact) / (k_off * k_D)

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
            y_mm = michaelis_menten(pu_vals, tK, k_cat, k_off, k_inact, k_D)

            # Plot
            ax = axs[i]
            ax.plot(pu_vals, y_nn, label="Neural Network", color='tab:blue')
            ax.plot(pu_vals, y_mm, label="Michaelis-Menten", color='tab:red', linestyle='--')
            ax.scatter(original_pu, y_true[idx], color='tab:blue', marker='o', s=100, label="Original P_u")
            ax.axvline(original_pu, color='gray', linestyle=':', linewidth=1.5, label='Original P_u')  # <-- Added
            ax.set_xlim(0.01 * original_pu, 5 * original_pu)
            ax.set_xlabel("P_u")
            if i % (n_samples // 2) == 0:
                ax.set_ylabel("Output")
            ax.set_title(f"Sample {i+1} | Log-MAE: NN = {log_mae[idx]:.3f} - MM {log_mae_mm[idx]:.3f}")

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.suptitle(f"NN vs MM — Response Curves (Scaled P_u) — {regime}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"nn_vs_mm_response_curves_scaled_{regime}.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate noise regimes using symbolic regression and compare with MM and NN.')
    parser.add_argument('--dataset', required=True, help='CSV dataset path')
    parser.add_argument('--dataset_size', type=int, help='Max samples to use')
    parser.add_argument('--features', type=str, help='Comma-separated list of features or "all"')
    args = parser.parse_args()

    data = load_dataset(args.dataset, args.dataset_size, args.features)
    evaluate_models(data, args.features, temp_dir='data/pysr_noise/', dataset_size=args.dataset_size)

if __name__ == '__main__':
    main()
