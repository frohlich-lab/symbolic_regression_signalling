import argparse
import os
import json
import numpy as np
import pandas as pd
from itertools import product
import wandb
from nn_model import load_dataset, train_model, evaluate_model

def run_nn_grid_search_script(
    dataset_path,
    dataset_size,
    features,
    output_dir,
    param_grid,
    n_trials_per_setting=1
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading data from {dataset_path}")
    sampled_data, full_data = load_dataset(dataset_path, dataset_size, features)

    all_settings = list(product(
        param_grid['learning_rate'],
        param_grid['batch_size'],
        param_grid['hidden_layers'],
        param_grid['dropout_rate'],
    ))

    results = []

    for i, (lr, batch_size, hidden_layers, dropout) in enumerate(all_settings):
        setting_name = f"lr{lr}_bs{batch_size}_hl{'-'.join(map(str, hidden_layers))}_do{dropout}"
        print(f"\n[{i+1}/{len(all_settings)}] Running {setting_name}")
        trial_maes = []

        for trial in range(n_trials_per_setting):
            trial_model_path = os.path.join(output_dir, f"{setting_name}_trial{trial}.pt")

            # Override global values
            import nn_model
            nn_model.LEARNING_RATE = lr
            nn_model.BATCH_SIZE = batch_size
            nn_model.HIDDEN_LAYERS = hidden_layers
            nn_model.DROPOUT_RATE = dropout

            run = wandb.init(
                project="michaelis-menten-nn-grid-search",
                name=f"{setting_name}_trial{trial}",
                config={
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "hidden_layers": hidden_layers,
                    "dropout_rate": dropout,
                    "dataset_size": dataset_size,
                    "features": features,
                    "trial": trial,
                    "model_path": trial_model_path
                },
                reinit=True,
                mode="online"
            )

            model = train_model(sampled_data, trial_model_path, verbose=False, retrain=True)
            mae, _ = evaluate_model(model, full_data)
            trial_maes.append(mae)

            run.log({"trial_mae": mae, "model_path": trial_model_path})
            run.finish()

        result = {
            "setting": setting_name,
            "learning_rate": lr,
            "batch_size": batch_size,
            "hidden_layers": hidden_layers,
            "dropout_rate": dropout,
            "mean_mae": float(np.mean(trial_maes)),
            "std_mae": float(np.std(trial_maes)),
            "trials": trial_maes,
        }
        results.append(result)

        # Summary logging
        summary_run = wandb.init(
            project="nn-grid-search",
            name=f"{setting_name}_summary",
            config=result,
            reinit=True,
            mode="online"
        )
        summary_run.log({
            "mean_mae": result["mean_mae"],
            "std_mae": result["std_mae"]
        })
        summary_run.finish()

    report_path = os.path.join(output_dir, "grid_search_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Grid search complete. Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NN hyperparameter grid search.")
    parser.add_argument("--dataset", required=True, help="Path to dataset CSV")
    parser.add_argument("--dataset_size", type=int, default=1000, help="Number of samples to use")
    parser.add_argument("--features", type=str, default="all", help="Comma-separated features or 'all'")
    parser.add_argument("--output_dir", default="grid_results", help="Directory to store models and report")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials per hyperparameter setting")

    args = parser.parse_args()

    # Define hyperparameter space
    param_grid = {
        "learning_rate": [1e-3, 3e-4],
        "batch_size": [128, 256],
        "hidden_layers": [[128, 64], [256, 256, 128]],
        "dropout_rate": [0.0, 0.1],
    }

    run_nn_grid_search_script(
        dataset_path=args.dataset,
        dataset_size=args.dataset_size,
        features=args.features,
        output_dir=args.output_dir,
        param_grid=param_grid,
        n_trials_per_setting=args.n_trials
    )
