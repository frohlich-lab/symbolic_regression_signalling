import argparse
import os
from pathlib import Path
# Removed unused import
# Removed unused import
import torch.nn as nn
import wandb
from nn_model import load_dataset, train_model, evaluate_model  # Removed unused NeuralNet

ACTIVATION_LOOKUP = {
    "ReLU": nn.ReLU,
    "SiLU": nn.SiLU,
}

def train_wandb_trial():
    wandb.init(settings=wandb.Settings(_service_wait=120))  # Allow more time for the service to start
    config = wandb.config

    setting_name = (
        f"lr{config.learning_rate}_bs{config.batch_size}_hl{'-'.join(map(str, config.hidden_layers))}"
        f"_do{config.dropout_rate}_wd{config.weight_decay}_act{config.activation}_opt{config.optimizer}"
    )

    model_path = os.path.join(config.output_dir, f"{setting_name}_trial{wandb.run.id}.pt")

    # Update global nn_model hyperparameters
    import nn_model
    nn_model.LEARNING_RATE = config.learning_rate
    nn_model.BATCH_SIZE = config.batch_size
    nn_model.HIDDEN_LAYERS = config.hidden_layers
    nn_model.ACTIVATION = ACTIVATION_LOOKUP[config.activation]
    nn_model.OPTIMIZER = config.optimizer
    nn_model.WEIGHT_DECAY = config.weight_decay

    sampled_data, full_data = load_dataset(config.dataset_path, config.dataset_size, config.features)

    model = train_model(sampled_data, model_path, verbose=False, retrain=True)
    mae, _ = evaluate_model(model, full_data)

    wandb.log({"mae": mae})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_size", type=int, default=1000)
    parser.add_argument("--features", type=str, default="all")
    parser.add_argument("--output_dir", default="sweep_results")
    parser.add_argument("--sweep_id", type=str, default=None, help="Optional sweep ID to resume")
    parser.add_argument("--project", type=str, default="michaelis-menten-nn-sweep-fixed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'mae', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {
                'min': 1e-5, 
                'max': 1e-2, 
                'distribution': 'log_uniform_values'
            },
            'batch_size': {
                'values': [512, 1024]
            },
            'hidden_layers': {
                'values': [
                    [256, 128],              # simple 2-layer
                    [256, 256, 128],         # your current best
                    [512, 256, 128],         # wide-start taper
                    [256, 128, 64],          # classic taper
                    [128, 128, 64],          # narrower stack
                    [256, 128, 128, 64],     # moderate depth
                    [128, 128, 128, 128]     # uniform deep
                ]
            },
            'dropout_rate': {
                'values': [0.0]
            },
            'weight_decay': {
                'min': 1e-5, 
                'max': 1e-2, 
                'distribution': 'log_uniform_values'
            },
            'activation': {
                'values': ["ReLU", "SiLU"]
            },
            'optimizer': {
                'values': ["Adam"]
            },
            'scheduler': {
                'values': ["None", "ReduceLROnPlateau", "CosineAnnealingLR"]
            },
            'early_stopping_patience': {
                'values': [10, 20, 30]
            },

            # Static passthroughs
            'dataset_path': {'value': args.dataset},
            'dataset_size': {'value': args.dataset_size},
            'features': {'value': args.features},
            'output_dir': {'value': args.output_dir},
        }
    }
    wandb.login(relogin=False)
    # Use per-run project names so each sweep gets its own W&B project
    project_name = args.project
    if args.sweep_id is None:
        # Derive enzyme/data type from dataset path (e.g., data/enzyme/dynamic/processed/...)
        parts = Path(args.dataset).parts
        try:
            enzyme_idx = parts.index('data') + 1
            enzyme = parts[enzyme_idx]
            regime = parts[enzyme_idx + 1]
            project_name = f"{args.project}_{enzyme}_{regime}"
        except (ValueError, IndexError):
            project_name = f"{args.project}_custom"

    sweep_id = args.sweep_id or wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, function=train_wandb_trial, project=project_name, count=100)

    # Save a summary report of the grid search results
    os.makedirs(args.output_dir, exist_ok=True)

    summary_path = os.path.join(args.output_dir, "grid_search_results.txt")
    api = wandb.Api()
    sweep_path = f"{api.default_entity}/{project_name}/{sweep_id}"
    sweep = api.sweep(sweep_path)

    with open(summary_path, "w") as summary_file:
        summary_file.write("Grid Search Results Report\n")
        summary_file.write("===========================\n")
        summary_file.write(f"Project: {args.project}\n")
        summary_file.write(f"Dataset: {args.dataset}\n")
        summary_file.write(f"Dataset Size: {args.dataset_size}\n")
        summary_file.write(f"Features: {args.features}\n")
        summary_file.write(f"Output Directory: {args.output_dir}\n")
        summary_file.write("\n")
        summary_file.write("Best Runs:\n")
        summary_file.write("---------------------------\n")

        # Sort runs by the metric (e.g., mae) and get the top results
        sorted_runs = sorted(sweep.runs, key=lambda run: run.summary.get("mae", float("inf")))
        for i, run in enumerate(sorted_runs[:10]):  # Top 10 runs
            summary_file.write(f"Rank {i + 1}:\n")
            summary_file.write(f"  Run ID: {run.id}\n")
            summary_file.write(f"  MAE: {run.summary.get('mae')}\n")
            summary_file.write(f"  Hyperparameters:\n")
            for param, value in run.config.items():
                summary_file.write(f"    {param}: {value}\n")
            summary_file.write("\n")

if __name__ == "__main__":
    main()
