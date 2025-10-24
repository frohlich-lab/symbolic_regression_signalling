"""
This module loads a dataset, trains a KAN model to find the best symbolic formula,
and saves the intermediate results to a temporary file.

Usage:
    python kan_model.py --dataset <path_to_dataset> --temp_file <path_to_temp_file> [--dataset_size <size>] [--features <feature_list>]
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import sympy as sp
import torch
from kan import KAN
from tqdm import tqdm

TARGET_COLUMN = 'kcat_cg'

# Hyperparameters
TRAIN_SIZE = 0.8  # Training set size as a percentage of the dataset
SEED = 1  # Random seed for reproducibility

# Model Settings
N_ITERATIONS = 3  # Number of KAN model iterations
WIDTH = [5, 7, 5, 1]  # Default width pattern (first entry is replaced with input dim)
GRID = 40  # Grid size for KAN
K = 3  # Degree of polynomial features
THRESHOLD = 0.01  # Pruning threshold

# Training Configuration
OPTIMIZER = "Adam"  # Optimizer for KAN training
STEPS = 150  # Optimization steps per iteration
LAMB = 0.001  # Regularization parameter
LAMB_ENTROPY = 1.0  # Entropy regularization

# Function Library for Symbolic Representation
LIBRARY = ['x', 'x^2', 'x^3', 'exp', 'log', 'abs']  # KAN function library

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_dataset(file_path, dataset_size=None, features=None, seed=None):
    """
    Load dataset from a CSV file, sample it if specified, 
    and select specific columns if features are provided.
    """
    data = pd.read_csv(file_path)

    target = TARGET_COLUMN if TARGET_COLUMN in data.columns else data.columns[-1]
    if features and features != "all":
        requested = [col.strip() for col in features.split(',')]
        available = [col for col in requested if col in data.columns]
        missing = [col for col in requested if col not in data.columns]
        if missing:
            print(f"Warning: missing features for KAN: {missing}. Using available columns {available}.")
        selected = available + ([target] if target not in available else [])
    else:
        selected = list(data.columns)
    data = data[selected]

    if dataset_size:
        data = data.sample(n=min(dataset_size, len(data)), random_state=seed)

    if not data.empty:
        input_cols = data.columns[:-1]
        data.loc[:, input_cols] = np.exp(data.loc[:, input_cols])
    return data

def _parse_width(width_spec: str) -> List[int]:
    return [int(part.strip()) for part in width_spec.split(',') if part.strip()]


def _resolve_width(width_values: Optional[List[int]], input_dim: int) -> List[int]:
    """Ensure the first width entry matches the number of input features."""
    base = width_values[:] if width_values else WIDTH[:]
    if not base:
        base = [input_dim, input_dim + 2, input_dim + 1, 1]
    base[0] = input_dim
    return base


def find_best_formula(
    data,
    temp_file,
    n_iterations: int = N_ITERATIONS,
    steps: int = STEPS,
    threshold: float = THRESHOLD,
    hidden_width: Optional[List[int]] = None,
    grid_size: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    Run KAN model to find the best formula, saving results 
    periodically and converting output to symbolic form.
    """
    # Convert data to torch tensors
    dataset = torch.tensor(data.values, dtype=torch.float32)
    dataset_size = len(dataset)
    test_size = int((1 - TRAIN_SIZE) * dataset_size)
    train_size = dataset_size - test_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # Verify feature and label dimensions match the dataset length
    train_input = torch.stack([train_dataset[i][:-1] for i in range(len(train_dataset))])
    train_label = torch.stack([train_dataset[i][-1] for i in range(len(train_dataset))]).unsqueeze(1)
    test_input = torch.stack([test_dataset[i][:-1] for i in range(len(test_dataset))])
    test_label = torch.stack([test_dataset[i][-1] for i in range(len(test_dataset))]).unsqueeze(1)
    
    # Prepare dataset dictionary
    dataset_dict = {
        'train_input': train_input,
        'train_label': train_label,
        'test_input': test_input,
        'test_label': test_label
    }

    # Initialize KAN model
    model_seed = seed if seed is not None else SEED
    seed_everything(model_seed)
    input_dim = data.shape[1] - 1
    width = _resolve_width(hidden_width, input_dim)
    grid_value = grid_size if grid_size is not None else GRID
    model = KAN(width=width, grid=grid_value, k=K, seed=model_seed)

    # Ensure temp file is reset before appending results
    Path(temp_file).write_text("")

    with tqdm(total=n_iterations, desc="KAN Model Training") as pbar:
        for iteration in range(n_iterations):
            try:
                # Train model
                results = model.fit(
                    dataset_dict,
                    opt=OPTIMIZER,
                    steps=steps,
                    lamb=LAMB,
                    lamb_entropy=LAMB_ENTROPY
                )
            
                # Prune the model and extract the formula
                model.prune(edge_th=threshold)
                model.auto_symbolic(lib=LIBRARY)
                current_formula = model.symbolic_formula()[0][0]
                loss_value = results['test_loss'][-1]
                
                # Save progress
                with open(temp_file, 'a') as f:
                    f.write(f"Iteration {iteration + 1}:\n")
                    f.write(f"Formula: {current_formula}\n")
                    f.write(f"Loss: {loss_value}\n\n")
                
                pbar.update(1)

            except Exception as e:
                print(f"Error in iteration {iteration + 1}: {e}")
                continue

def main():
    parser = argparse.ArgumentParser(description='Run KAN for Symbolic Regression')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--dataset_size', type=int, help='Max number of samples from the dataset')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use')
    parser.add_argument('--temp_file', required=True, help='Path to save intermediate results')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--n_iterations', type=int, default=N_ITERATIONS, help='KAN outer iterations')
    parser.add_argument('--steps', type=int, default=STEPS, help='Optimizer steps per iteration')
    parser.add_argument('--threshold', type=float, default=THRESHOLD, help='Pruning threshold')
    parser.add_argument('--hidden_width', type=str, help='Comma-separated hidden widths (e.g., 5,7,5,1)')
    parser.add_argument('--grid', type=int, help='Grid size for the spline basis')

    args = parser.parse_args()
    data = load_dataset(args.dataset, args.dataset_size, args.features, args.seed)
    width_override = _parse_width(args.hidden_width) if args.hidden_width else None
    find_best_formula(
        data,
        args.temp_file,
        n_iterations=args.n_iterations,
        steps=args.steps,
        threshold=args.threshold,
        hidden_width=width_override,
        grid_size=args.grid,
        seed=args.seed,
    )

if __name__ == '__main__':
    main()
