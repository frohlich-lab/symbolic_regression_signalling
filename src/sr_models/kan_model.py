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
# A shallow, narrow network is deliberate: KAN's symbolic extraction only yields
# a compact, human-readable formula on small architectures. The previous default
# ([n,7,5,1], grid 40) was so over-parameterised that auto_symbolic fell back to a
# ~18-term sum of exponentials with poor loss - technically valid but useless.
N_ITERATIONS = 3  # Number of grid-refinement rounds before symbolic extraction
# The target is log(kcat_cg), so KAN is fitting log(MM) = log(k) + log(s) -
# log(K+s) - a *sum* of univariate log terms, not a literal product. Plain
# addition-node KAN handles that natively (no MultKAN needed): one hidden layer
# forms the sum (K+s), and 'log' edges + the output sum assemble log(k)+log(s)-
# log(K+s). Keep a single moderate hidden layer; the earlier 18-term-exp result
# came from an over-large net + weak training, not from missing capacity.
WIDTH = [1, 6, 1]  # first entry replaced with input dim -> [n, 6, 1]
GRID = 10  # Grid size for KAN (refined across rounds)
K = 3  # Spline order
THRESHOLD = 0.02  # Pruning threshold (lower -> keeps more structure for accuracy)

# Training Configuration
OPTIMIZER = "LBFGS"  # LBFGS is pykan's recommended optimiser for SR fitting
STEPS = 100  # Optimizer steps per refinement round
LAMB = 0.001  # Regularization parameter
LAMB_ENTROPY = 2.0  # Entropy regularization (encourages sparse activations)

# Function Library for Symbolic Representation. Include 1/x and sqrt so KAN can
# express rational / Michaelis-Menten-like forms (k/(K+s)), not just poly/exp/log.
LIBRARY = ['x', 'x^2', 'x^3', '1/x', 'sqrt', 'exp', 'log', 'abs']  # KAN function library

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

    import traceback

    # Stage 1: fit the spline network, refining the grid across rounds. Grid
    # refinement (coarse -> fine) is pykan's recommended recipe: fit a coarse
    # model, then progressively `refine` to a finer grid. Crucially we do NOT
    # call auto_symbolic here - fixing edges to symbolic functions and then
    # re-fitting raises "stack expects a non-empty TensorList" in pykan >=0.2.
    last_loss = float("nan")
    grids = [grid_value * (2 ** i) for i in range(max(1, n_iterations))]
    with tqdm(total=len(grids), desc="KAN grid refinement") as pbar:
        for round_idx, grid in enumerate(grids):
            try:
                if round_idx > 0:
                    model = model.refine(grid)
                results = model.fit(
                    dataset_dict,
                    opt=OPTIMIZER,
                    steps=steps,
                    lamb=LAMB,
                    lamb_entropy=LAMB_ENTROPY,
                )
                last_loss = float(results["test_loss"][-1])
            except Exception as e:
                print(f"Error in refinement round {round_idx + 1}: {e}")
                traceback.print_exc()
            pbar.update(1)

    # Stage 2: prune, then convert the trained splines to a symbolic formula.
    # In pykan >=0.2 ``prune`` returns a *new* compacted model, so reassign it;
    # otherwise auto_symbolic runs on the full-width network and yields an
    # unusable, bloated formula.
    best_formula = None
    try:
        pruned = model.prune(edge_th=threshold)
        if pruned is not None:
            model = pruned
        model.auto_symbolic(lib=LIBRARY)
        best_formula = model.symbolic_formula()[0][0]
    except Exception as e:
        print(f"KAN symbolic extraction failed: {e}")
        traceback.print_exc()

    # Guarantee a parseable block so downstream extraction never silently drops
    # KAN from the comparison.
    if best_formula is not None:
        with open(temp_file, "w") as f:
            f.write("Best:\n")
            f.write(f"Formula: {best_formula}\n")
            f.write(f"Loss: {last_loss}\n\n")
        print(f"KAN best formula (test_loss={last_loss}): {best_formula}")
    else:
        print("KAN produced no valid symbolic formula.")

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
