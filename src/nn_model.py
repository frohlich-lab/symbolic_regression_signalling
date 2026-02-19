import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os
import copy
import sys
from pathlib import Path

from utils.seeding import resolve_seed, seed_everything

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from regime_variants import VARIANTS, augment_for_variant


def _build_optimizer(model):
    name = OPTIMIZER_NAME.lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.9)
    raise ValueError(f"Unsupported optimizer '{OPTIMIZER_NAME}'.")


def _build_scheduler(optimizer):
    name = SCHEDULER_NAME.lower()
    if name in {"none", ""}:
        return None
    if name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    if name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(10, EPOCHS // 2))
    if name == "exponentiallr":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    raise ValueError(f"Unsupported scheduler '{SCHEDULER_NAME}'.")

# Hyperparameters for the neural network
LEARNING_RATE = 6.533982776597072e-04
BATCH_SIZE = 256
EPOCHS = 600
HIDDEN_LAYERS = [256, 128]
ACTIVATION = nn.SiLU  # tuned surrogate behaves best with SiLU
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 1.3520544729384474e-04
OPTIMIZER_NAME = "RMSprop"
SCHEDULER_NAME = "CosineAnnealingLR"
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
EARLY_STOP_PATIENCE = 70
VAL_CHECK_INTERVAL = 10
MIN_DELTA = 1e-4
MIN_N_SAMPLES = 10_000
GRAD_CLIP_NORM = 4.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=DROPOUT_RATE):
        super(NeuralNet, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in HIDDEN_LAYERS:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # Add BatchNorm layer
            layers.append(ACTIVATION())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def load_dataset(
    file_path,
    dataset_size=None,
    features=None,
    seed: int = 42,
    variant: str = "sQSSA",
    return_metadata: bool = False,
):
    """
    Load dataset, normalize input features, and optionally sample.
    """
    data = pd.read_csv(file_path)

    # Optional: add meaningful ratios if helpful for MM modeling
    # data['P_u_over_tK'] = data['P_u'] / data['tK']
    # data['combined_rate'] = (data['k_off'] + data['k_cat'] + data['k_inact']) / (data['k_D'] * data['k_off'])

    target = 'kcat_cg' if 'kcat_cg' in data.columns else data.columns[-1]

    if features and features != "all":
        requested = [col.strip() for col in features.split(',') if col.strip()]
        available = [col for col in requested if col in data.columns]
        missing = [col for col in requested if col not in data.columns]
        if missing:
            raise KeyError(f"Requested NN features missing from dataset: {missing}")
        selected = available
        if target not in selected:
            selected.append(target)
        if not selected:
            raise ValueError("No valid features found for NN training.")
        data = data[selected]
    elif target not in data.columns:
        raise ValueError("Target column not present in dataset.")
    
    if variant:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Expected one of: {list(VARIANTS.keys())}")
        augmented = augment_for_variant(data, variant)
        augmented.index = data.index
        data = augmented

    # Drop non-numeric input columns (e.g., condition_id) before scaling
    non_numeric_cols = data.iloc[:, :-1].select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(
            "[WARN nn_model] Dropping non-numeric feature columns before scaling:",
            ", ".join(non_numeric_cols),
        )
        data = data.drop(columns=non_numeric_cols)
        if target not in data.columns:
            raise ValueError(
                "Target column removed when dropping non-numeric features. Please adjust feature selection."
            )
        # Ensure target remains last
        ordered_cols = [c for c in data.columns if c != target] + [target]
        data = data[ordered_cols]

    # Convert to numeric and clean infinities/NaNs before scaling
    feature_frame = data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
    target_series = pd.to_numeric(data.iloc[:, -1], errors='coerce')

    feature_cols = list(feature_frame.columns)
    combined = pd.concat([feature_frame, target_series], axis=1)
    combined.columns = feature_cols + [target]

    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    before_drop = len(combined)
    combined.dropna(inplace=True)
    if len(combined) < before_drop:
        print(
            f"[WARN nn_model] Dropped {before_drop - len(combined)} rows containing NaN/inf values before scaling."
        )

    if combined.empty:
        raise ValueError("No valid rows remaining after removing NaN/inf values.")

    data = combined.reset_index(drop=True)

    if dataset_size:
        desired = max(dataset_size, MIN_N_SAMPLES)
    else:
        desired = max(len(data), MIN_N_SAMPLES)

    sample_size = min(desired, len(data))
    if sample_size < MIN_N_SAMPLES and sample_size < desired:
        print(
            f"[WARN nn_model] Dataset provides only {sample_size} samples (< {MIN_N_SAMPLES}); using all available rows."
        )

    sampled_data = data.sample(n=sample_size, random_state=resolve_seed(seed)) if sample_size < len(data) else data

    total = len(sampled_data)
    if total < 3:
        raise ValueError("Need at least 3 samples to form train/val/test splits")

    rng = np.random.default_rng(resolve_seed(seed))
    indices = np.arange(total)
    rng.shuffle(indices)

    n_test = max(1, int(total * TEST_FRACTION))
    n_val = max(1, int(total * VAL_FRACTION))
    if total - n_test - n_val < 1:
        n_train = total - 2
        n_test = max(1, n_test)
        n_val = max(1, total - n_train - n_test)
    else:
        n_train = total - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def _reset(df):
        return df.reset_index(drop=True)

    train_data = _reset(sampled_data.iloc[train_idx])
    val_data = _reset(sampled_data.iloc[val_idx])
    test_data = _reset(sampled_data.iloc[test_idx])

    scaler_X = StandardScaler()
    scaler_X.fit(train_data[feature_cols])

    def _apply_scaler(df: pd.DataFrame) -> pd.DataFrame:
        transformed = scaler_X.transform(df[feature_cols])
        transformed_df = pd.DataFrame(transformed, columns=feature_cols, index=df.index)
        transformed_df[target] = df[target].values
        return transformed_df.reset_index(drop=True)

    train_scaled = _apply_scaler(train_data)
    val_scaled = _apply_scaler(val_data) if len(val_data) else val_data.copy()
    test_scaled = _apply_scaler(test_data)
    full_scaled = _apply_scaler(data)

    if return_metadata:
        metadata = {
            "scaler": scaler_X,
            "feature_cols": feature_cols,
            "target_col": target,
        }
        return train_scaled, val_scaled, test_scaled, full_scaled, metadata

    return train_scaled, val_scaled, test_scaled, full_scaled

def train_model(train_data, val_data, output_path, verbose=False, retrain=True, seed: int = 42):
    """
    Train a neural network on log-transformed targets with early stopping.
    """

    seed_everything(resolve_seed(seed))

    input_dim = train_data.shape[1] - 1
    model = NeuralNet(input_dim=input_dim, output_dim=1, dropout_rate=DROPOUT_RATE).to(device)

    if not retrain and os.path.exists(output_path):
        try:
            if verbose:
                print(f"Loading pretrained model from {output_path}")
            state_dict = torch.load(output_path, map_location=device)
            model.load_state_dict(state_dict)
            return model
        except (RuntimeError, KeyError) as exc:
            print(
                "[WARN nn_model] Checkpoint incompatible with current architecture; retraining from scratch."
            )
            if verbose:
                print(f"[DEBUG nn_model] load_state_dict error: {exc}")
    X_train = torch.from_numpy(train_data.iloc[:, :-1].values.astype(np.float32))
    y_train = torch.from_numpy(train_data.iloc[:, -1].values.astype(np.float32).reshape(-1, 1))

    train_dataset = TensorDataset(X_train, y_train)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    if len(val_data) > 0:
        X_val = torch.from_numpy(val_data.iloc[:, :-1].values.astype(np.float32)).to(device)
        y_val = torch.from_numpy(val_data.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)).to(device)
    else:
        X_val = None
        y_val = None

    criterion = nn.L1Loss()
    optimizer = _build_optimizer(model)
    scheduler = _build_scheduler(optimizer)

    best_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float('inf')

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val)
                val_loss = criterion(val_predictions, y_val).item()
        else:
            val_loss = train_loss

        monitor = val_loss
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor)
            else:
                scheduler.step()

        if monitor < best_loss - MIN_DELTA:
            best_loss = monitor
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        if verbose and (epoch % VAL_CHECK_INTERVAL == 0 or epoch == EPOCHS - 1):
            if X_val is not None:
                print(f"Epoch {epoch}: train Log-MAE = {train_loss:.4f} | val Log-MAE = {val_loss:.4f}")
            else:
                print(f"Epoch {epoch}: train Log-MAE = {train_loss:.4f}")

    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), output_path)
    if verbose:
        print(f"Best model saved to {output_path}")

    return model


def load_model_checkpoint(checkpoint_path, input_dim, device_override=None):
    resolved_device = device_override or device
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"NN checkpoint not found: {checkpoint_path}")
    model = NeuralNet(input_dim=input_dim, output_dim=1, dropout_rate=DROPOUT_RATE).to(resolved_device)
    state_dict = torch.load(checkpoint_path, map_location=resolved_device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def evaluate_model(model, full_data):
    """
    Evaluate model on full dataset (in original space), return MAE and predictions.
    """
    X_full = full_data.iloc[:, :-1].values.astype(np.float32)
    y_full = full_data.iloc[:, -1].values.astype(np.float32)

    X_tensor = torch.tensor(X_full, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        pred_log = model(X_tensor).cpu().numpy().flatten()
        pred = np.exp(pred_log)  # invert log-transform

    mae = np.mean(np.abs(pred_log - y_full))

    return mae, torch.tensor(pred)

def main():
    parser = argparse.ArgumentParser(description='Train NN to approximate MM')
    parser.add_argument('--dataset', required=True, help='CSV path')
    parser.add_argument('--dataset_size', type=int, help='Max sample size')
    parser.add_argument('--features', type=str, help='Comma-separated features or "all"')
    parser.add_argument('--output', required=True, help='Model output path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for training')
    parser.add_argument('--variant', type=str, choices=list(VARIANTS.keys()), default='sQSSA', help='Model variant controlling feature augmentation')
    
    args = parser.parse_args()
    seed_everything(resolve_seed(args.seed))
    train_data, val_data, test_data, full_data = load_dataset(
        args.dataset,
        args.dataset_size,
        args.features,
        seed=args.seed,
        variant=args.variant,
    )
    print(f"Training NN variant: {args.variant}")
    model = train_model(train_data, val_data, args.output, verbose=True, seed=args.seed)
    mae, _ = evaluate_model(model, test_data)
    print("MAE on test split:", mae) 

if __name__ == '__main__':
    main()
