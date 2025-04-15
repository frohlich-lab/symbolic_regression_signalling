import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os

# Hyperparameters for the neural network
LEARNING_RATE = 1
BATCH_SIZE = 512
EPOCHS = 600
HIDDEN_LAYERS = [256, 256, 128, 64]
ACTIVATION = nn.ReLU  # smooth saturation curve approximation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.03):
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

def load_dataset(file_path, dataset_size=None, features=None):
    """
    Load dataset, normalize input features, and optionally sample.
    """
    data = pd.read_csv(file_path)

    # Optional: add meaningful ratios if helpful for MM modeling
    # data['P_u_over_tK'] = data['P_u'] / data['tK']
    # data['combined_rate'] = (data['k_off'] + data['k_cat'] + data['k_inact']) / (data['k_D'] * data['k_off'])

    if features and features != "all":
        data = data[features.split(',')]
    
    # Normalize inputs only (not output)
    scaler_X = StandardScaler()
    data.iloc[:, :-1] = scaler_X.fit_transform(data.iloc[:, :-1])
    
    sampled_data = data.sample(n=min(dataset_size, len(data))) if dataset_size else data

    return sampled_data, data

def train_model(sampled_data, output_path, verbose=False, retrain=True):
    """
    Train a neural network on log-transformed targets with early stopping.
    """

    input_dim = sampled_data.shape[1] - 1

    model = NeuralNet(input_dim=input_dim, output_dim=1).to(device)

    if not retrain and os.path.exists(output_path):
        if verbose:
            print(f"Loading pretrained model from {output_path}")
        model.load_state_dict(torch.load(output_path, map_location=device))
        return model
    
    X_sampled = sampled_data.iloc[:, :-1].values
    y_sampled = sampled_data.iloc[:, -1].values

    X_sampled = torch.tensor(X_sampled, dtype=torch.float32).to(device)
    y_sampled = torch.tensor(y_sampled, dtype=torch.float32).view(-1, 1).to(device)

    dataset = TensorDataset(X_sampled, y_sampled)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = NeuralNet(input_dim=X_sampled.shape[1], output_dim=1).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    early_patience = 40

    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        scheduler.step(avg_loss)

        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: Log-MAE = {avg_loss:.4f}")

    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), output_path)
    if verbose:
        print(f"Best model saved to {output_path}")

    return model

def evaluate_model(model, full_data):
    """
    Evaluate model on full dataset (in original space), return MAE and predictions.
    """
    X_full = full_data.iloc[:, :-1].values
    y_full = full_data.iloc[:, -1].values

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
    
    args = parser.parse_args()
    sampled_data, full_data = load_dataset(args.dataset, args.dataset_size, args.features)
    model = train_model(sampled_data, args.output, verbose=True)
    mae, _ = evaluate_model(model, full_data)
    print("MAE on full data:", mae) 

if __name__ == '__main__':
    main()