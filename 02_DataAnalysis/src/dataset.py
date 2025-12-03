import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.optim as optim


class ActivityDataset(Dataset):
    """PyTorch Dataset for activity data"""
    
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
# ============================================================================
# 3. AUTOENCODER MODEL
# ============================================================================

class ActivityAutoencoder(nn.Module):
    """Autoencoder for activity anomaly detection"""
    
    def __init__(self, input_dim, encoding_dim=16, dropout=0.2):
        super(ActivityAutoencoder, self).__init__()
        
        # Calculate intermediate dimensions
        dim1 = max(64, input_dim * 2)
        dim2 = max(32, input_dim)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.BatchNorm1d(dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(dim1, dim2),
            nn.BatchNorm1d(dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(dim2, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, dim2),
            nn.BatchNorm1d(dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(dim2, dim1),
            nn.BatchNorm1d(dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(dim1, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Get latent representation"""
        return self.encoder(x)

def train_autoencoder(
    X_train, 
    X_val, 
    X_test, 
    model_class, 
    input_dim, 
    encoding_dim, 
    lr = 3e-4, 
    batch_size = 64, 
    patience = 30, 
    epochs = 100, 
    device = None):
    """
Train an autoencoder model for activity data.

Parameters
----------
X_train : array-like or torch.Tensor
    Training dataset.
X_val : array-like or torch.Tensor
    Validation dataset.
X_test : array-like or torch.Tensor
    Testing dataset.
model_class : type
    Autoencoder model class to be instantiated and trained.
input_dim : int
    Dimensionality of the input data.
encoding_dim : int
    Dimensionality of the latent (encoded) representation.
lr : float
    Learning rate for the optimizer.
batch_size : int
    Number of samples per training batch.
patience : int
    Patience parameter for early stopping.
epochs : int
    Number of training iterations (epochs).
device : torch.device
    Device on which the model will be trained (e.g., CPU or CUDA).

Returns
-------
model : torch.nn.Module
    The trained autoencoder model.
history : dict
    Training history containing loss curves and metrics.
mean_error: float
    Mean of reconstruction error
"""

    
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # DataLoaders
    # -----------------------------
    train_loader = DataLoader(ActivityDataset(X_train), batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(ActivityDataset(X_val),   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(ActivityDataset(X_test),  batch_size=batch_size, shuffle=False, drop_last=False)

    model = model_class(input_dim=input_dim, encoding_dim=encoding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs): 
        
        model.train()
        train_loss = 0.0
        
        for batch in train_loader: 
            batch = batch.to(device)

            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        ## -- validation

        model.eval()
        val_loss = 0.0
        recon_errors_val = []

        with torch.no_grad(): 
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed = model(batch)

                errors = torch.mean((batch - reconstructed) ** 2, dim=1)
                recon_errors_val.extend(errors.cpu().numpy())

                val_loss += criterion(reconstructed, batch).item()

        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)
        # -------------------------
        # Learning Rate Scheduling
        # -------------------------
        scheduler.step(val_loss)

        # -------------------------
        # Early Stopping
        # -------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[Early Stopping] Epoch {epoch+1}")
            break

        # Logging
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Recon Err Mean: {np.mean(recon_errors_val):.6f}"
            )

    # Load best
    model.load_state_dict(best_model_state)

    # -------------------------
    # Test Final Error
    # -------------------------
    model.eval()
    test_errors = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            errors = torch.mean((batch - reconstructed) ** 2, dim=1)
            test_errors.extend(errors.cpu().numpy())

    print("Final Test Reconstruction Error Mean:", np.mean(test_errors))

    return model, history, np.mean(test_errors)

# ==========================================================
# Function 2 — Compute Reconstruction Error on Any df
# ==========================================================
def compute_reconstruction_error(model, df, feature_cols, device=None, batch_size=512, result_cols = None, save_path=None):

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    X = df[feature_cols].values.astype(np.float32)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    recon_errors = []

    with torch.no_grad():
        # batch inference
        for i in range(0, len(X), batch_size):
            batch = X_tensor[i:i+batch_size]
            reconstructed = model(batch)
            errors = torch.mean((batch - reconstructed)**2, dim=1)
            recon_errors.extend(errors.cpu().numpy())

    df["reconstruction_error"] = recon_errors
    
    threshold = df["reconstruction_error"].quantile(0.95)

    df["anomaly"] = df["reconstruction_error"] > threshold

    if result_cols != None: 
        df = df[result_cols]
        
    if save_path:
        df.to_parquet(save_path)
        print(f"Saved parquet → {save_path}")

    return df