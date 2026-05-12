"""
Barlow Twins for per‑ETF embedding learning.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class BarlowTwins(nn.Module):
    def __init__(self, input_dim, hidden_dims, proj_dim, lambda_param=0.005):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dims, proj_dim)
        self.lambda_param = lambda_param
    
    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        return z1, z2
    
    def loss(self, z1, z2):
        batch_size = z1.size(0)
        feature_dim = z1.size(1)
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        c = (z1.T @ z2) / batch_size
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambda_param * off_diag
        return loss
    
    def _off_diagonal(self, x):
        n, m = x.shape
        return x.flatten()[:-1].view(n-1, m+1)[:, 1:].flatten()
    
    def get_embedding(self, x):
        return self.encoder(x).detach().cpu().numpy()

def train_bt_model(data, input_dim, hidden_dims, proj_dim, epochs, batch_size, lr, lambda_param):
    """
    data: numpy array (n_etfs, feature_dim)
    Returns trained model and scaler.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    dataset = torch.tensor(data_scaled, dtype=torch.float32)
    
    def augment(x):
        noise = torch.randn_like(x) * 0.05
        return x + noise
    
    model = BarlowTwins(input_dim, hidden_dims, proj_dim, lambda_param)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    n_samples = len(dataset)
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i+batch_size]
            batch = dataset[batch_idx]
            view1 = augment(batch)
            view2 = augment(batch)
            z1, z2 = model(view1, view2)
            loss = model.loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss: {epoch_loss/ (n_samples/batch_size + 1e-6):.4f}")
    return model, scaler
