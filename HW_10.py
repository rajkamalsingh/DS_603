import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv('Pizza.csv')
X = data.iloc[:, 2:9].values  # Extract columns 3 to 9
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
#a
# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_dim, code_dim, nonlinear=False):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, code_dim)
        self.decoder = nn.Linear(code_dim, input_dim)
        self.nonlinear = nonlinear
        if nonlinear:
            self.activation = nn.ReLU()

    def forward(self, x):
        if self.nonlinear:
            encoded = self.activation(self.encoder(x))
        else:
            encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Training function
def train_autoencoder(model, data, epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
    return loss.item()

# Compute MSE for varying code dimensions
input_dim = X_tensor.shape[1]
mse_results_linear = []
mse_results_relu = []

for h in range(1, 7):
    # Linear autoencoder
    model_linear = Autoencoder(input_dim, h, nonlinear=False)
    train_autoencoder(model_linear, X_tensor)
    X_reconstructed = model_linear(X_tensor).detach().numpy()
    mse_linear = np.mean((X_scaled - X_reconstructed) ** 2)
    mse_results_linear.append(mse_linear)

    # ReLU autoencoder
    model_relu = Autoencoder(input_dim, h, nonlinear=True)
    train_autoencoder(model_relu, X_tensor)
    X_reconstructed_relu = model_relu(X_tensor).detach().numpy()
    mse_relu = np.mean((X_scaled - X_reconstructed_relu) ** 2)
    mse_results_relu.append(mse_relu)

# Plot results
plt.plot(range(1, 7), mse_results_linear, label="Linear Autoencoder")
plt.plot(range(1, 7), mse_results_relu, label="ReLU Autoencoder")
plt.xlabel("Code Dimension (h)")
plt.ylabel("MSE")
plt.title("MSE vs Code Dimension")
plt.legend()
plt.show()
