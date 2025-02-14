import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Mean Squared Error Function
def compute_mse(X, X_hat):
    """
    Compute the Mean Squared Error (MSE) as per the given formula.

    Parameters:
    X (numpy.ndarray): Original data matrix.
    X_hat (numpy.ndarray): Reconstructed data matrix.

    Returns:
    float: Mean Squared Error.
    """
    n = X.shape[0]
    mse = (1 / n) * np.linalg.norm(X - X_hat, 'fro') ** 2
    return mse

# (a) Optimal Autoencoder with Linear Encoder and Decoder
class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim, code_dim):
        super(LinearAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, code_dim, bias=False)
        self.decoder = nn.Linear(code_dim, input_dim, bias=False)

    def forward(self, x):
        code = self.encoder(x)
        reconstructed = self.decoder(code)
        return reconstructed

# (b) Autoencoder with ReLU Activation
class ReLUAutoencoder(nn.Module):
    def __init__(self, input_dim, code_dim):
        super(ReLUAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, code_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        code = self.encoder(x)
        reconstructed = self.decoder(code)
        return reconstructed

# Train Autoencoder
def train_autoencoder(model, data_loader, num_epochs=50, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            inputs = batch[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

    return model

# Main script
if __name__ == "__main__":
    # Example data
    np.random.seed(0)
    torch.manual_seed(0)
    X = np.random.rand(100, 10)  # 100 samples, 10 features

    # Convert to PyTorch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    mse_results_linear = []
    mse_results_relu = []

    for code_dim in range(1, 7):  # Vary code dimension from 1 to 6
        # Linear Autoencoder
        linear_model = LinearAutoencoder(input_dim=X.shape[1], code_dim=code_dim)
        linear_model = train_autoencoder(linear_model, data_loader)
        X_hat_linear = linear_model(X_tensor).detach().numpy()
        mse_linear = compute_mse(X, X_hat_linear)
        mse_results_linear.append(mse_linear)

        # ReLU Autoencoder
        relu_model = ReLUAutoencoder(input_dim=X.shape[1], code_dim=code_dim)
        relu_model = train_autoencoder(relu_model, data_loader)
        X_hat_relu = relu_model(X_tensor).detach().numpy()
        mse_relu = compute_mse(X, X_hat_relu)
        mse_results_relu.append(mse_relu)

    print("MSE for Linear Autoencoder:", mse_results_linear)
    print("MSE for ReLU Autoencoder:", mse_results_relu)
