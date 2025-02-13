import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#
# Load the dataset
data = pd.read_csv('moonDataset.csv', header=0)
X = data.iloc[:, :3].astype(float).values  # Convert to float explicitly
y = data.iloc[:, 3].astype(int).values     # Convert to integer explicitly


# Split data into training (150 samples) and testing (50 samples)
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Bootstrap: Create 50 bootstrapped datasets
bootstrapped_datasets = []
for _ in range(50):
    indices = np.random.choice(150, size=150, replace=True)
    X_b = X_train[indices].astype(np.float32)  # Convert to float32 explicitly
    y_b = y_train[indices].astype(np.float32)  # Convert to float32 explicitly
    bootstrapped_datasets.append((X_b, y_b))


# Define the Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(3, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


# Train 50 neural networks and calculate error rates
error_rates = []
trained_models = []

for dataset in bootstrapped_datasets:
    X_b, y_b = dataset
    model = NeuralNetwork()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Convert to torch tensors
    #print(X_b.dtype)  # Should be something like float64 or int64

    X_b = torch.tensor(X_b, dtype=torch.float32)
    y_b = torch.tensor(y_b, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Train the model
    for epoch in range(100):  # Adjust epochs as needed
        model.train()  # Set the model to training mode
        optimizer.zero_grad()
        outputs = model(X_b)
        loss = criterion(outputs, y_b)
        loss.backward()
        optimizer.step()

    # Append the trained model to the list
    trained_models.append(model)

    # Evaluate on the test dataset
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predictions = (predictions >= 0.5).float()  # Binarize predictions
        error_rate = (predictions != y_test_tensor).sum().item() / len(y_test)
        error_rates.append(error_rate)
        #print(f"Model {i + 1} Error Rate: {error_rate:.4f}")

# Plot histogram of error rates
plt.hist(error_rates, bins=10, edgecolor='k')
plt.title('Error Rates of 50 Neural Networks')
plt.xlabel('Error Rate')
plt.ylabel('Frequency')
plt.show()

# Bagging: Combine classifiers with ensemble sizes m = {5, 10, 15, 20}
ensemble_sizes = [5, 10, 15, 20]
ensemble_error_rates = []

for m in ensemble_sizes:
    ensemble_predictions = []
    for _ in range(m):
        model_idx = np.random.choice(range(50))
        model = trained_models[model_idx]  # Assuming `trained_models` stores all trained networks
        with torch.no_grad():
            predictions = model(X_test_tensor)
            ensemble_predictions.append((predictions >= 0.5).float().numpy())

    # Majority voting
    ensemble_predictions = np.array(ensemble_predictions).squeeze()
    final_predictions = (ensemble_predictions.mean(axis=0) >= 0.5).astype(float)
    error_rate = (final_predictions != y_test).mean()
    ensemble_error_rates.append(error_rate)

# Plot ensemble size vs error rate
plt.plot(ensemble_sizes, ensemble_error_rates, marker='o')
plt.title('Error Rate vs Ensemble Size')
plt.xlabel('Ensemble Size')
plt.ylabel('Error Rate')
plt.show()
