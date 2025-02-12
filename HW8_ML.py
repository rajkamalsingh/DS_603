import numpy as np
import matplotlib.pyplot as plt
##
# Activation function (sigmoid)
def g(z):
    return 1 / (1 + np.exp(-z))

# Derivative of sigmoid
def g_prime(z):
    sigmoid = g(z)
    return sigmoid * (1 - sigmoid)

# Initialize random weights with uniform distribution (0, 1)
np.random.seed(0)
theta_1 = np.random.uniform(0, 1, (2, 2))  # Weights from input to hidden layer
theta_2 = np.random.uniform(0, 1, (1, 2))  # Weights from hidden to output layer

# Input data
x = np.array([[2], [1]])  # x1 = 2, x2 = 1
y = 3  # True output

# Learning rate
gamma = 0.05

# Store the squared error loss for each iteration
losses = []

# Training for 50 iterations
for iteration in range(50):
    # Forward pass
    z1 = np.dot(theta_1, x)               # Linear combination for the hidden layer
    a1 = g(z1)                            # Activation output for the hidden layer
    z2 = np.dot(theta_2, a1)              # Linear combination for the output layer
    y_hat = z2[0]                         # Predicted output

    # Compute squared error loss
    loss = (y_hat - y) ** 2
    losses.append(loss)

    # Backward pass
    d_loss_y_hat = 2 * (y_hat - y)        # Derivative of loss w.r.t. y_hat

    # Gradients for theta_2
    d_z2 = d_loss_y_hat                   # Derivative of z2 (output layer)
    d_theta_2 = d_z2 * a1.T               # Gradient w.r.t. theta_2

    # Gradients for theta_1
    d_a1 = d_z2 * theta_2.T               # Derivative of a1 from output gradient
    d_z1 = d_a1 * g_prime(z1)             # Derivative of z1 (hidden layer)
    d_theta_1 = np.dot(d_z1, x.T)         # Gradient w.r.t. theta_1

    # Update weights
    theta_2 -= gamma * d_theta_2
    theta_1 -= gamma * d_theta_1

# Plot the squared error loss over iterations
plt.plot(range(50), losses, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Squared Error Loss')
plt.title('Training Loss over Iterations')
plt.yticks(np.arange(0, max(losses) + 0.5, 0.5))  # Set y-axis to smaller intervals
plt.show()
