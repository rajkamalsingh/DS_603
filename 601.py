import numpy as np
import scipy.linalg as la

# Define the transition matrix Q (example with a 6-state Markov chain)
Q = np.array([[0.5, 0.5, 0, 0, 0, 0],
              [0.2, 0.8, 0, 0, 0, 0],
              [0, 0.5, 0.5, 0, 0, 0],
              [0, 0, 0.5, 0.5, 0, 0],
              [0, 0, 0, 0.5, 0.5, 0],
              [0, 0, 0, 0, 0.5, 0.5]])

# Find the left eigenvector corresponding to eigenvalue 1
eigvals, eigvecs = la.eig(Q.T, left=True, right=False)

# Find the eigenvector corresponding to eigenvalue 1
stationary_vec = eigvecs[:, np.isclose(eigvals, 1)]

# Normalize the stationary vector (1-norm normalization)
stationary_vec = stationary_vec.real[:, 0]  # Take real part
stationary_vec /= np.sum(stationary_vec)  # Normalize to 1

# Print the stationary distribution
print("Stationary Distribution:", stationary_vec)
