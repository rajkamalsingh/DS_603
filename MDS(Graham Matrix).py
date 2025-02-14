import numpy as np

def classical_mds_1d(X):
    """
    Perform Classical Multidimensional Scaling (MDS) to find a 1D embedding of the dataset.

    Parameters:
    X (numpy.ndarray): The distance matrix, a symmetric matrix where element (i, j)
                       represents the distance between points i and j.

    Returns:
    numpy.ndarray: 1D embedding of the dataset.
    """
    # Step 1: Ensure the input matrix is square
    n_samples = X.shape[0]
    #assert X.shape[0] == X.shape[1], "Input distance matrix must be square."

    # Step 2: Double centering of the distance matrix
    H = np.eye(n_samples) - (1 / n_samples) * np.ones((n_samples, n_samples))
    B = -0.5 * H @ (X ** 2) @ H  # Compute the B matrix (inner-product matrix)
    print(B)

    # Step 3: Compute eigenvalues and eigenvectors of B
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Step 4: Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Select the top eigenvector for 1D embedding
    top_eigenvalue = eigenvalues[0]
    top_eigenvector = eigenvectors[:, 0]

    # Step 6: Compute 1D embedding by scaling the top eigenvector
    embedding_1d = np.sqrt(top_eigenvalue) * top_eigenvector

    return embedding_1d

# Example usage
X = np.array([
    [25, 155,0,0,0,0],
    [22, 118,0,0,0,0],
    [28, 160,0,0,0,0],
    [24, 172,0,0,0,0],
    [23, 107,0,0,0,0],
    [18, 112,0,0,0,0]
])

embedding_1d = classical_mds_1d(X)
print("1D Embedding:")
print(embedding_1d)
