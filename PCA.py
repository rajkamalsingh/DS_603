import numpy as np

def pca(data, num_components):
    """
    Perform Principal Component Analysis (PCA) on the given data.

    Parameters:
    data (numpy.ndarray): The data matrix where rows represent samples and columns represent features.
    num_components (int): The number of principal components to retain.

    Returns:
    numpy.ndarray: Transformed data with reduced dimensions.
    numpy.ndarray: Principal components (eigenvectors).
    numpy.ndarray: Explained variance (eigenvalues).
    """
    # Step 1: Standardize the data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std

    # Step 2: Compute the covariance matrix
    covariance_matrix = np.cov(standardized_data, rowvar=False)

    # Step 3: Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Step 4: Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Select the top `num_components` eigenvectors
    selected_eigenvectors = eigenvectors[:, :num_components]

    # Step 6: Transform the data to the new subspace
    reduced_data = np.dot(standardized_data, selected_eigenvectors)

    return reduced_data, selected_eigenvectors, eigenvalues[:num_components]

# Example usage
data = np.array([
    [2.8, 1.5],
    [0.7, 0.4],
    [-1.4, -0.6],
    [-0.2, -0.1],
    [1.2, 0.5],
    [-3.1, -1.7]
])

num_components = 1
reduced_data, principal_components, explained_variance = pca(data, num_components)

print("Reduced Data:")
print(reduced_data)
print("\nPrincipal Components:")
print(principal_components)
print("\nExplained Variance:")
print(explained_variance)
