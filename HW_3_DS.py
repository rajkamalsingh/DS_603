import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist


# Assuming you have a function to load the dataset and preprocess images
from PIL import Image


def load_data(dataset_dir="ATT"):
    # Lists to store the flattened images and corresponding labels
    images = []
    labels = []

    # Loop through each image in the ATT folder
    for image_file in os.listdir(dataset_dir):
        # Construct the full path to the image
        image_path = os.path.join(dataset_dir, image_file)

        if os.path.isfile(image_path) and image_file.endswith('.png'):
            # The image name is in the format x_y.png, where x is the label and y is the image number
            # We split by '_' and ignore the part after '.png'
            label, _ = image_file.split('_')  # x_y.png -> (label, image_number.png)
            label = int(label)  # Convert label (e.g., '1') to an integer

            # Load the image
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            img = img.resize((92, 112))  # Resize to 92x112 if necessary

            # Flatten the image into a 1D array (size: 10304)
            img_array = np.asarray(img).flatten()

            # Append the image array and label
            images.append(img_array)
            labels.append(label)

    # Convert images and labels to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    print("loaded images")
    return images, labels


# 1NN Classifier implementation
def one_nn_classifier(train_data, train_labels, test_data):
    # Calculate Euclidean distance between test_data and all train_data
    distances = cdist(test_data, train_data, 'euclidean')
    # Get the index of the nearest neighbor
    nearest_idx = np.argmin(distances, axis=1)
    # Return the label of the nearest neighbor
    return train_labels[nearest_idx]

def pca_reduce(data, n_components=60):
    """
    Perform PCA using eigen decomposition to reduce data dimensionality.
    :param data: Data matrix of shape (n_samples, n_features)
    :param n_components: Number of principal components to retain
    :return: Reduced data of shape (n_samples, n_components)
    """
    # Step 1: Center the data
    mean_vector = np.mean(data, axis=0)
    centered_data = data - mean_vector

    # Step 2: Compute the covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Step 3: Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 4: Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(-eigenvalues)
    top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]

    # Ensure eigenvectors are real
    top_eigenvectors = np.real(top_eigenvectors)

    # Step 5: Project the data onto the top eigenvectors
    reduced_data = np.dot(centered_data, top_eigenvectors)

    # Return reduced data, mean vector, and eigenvectors
    return reduced_data, mean_vector, top_eigenvectors


# 5-Fold Cross Validation
def cross_validation_with_pca(data, labels, n_splits=5, n_components=60):
    """
    Perform 5-fold cross-validation with PCA dimensionality reduction.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_index, test_index) in enumerate(kf.split(data), 1):
        # Split into training and testing sets
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # Apply PCA on the training data
        train_data_reduced, mean_vector, top_eigenvectors = pca_reduce(train_data, n_components)

        # Project test data onto the same PCA subspace
        centered_test_data = test_data - mean_vector
        test_data_reduced = np.dot(centered_test_data, top_eigenvectors)

        # Classify using 1NN
        predictions = one_nn_classifier(train_data_reduced, train_labels, test_data_reduced)

        # Print the true labels and predicted labels for the test set
        print(f"Fold {fold}:")
        print(f"True labels: {test_labels}")
        print(f"Predicted labels: {predictions}")

        # Calculate accuracy for the fold
        accuracy = accuracy_score(test_labels, predictions)
        print(f"Accuracy for Fold {fold}: {accuracy * 100:.2f}%\n")

        # Append accuracy to the list of accuracies
        accuracies.append(accuracy)

    # Return average accuracy
    avg_accuracy = np.mean(accuracies)
    return avg_accuracy


# Main function to load data, apply PCA + 1NN, and report accuracy
def main():
    # Step 1: Load and preprocess the dataset
    images, labels = load_data()  # Replace with your actual data loading function

    # Step 2: Perform 5-fold cross-validation with PCA
    average_accuracy = cross_validation_with_pca(images, labels, n_splits=5, n_components=60)

    # Step 3: Report the average accuracy
    print(f'Average Accuracy from 5-Fold Cross-Validation with PCA: {average_accuracy * 100:.2f}%')


# Execute main function
if __name__ == "__main__":
    main()