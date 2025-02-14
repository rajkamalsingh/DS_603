import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from tabulate import tabulate


# Load the dataset
file_path = "HW11-ClusteringData.csv"  # Update if needed
data = pd.read_csv(file_path, header=None)
points = data.iloc[:, :2].values  # First two columns (x, y)
labels = data.iloc[:, 2].values   # Third column (true labels)

# Step 1: Visualization of true labels
unique_labels = sorted(set(labels))
colors = plt.cm.tab10.colors
plt.figure(figsize=(8, 6))
for i, label in enumerate(unique_labels):
    cluster_points = points[labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                color=colors[i % len(colors)], label=f"Cluster {label}", alpha=0.6)
plt.title("True Labels of Data Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(alpha=0.5)
plt.show()

# Step 2: K-means clustering and silhouette scores
k_values = range(2, 8)
silhouette_scores_euclidean = []
silhouette_scores_manhattan = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(points)
    labels_kmeans = kmeans.labels_
    score_euclidean = silhouette_score(points, labels_kmeans, metric='euclidean')
    score_manhattan = silhouette_score(points, labels_kmeans, metric='manhattan')
    silhouette_scores_euclidean.append(score_euclidean)
    silhouette_scores_manhattan.append(score_manhattan)

# Plot silhouette scores
plt.figure(figsize=(12, 6))
plt.plot(k_values, silhouette_scores_euclidean, label='Euclidean Distance', marker='o')
plt.plot(k_values, silhouette_scores_manhattan, label='Manhattan Distance', marker='o')
plt.title("Average Silhouette Scores vs. Number of Clusters (k)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Average Silhouette Score")
plt.legend()
plt.grid(alpha=0.5)
plt.show()

# Step 3: Expectation-Maximization (EM) Algorithm
gmm = GaussianMixture(n_components=4, random_state=42).fit(points)
estimated_means = gmm.means_
estimated_weights = gmm.weights_
true_means = np.array([[4, 6], [-3, 3], [2, -2], [-1, -7]])
true_weights = np.array([0.1875, 0.25, 0.3438, 0.2188])

# Comparison Table
comparison_table = pd.DataFrame({
    'True Means': [list(map(float, mean)) for mean in true_means],
    'Estimated Means': [list(map(float, mean)) for mean in estimated_means],
    'True Weights': true_weights.astype(float),
    'Estimated Weights': estimated_weights.astype(float),
})

# Display the table
table_headers = ["True Means", "Estimated Means", "True Weights", "Estimated Weights"]
print(tabulate(comparison_table, headers=table_headers, tablefmt="grid"))


# Step 4: Visualization of GMM Components
def plot_gmm(points, means, covariances, ax):
    colors = plt.cm.tab10.colors
    ax.scatter(points[:, 0], points[:, 1], s=10, alpha=0.5, label="Data Points")
    for i, (mean, covar) in enumerate(zip(means, covariances)):
        eigvals, eigvecs = np.linalg.eigh(covar)
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigvals)  # 2 standard deviations
        ell = Ellipse(mean, width, height, angle=angle, color=colors[i % len(colors)], alpha=0.3)
        ax.add_patch(ell)
        ax.scatter(*mean, color=colors[i % len(colors)], edgecolor='k', s=100, marker='x', label=f"Mean {i+1}")

fig, ax = plt.subplots(figsize=(10, 8))
plot_gmm(points, estimated_means, gmm.covariances_, ax)
ax.set_title("Gaussian Mixture Model: Estimated Components")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
plt.grid(alpha=0.5)
plt.show()
