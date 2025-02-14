import numpy as np
import pandas as pd

# Load training data
training_data = pd.read_csv('TrainingData.csv')
X_train = training_data.iloc[:, :-1]
y_train = training_data.iloc[:, -1]


# Function to calculate entropy
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


# Function to calculate information gain
def information_gain(y, y_left, y_right):
    parent_entropy = entropy(y)
    n = len(y)
    n_left, n_right = len(y_left), len(y_right)
    child_entropy = (n_left / n) * entropy(y_left) + (n_right / n) * entropy(y_right)
    return parent_entropy - child_entropy


# Find the best split based on information gain
def find_best_split(X, y, used_features):
    best_gain = -1
    best_feature, best_threshold = None, None
    n_features = X.shape[1]

    for feature_index in range(n_features):
        if feature_index in used_features:  # Skip if feature already used in this branch
            continue
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_indices = X[:, feature_index] <= threshold
            right_indices = X[:, feature_index] > threshold
            y_left, y_right = y[left_indices], y[right_indices]

            if len(y_left) > 0 and len(y_right) > 0:
                gain = information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
    return best_feature, best_threshold, best_gain


# Recursive function to build the decision tree
def build_tree(X, y, used_features=set(), depth=0, max_depth=3, min_gain=0.01):
    if len(set(y)) == 1 or depth == max_depth:
        return {'label': np.bincount(y).argmax()}

    feature, threshold, gain = find_best_split(X, y, used_features)

    if gain < min_gain or feature is None:  # Stop if gain is too small or no feature found
        return {'label': np.bincount(y).argmax()}

    used_features.add(feature)  # Mark feature as used

    left_indices = X[:, feature] <= threshold
    right_indices = X[:, feature] > threshold

    left_branch = build_tree(X[left_indices], y[left_indices], used_features.copy(), depth + 1, max_depth, min_gain)
    right_branch = build_tree(X[right_indices], y[right_indices], used_features.copy(), depth + 1, max_depth, min_gain)

    return {'feature': feature, 'threshold': threshold, 'gain': gain,
            'left': left_branch, 'right': right_branch}


# Convert X_train to numpy array for indexing
X_train_np = X_train.values
y_train_np = y_train.values

# Build the decision tree
decision_tree = build_tree(X_train_np, y_train_np)


# Function to print the sequence of questions (decision rules)
def print_tree(node, feature_names, spacing=""):
    if 'label' in node:
        print(spacing + "Predict:", "Denied" if node['label'] == 1 else "Approved")
        return

    feature_name = feature_names[node['feature']]
    print(f"{spacing}Is {feature_name} <= {node['threshold']}? (Gain: {node['gain']:.3f})")

    # Print the left branch
    print(spacing + '--> True:')
    print_tree(node['left'], feature_names, spacing + "  ")

    # Print the right branch
    print(spacing + '--> False:')
    print_tree(node['right'], feature_names, spacing + "  ")


# Print the decision tree rules
print("\nDecision Tree Rules (Information Gain):")
print_tree(decision_tree, feature_names=X_train.columns)
