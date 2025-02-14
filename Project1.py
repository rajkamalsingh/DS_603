import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tabulate import tabulate

# Load training and testing data
train_data = pd.read_csv('TrainingData.csv')
test_data = pd.read_csv('TestingData.csv')

results = []
# Separate features and labels
X_train = train_data.iloc[:, :-1]  # features
y_train = train_data.iloc[:, -1]  # label
X_test = test_data.iloc[:, :-1]  # features
y_test = test_data.iloc[:, -1]  # label


# Helper function to calculate type 1 and type 2 error rates
def calculate_errors(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    type1_error_rate = fp / (fp + tn)
    type2_error_rate = fn / (fn + tp)
    return type1_error_rate, type2_error_rate


# Part 1 - Binary Classifiers Using Original Features
print("Part 1 - Binary Classifiers Using Original Features")

# 1. Linear Discriminant Analysis (LDA)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)
type1_error_lda, type2_error_lda = calculate_errors(y_test, y_pred_lda)
print("\nLDA Type 1 Error Rate:", type1_error_lda)
print("LDA Type 2 Error Rate:", type2_error_lda)
results.append(["LDA", "Original", "-", type1_error_lda, type2_error_lda])

# Plot type 1 and type 2 error rates as threshold varies for LDA
thresholds = np.linspace(min(lda.decision_function(X_test)), max(lda.decision_function(X_test)), 100)
type1_errors = []
type2_errors = []
for threshold in thresholds:
    y_pred_threshold = (lda.decision_function(X_test) > threshold).astype(int)
    type1, type2 = calculate_errors(y_test, y_pred_threshold)
    type1_errors.append(type1)
    type2_errors.append(type2)

plt.plot(thresholds, type1_errors, label='Type 1 Error Rate')
plt.plot(thresholds, type2_errors, label='Type 2 Error Rate')
plt.xlabel('Threshold')
plt.ylabel('Error Rate')
plt.legend()
plt.title('LDA Error Rates by Threshold')
plt.show()

# 2. Decision Tree
tree = DecisionTreeClassifier(criterion='gini')
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
type1_error_tree, type2_error_tree = calculate_errors(y_test, y_pred_tree)
print("\nDecision Tree Type 1 Error Rate:", type1_error_tree)
print("Decision Tree Type 2 Error Rate:", type2_error_tree)
results.append(["Decision Tree", "Original", "-", type1_error_tree, type2_error_tree])

# 3. k-Nearest Neighbors (kNN) for k = 1, 3, 5, 10
k_values = [1, 3, 5, 10]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    type1_error_knn, type2_error_knn = calculate_errors(y_test, y_pred_knn)
    print(f"\nkNN (k={k}) Type 1 Error Rate:", type1_error_knn)
    print(f"kNN (k={k}) Type 2 Error Rate:", type2_error_knn)
    results.append([f"kNN (k={k})", "Original", "-", type1_error_knn, type2_error_knn])

# 4. Support Vector Machine (SVM) with soft margin
svm = LinearSVC(C=1)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
type1_error_svm, type2_error_svm = calculate_errors(y_test, y_pred_svm)
print("\nSVM Type 1 Error Rate:", type1_error_svm)
print("SVM Type 2 Error Rate:", type2_error_svm)
results.append(["SVM", "Original", "-", type1_error_svm, type2_error_svm])


# Part 2 - Binary Classifiers Using PCA-Reduced Features
print("\nPart 2 - Binary Classifiers Using PCA-Reduced Features")


# Function to apply PCA and transform both training and testing data
def apply_pca(n_components, X_train, X_test):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca


# Function to evaluate a classifier and print type 1 and type 2 errors
def evaluate_classifier(clf, X_train_pca, X_test_pca, y_train, y_test, classifier_name="Classifier"):
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    type1_error, type2_error = calculate_errors(y_test, y_pred)
    print(f"{classifier_name} - Type 1 Error Rate: {type1_error:.3f}, Type 2 Error Rate: {type2_error:.3f}")
    return type1_error, type2_error


# PCA components to test
components = [5, 10, 15]

# 1. k-Nearest Neighbors (kNN) with PCA-Reduced Features
for n_components in components:
    print(f"\nEvaluating kNN with {n_components} PCA components")
    X_train_pca, X_test_pca = apply_pca(n_components, X_train, X_test)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        type1_error_knn, type2_error_knn=evaluate_classifier(knn, X_train_pca, X_test_pca, y_train, y_test,
                            f"kNN (k={k}) with {n_components} components")
        results.append([f"kNN (k={k})", "PCA-Reduced", f"{n_components} components", type1_error_knn, type2_error_knn])

# 2. Support Vector Machine (SVM) with PCA-Reduced Features
for n_components in components:
    print(f"\nEvaluating SVM with {n_components} PCA components")
    X_train_pca, X_test_pca = apply_pca(n_components, X_train, X_test)

    svm = LinearSVC(C=2)
    type1_error_svm, type2_error_svm =evaluate_classifier(svm, X_train_pca, X_test_pca, y_train, y_test, f"SVM with {n_components} components")
    results.append(["SVM", "PCA-Reduced", f"{n_components} components", type1_error_svm, type2_error_svm])

# Print summary table
print("\nComparison of Classifier Performance on Original vs. PCA-Reduced Features")
print(tabulate(results, headers=["Classifier", "Feature Type", "PCA Components", "Type 1 Error Rate", "Type 2 Error Rate"], tablefmt="grid"))
