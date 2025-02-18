import os
import re
from collections import Counter
from math import log
import numpy as np
from sklearn.svm import SVC, LinearSVC


# Function to load and clean data
def load_and_clean_data(folder_path):
    """
        Load and preprocess the dataset, removing the first 4 lines of each file
        and returning the text with associated labels.
        """
    data = []
    labels = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            label = os.path.basename(root)
            with open(os.path.join(root, file), 'r', encoding='latin1') as f:
                lines = f.readlines()
                content = ' '.join(lines[4:])  # Remove first 4 lines
                data.append(content)
                labels.append(label)
    return data, labels


# Build vocabulary and remove the top 300 most frequent words
def build_vocabulary(data, top_n=300):
    word_freq = Counter()
    for text in data:
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq.update(words)

    # Identify the top `top_n` most frequent words
    most_frequent_words = set(word for word, _ in word_freq.most_common(top_n))

    # Remove the top 300 words from the vocabulary
    vocabulary = [word for word in word_freq if word not in most_frequent_words]
    return vocabulary, most_frequent_words


# Function to calculate term frequency (TF)
def compute_tf(doc, vocabulary):
    words = doc.split()
    word_count = Counter(words)
    total_words = len(words)
    tf = {word: (word_count[word] / total_words) for word in vocabulary}
    return tf


# Function to calculate inverse document frequency (IDF)
def compute_idf(corpus, vocabulary):
    N = len(corpus)
    doc_count = Counter()
    for doc in corpus:
        unique_words = set(re.findall(r'\b\w+\b', doc.lower()))
        for word in vocabulary:
            if word in unique_words:
                doc_count[word] += 1
    idf = {word: log(N / (1 + count)) for word, count in doc_count.items()}  # Add 1 to avoid division by zero
    return idf


# Function to compute TF-IDF for a corpus
def compute_tfidf(corpus, vocabulary):
    idf = compute_idf(corpus, vocabulary)
    tfidf_matrix = []
    for doc in corpus:
        tf = compute_tf(doc, vocabulary)
        tfidf = {word: tf[word] * idf[word] for word in vocabulary}
        tfidf_matrix.append(tfidf)
    return tfidf_matrix


# Convert TF-IDF to a consistent matrix format
def vectorize_tfidf(tfidf_matrix, vocabulary):
    vectors = []
    for tfidf in tfidf_matrix:
        vector = [tfidf.get(word, 0) for word in vocabulary]
        vectors.append(vector)
    return np.array(vectors)


# Manual implementation of 5-fold cross-validation
def manual_k_fold_cross_validation(X, y, k=5):
    n = len(X)
    fold_size = n // k
    indices = np.arange(n)
    np.random.shuffle(indices)
    accuracies = []

    for i in range(k):
        # Split data into train and test
        test_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train SVM
        clf = LinearSVC(C=1)
        clf.fit(X_train, y_train)

        # Test SVM
        accuracy = clf.score(X_test, y_test)
        accuracies.append(accuracy)

    return accuracies


# Main workflow
def main():
    # Set path to dataset
    folder_path = "./20_newsgroups"  # Replace with your dataset folder path

    # Load and clean data
    print("Loading and cleaning data...")
    data, labels = load_and_clean_data(folder_path)

    # Convert string labels to numeric
    label_mapping = {label: idx for idx, label in enumerate(set(labels))}
    labels = [label_mapping[label] for label in labels]

    # Build vocabulary and remove top 300 words
    print("Building vocabulary...")
    vocabulary, stop_words = build_vocabulary(data, top_n=300)
    print(f"Vocabulary size after removing stop words: {len(vocabulary)}")
    print(f"Top 300 stop words removed: {list(stop_words)[:10]}...")

    # Compute TF-IDF
    print("Computing TF-IDF...")
    tfidf_matrix = compute_tfidf(data, vocabulary)
    X = vectorize_tfidf(tfidf_matrix, vocabulary)
    y = np.array(labels)

    # Perform 5-fold cross-validation
    print("Performing 5-fold cross-validation...")
    accuracies = manual_k_fold_cross_validation(X, y, k=5)

    # Print results
    print(f"Accuracies from 5-fold cross-validation: {accuracies}")
    print(f"Mean Accuracy: {np.mean(accuracies):.2f}")


if __name__ == "__main__":
    main()
