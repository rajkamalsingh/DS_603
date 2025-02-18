import os
import re
from collections import Counter
from math import log
import numpy as np
from scipy.sparse import csr_matrix
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
def build_vocabulary(data, top_n=300, max_vocab_size=10000):
    word_freq = Counter()
    for text in data:
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq.update(words)

    # Identify the top `top_n` most frequent words
    most_frequent_words = set(word for word, _ in word_freq.most_common(top_n))
    #print(len(word_freq))
    #I tried different vocab sizes to see if there would be any major effect on accuracy but it remains almost unchanged; you can un-comment the next line to run it on the whole vocab
    #max_vocab_size=len(word_freq)
    # Build the final vocabulary limited to `max_vocab_size`
    vocabulary = [word for word, _ in word_freq.most_common(max_vocab_size) if word not in most_frequent_words]
    return vocabulary, most_frequent_words


# Function to compute TF-IDF using sparse matrices
def compute_tfidf_sparse(corpus, vocabulary):
    vocab_index = {word: idx for idx, word in enumerate(vocabulary)}
    rows, cols, data = [], [], []
    idf_count = Counter()

    for doc_id, doc in enumerate(corpus):
        words = re.findall(r'\b\w+\b', doc.lower())
        word_count = Counter(words)
        total_words = len(words)

        # Update IDF counts
        unique_words = set(words)
        for word in unique_words:
            if word in vocab_index:
                idf_count[word] += 1

        # Compute TF and populate sparse matrix data
        for word, count in word_count.items():
            if word in vocab_index:
                rows.append(doc_id)
                cols.append(vocab_index[word])
                data.append(count / total_words)  # TF

    # Compute IDF
    N = len(corpus)
    idf = {word: log(N / (1 + idf_count[word])) for word in vocabulary}

    # Apply IDF to TF and create sparse matrix
    for i in range(len(data)):
        data[i] *= idf[vocabulary[cols[i]]]

    return csr_matrix((data, (rows, cols)), shape=(len(corpus), len(vocabulary)))


# Manual implementation of 5-fold cross-validation
# Manual implementation of 5-fold cross-validation
def manual_k_fold_cross_validation(X, y, k=5):
    n = X.shape[0]  # Use shape[0] to get the number of rows in the sparse matrix
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
    vocabulary, stop_words = build_vocabulary(data, top_n=300, max_vocab_size=50000)
    print(f"Vocabulary size after removing stop words: {len(vocabulary)}")
    print(f"Top 300 stop words removed: {list(stop_words)[:10]}...")

    # Compute sparse TF-IDF
    print("Computing TF-IDF...")
    X = compute_tfidf_sparse(data, vocabulary)
    y = np.array(labels)

    # Perform 5-fold cross-validation
    print("Performing 5-fold cross-validation...")
    accuracies = manual_k_fold_cross_validation(X, y, k=5)

    # Print results
    print(f"Accuracies from 5-fold cross-validation: {accuracies}")
    print(f"Mean Accuracy: {np.mean(accuracies):.2f}")


if __name__ == "__main__":
    main()
