import numpy as np
import struct
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Helper function to load IDX files
def load_idx(file_path):
    with open(file_path, 'rb') as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        if magic == 2051:  # Images
            rows, cols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.uint8).reshape(num_items, rows, cols)
        elif magic == 2049:  # Labels
            data = np.fromfile(f, dtype=np.uint8)
        else:
            raise ValueError("Invalid IDX file.")
    return data

# File paths (update these to your actual file locations)
train_images_path = "train-images.idx3-ubyte"
train_labels_path = "train-labels.idx1-ubyte"
test_images_path = "t10k-images.idx3-ubyte"
test_labels_path = "t10k-labels.idx1-ubyte"

# Load the dataset
X_train = load_idx(train_images_path)
y_train = load_idx(train_labels_path)
X_test = load_idx(test_images_path)
y_test = load_idx(test_labels_path)

# Preprocess the data
X_train = X_train / 255.0  # Normalize pixel values
X_test = X_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)  # One-hot encode labels
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Reshape the data for CNN
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

# Function to build Feedforward Neural Network (FNN)
def build_fnn():
    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to build Convolutional Neural Network (CNN)
def build_cnn():
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and test multiple models
def train_and_evaluate(build_model_func, X_train, y_train, X_test, y_test, num_models=5, model_type="FNN"):
    accuracies = []
    for i in range(num_models):
        print(f"Training {model_type} Model {i+1}...")
        model = build_model_func()
        model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)
        accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
        accuracies.append(accuracy)
        print(f"{model_type} Model {i+1} Test Accuracy: {accuracy:.4f}\n")
    print(f"Average {model_type} Accuracy: {np.mean(accuracies):.4f}")
    return accuracies

# Train and evaluate 5 FNN models
print("==== Training FNN Models ====")
fnn_accuracies = train_and_evaluate(build_fnn, X_train, y_train, X_test, y_test, model_type="FNN")

# Train and evaluate 5 CNN models
print("==== Training CNN Models ====")
cnn_accuracies = train_and_evaluate(build_cnn, X_train_cnn, y_train, X_test_cnn, y_test, model_type="CNN")
