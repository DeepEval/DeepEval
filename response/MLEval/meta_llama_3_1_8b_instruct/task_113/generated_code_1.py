# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess data
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the model
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train the model
def method():
    model = create_model()
    history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_val, y_val))
    # Make predictions on the test set
    predictions = model.predict(X_test)
    # Convert predictions to labels
    predicted_labels = np.argmax(predictions, axis=1)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predicted_labels)
    return accuracy

# Call the method for validation
output = method()
print(f'The accuracy of the model is: {output:.2f}')