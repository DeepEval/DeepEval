import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape the images from (28, 28) to (28*28,)
train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# One-hot encode labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, train_images, train_labels):
    model.fit(train_images, train_labels, epochs=5)

def evaluate_model(model, test_images, test_labels):
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=2)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy*100:.2f}%')

# Create and train the model
model = create_model()
train_model(model, train_images, train_labels)

# Evaluate the model
evaluate_model(model, test_images, test_labels)