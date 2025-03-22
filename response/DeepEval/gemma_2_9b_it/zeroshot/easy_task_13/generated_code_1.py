import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),  # Input layer for MNIST images (28x28 pixels, 1 channel)

        # First 1x1 Convolutional Layer
        layers.Conv1D(filters=32, kernel_size=1, activation='relu', padding='same'),
        layers.Dropout(0.25),

        # Second 1x1 Convolutional Layer
        layers.Conv1D(filters=64, kernel_size=1, activation='relu', padding='same'),
        layers.Dropout(0.25),

        # 3x1 Convolutional Layer
        layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        layers.Dropout(0.25),

        # 1x3 Convolutional Layer
        layers.Conv1D(filters=64, kernel_size=1, activation='relu', padding='same'),
        layers.Dropout(0.25),

        # Restore Channels
        layers.Conv1D(filters=1, kernel_size=1, activation='relu', padding='same'),

        # Add original input to processed features
        layers.Add() ,

        layers.Flatten(),  # Flatten the output for the dense layer

        layers.Dense(10, activation='softmax')  # Output layer with 10 classes (MNIST digits)
    ])
    return model