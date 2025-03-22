import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.models.Sequential([
        layers.Input(shape=(28, 28, 1)),  # Input layer for MNIST images (28x28 grayscale)
        layers.Conv2D(32, kernel_size=(1, 1), activation='relu'), # 1x1 convolution for dimensionality reduction
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), # 3x3 convolution for feature extraction
        layers.Conv2D(32, kernel_size=(1, 1), activation='relu'), # 1x1 convolution to restore dimensionality
        layers.Flatten(), # Flatten the output for the dense layer
        layers.Dense(10, activation='softmax') # Output layer with 10 neurons for MNIST classification
    ])
    return model