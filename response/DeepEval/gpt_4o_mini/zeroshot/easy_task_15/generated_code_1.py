import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(28, 28, 1))  # MNIST images are 28x28 pixels with a single channel

    # First specialized block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    x = layers.AveragePooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Second specialized block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = layers.AveragePooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for digits 0-9

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()