import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 channel
    inputs = layers.Input(shape=input_shape)

    # First block of convolution and max pooling
    x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x1 = layers.MaxPooling2D((2, 2))(x1)

    # Second block of convolution and max pooling
    x2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    # Combine the outputs from both blocks with the input using addition
    x = layers.Add()([x1, x2])

    # Flatten the combined output
    x = layers.Flatten()(x)

    # Fully connected layer for classification
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()