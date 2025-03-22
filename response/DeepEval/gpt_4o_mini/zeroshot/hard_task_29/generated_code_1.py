import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer for 28x28 grayscale images
    input_shape = (28, 28, 1)
    inputs = layers.Input(shape=input_shape)

    # First block
    # Main path
    x_main = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)  # Increase feature map dimensions
    x_main = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x_main)  # Restore the number of channels

    # Branch path
    x_branch = inputs  # Connects directly to the input

    # Combine paths
    x_combined = layers.Add()([x_main, x_branch])  # Element-wise addition

    # Second block - Max pooling layers
    x_pool1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x_combined)
    x_pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_combined)
    x_pool3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x_combined)

    # Flatten each pooling layer output
    x_flat1 = layers.Flatten()(x_pool1)
    x_flat2 = layers.Flatten()(x_pool2)
    x_flat3 = layers.Flatten()(x_pool3)

    # Concatenate flattened outputs
    x_concat = layers.Concatenate()([x_flat1, x_flat2, x_flat3])

    # Fully connected layers
    x_dense1 = layers.Dense(128, activation='relu')(x_concat)
    x_dense2 = layers.Dense(10, activation='softmax')(x_dense1)  # 10 classes for MNIST

    # Construct the model
    model = models.Model(inputs=inputs, outputs=x_dense2)

    return model

# Example usage:
model = dl_model()
model.summary()  # To show the model architecture