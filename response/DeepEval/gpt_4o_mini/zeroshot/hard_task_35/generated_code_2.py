import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 RGB images
    inputs = layers.Input(shape=input_shape)

    # Define the block
    def block(x):
        # Global Average Pooling
        pooled = layers.GlobalAveragePooling2D()(x)
        # Fully connected layers
        fc1 = layers.Dense(128, activation='relu')(pooled)
        fc2 = layers.Dense(3, activation='sigmoid')(fc1)  # Output dimension = channel dimension
        # Reshape to match input shape
        weights = layers.Reshape((1, 1, 3))(fc2)
        # Element-wise multiplication
        x = layers.multiply([x, weights])
        return x

    # Create two branches
    branch1 = block(inputs)
    branch2 = block(inputs)

    # Concatenate the outputs of both branches
    merged = layers.Concatenate()([branch1, branch2])

    # Flatten and fully connected layer for classification
    flattened = layers.Flatten()(merged)
    outputs = layers.Dense(10, activation='softmax')(flattened)  # CIFAR-10 has 10 classes

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of building the model
model = dl_model()
model.summary()