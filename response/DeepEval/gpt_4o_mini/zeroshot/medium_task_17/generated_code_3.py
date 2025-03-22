import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Input shape for CIFAR-10 images
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Reshape the input tensor to (height, width, groups, channels_per_group)
    groups = 3
    channels = input_shape[2]
    channels_per_group = channels // groups  # In CIFAR-10, this is 1
    reshaped = layers.Reshape((32, 32, groups, channels_per_group))(inputs)

    # Permute the dimensions to swap the last two dimensions (groups and channels_per_group)
    permuted = layers.Permute((1, 2, 4, 3))(reshaped)

    # Reshape back to the original input shape
    reshaped_back = layers.Reshape((32, 32, 3))(permuted)

    # Add a flatten layer to prepare for the fully connected layer
    flattened = layers.Flatten()(reshaped_back)

    # Fully connected layer with softmax activation for classification
    outputs = layers.Dense(10, activation='softmax')(flattened)  # CIFAR-10 has 10 classes

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to create the model
model = dl_model()
model.summary()