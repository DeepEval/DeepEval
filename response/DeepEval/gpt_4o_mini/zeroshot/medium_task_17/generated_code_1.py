import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Define input shape
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)
    inputs = layers.Input(shape=input_shape)

    # Step 1: Reshape the input tensor into groups
    height, width, channels = input_shape
    groups = 3
    channels_per_group = channels // groups  # channels / groups = 3 / 3 = 1

    reshaped = layers.Reshape((height, width, groups, channels_per_group))(inputs)
    
    # Step 2: Permute the dimensions to enable channel shuffling
    permuted = layers.Permute((1, 2, 4, 3))(reshaped)  # (height, width, channels_per_group, groups)

    # Step 3: Reshape back to original input shape
    reshaped_back = layers.Reshape((height, width, channels))(permuted)

    # Step 4: Add a Flatten layer and a Fully Connected layer with Softmax activation
    flattened = layers.Flatten()(reshaped_back)
    outputs = layers.Dense(10, activation='softmax')(flattened)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to use the model
if __name__ == "__main__":
    model = dl_model()
    model.summary()