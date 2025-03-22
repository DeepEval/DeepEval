import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image dimensions
    inputs = layers.Input(shape=input_shape)

    # Branch 1: 1x1 convolution
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)

    # Branch 3: 1x1 convolution followed by two 3x3 convolutions
    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)

    # Concatenate the outputs of the branches
    concatenated = layers.concatenate([branch1, branch2, branch3])

    # 1x1 convolution to adjust the output dimensions
    output_branch = layers.Conv2D(3, (1, 1), activation='relu')(concatenated)

    # Add the original input to the output branch
    added = layers.add([output_branch, inputs])

    # Flatten and pass through fully connected layers for classification
    x = layers.Flatten()(added)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=x)

    return model

# Example of how to create the model
model = dl_model()
model.summary()