# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Create the input layer
    inputs = keras.Input(shape=input_shape)

    # Define the feature extraction branches
    # Branch 1: 1x1 convolution
    branch1 = layers.Conv2D(32, 1, activation='relu')(inputs)
    branch1 = layers.Conv2D(32, 3, activation='relu')(branch1)

    # Branch 2: 1x1 convolution followed by a 3x3 convolution
    branch2 = layers.Conv2D(64, 1, activation='relu')(inputs)
    branch2 = layers.Conv2D(64, 3, activation='relu')(branch2)

    # Branch 3: 1x1 convolution followed by two 3x3 convolutions
    branch3 = layers.Conv2D(128, 1, activation='relu')(inputs)
    branch3 = layers.Conv2D(128, 3, activation='relu')(branch3)
    branch3 = layers.Conv2D(128, 3, activation='relu')(branch3)

    # Concatenate the outputs from the three branches
    concatenated = layers.Concatenate()([branch1, branch2, branch3])

    # Adjust the output dimensions to match the input image's channel size
    adjusted = layers.Conv2D(32, 1, activation='relu')(concatenated)

    # Fuse the branch directly connected to the input with the main path
    fused = layers.Add()([adjusted, inputs])

    # Flatten the output
    flattened = layers.Flatten()(fused)

    # Define the classification layers
    x = layers.Dense(128, activation='relu')(flattened)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model