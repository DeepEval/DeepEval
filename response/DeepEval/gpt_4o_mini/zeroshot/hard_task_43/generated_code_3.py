import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Concatenate, Reshape, Conv2D
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with a single channel (grayscale)
    inputs = Input(shape=input_shape)

    # Block 1: Three parallel average pooling paths
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)

    # Flatten the outputs of the three paths
    path1_flat = Flatten()(path1)
    path2_flat = Flatten()(path2)
    path3_flat = Flatten()(path3)

    # Concatenate the flattened outputs
    concatenated = Concatenate()([path1_flat, path2_flat, path3_flat])

    # Fully connected layer
    dense1 = Dense(128, activation='relu')(concatenated)

    # Reshape the output to 4D for Block 2
    reshaped = Reshape((1, 1, 128))(dense1)

    # Block 2: Three branches for feature extraction
    branch1 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    branch1 = Conv2D(64, (3, 3), activation='relu')(branch1)

    branch2 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    branch2 = Conv2D(32, (1, 7), activation='relu')(branch2)
    branch2 = Conv2D(32, (7, 1), activation='relu')(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)

    branch3 = AveragePooling2D(pool_size=(2, 2))(reshaped)

    # Concatenate the outputs of the branches
    concatenated_branches = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated outputs
    flattened_branches = Flatten()(concatenated_branches)

    # Fully connected layers for classification
    dense2 = Dense(128, activation='relu')(flattened_branches)
    outputs = Dense(10, activation='softmax')(dense2)  # 10 classes for MNIST

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
model = dl_model()
model.summary()