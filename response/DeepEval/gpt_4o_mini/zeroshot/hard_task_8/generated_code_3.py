import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for MNIST images (28x28, 1 channel)
    input_shape = (28, 28, 1)
    inputs = Input(shape=input_shape)

    # Block 1
    # Primary path
    x1 = Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    x1 = DepthwiseConv2D((3, 3), padding='same', activation='relu')(x1)
    x1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x1)

    # Branch path
    x2 = DepthwiseConv2D((3, 3), padding='same', activation='relu')(inputs)
    x2 = Conv2D(32, (1, 1), padding='same', activation='relu')(x2)

    # Concatenate both paths
    x = Concatenate(axis=-1)([x1, x2])  # Concatenating along the channel dimension

    # Block 2: Channel Shuffling
    # Getting the shape of features
    shape = tf.shape(x)
    height, width = shape[1], shape[2]
    channels = shape[3]
    
    # Reshape into groups (assuming we want to create 4 groups)
    groups = 4
    channels_per_group = channels // groups
    
    x = Reshape((height, width, groups, channels_per_group))(x)
    x = Permute((0, 1, 3, 2, 4))(x)  # Swap the last two dimensions
    x = Reshape((height, width, channels))(x)  # Reshape back to original shape

    # Fully connected layer for classification
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    # Create the model
    model = Model(inputs, outputs)

    return model

# Example usage:
model = dl_model()
model.summary()