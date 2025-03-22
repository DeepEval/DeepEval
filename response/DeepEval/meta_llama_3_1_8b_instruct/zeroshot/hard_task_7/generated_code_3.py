# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():
    # Define the input shape of the MNIST dataset
    input_shape = (28, 28, 1)

    # Create the initial convolutional layer
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)

    # Block 1: split the input into two groups and process them separately
    x = layers.Lambda(lambda x: tf.split(x, 2, axis=-1))(x)
    group1 = layers.Conv2D(32, (1, 1), activation='relu')(x[0])
    group1 = layers.DepthwiseConv2D((3, 3), activation='relu')(group1)
    group1 = layers.Conv2D(32, (1, 1), activation='relu')(group1)
    group2 = x[1]
    x = layers.Concatenate()([group1, group2])

    # Block 2: channel shuffling
    input_shape = keras.Input.shape(x).as_list()
    height, width, channels = input_shape[1:]
    groups = 4
    channels_per_group = channels // groups
    x = layers.Lambda(lambda x: tf.reshape(x, (-1, height, width, groups, channels_per_group)))(x)
    x = layers.Permute((1, 2, 4, 3, 5))(x)
    x = layers.Reshape((height, width, channels))(x)

    # Flatten the output and pass it through a fully connected layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Usage
model = dl_model()
model.summary()