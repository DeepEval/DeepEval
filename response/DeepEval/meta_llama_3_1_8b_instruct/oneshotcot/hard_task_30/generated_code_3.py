import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():
    # Define input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # Define the first block
    def block1(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path1 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path2 = input_tensor
        output_tensor = layers.Add()([path1, path2])
        return output_tensor

    # Apply the first block
    block1_output = block1(input_layer)

    # Define the second block
    def block2(input_tensor):
        def depthwise_separable_convolution(kernel_size):
            return layers.SeparableConv2D(filters=64, kernel_size=kernel_size, activation='relu')(input_tensor)
        
        group1 = depthwise_separable_convolution((1, 1))
        group2 = depthwise_separable_convolution((3, 3))
        group3 = depthwise_separable_convolution((5, 5))
        output_tensor = layers.Concatenate()([group1, group2, group3])
        return output_tensor

    # Apply the second block
    block2_output = block2(block1_output)

    # Add batch normalization and flatten the result
    batch_norm = layers.BatchNormalization()(block2_output)
    flatten_layer = layers.Flatten()(batch_norm)

    # Add dense layers to produce the final classification
    dense1 = layers.Dense(units=128, activation='relu')(flatten_layer)
    output_layer = layers.Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

dl_model()