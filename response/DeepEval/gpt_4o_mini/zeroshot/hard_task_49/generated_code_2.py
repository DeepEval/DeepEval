import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for 28x28 grayscale images
    input_shape = (28, 28, 1)
    inputs = layers.Input(shape=input_shape)

    # Block 1: Average pooling with varying scales
    pool1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    pool2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    pool3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)

    # Flatten the pooling results
    flat1 = layers.Flatten()(pool1)
    flat2 = layers.Flatten()(pool2)
    flat3 = layers.Flatten()(pool3)

    # Concatenate the flattened outputs
    concatenated = layers.Concatenate()([flat1, flat2, flat3])

    # Fully connected layer to reshape
    dense1 = layers.Dense(128, activation='relu')(concatenated)
    reshaped = layers.Reshape((4, 4, 8))(dense1)  # Reshape into a 4D tensor (4, 4, 8)

    # Block 2: Depthwise separable convolution with different kernel sizes
    def depthwise_separable_conv(inputs, kernel_size):
        return layers.SeparableConv2D(filters=8, kernel_size=kernel_size, padding='same', activation='relu')(inputs)

    # Split the tensor into 4 groups
    split_tensors = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshaped)

    # Process each group with different kernel sizes
    conv_outputs = [
        depthwise_separable_conv(split_tensor, kernel_size) for split_tensor, kernel_size in zip(split_tensors, [(1, 1), (3, 3), (5, 5), (7, 7)])
    ]

    # Concatenate the outputs from the depthwise separable convolutions
    concatenated_outputs = layers.Concatenate()(conv_outputs)

    # Flatten the final output
    flattened_output = layers.Flatten()(concatenated_outputs)

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(flattened_output)  # 10 classes for MNIST

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of creating the model
model = dl_model()
model.summary()