import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Step 1: Increase the dimensionality of the input's channels threefold using a 1x1 convolution
    x = layers.Conv2D(9, (1, 1), padding='same', activation='relu')(inputs)  # 3 * 3 = 9 channels

    # Step 2: Extract initial features using a 3x3 depthwise separable convolution
    x = layers.SeparableConv2D(9, (3, 3), padding='same', activation='relu')(x)

    # Step 3: Compute channel attention weights through global average pooling
    gap = layers.GlobalAveragePooling2D()(x)
    
    # Fully connected layers for channel attention
    fc1 = layers.Dense(9, activation='relu')(gap)
    fc2 = layers.Dense(9, activation='sigmoid')(fc1)  # Output layer for attention weights

    # Reshape the weights to match the initial features
    attention_weights = layers.Reshape((1, 1, 9))(fc2)

    # Multiply the initial features with attention weights
    x = layers.multiply([x, attention_weights])

    # Step 4: Reduce dimensionality with a 1x1 convolution
    x = layers.Conv2D(3, (1, 1), padding='same')(x)  # Reducing back to 3 channels

    # Combine output with the initial input
    x = layers.add([x, inputs])

    # Flattening layer
    x = layers.Flatten()(x)

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs, outputs)

    return model

# Example of how to use the function
if __name__ == "__main__":
    model = dl_model()
    model.summary()  # To see the model summary