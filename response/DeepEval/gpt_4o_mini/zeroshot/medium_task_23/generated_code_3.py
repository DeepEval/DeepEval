import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Path 1: Single 1x1 convolution
    path1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)

    # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    path2 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    path2 = layers.Conv2D(32, (1, 7), padding='same', activation='relu')(path2)
    path2 = layers.Conv2D(32, (7, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
    path3 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    path3 = layers.Conv2D(32, (1, 7), padding='same', activation='relu')(path3)
    path3 = layers.Conv2D(32, (7, 1), padding='same', activation='relu')(path3)
    path3 = layers.Conv2D(32, (1, 7), padding='same', activation='relu')(path3)
    path3 = layers.Conv2D(32, (7, 1), padding='same', activation='relu')(path3)

    # Path 4: Average pooling followed by a 1x1 convolution
    path4 = layers.AveragePooling2D(pool_size=(2, 2))(inputs)
    path4 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(path4)

    # Concatenating all paths
    concatenated = layers.concatenate([path1, path2, path3, path4])

    # Flattening the output
    flatten = layers.Flatten()(concatenated)

    # Fully connected layer
    dense = layers.Dense(128, activation='relu')(flatten)
    outputs = layers.Dense(10, activation='softmax')(dense)  # CIFAR-10 has 10 classes

    # Constructing the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to use the model
if __name__ == "__main__":
    model = dl_model()
    model.summary()  # Print the model summary to verify its architecture