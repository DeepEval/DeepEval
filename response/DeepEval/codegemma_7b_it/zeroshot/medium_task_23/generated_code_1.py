import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Path 1: Single 1x1 convolution
    path_1 = layers.Conv2D(filters=16, kernel_size=1, activation='relu')(inputs)

    # Path 2: 1x1 -> 1x7 -> 7x1 convolutions
    path_2 = layers.Conv2D(filters=16, kernel_size=1, activation='relu')(inputs)
    path_2 = layers.Conv2D(filters=16, kernel_size=1, activation='relu')(path_2)
    path_2 = layers.Conv2D(filters=16, kernel_size=7, activation='relu')(path_2)
    path_2 = layers.Conv2D(filters=16, kernel_size=7, activation='relu')(path_2)

    # Path 3: 1x1 -> (1x7 -> 7x1) x 2
    path_3 = layers.Conv2D(filters=16, kernel_size=1, activation='relu')(inputs)
    path_3_1 = layers.Conv2D(filters=16, kernel_size=7, activation='relu')(path_3)
    path_3_1 = layers.Conv2D(filters=16, kernel_size=1, activation='relu')(path_3_1)
    path_3_2 = layers.Conv2D(filters=16, kernel_size=7, activation='relu')(path_3)
    path_3_2 = layers.Conv2D(filters=16, kernel_size=1, activation='relu')(path_3_2)
    path_3 = layers.concatenate([path_3_1, path_3_2])

    # Path 4: Average pooling -> 1x1 convolution
    path_4 = layers.AveragePooling2D()(inputs)
    path_4 = layers.Conv2D(filters=16, kernel_size=1, activation='relu')(path_4)

    # Concatenate outputs from all paths
    concat_output = layers.concatenate([path_1, path_2, path_3, path_4])

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(concat_output)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model