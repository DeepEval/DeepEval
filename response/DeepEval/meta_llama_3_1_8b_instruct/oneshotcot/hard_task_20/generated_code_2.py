import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():
    # Input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # Main path
    def split_input(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=3, axis=-1)

    main_path = layers.Lambda(split_input)(input_layer)
    group1 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(main_path[0])
    group2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(main_path[1])
    group3 = layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(main_path[2])

    # Branch path
    branch_path = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)

    # Concatenate and add
    output = layers.Concatenate()([group1, group2, group3])
    output = layers.Add()([output, branch_path])

    # Batch normalization and activation
    output = layers.BatchNormalization()(output)
    output = layers.ReLU()(output)

    # Flatten and fully connected layers
    output = layers.Flatten()(output)
    output = layers.Dense(128, activation='relu')(output)
    output = layers.Dense(10, activation='softmax')(output)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model