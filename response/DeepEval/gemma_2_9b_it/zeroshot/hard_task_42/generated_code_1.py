import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1: Parallel Max Pooling Paths
    x1_1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    x1_2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    x1_3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)

    x1_1 = layers.Flatten()(x1_1)
    x1_1 = layers.Dropout(0.2)(x1_1)
    x1_2 = layers.Flatten()(x1_2)
    x1_2 = layers.Dropout(0.2)(x1_2)
    x1_3 = layers.Flatten()(x1_3)
    x1_3 = layers.Dropout(0.2)(x1_3)

    x1_output = layers.Concatenate()([x1_1, x1_2, x1_3])

    # Fully Connected Layer and Reshaping
    x2_input = layers.Dense(128, activation='relu')(x1_output)
    x2_input = layers.Reshape((128, 1))(x2_input)

    # Block 2: Parallel Convolutional Paths
    x2_1 = layers.Conv2D(32, (1, 1))(x2_input)
    x2_2 = layers.Conv2D(32, (1, 1))(x2_input)
    x2_2 = layers.Conv2D(32, (7, 1))(x2_2)
    x2_2 = layers.Conv2D(32, (1, 7))(x2_2)
    x2_3 = layers.Conv2D(32, (1, 1))(x2_input)
    x2_3 = layers.Conv2D(32, (7, 1))(x2_3)
    x2_3 = layers.Conv2D(32, (1, 7))(x2_3)
    x2_3 = layers.Conv2D(32, (7, 1))(x2_3)
    x2_3 = layers.Conv2D(32, (1, 7))(x2_3)
    x2_4 = layers.AveragePooling2D(pool_size=(2, 2))(x2_input)
    x2_4 = layers.Conv2D(32, (1, 1))(x2_4)

    x2_output = layers.Concatenate(axis=1)([x2_1, x2_2, x2_3, x2_4])

    # Classification Layers
    x = layers.Flatten()(x2_output)
    x = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=x)

    return model