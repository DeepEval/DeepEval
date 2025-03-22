import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Input layer for the CIFAR-10 dataset
    inputs = keras.Input(shape=(32, 32, 3))

    # Main path
    x = layers.Conv2D(filters=32, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Branch path
    branch = layers.Conv2D(filters=32, kernel_size=7, strides=2, padding='same')(inputs)
    branch = layers.BatchNormalization()(branch)
    branch = layers.Activation('relu')(branch)

    branch = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(branch)
    branch = layers.BatchNormalization()(branch)
    branch = layers.Activation('relu')(branch)

    # Combine paths
    combined = layers.Add()([x, branch])

    # Classification layers
    x = layers.Flatten()(combined)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model