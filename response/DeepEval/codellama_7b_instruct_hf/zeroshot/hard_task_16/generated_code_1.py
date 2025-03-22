import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    x = layers.Conv2D(64, (1, 1), padding='same')(x[0])
    x = layers.Conv2D(64, (3, 3), padding='same')(x[1])
    x = layers.Conv2D(64, (1, 1), padding='same')(x[2])
    x = layers.Concatenate()(x)

    # Transition Convolution
    x = layers.Conv2D(64, (3, 3), padding='same', strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Block 2
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Branch
    branch = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    branch = layers.Conv2D(64, (1, 1), padding='same')(branch[0])
    branch = layers.Conv2D(64, (3, 3), padding='same')(branch[1])
    branch = layers.Conv2D(64, (1, 1), padding='same')(branch[2])
    branch = layers.Concatenate()(branch)

    # Addition
    x = layers.Add()([x, branch])

    # Output
    output = layers.Dense(10, activation='softmax')(x)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model