import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Initial 1x1 convolution
    x = layers.Conv2D(32, (1, 1), activation='relu')(inputs)

    # Branch 1: Local Feature Extraction
    branch1 = layers.Conv2D(64, (3, 3), activation='relu')(x)
    branch1 = layers.Conv2D(64, (3, 3), activation='relu')(branch1)

    # Branch 2: Downsampling and Upsampling
    branch2 = layers.AveragePooling2D((2, 2))(x)
    branch2 = layers.Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = layers.Conv2DTranspose((32, 32), (2, 2), strides=(2, 2), activation='relu')(branch2)

    # Branch 3: Downsampling and Upsampling
    branch3 = layers.AveragePooling2D((2, 2))(x)
    branch3 = layers.Conv2D(64, (3, 3), activation='relu')(branch3)
    branch3 = layers.Conv2DTranspose((32, 32), (2, 2), strides=(2, 2), activation='relu')(branch3)

    # Concatenate branches
    x = layers.Concatenate()([branch1, branch2, branch3])

    # Final 1x1 convolution
    x = layers.Conv2D(10, (1, 1), activation='relu')(x)

    # Flatten and output layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model