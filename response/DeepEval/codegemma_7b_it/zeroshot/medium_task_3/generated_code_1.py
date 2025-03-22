import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer for the MNIST dataset
    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1
    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    y = layers.Conv2D(64, (3, 3), activation="relu")(x)
    y = layers.MaxPooling2D((2, 2))(y)

    # Combine outputs from both paths with input
    combined = layers.Add()([inputs, y])
    combined = layers.Activation("relu")(combined)

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation="softmax")(combined)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model