import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Main path
    x_main = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x_main = layers.MaxPooling2D()(x_main)
    x_main = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x_main)
    x_main = layers.MaxPooling2D()(x_main)
    x_main = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x_main)
    x_main = layers.MaxPooling2D()(x_main)

    # Branch path
    x_branch = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x_branch = layers.MaxPooling2D()(x_branch)
    x_branch = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x_branch)
    x_branch = layers.MaxPooling2D()(x_branch)

    # Global average pooling in the main path
    x_main = layers.GlobalAveragePooling2D()(x_main)

    # Fully connected layers for weight generation
    x_main = layers.Dense(128, activation="relu")(x_main)
    x_main = layers.Dense(32 * 32 * 3, activation="relu")(x_main)
    x_main = layers.Reshape((32, 32, 3))(x_main)

    # Element-wise multiplication and addition
    x = layers.Add()([x_main, x_branch])

    # Fully connected layers for classification
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model