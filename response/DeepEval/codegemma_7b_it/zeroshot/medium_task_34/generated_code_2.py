import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Feature Extraction
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    # Feature Enhancement
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)

    # Upsampling and Skip Connections
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Add()([x, layers.Conv2D(128, (3, 3), activation="relu", padding="same")(inputs)])
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Add()([x, layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)])
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Add()([x, layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)])

    # Output Layer
    outputs = layers.Conv2D(10, (1, 1), activation="softmax")(x)

    # Model Definition
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model