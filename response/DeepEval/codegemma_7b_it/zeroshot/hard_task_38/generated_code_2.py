import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

def residual_block(x, filters):
    # Path 1:
    path1 = layers.BatchNormalization()(x)
    path1 = layers.Activation("relu")(path1)
    path1 = layers.Conv2D(filters, (3, 3), padding="same")(path1)

    # Path 2:
    path2 = layers.Conv2D(filters, (3, 3), padding="same")(x)
    path2 = layers.BatchNormalization()(path2)
    path2 = layers.Activation("relu")(path2)
    path2 = layers.Conv2D(filters, (3, 3), padding="same")(path2)

    # Concatenate paths
    output = layers.Concatenate()([path1, path2])
    output = layers.BatchNormalization()(output)
    output = layers.Activation("relu")(output)

    return output

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # First convolutional layer
    x = layers.Conv2D(16, (3, 3), padding="same")(inputs)

    # Repeated residual blocks
    for _ in range(3):
        x = residual_block(x, 16)

    # Output layers
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Load MNIST dataset
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
# x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# # Train the model
# model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))