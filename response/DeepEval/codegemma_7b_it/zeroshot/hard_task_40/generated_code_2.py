import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # First block
    x = inputs
    for size in [(1, 1), (2, 2), (4, 4)]:
        avg_pool = layers.AveragePooling2D(size, strides=size, padding="same")(x)
        avg_pool = layers.Conv2D(8, (1, 1), padding="same")(avg_pool)
        x = layers.add([x, avg_pool])

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Reshape((4, 4, 2))(x)

    # Second block
    path1 = layers.Conv2D(16, (1, 1), padding="same")(x)
    path1 = layers.Activation("relu")(path1)

    path2 = layers.Conv2D(16, (1, 1), padding="same")(x)
    path2 = layers.Activation("relu")(path2)
    path2 = layers.Conv2D(16, (3, 3), padding="same")(path2)
    path2 = layers.Activation("relu")(path2)

    path3 = layers.Conv2D(16, (1, 1), padding="same")(x)
    path3 = layers.Activation("relu")(path3)
    path3 = layers.Conv2D(16, (3, 3), padding="same")(path3)
    path3 = layers.Activation("relu")(path3)

    path4 = layers.AveragePooling2D((2, 2), strides=(2, 2), padding="same")(x)
    path4 = layers.Conv2D(16, (1, 1), padding="same")(path4)
    path4 = layers.Activation("relu")(path4)

    x = layers.concatenate([path1, path2, path3, path4])

    x = layers.Dropout(0.2)(x)

    # Output layer
    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model