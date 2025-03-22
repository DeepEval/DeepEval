from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential(
        [
            layers.Conv2D(
                filters=28,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="valid",
                activation="relu",
                input_shape=(28, 28, 1),
            ),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
            layers.Conv2D(
                filters=28,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="valid",
                activation="relu",
            ),
            layers.Flatten(),
            layers.Dense(units=128, activation="relu"),
            layers.Dense(units=10, activation="softmax"),
        ]
    )

    return model