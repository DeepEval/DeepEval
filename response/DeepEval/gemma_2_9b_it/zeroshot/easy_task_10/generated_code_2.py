from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential(
        [
            layers.Conv2D(filters=32, kernel_size=1, strides=2, activation="relu", input_shape=(28, 28, 1)),
            layers.DepthwiseConv2D(kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(filters=16, kernel_size=1, strides=2, activation="relu"),
            layers.Flatten(),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model