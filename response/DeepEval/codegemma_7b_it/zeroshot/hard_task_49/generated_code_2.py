import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_img = keras.Input(shape=(28, 28, 1))

    # First block
    x = input_img
    x = layers.AveragePooling2D(pool_size=1, strides=1)(x)
    x = layers.AveragePooling2D(pool_size=2, strides=2)(x)
    x = layers.AveragePooling2D(pool_size=4, strides=4)(x)
    x = layers.Flatten()(x)

    # Fully connected layer and reshape
    x = layers.Dense(2048, activation="relu")(x)
    x = layers.Reshape((-1, 1, 2048))(x)

    # Second block
    x = tf.split(x, 4, axis=-1)
    x = [
        layers.DepthwiseConv2D(kernel_size=1, padding="same", use_bias=False)(group)
        for group in x
    ]
    x = [layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=False)(group) for group in x]
    x = [layers.DepthwiseConv2D(kernel_size=5, padding="same", use_bias=False)(group) for group in x]
    x = [layers.DepthwiseConv2D(kernel_size=7, padding="same", use_bias=False)(group) for group in x]

    x = [layers.Conv2D(10, kernel_size=1, padding="same", activation="softmax")(group) for group in x]
    x = layers.Concatenate(axis=-1)(x)

    # Flatten and output layer
    x = layers.Flatten()(x)
    output = layers.Dense(10, activation="softmax")(x)

    # Model creation
    model = keras.Model(input_img, output)

    return model