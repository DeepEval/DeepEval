import tensorflow as tf
from tensorflow import keras

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1
    x = keras.layers.Conv2D(32, (1, 1))(inputs)
    x = keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding='same')(x)
    x = keras.layers.Conv2D(32, (1, 1))(x)

    branch = keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch = keras.layers.Conv2D(32, (1, 1))(branch)

    x = keras.layers.Concatenate(axis=3)([x, branch])

    # Block 2
    shape = keras.layers.Lambda(lambda x: tf.shape(x))(x)
    x = keras.layers.Reshape(target_shape=(shape[1], shape[2], 4, 8))(x)
    x = keras.layers.Permute((2, 3, 1, 4))(x)
    x = keras.layers.Reshape(target_shape=(shape[1], shape[2], 32))(x)

    # Output layer
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model