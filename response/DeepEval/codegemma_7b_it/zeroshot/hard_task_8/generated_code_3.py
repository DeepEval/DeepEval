import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1
    x = layers.Conv2D(filters=32, kernel_size=1, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    residual = layers.Conv2D(filters=32, kernel_size=3, padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=32, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    x = layers.ReLU()(x)

    # Block 2
    residual = layers.Conv2D(filters=64, kernel_size=1, use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=64, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    x = layers.ReLU()(x)

    x = layers.Reshape((-1, 7*7*64))(x)

    # Channel shuffling
    group_num = 4
    group_size = 64 // group_num
    x = layers.Reshape((-1, 7, 7, group_num, group_size))(x)
    x = layers.Permute((0, 1, 3, 2, 4))(x)
    x = layers.Reshape((-1, 7, 7, 64))(x)

    # Output layer
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model