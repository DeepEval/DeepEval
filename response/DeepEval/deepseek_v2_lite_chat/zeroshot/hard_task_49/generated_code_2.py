import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, Flatten, concatenate, Reshape
from tensorflow.keras.layers import Layer
import numpy as np


def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape data
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

    # First block input
    input_layer = Input((28, 28, 1))

    # First block
    x = input_layer
    for pool_size in [1, 2, 4]:
        x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=1, padding='same')(x)
    x = Flatten()(x)

    # Second block input
    input_layer_second = Input((14, 14, 4))

    # Splitting input into four groups for second block
    split_1 = Lambda(lambda tensors: tf.split(tensors, 4, axis=-1))(input_layer_second)
    split_2 = Lambda(lambda tensors: tf.concat(tensors, axis=-1))(split_1)

    # Second block
    x = split_2
    for kernel_size in [1, 3, 5, 7]:
        x = tf.keras.layers.Conv2D(64, kernel_size=kernel_size, activation='relu')(x)
    x = Flatten()(x)

    # Output layer
    output_layer = Dense(10, activation='softmax')(x)

    # Model
    model = Model(inputs=[input_layer, input_layer_second], outputs=[output_layer])

    return model