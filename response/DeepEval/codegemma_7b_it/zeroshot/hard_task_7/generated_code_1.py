import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # Initial convolutional layer to adjust input dimensions
    x = layers.Conv2D(32, kernel_size=(1, 1), padding='same')(inputs)

    # Block 1
    x_split = layers.Lambda(lambda x: tf.split(x, 2, axis=-1))(x)
    x_1 = layers.Conv2D(64, kernel_size=(1, 1), padding='same')(x_split[0])
    x_1 = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x_1)
    x_1 = layers.Conv2D(64, kernel_size=(1, 1), padding='same')(x_1)
    x_2 = x_split[1]
    concat_1 = layers.Concatenate(axis=-1)([x_1, x_2])

    # Block 2
    shape_x = tf.shape(concat_1)
    reshape_x = layers.Reshape((shape_x[1], shape_x[2], shape_x[3] // 4, 4))(concat_1)
    permute_x = layers.Permute((1, 2, 4, 3))(reshape_x)
    reshape_x_back = layers.Reshape((shape_x[1], shape_x[2], shape_x[3]))(permute_x)

    # Classification layer
    flatten_x = layers.Flatten()(reshape_x_back)
    outputs = layers.Dense(10, activation='softmax')(flatten_x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model