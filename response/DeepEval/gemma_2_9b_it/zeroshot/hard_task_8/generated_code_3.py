import tensorflow as tf
from tensorflow import keras

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1
    x = keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu')(inputs)
    x = keras.layers.DepthwiseConv1D(kernel_size=3, strides=1, activation='relu')(x)
    x = keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu')(x)

    branch_x = keras.layers.DepthwiseConv1D(kernel_size=3, strides=1, activation='relu')(inputs)
    branch_x = keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu')(branch_x)

    x = keras.layers.concatenate([x, branch_x], axis=-1)

    # Block 2
    shape = keras.layers.Lambda(lambda x: tf.shape(x))(x)
    x = keras.layers.Reshape((shape[1], shape[2], 4, 16))(x)
    x = keras.layers.Permute((1, 2, 4, 3))(x)
    x = keras.layers.Reshape((shape[1], shape[2], 64))(x)

    # Fully connected layer
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model