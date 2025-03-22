import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    DepthwiseConv2D,
    SeparableConv2D,
    Activation,
    BatchNormalization,
    Concatenate,
    Lambda,
    Reshape,
    Permute,
)
from tensorflow.keras.utils import plot_model

def block1(x):
    # Split the input into two groups
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)

    # Apply operations to the first group
    x1 = Conv2D(32, (1, 1), padding="same", use_bias=False)(x1)
    x1 = DepthwiseConv2D(3, padding="same", use_bias=False)(x1)
    x1 = Conv2D(32, (1, 1), padding="same", use_bias=False)(x1)
    x1 = BatchNormalization(axis=-1)(x1)

    # Concatenate the outputs
    return Concatenate(axis=-1)([x1, x2])

def block2(x):
    # Get the input shape
    input_shape = tf.keras.backend.int_shape(x)

    # Reshape the input into four groups
    x = Reshape((input_shape[1], input_shape[2], 4, input_shape[3] // 4))(x)

    # Swap the third and fourth dimensions
    x = Permute((1, 2, 4, 3))(x)

    # Reshape the input back to the original shape
    x = Reshape((input_shape[1], input_shape[2], input_shape[3]))(x)

    return x

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))

    # Initial convolutional layer
    conv1 = Conv2D(32, (3, 3), strides=(2, 2), padding="same")(inputs)
    conv1 = Activation("relu")(conv1)

    # Block 1
    block1_output = block1(conv1)

    # Block 2
    block2_output = block2(block1_output)

    # Fully connected layer
    flatten = tf.keras.layers.Flatten()(block2_output)
    dense = tf.keras.layers.Dense(units=10, activation="softmax")(flatten)

    # Model definition
    model = Model(inputs=inputs, outputs=dense)

    # Compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model