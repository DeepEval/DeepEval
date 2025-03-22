import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Dense, Flatten, Lambda, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with a single channel

    # Input Layer
    inputs = Input(shape=input_shape)

    # Initial Convolutional Layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)

    # Block 1: Split and process
    def split_and_process(x):
        # Split the input into two groups along the last dimension
        group1, group2 = tf.split(x, num_or_size_splits=2, axis=-1)

        # First group operations
        # 1x1 Convolution
        group1 = Conv2D(16, (1, 1), activation='relu', padding='same')(group1)
        # Depthwise Separable Convolution
        group1 = SeparableConv2D(16, (3, 3), activation='relu', padding='same')(group1)
        # Another 1x1 Convolution
        group1 = Conv2D(16, (1, 1), activation='relu', padding='same')(group1)

        # Concatenate the processed group1 and untouched group2
        return Concatenate(axis=-1)([group1, group2])

    # Block 1
    x = Lambda(split_and_process)(x)

    # Block 2: Channel Shuffle
    def channel_shuffle(x):
        # Get shape of input
        batch_size, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        groups = 4
        channels_per_group = channels // groups

        # Reshape into (height, width, groups, channels_per_group)
        x = tf.reshape(x, (batch_size, height, width, groups, channels_per_group))

        # Transpose to swap the third and fourth dimensions
        x = tf.transpose(x, [0, 1, 2, 4, 3])

        # Reshape back to original shape
        x = tf.reshape(x, (batch_size, height, width, channels))
        return x

    # Block 2
    x = Lambda(channel_shuffle)(x)

    # Flatten and Fully Connected Layer for classification
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)  # MNIST has 10 classes

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model