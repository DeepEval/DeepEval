import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Lambda, Concatenate, Flatten, Dense, Reshape, Permute

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer to adjust the dimensions
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block_1(input_tensor):
        # Split the input into two groups along the last dimension
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_tensor)

        # First group: series of 1x1, Depthwise Separable, and 1x1 convolutions
        group1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_groups[0])
        group1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1)
        group1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)

        # Second group: passed through without modification
        group2 = split_groups[1]

        # Concatenate the outputs from both groups
        output_tensor = Concatenate()([group1, group2])
        return output_tensor

    def block_2(input_tensor):
        # Obtain the shape and perform reshaping for channel shuffling
        shape = tf.shape(input_tensor)
        height, width, channels = shape[1], shape[2], shape[3]
        groups = 4
        channels_per_group = channels // groups

        # Reshape to (height, width, groups, channels_per_group)
        reshaped = Reshape((height, width, groups, channels_per_group))(input_tensor)

        # Swap the third and fourth dimensions
        permuted = Permute((1, 2, 4, 3))(reshaped)

        # Reshape back to original shape
        shuffled_output = Reshape((height, width, channels))(permuted)

        return shuffled_output

    # Sequentially apply Block 1 and Block 2
    block1_output = block_1(initial_conv)
    block2_output = block_2(block1_output)

    # Flatten and pass through a fully connected layer for classification
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model