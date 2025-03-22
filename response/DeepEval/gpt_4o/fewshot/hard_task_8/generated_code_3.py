import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Lambda, Reshape, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Main path
    main_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1)
    main_conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_depthwise_conv)

    # Branch path
    branch_depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_depthwise_conv)

    # Concatenate the outputs of the main and branch paths
    block1_output = Concatenate(axis=-1)([main_conv2, branch_conv])

    # Block 2
    def channel_shuffle(input_tensor, groups):
        height, width, channels = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]
        channels_per_group = channels // groups

        # Reshape to (height, width, groups, channels_per_group)
        reshaped = Reshape((height, width, groups, channels_per_group))(input_tensor)

        # Transpose to (height, width, channels_per_group, groups)
        permuted = Lambda(lambda x: tf.transpose(x, [0, 1, 2, 4, 3]))(reshaped)

        # Reshape back to (height, width, channels)
        output_tensor = Reshape((height, width, channels))(permuted)
        return output_tensor

    # Apply channel shuffle
    block2_output = channel_shuffle(input_tensor=block1_output, groups=4)

    # Final classification layers
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model