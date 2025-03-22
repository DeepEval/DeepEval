import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Dense, Lambda, Reshape, Permute, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Split into 3 groups and apply 1x1 convolutions
    def block1(input_tensor):
        def split_channels(x):
            return tf.split(x, num_or_size_splits=3, axis=-1)

        splits = Lambda(split_channels)(input_tensor)
        convs = [Conv2D(filters=x.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(x) for x in splits]
        output_tensor = tf.keras.layers.Concatenate(axis=-1)(convs)
        return output_tensor

    # Block 2: Channel Shuffling
    def block2(input_tensor):
        shape = tf.shape(input_tensor)
        height, width, channels = shape[1], shape[2], shape[3]
        groups = 3
        channels_per_group = channels // groups

        reshaped = Reshape((height, width, groups, channels_per_group))(input_tensor)
        permuted = Permute((1, 2, 4, 3))(reshaped)  # Swap the last two dimensions
        shuffled = Reshape((height, width, channels))(permuted)
        return shuffled

    # Block 3: Depthwise Separable Convolution
    def block3(input_tensor):
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return depthwise_conv

    # Constructing the main path
    b1_output = block1(input_layer)
    b2_output = block2(b1_output)
    b3_output = block3(b2_output)

    # Direct branch
    direct_branch = input_layer

    # Combine main path and direct branch
    combined_output = Add()([b3_output, direct_branch])

    # Final classification layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model