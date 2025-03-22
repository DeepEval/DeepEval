import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Lambda, Add, Flatten, Dense, Concatenate, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Split the input into three groups
        groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        # Apply 1x1 convolutions to each group
        convs = [Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(group) for group in groups]
        # Concatenate the results along the channel dimension
        output_tensor = Concatenate(axis=-1)(convs)
        return output_tensor

    def block_2(input_tensor):
        # Get shape and split into three groups
        shape = tf.shape(input_tensor)
        height, width, channels = shape[1], shape[2], shape[3]
        channels_per_group = channels // 3
        reshaped = Reshape((height, width, 3, channels_per_group))(input_tensor)
        # Permute dimensions to shuffle channels
        permuted = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 2, 4, 3]))(reshaped)
        # Reshape back to original shape
        output_tensor = Reshape((height, width, channels))(permuted)
        return output_tensor

    def block_3(input_tensor):
        # Apply 3x3 depthwise separable convolution
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return depthwise_conv

    # Main path with blocks
    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)
    block3_output = block_3(block2_output)
    final_block1_output = block_1(block3_output)

    # Branch path directly from input
    branch_path = input_layer

    # Combine main path and branch path using addition
    added = Add()([final_block1_output, branch_path])

    # Flatten and apply fully connected layer for classification
    flatten = Flatten()(added)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model