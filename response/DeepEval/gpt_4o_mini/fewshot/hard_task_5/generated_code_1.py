import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block_1(input_tensor):
        # Split the input tensor into 3 groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        output_tensor = Concatenate(axis=-1)([conv1, conv2, conv3])
        return output_tensor

    # Block 2
    def block_2(input_tensor):
        shape = tf.shape(input_tensor)
        # Reshape the input tensor into groups
        reshaped = tf.reshape(input_tensor, (shape[0], shape[1], shape[2], 3, -1))  # (batch_size, height, width, groups, channels_per_group)
        permuted = tf.transpose(reshaped, perm=[0, 1, 2, 4, 3])  # Swap the last two dimensions
        shuffled = tf.reshape(permuted, (shape[0], shape[1], shape[2], shape[3]))  # Reshape back to (batch_size, height, width, channels)
        return shuffled

    # Block 3
    def block_3(input_tensor):
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return depthwise_conv

    # Create the main path
    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)
    block3_output = block_3(block2_output)

    # Branch path directly from the input
    branch_output = input_layer

    # Combine the outputs from the main path and the branch
    combined_output = Add()([block3_output, branch_output])

    # Final classification layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model