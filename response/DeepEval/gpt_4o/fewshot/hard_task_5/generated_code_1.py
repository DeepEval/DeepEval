import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense, Lambda, Concatenate, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=input_tensor.shape[-1]//3, kernel_size=(1, 1), activation='relu', padding='same')(split_inputs[0])
        conv2 = Conv2D(filters=input_tensor.shape[-1]//3, kernel_size=(1, 1), activation='relu', padding='same')(split_inputs[1])
        conv3 = Conv2D(filters=input_tensor.shape[-1]//3, kernel_size=(1, 1), activation='relu', padding='same')(split_inputs[2])
        output_tensor = Concatenate(axis=-1)([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        shape = tf.shape(input_tensor)
        height, width, channels = shape[1], shape[2], shape[3]
        groups = 3
        channels_per_group = channels // groups

        reshaped = Reshape(target_shape=(height, width, groups, channels_per_group))(input_tensor)
        permuted = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 2, 4, 3]))(reshaped)  # Swapping the last two dimensions
        shuffled = Reshape(target_shape=(height, width, channels))(permuted)
        return shuffled

    def block_3(input_tensor):
        output_tensor = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return output_tensor

    # Main Path
    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)
    block3_output = block_3(block2_output)
    block1_again_output = block_1(block3_output)

    # Branch Path
    branch_output = input_layer

    # Combine Main Path and Branch
    combined_output = Add()([block1_again_output, branch_output])

    # Classification Layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model