import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block_1(input_tensor):
        # Primary path
        primary_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        primary_path = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(primary_path)
        primary_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(primary_path)

        # Branch path
        branch_path = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)

        # Concatenate both paths
        output_tensor = Concatenate(axis=-1)([primary_path, branch_path])
        return output_tensor

    # Block 2
    def block_2(input_tensor):
        # Get the shape of the features from Block 1
        shape = tf.shape(input_tensor)
        height, width = shape[1], shape[2]

        # Reshape into four groups
        reshaped = Reshape(target_shape=(height, width, 2, 32))(input_tensor)  # 32 filters from concatenation
        # Swap the third and fourth dimensions
        permuted = Permute((1, 2, 4, 3))(reshaped)  # (height, width, channels_per_group, groups)
        # Reshape back to original shape
        shuffled = Reshape(target_shape=(height, width, 64))(permuted)  # 64 = 4*16 (4 groups of 16 channels each)

        return shuffled

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model