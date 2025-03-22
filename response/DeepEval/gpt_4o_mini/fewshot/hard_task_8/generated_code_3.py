import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block_1(input_tensor):
        # Primary path
        primary_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        primary_path = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(primary_path)
        primary_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(primary_path)

        # Branch path
        branch_path = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch_path)

        # Concatenate both paths
        output_tensor = Concatenate(axis=-1)([primary_path, branch_path])
        return output_tensor

    block1_output = block_1(input_layer)

    # Block 2
    def block_2(input_tensor):
        # Get shape of the output from Block 1
        shape = tf.shape(input_tensor)
        # Reshape into four groups
        reshaped = Reshape(target_shape=(shape[1], shape[2], 4, -1))(input_tensor)  # Last dimension will be channels_per_group

        # Permute dimensions to shuffle channels
        permuted = Permute((1, 2, 4, 3))(reshaped)  # swap the last two dimensions

        # Reshape back to its original shape
        reshuffled = Reshape(target_shape=(shape[1], shape[2], shape[3]))(permuted)
        return reshuffled

    block2_output = block_2(block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model