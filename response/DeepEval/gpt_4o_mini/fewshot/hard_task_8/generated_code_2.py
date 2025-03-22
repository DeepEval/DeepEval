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
        output_tensor = Concatenate()([primary_path, branch_path])
        return output_tensor

    # Block 1 output
    block1_output = block_1(input_tensor=input_layer)

    # Block 2
    def block_2(input_tensor):
        # Get the shape of the input tensor for reshaping
        height, width, channels = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]

        # Reshape into four groups
        reshaped = Reshape(target_shape=(height, width, 4, channels // 4))(input_tensor)

        # Permute to shuffle channels
        shuffled = Permute((1, 2, 4, 3))(reshaped)

        # Reshape back to original shape
        reshaped_back = Reshape(target_shape=(height, width, channels))(shuffled)

        return reshaped_back

    # Block 2 output
    block2_output = block_2(input_tensor=block1_output)

    # Flatten and classification layer
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model