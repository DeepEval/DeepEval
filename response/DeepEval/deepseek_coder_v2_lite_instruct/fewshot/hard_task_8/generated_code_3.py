import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Permute, Reshape, Flatten, Dense

def dl_model():
    # Block 1
    def block_1(input_tensor):
        # Primary path
        conv1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        dw_conv3x3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(conv1x1_1)
        conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(dw_conv3x3)

        # Branch path
        dw_conv3x3_branch = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv1x1_branch = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(dw_conv3x3_branch)

        # Concatenate along the channel dimension
        concat = Concatenate(axis=-1)([conv1x1_2, conv1x1_branch])
        return concat

    # Block 2
    def block_2(input_tensor):
        # Get the shape of the features from Block 1
        input_shape = keras.backend.int_shape(input_tensor)
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]

        # Reshape into four groups
        reshaped = Reshape(target_shape=(height, width, 4, int(channels / 4)))(input_tensor)

        # Swap the third and fourth dimensions
        permuted = Permute((1, 2, 4, 3))(reshaped)

        # Reshape back to the original shape
        final_shape = (height * width * 4, int(channels / 4))
        reshaped_back = Reshape(target_shape=final_shape)(permuted)
        return reshaped_back

    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Apply Block 1
    block1_output = block_1(input_layer)

    # Apply Block 2
    block2_output = block_2(block1_output)

    # Flatten the output
    flatten = Flatten()(block2_output)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model