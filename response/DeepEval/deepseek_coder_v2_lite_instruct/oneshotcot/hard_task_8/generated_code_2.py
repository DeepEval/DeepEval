import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Permute

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block1(input_tensor):
        # Primary path
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_mode=True)(path1)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)

        # Branch path
        branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_mode=True)(input_tensor)
        branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch)

        # Concatenate along the channel dimension
        output_tensor = Concatenate(axis=-1)([path3, branch])
        return output_tensor

    block1_output = block1(input_layer)
    batch_norm1 = BatchNormalization()(block1_output)
    flatten_layer = Flatten()(batch_norm1)

    # Block 2
    def block2(input_tensor):
        # Reshape to (height, width, groups, channels_per_group)
        reshaped_tensor = Reshape((7, 7, 4, 32))(flatten_layer)

        # Swap the third and fourth dimensions
        permuted_tensor = Permute((1, 2, 4, 3))(reshaped_tensor)

        # Reshape back to the original shape
        final_tensor = Reshape((28, 28, 128))(permuted_tensor)
        return final_tensor

    block2_output = block2(flatten_layer)
    batch_norm2 = BatchNormalization()(block2_output)

    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(batch_norm2)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model