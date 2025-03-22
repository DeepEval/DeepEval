import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Primary path
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    primary_path = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(primary_path)
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(primary_path)
    
    # Branch path
    branch_path = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)

    # Concatenate both paths
    block_output = Concatenate()([primary_path, branch_path])

    # Block 2
    # Reshape to (height, width, groups, channels_per_group)
    height, width, channels = block_output.shape[1], block_output.shape[2], block_output.shape[3]
    groups = 4
    channels_per_group = channels // groups
    reshaped_output = Reshape((height, width, groups, channels_per_group))(block_output)

    # Permute to swap dimensions for channel shuffling
    permuted_output = Permute((1, 2, 4, 3))(reshaped_output)

    # Reshape back to original shape
    channel_shuffled_output = Reshape((height, width, channels))(permuted_output)

    # Flatten and fully connected layer
    flatten_layer = Flatten()(channel_shuffled_output)
    dense_output = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_output)

    return model