import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    # Primary path
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    primary_path = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(primary_path)
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(primary_path)

    # Branch path
    branch_path = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch_path)

    # Concatenate both paths
    block1_output = Concatenate(axis=-1)([primary_path, branch_path])
    
    # Block 2
    # Reshape for channel shuffling
    height, width, channels = block1_output.shape[1], block1_output.shape[2], block1_output.shape[3]
    groups = 4
    channels_per_group = channels // groups
    
    reshaped = Reshape((height, width, groups, channels_per_group))(block1_output)
    
    # Permute dimensions for channel shuffling
    permuted = Permute((0, 1, 3, 2))(reshaped)
    
    # Reshape back to original shape
    reshuffled_output = Reshape((height, width, channels))(permuted)

    # Flatten and fully connected layer
    flatten_layer = Flatten()(reshuffled_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model