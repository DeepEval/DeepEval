import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Primary Path
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    primary_path = DepthwiseConv2D(kernel_size=(3, 3), activation='relu', padding='same')(primary_path)
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(primary_path)

    # Branch Path
    branch_path = DepthwiseConv2D(kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(branch_path)

    # Concatenate both paths
    block1_output = Concatenate()([primary_path, branch_path])

    # Block 2 - Channel Shuffling
    # Get shape
    _, height, width, channels = block1_output.shape

    # Reshape into (height, width, groups, channels_per_group)
    groups = 4
    channels_per_group = channels // groups
    reshaped = Reshape((height, width, groups, channels_per_group))(block1_output)

    # Permute the dimensions to swap the group and channel dimensions
    permuted = Permute((1, 2, 4, 3))(reshaped)

    # Reshape back to the original shape
    block2_output = Reshape((height, width, channels))(permuted)

    # Flatten and Fully Connected Layer for classification
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model