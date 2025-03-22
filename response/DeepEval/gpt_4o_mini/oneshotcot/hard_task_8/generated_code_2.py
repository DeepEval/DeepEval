import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    # Primary path
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    primary_path = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(primary_path)
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(primary_path)
    
    # Branch path
    branch_path = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch_path)

    # Concatenating both paths
    block1_output = Concatenate()([primary_path, branch_path])
    
    # Block 2
    # Get shape and reshape into (height, width, groups, channels_per_group)
    block2_shape = block1_output.shape[1:3] + (2, 32)  # 2 groups, 32 channels per group
    reshaped = Reshape(block2_shape)(block1_output)

    # Permute to shuffle channels
    permuted = Permute((0, 1, 3, 2))(reshaped)

    # Reshape back to original shape
    reshaped_back = Reshape((28, 28, 64))(permuted)  # 64 total channels after reshuffle

    # Flatten and output layer
    flatten_layer = Flatten()(reshaped_back)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model