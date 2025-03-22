import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Primary path
    primary_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    primary_depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(primary_1x1)
    primary_1x1_out = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(primary_depthwise)

    # Branch path
    branch_depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_1x1_out = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_depthwise)

    # Concatenate
    block1_output = Concatenate(axis=-1)([primary_1x1_out, branch_1x1_out])

    # Block 2 - Channel Shuffling
    # Reshape to (height, width, groups, channels_per_group)
    _, h, w, c = block1_output.shape  # Getting the shape
    groups = 4
    channels_per_group = c // groups
    reshaped = Reshape((h, w, groups, channels_per_group))(block1_output)
    
    # Permute dimensions to swap groups and channels_per_group
    permuted = Permute((1, 2, 4, 3))(reshaped)
    
    # Reshape back to original shape
    block2_output = Reshape((h, w, c))(permuted)

    # Flatten and Fully Connected Layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model