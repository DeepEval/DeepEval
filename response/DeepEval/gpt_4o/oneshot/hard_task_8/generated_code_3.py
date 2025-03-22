import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Concatenate, Reshape, Permute, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Primary path
    primary_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    primary_sep_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(primary_conv1)
    primary_conv2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(primary_sep_conv)

    # Branch path
    branch_sep_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(branch_sep_conv)

    # Concatenate features from both paths
    block1_output = Concatenate()([primary_conv2, branch_conv])

    # Block 2
    # Get the shape of the block1_output
    shape = keras.backend.int_shape(block1_output)
    height, width, channels = shape[1], shape[2], shape[3]

    # Assuming channels are divisible by 4 for the groups
    groups = 4
    channels_per_group = channels // groups

    # Reshape for grouping
    reshaped = Reshape((height, width, groups, channels_per_group))(block1_output)

    # Permute to shuffle channels
    permuted = Permute((1, 2, 4, 3))(reshaped)

    # Reshape back to the original shape
    shuffled_output = Reshape((height, width, channels))(permuted)

    # Fully connected layer for classification
    flatten_layer = Flatten()(shuffled_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model