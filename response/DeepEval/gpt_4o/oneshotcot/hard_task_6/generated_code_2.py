import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Dense, Concatenate, Reshape, Permute, SeparableConv2D, Flatten
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        channels = input_tensor.shape[-1]
        split_channels = channels // 3

        # Splitting into 3 groups
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Process each split with 1x1 Convolution
        path1 = Conv2D(filters=split_channels, kernel_size=(1, 1), activation='relu')(splits[0])
        path2 = Conv2D(filters=split_channels, kernel_size=(1, 1), activation='relu')(splits[1])
        path3 = Conv2D(filters=split_channels, kernel_size=(1, 1), activation='relu')(splits[2])

        # Concatenate the outputs
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    # Block 2
    def block2(input_tensor):
        shape = tf.shape(input_tensor)
        height, width, channels = shape[1], shape[2], shape[3]
        groups = 3
        channels_per_group = channels // groups

        # Reshape and Permute
        reshaped = Reshape((height, width, groups, channels_per_group))(input_tensor)
        permuted = Permute((1, 2, 4, 3))(reshaped)
        output_tensor = Reshape((height, width, channels))(permuted)
        return output_tensor

    # Block 3
    def block3(input_tensor):
        # Apply 3x3 depthwise separable convolution
        output_tensor = SeparableConv2D(filters=input_tensor.shape[-1], kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return output_tensor

    # Main Path
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    block3_output = block3(block2_output)
    block1_repeated_output = block1(block3_output)

    # Branch Path
    branch_output = AveragePooling2D(pool_size=(4, 4))(input_layer)

    # Concatenate Main Path and Branch Path
    concatenated_output = Concatenate()([block1_repeated_output, branch_output])

    # Fully connected layer
    flattened_output = Flatten()(concatenated_output)
    output_layer = Dense(units=10, activation='softmax')(flattened_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model