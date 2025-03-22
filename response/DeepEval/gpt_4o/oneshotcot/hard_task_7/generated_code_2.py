import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Lambda, Concatenate, Flatten, Dense, Reshape, Permute
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1
    def block1(input_tensor):
        # Split the input into two groups
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_tensor)
        
        # First group processing
        group1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
        group1 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(group1)
        group1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(group1)
        
        # Second group is passed through without modification
        group2 = split_layer[1]
        
        # Concatenate the outputs from both groups
        output_tensor = Concatenate(axis=-1)([group1, group2])
        
        return output_tensor

    block1_output = block1(initial_conv)

    # Block 2
    def block2(input_tensor):
        # Get shape and calculate necessary parameters for reshaping
        input_shape = tf.shape(input_tensor)
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]
        groups = 4
        channels_per_group = channels // groups

        # Reshape and permute for channel shuffling
        reshaped = Reshape((height, width, groups, channels_per_group))(input_tensor)
        permuted = Permute((1, 2, 4, 3))(reshaped)
        reshuffled = Reshape((height, width, channels))(permuted)
        
        return reshuffled

    block2_output = block2(block1_output)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model