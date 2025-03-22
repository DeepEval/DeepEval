import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense, Reshape, Permute

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    
    # Initial convolutional layer to adjust dimensions
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block_1(input_tensor):
        # Split input into two groups
        input_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_tensor)

        # First group operations
        group1_conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[0])
        group1_depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1_conv1)
        group1_conv2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1_depthwise)
        
        # Second group remains unchanged
        group2 = input_groups[1]

        # Concatenate the results of both groups
        output_tensor = Concatenate()([group1_conv2, group2])
        return output_tensor
    
    def block_2(input_tensor):
        # Obtain input shape
        input_shape = tf.shape(input_tensor)
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]
        groups = 4  # Define number of groups
        channels_per_group = channels // groups
        
        # Reshape input to (height, width, groups, channels_per_group)
        reshaped = Reshape((height, width, groups, channels_per_group))(input_tensor)
        
        # Permute to swap third and fourth dimensions
        permuted = Permute((1, 2, 4, 3))(reshaped)
        
        # Reshape back to original shape
        output_tensor = Reshape((height, width, channels))(permuted)
        return output_tensor

    # Apply the two blocks
    block1_output = block_1(initial_conv)
    block2_output = block_2(block1_output)

    # Final classification layer
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model