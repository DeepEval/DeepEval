import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense, Reshape, DepthwiseConv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolution layer
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block_1(input_tensor):
        # Split input into two groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(input_tensor)
        
        # First group processing
        group_1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        group_1 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(group_1)
        group_1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(group_1)

        # Second group processing (no modification)
        group_2 = inputs_groups[1]

        # Concatenate outputs of both groups
        output_tensor = Concatenate()([group_1, group_2])
        return output_tensor

    def block_2(input_tensor):
        # Get shape of input for reshaping
        input_shape = tf.shape(input_tensor)
        # Reshape to (height, width, groups, channels_per_group)
        reshaped = Reshape(target_shape=(input_shape[1], input_shape[2], 4, -1))(input_tensor)

        # Permute dimensions to shuffle channels
        permuted = tf.transpose(reshaped, perm=[0, 1, 3, 2])
        
        # Reshape back to original shape
        shuffled = Reshape(target_shape=(input_shape[1], input_shape[2], -1))(permuted)
        
        return shuffled

    # Block 1 output
    block1_output = block_1(input_tensor=initial_conv)
    
    # Block 2 output
    block2_output = block_2(input_tensor=block1_output)

    # Flatten the output and pass through a fully connected layer
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model