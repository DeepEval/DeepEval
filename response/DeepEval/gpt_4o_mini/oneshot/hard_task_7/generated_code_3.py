import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Lambda, Concatenate, Flatten, Dense, Reshape
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1
    def block1(input_tensor):
        # Split the input into two groups along the last dimension
        split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_tensor)
        
        # First group operations
        group1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
        group1 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(group1)
        group1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(group1)
        
        # Second group (passed through without modification)
        group2 = split_tensors[1]

        # Concatenate the outputs of both groups
        output_tensor = Concatenate()([group1, group2])

        return output_tensor

    block1_output = block1(initial_conv)

    # Block 2
    def block2(input_tensor):
        # Get the shape of the input
        shape = tf.shape(input_tensor)
        height, width, channels = shape[1], shape[2], shape[3]

        # Reshape the input into four groups
        groups = 4
        channels_per_group = channels // groups
        reshaped = Reshape((height, width, groups, channels_per_group))(input_tensor)

        # Permute the dimensions to achieve channel shuffling
        permuted = tf.transpose(reshaped, perm=[0, 1, 3, 2, 4])

        # Reshape back to original shape
        shuffled_output = Reshape((height, width, channels))(permuted)

        return shuffled_output

    block2_output = block2(block1_output)

    # Flatten and fully connected layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model