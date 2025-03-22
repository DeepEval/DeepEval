import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense, DepthwiseConv2D, Reshape, Permute
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Initial convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Block 1
    def block1(input_tensor):
        # Split the input into two groups along the last dimension
        split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_tensor)
        group1 = split_tensor[0]
        group2 = split_tensor[1]

        # First group operations
        group1_path1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(group1)
        group1_path2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(group1_path1)
        group1_path3 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(group1_path2)

        # Concatenate outputs of group1 and group2
        output_tensor = Concatenate()([group1_path3, group2])

        return output_tensor

    block1_output = block1(initial_conv)

    # Block 2
    def block2(input_tensor):
        # Obtain shape of input
        shape = tf.shape(input_tensor)
        height, width = shape[1], shape[2]
        channels = shape[3]

        # Reshape into four groups
        groups = 4
        channels_per_group = channels // groups
        reshaped_input = Reshape((height, width, groups, channels_per_group))(input_tensor)

        # Permute dimensions to achieve channel shuffling
        permuted_input = Permute((1, 2, 4, 3))(reshaped_input)

        # Reshape back to original shape
        shuffled_output = Reshape((height, width, channels))(permuted_input)

        return shuffled_output

    block2_output = block2(block1_output)

    # Flatten and fully connected layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model