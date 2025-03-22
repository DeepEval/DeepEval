import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block_1(input_tensor):
        # Split the input into two groups
        groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(input_tensor)

        # Process the first group
        group1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(groups[0])
        group1 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(group1)
        group1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(group1)

        # The second group remains unchanged
        group2 = groups[1]

        # Concatenate the outputs from both groups
        output_tensor = Concatenate()([group1, group2])
        return output_tensor

    def block_2(input_tensor):
        # Get the shape of the input
        shape = tf.shape(input_tensor)
        # Reshape into four groups
        reshaped = Reshape(target_shape=(shape[1], shape[2], 4, shape[3] // 4))(input_tensor)

        # Permute dimensions to shuffle the channels
        permuted = tf.transpose(reshaped, perm=[0, 1, 3, 2, 4])  # swap the last two dimensions

        # Reshape back to original shape
        shuffled = Reshape(target_shape=(shape[1], shape[2], shape[3]))(permuted)
        return shuffled

    # Create the blocks
    block1_output = block_1(input_tensor=initial_conv)
    block2_output = block_2(input_tensor=block1_output)

    # Flatten the final output and apply a fully connected layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model