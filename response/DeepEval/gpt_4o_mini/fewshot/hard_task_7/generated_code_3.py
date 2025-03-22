import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer to adjust dimensions
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block_1(input_tensor):
        # Split the input into two groups along the last dimension
        groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(input_tensor)
        
        # First group: series of convolutions
        group1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(groups[0])
        group1 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(group1)
        group1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(group1)

        # Second group: passed through without modification
        group2 = groups[1]

        # Concatenate the outputs from both groups
        output_tensor = Concatenate()([group1, group2])
        return output_tensor

    def block_2(input_tensor):
        # Obtain the shape of input
        input_shape = tf.shape(input_tensor)
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]

        # Reshape into four groups
        reshaped = Reshape(target_shape=(height, width, 4, channels // 4))(input_tensor)

        # Permute to swap the third and fourth dimensions
        permuted = tf.transpose(reshaped, perm=[0, 1, 3, 2, 4])
        
        # Reshape back to original shape
        output_tensor = Reshape(target_shape=(height, width, channels))(permuted)
        return output_tensor

    # Block 1
    block1_output = block_1(input_tensor=initial_conv)

    # Block 2
    block2_output = block_2(input_tensor=block1_output)

    # Flatten and fully connected layer for classification
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model