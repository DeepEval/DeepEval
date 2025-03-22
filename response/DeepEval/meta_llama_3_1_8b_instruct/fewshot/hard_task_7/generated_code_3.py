import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape, Permute

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1
    def block_1(input_tensor):
        # Split the input into two groups along the last dimension
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(input_tensor)
        
        # First group operations
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
        
        # Second group (no modification)
        second_group = inputs_groups[1]
        
        # Merge the outputs from both groups
        output_tensor = Concatenate()([conv2, second_group])
        return output_tensor

    block1_output = block_1(input_tensor=conv)

    # Block 2
    def block_2(input_tensor):
        # Get the shape of the input
        shape = Lambda(lambda x: tf.shape(x))(input_tensor)
        
        # Reshape the input into four groups
        reshaped = Reshape(target_shape=(28, 28, 2, 32))(input_tensor)
        
        # Permute the dimensions to achieve channel shuffling
        permuted = Permute((2, 3, 1, 4))(reshaped)
        
        # Reshape the input back to its original shape
        output_tensor = Reshape(target_shape=(28, 28, 128))(permuted)
        return output_tensor

    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model