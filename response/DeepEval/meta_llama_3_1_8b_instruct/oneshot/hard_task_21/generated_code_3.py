import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, Conv2D, Concatenate, Add, BatchNormalization, Flatten, Dense
from tensorflow.keras import regularizers
import tensorflow as tf

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Define a Lambda layer to split the input along the channel dimension
    def split_input(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=3, axis=-1)

    split_input_layer = Lambda(split_input)(input_layer)

    # Define a function to apply depthwise separable convolutional layers with varying kernel sizes
    def depthwise_convolution(input_tensor, kernel_size):
        conv1 = DepthwiseConv2D(kernel_size=kernel_size, padding='same')(input_tensor)
        conv2 = Conv2D(kernel_size=1, padding='same')(conv1)
        return conv2

    # Apply depthwise separable convolutional layers with kernel sizes 1x1, 3x3, and 5x5
    conv1 = depthwise_convolution(split_input_layer[0], kernel_size=(1, 1))
    conv2 = depthwise_convolution(split_input_layer[1], kernel_size=(3, 3))
    conv3 = depthwise_convolution(split_input_layer[2], kernel_size=(5, 5))

    # Concatenate the outputs from the three groups
    main_path_output = Concatenate()([conv1, conv2, conv3])

    # Define the branch path
    branch_output = Conv2D(kernel_size=1, padding='same')(input_layer)

    # Add the outputs from both the main and branch paths
    added_output = Add()([main_path_output, branch_output])

    # Apply batch normalization and flatten the result
    bath_norm = BatchNormalization()(added_output)
    flatten_layer = Flatten()(bath_norm)

    # Apply two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model