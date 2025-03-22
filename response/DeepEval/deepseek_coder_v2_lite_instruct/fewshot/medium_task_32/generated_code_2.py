import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    # Split the input into three groups along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))
    groups = split_layer(input_layer)

    # Define the depthwise separable convolutional layers for each group
    def depthwise_conv_block(input_tensor, kernel_size):
        return DepthwiseConv2D(kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(input_tensor)

    # Apply depthwise separable convolutional layers with different kernel sizes
    conv1x1 = depthwise_conv_block(groups[0], (1, 1))
    conv3x3 = depthwise_conv_block(groups[1], (3, 3))
    conv5x5 = depthwise_conv_block(groups[2], (5, 5))

    # Concatenate the outputs of the three groups
    concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])

    # Flatten the concatenated features
    flattened = Flatten()(concatenated)

    # Pass through a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model