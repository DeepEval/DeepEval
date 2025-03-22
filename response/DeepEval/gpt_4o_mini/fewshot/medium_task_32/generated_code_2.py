import keras
import tensorflow as tf
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape 32x32 with 3 color channels

    # Split the input into three groups along the last dimension
    split_inputs = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    def depthwise_separable_conv_group(input_tensor, kernel_size):
        # Apply depthwise separable convolution
        conv = DepthwiseConv2D(kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    # Process each group with different kernel sizes
    group1 = depthwise_separable_conv_group(split_inputs[0], kernel_size=(1, 1))
    group2 = depthwise_separable_conv_group(split_inputs[1], kernel_size=(3, 3))
    group3 = depthwise_separable_conv_group(split_inputs[2], kernel_size=(5, 5))

    # Concatenate the outputs of the three groups
    concatenated = Concatenate()([group1, group2, group3])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)  # CIFAR-10 has 10 classes

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model