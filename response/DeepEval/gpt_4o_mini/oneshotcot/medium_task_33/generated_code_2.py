import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    # Input layer for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three channels
    channel_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define a function for separable convolution processing
    def separable_conv_block(channel_input):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', use_bias=False)(channel_input)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', use_bias=False)(channel_input)
        conv5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', use_bias=False)(channel_input)
        return conv1, conv3, conv5

    # Process each channel group
    path1 = separable_conv_block(channel_groups[0])
    path2 = separable_conv_block(channel_groups[1])
    path3 = separable_conv_block(channel_groups[2])

    # Concatenate all the outputs
    concatenated = Concatenate()([path1[0], path1[1], path1[2], path2[0], path2[1], path2[2], path3[0], path3[1], path3[2]])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model