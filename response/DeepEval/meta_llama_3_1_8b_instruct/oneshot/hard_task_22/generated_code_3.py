import keras
from keras.layers import Input, Lambda, Concatenate, BatchNormalization, Flatten, Dense, SeparableConv2D, Conv2D
from tensorflow.keras import backend as K
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    def split_layer(input_tensor):
        ch1, ch2, ch3 = tf.split(input_tensor, num_or_size_splits=3, axis=-1)
        return ch1, ch2, ch3

    def separable_convolution(channel_input, kernel_size):
        conv = SeparableConv2D(filters=64, kernel_size=kernel_size, padding='same')(channel_input)
        return conv

    def main_path(input_tensor):
        split_output = Lambda(split_layer)(input_tensor)
        conv1x1 = separable_convolution(split_output[0], kernel_size=(1, 1))
        conv3x3 = separable_convolution(split_output[1], kernel_size=(3, 3))
        conv5x5 = separable_convolution(split_output[2], kernel_size=(5, 5))
        output_tensor = Concatenate()([conv1x1, conv3x3, conv5x5])
        return output_tensor

    def branch_path(input_tensor):
        conv1x1 = Conv2D(filters=192, kernel_size=(1, 1), padding='same')(input_tensor)
        return conv1x1

    main_output = main_path(input_tensor=input_layer)
    branch_output = branch_path(input_tensor=input_layer)

    fusion_layer = tf.add(main_output, branch_output)
    batch_norm = BatchNormalization()(fusion_layer)
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model