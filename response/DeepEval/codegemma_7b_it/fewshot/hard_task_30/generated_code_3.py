import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape
from tensorflow.keras import layers, models

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First Block: Dual-Path Structure
    def main_path(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    def branch_path(input_tensor):
        conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv4

    main_output = main_path(input_tensor)
    branch_output = branch_path(input_tensor)
    concat_output = keras.layers.Add()([main_output, branch_output])

    # Second Block: Grouped Depthwise Separable Convolutions
    def grouped_convolutions(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        concat_output = keras.layers.Concatenate()([conv1, conv2, conv3])
        return concat_output

    group_output = grouped_convolutions(concat_output)

    # Fully Connected Layers for Classification
    flatten = Flatten()(group_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model