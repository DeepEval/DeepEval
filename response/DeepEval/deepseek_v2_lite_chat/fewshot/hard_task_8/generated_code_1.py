import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Lambda, Flatten, Concatenate, Reshape, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        depthwise_conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        depthwise_conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        concatenated = Concatenate(axis=-1)([conv1, depthwise_conv1, conv2, depthwise_conv2])
        return concatenated

    def block_2(input_tensor):
        shape = Lambda(lambda x: tf.shape(x))(input_tensor)
        height, width = shape[1], shape[2]
        groups = shape[3]
        channels_per_group = shape[1] // groups
        input_tensor = Reshape((groups, channels_per_group, height, width))(input_tensor)
        input_tensor = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 3, 1]))(input_tensor)
        input_tensor = Reshape((height, width, groups, channels_per_group))(input_tensor)
        depthwise_conv3 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        depthwise_conv4 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        concatenated = Concatenate(axis=-1)([depthwise_conv3, depthwise_conv4])
        dense = Dense(units=128, activation='relu')(concatenated)
        output_layer = Dense(units=10, activation='softmax')(dense)
        return output_layer

    block1_output = block_1(input_tensor=input_layer)
    model = block_2(input_tensor=block1_output)

    return model