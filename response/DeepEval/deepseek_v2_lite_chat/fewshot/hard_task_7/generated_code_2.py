import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape, MaxPooling2D

def dl_model():

    input_layer = Input(shape=(28,28,1))

    def block_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        depthwise_conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        depthwise_conv2 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        output_tensor = Concatenate()([depthwise_conv1, depthwise_conv2, conv1, conv2])
        return output_tensor

    def block_2(input_tensor):
        input_shape = tf.shape(input_tensor)
        output_shape = (input_shape[1], input_shape[2] // 2, 2, 2)
        input_tensor = Reshape(target_shape=output_shape)(input_tensor)
        input_tensor = tf.transpose(input_tensor, perm=[0, 3, 1, 2])  # Swap third and fourth dimensions
        reshaped_tensor = Reshape(target_shape=input_shape)(input_tensor)
        output_tensor = Dense(units=64, activation='relu')(reshaped_tensor)
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model