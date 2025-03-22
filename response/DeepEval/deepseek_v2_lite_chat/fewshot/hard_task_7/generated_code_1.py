import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(28,28,1))

    def block_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        ds_conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        merge1 = Concatenate()([ds_conv1, conv2])
        return merge1

    def block_2(input_tensor):
        input_shape = tf.shape(input_tensor)
        height, width, channels = input_shape[1], input_shape[2], 1
        groups = 4
        channels_per_group = channels // groups
        input_tensor = Reshape(target_shape=(height, width, groups, channels_per_group))(input_tensor)
        input_tensor = tf.transpose(input_tensor, [0, 2, 3, 1])
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        reshaped = Reshape(target_shape=(height, width, channels)) (conv1)
        return reshaped

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model