import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28,28,1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block_1(input_tensor):
        group1 = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(input_tensor)
        conv1_group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1[0])
        depthwise_conv_group1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_group1)
        conv2_group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv_group1)
        group2 = input_tensor
        output_tensor = Concatenate()([conv2_group1, group2])
        return output_tensor

    block1_output = block_1(input_tensor=conv1)

    def block_2(input_tensor):
        input_shape = input_tensor.shape
        height, width, groups, channels_per_group = input_shape[1], input_shape[2], input_shape[3], input_shape[4]//4
        input_reshaped = Reshape((height, width, groups, channels_per_group))(input_tensor)
        permuted = tf.transpose(input_reshaped, [0, 1, 3, 2])
        output_reshaped = Reshape((height, width, groups, channels_per_group))(permuted)
        return output_tensor

    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model