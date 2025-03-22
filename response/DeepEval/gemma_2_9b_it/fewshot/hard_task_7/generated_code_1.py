import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape, Permute

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(input_tensor)
        conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
        conv1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
        output_tensor = Concatenate()([conv1_2, inputs_groups[1]])
        return output_tensor

    block1_output = block1(conv1)

    def block2(input_tensor):
        shape = tf.shape(input_tensor)
        reshaped_input = Reshape(target_shape=(shape[1], shape[2], 4, shape[3] // 4))(input_tensor)
        permuted_input = Permute(dims=(0, 1, 3, 2))(reshaped_input)
        reshaped_output = Reshape(target_shape=(shape[1], shape[2], shape[3]))(permuted_input)
        return reshaped_output

    block2_output = block2(block1_output)

    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model