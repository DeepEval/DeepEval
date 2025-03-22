import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block_1(input_tensor):
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1_2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
        conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_2)
        output_tensor = Concatenate()([conv1_1, conv1_2, conv1_3])
        return output_tensor

    # Block 2
    def block_2(input_tensor):
        shape = tf.keras.backend.int_shape(input_tensor)
        reshaped_input = Reshape(target_shape=(shape[1], shape[2], shape[3] // 4, 4))(input_tensor)
        swapped_input = Lambda(lambda x: tf.transpose(x, [0, 1, 2, 3, 4]))(reshaped_input)
        reshaped_output = Reshape(target_shape=(shape[1], shape[2], 4, shape[3]))(swapped_input)
        conv2_1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
        conv2_2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
        conv2_3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
        conv2_4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
        output_tensor = Concatenate()([conv2_1, conv2_2, conv2_3, conv2_4])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model