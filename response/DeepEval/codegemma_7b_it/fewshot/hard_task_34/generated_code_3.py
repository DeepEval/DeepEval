import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(28,28,1))

    def block_main(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_tensor)
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
        concat = Concatenate(axis=-1)([conv, depthwise_conv])
        return concat

    block1_output = block_main(input_tensor=input_layer)
    block2_output = block_main(input_tensor=block1_output)
    block3_output = block_main(input_tensor=block2_output)

    input_branch = Input(shape=block1_output.shape[1:])
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_branch)

    concat_features = Concatenate(axis=-1)([block3_output, conv_branch])

    flatten = Flatten()(concat_features)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=[input_layer, input_branch], outputs=output_layer)

    return model