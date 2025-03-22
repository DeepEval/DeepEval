import keras
from keras.layers import Input, Conv2D, LayerNormalization, Add, Dense, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def depthwise_separable_conv(input_tensor, kernel_size, strides):
        conv = Conv2D(filters=64, kernel_size=kernel_size, strides=strides, padding='same', activation='relu', use_bias=False)(input_tensor)
        conv = DepthwiseSeparableConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(conv)
        return conv

    def dense_block(input_tensor):
        dense1 = Dense(units=256, activation='relu')(input_tensor)
        dense2 = Dense(units=128, activation='relu')(dense1)
        return dense2

    dwc_layer = depthwise_separable_conv(input_tensor=input_layer, kernel_size=7, strides=1)
    lnorm_layer = LayerNormalization()(dwc_layer)
    dense_block_output = dense_block(input_tensor=lnorm_layer)

    add_layer = Add()([dwc_layer, dense_block_output])
    output_layer = Dense(units=10, activation='softmax')(add_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model