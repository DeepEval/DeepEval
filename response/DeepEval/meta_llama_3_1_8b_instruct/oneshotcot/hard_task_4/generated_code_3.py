import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Multiply, GlobalAveragePooling2D, Dense, Reshape, Concatenate, Flatten

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv_1x1 = Conv2D(filters=3*32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_1x1)

    def channel_attention(input_tensor):
        gap = GlobalAveragePooling2D()(input_tensor)
        gap = Dense(32, activation='relu')(gap)
        gap = Dense(32)(gap)
        gap = Reshape((1, 1, 32))(gap)
        weights = Multiply()([input_tensor, gap])
        return weights
    
    weights = channel_attention(dw_conv)
    weighted_dw_conv = Multiply()([dw_conv, weights])
    conv_1x1_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(weighted_dw_conv)
    merged = Concatenate()([conv_1x1_output, dw_conv])

    flatten_layer = Flatten()(merged)
    dense = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model