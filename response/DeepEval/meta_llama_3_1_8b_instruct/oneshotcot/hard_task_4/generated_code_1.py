import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add
from keras import layers

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=6, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    def channel_attention(input_tensor):
        gap = GlobalAveragePooling2D()(input_tensor)
        gap = Dense(128, activation='relu')(gap)
        gap = Dense(6, activation='sigmoid')(gap)
        gap = Reshape((1, 1, 6))(gap)
        output_tensor = Multiply()([gap, input_tensor])
        output_tensor = Conv2D(filters=6, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)
        output_tensor = Add()([output_tensor, conv1])
        return output_tensor

    attention_output = channel_attention(dw_conv)
    flatten_layer = layers.Flatten()(attention_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model