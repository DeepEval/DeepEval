import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pointwise_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv)
        output_tensor = Add()([conv, pointwise_conv])
        output_tensor = Add()([output_tensor, input_tensor])
        return output_tensor

    branch1_output = block(input_layer)
    branch2_output = block(branch1_output)
    branch3_output = block(branch2_output)

    concat_output = Concatenate()([branch1_output, branch2_output, branch3_output])
    bath_norm = BatchNormalization()(concat_output)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model