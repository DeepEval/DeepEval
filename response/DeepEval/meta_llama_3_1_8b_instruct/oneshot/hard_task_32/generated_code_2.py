import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Concatenate, Dropout, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):

        dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv)
        dropout = Dropout(0.2)(conv)
        output_tensor = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout)
        dropout2 = Dropout(0.2)(output_tensor)

        return dropout2

    branch1_output = block(input_layer)
    branch2_output = block(input_layer)
    branch3_output = block(input_layer)

    concat_output = Concatenate()([branch1_output, branch2_output, branch3_output])

    flatten_layer = keras.layers.Flatten()(concat_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model