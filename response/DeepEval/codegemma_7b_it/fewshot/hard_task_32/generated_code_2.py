import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense, Concatenate

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def specialized_block(input_tensor):
        conv_dw = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_pw = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_dw)
        dropout1 = Dropout(0.2)(conv_pw)
        conv_pw2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(conv_pw2)
        return dropout2

    branch_1_output = specialized_block(input_tensor)
    branch_2_output = specialized_block(input_tensor)
    branch_3_output = specialized_block(input_tensor)

    concat_branch_outputs = Concatenate()([branch_1_output, branch_2_output, branch_3_output])
    flatten_layer = Flatten()(concat_branch_outputs)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model