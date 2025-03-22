import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Concatenate, Dropout, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    branch1 = block(input_layer, name='branch1')
    branch2 = block(input_layer, name='branch2')
    branch3 = block(input_layer, name='branch3')
    concat_layer = Concatenate()([branch1, branch2, branch3])

    bath_norm = BatchNormalization()(concat_layer)
    dropout = Dropout(0.2)(bath_norm)
    flatten_layer = Flatten()(dropout)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout2 = Dropout(0.2)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout2)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

def block(input_tensor, name=None):

    dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name=name + '_dw_conv')(input_tensor)
    conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name=name + '_conv')(dw_conv)
    dropout = Dropout(0.2, name=name + '_dropout')(conv)

    return dropout