import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, SeparableConv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    conv_inc = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv_ext = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_inc)
    conv_dec = Conv2D(filters=16, kernel_size=(1, 1), strides=(2, 2), padding='valid', activation='relu')(conv_ext)

    flatten_layer = Flatten()(conv_dec)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model