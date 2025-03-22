import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten, Dense, Dropout, concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', use_bias=False)(input_layer)
    bn = BatchNormalization()(conv)
    act = Activation('relu')(bn)
    model = keras.Model(inputs=input_layer, outputs=act)
    return model