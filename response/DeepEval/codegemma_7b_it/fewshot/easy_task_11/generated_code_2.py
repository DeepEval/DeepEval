import keras
import tensorflow as tf
from keras.layers import Input, AveragePooling2D, Lambda, Flatten, Dense, Dropout

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    maxpool1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool1)

    flatten_layer = Flatten()(conv1)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    drop_out = Dropout(rate=0.5)(dense1)
    dense2 = Dense(units=10, activation='softmax')(drop_out)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model