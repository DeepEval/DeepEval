import keras
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(input_layer)
    conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool)
    flatten = Flatten()(conv)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dropout = Dropout(0.2)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model