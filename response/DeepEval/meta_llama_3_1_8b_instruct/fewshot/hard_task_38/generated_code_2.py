import keras
from keras.layers import Input, Conv2D, BatchNormalization, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        norm = BatchNormalization()(input_tensor)
        relu = keras.layers.ReLU()(norm)
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu)
        cat = Concatenate()([input_tensor, conv])
        return cat

    pathway1 = block(input_layer)
    pathway1 = block(pathway1)
    pathway1 = block(pathway1)

    pathway2 = block(input_layer)
    pathway2 = block(pathway2)
    pathway2 = block(pathway2)

    merge = Concatenate()([pathway1, pathway2])
    flatten = Flatten()(merge)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model