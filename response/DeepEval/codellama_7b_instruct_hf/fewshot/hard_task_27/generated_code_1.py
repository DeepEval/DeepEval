import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, LayerNormalization

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    conv_layer = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    norm_layer = LayerNormalization()(conv_layer)
    flatten_layer = Flatten()(norm_layer)
    dense_layer_1 = Dense(units=64, activation='relu')(flatten_layer)
    dense_layer_2 = Dense(units=64, activation='relu')(dense_layer_1)
    adding_layer = Add()([input_layer, dense_layer_2])
    output_layer = Dense(units=10, activation='softmax')(adding_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model