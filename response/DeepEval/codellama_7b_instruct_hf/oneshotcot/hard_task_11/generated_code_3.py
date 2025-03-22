import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense


def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    first_layer = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    second_layer = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(first_layer)
    third_layer = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(second_layer)
    fourth_layer = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(third_layer)
    fifth_layer = Concatenate()([first_layer, second_layer, third_layer, fourth_layer])
    sixth_layer = BatchNormalization()(fifth_layer)
    seventh_layer = Flatten()(sixth_layer)
    eighth_layer = Dense(units=128, activation='relu')(seventh_layer)
    ninth_layer = Dense(units=64, activation='relu')(eighth_layer)
    tenth_layer = Dense(units=10, activation='softmax')(ninth_layer)

    model = keras.Model(inputs=input_layer, outputs=tenth_layer)

    return model