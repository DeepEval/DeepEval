import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, add

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Pathway 1
    pathway1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pathway1 = BatchNormalization()(pathway1)
    for i in range(3):
        pathway1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pathway1)
        pathway1 = BatchNormalization()(pathway1)

    # Pathway 2
    pathway2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pathway2 = BatchNormalization()(pathway2)
    for i in range(3):
        pathway2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pathway2)
        pathway2 = BatchNormalization()(pathway2)

    # Concatenate outputs
    concat = Concatenate()([pathway1, pathway2])

    # Fully connected layers
    flatten_layer = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model