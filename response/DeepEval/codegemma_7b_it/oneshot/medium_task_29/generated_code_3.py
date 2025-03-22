import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling_1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(conv_1)

    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling_1)
    max_pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_2)

    conv_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling_2)
    max_pooling_3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(conv_3)

    flatten_layer = Flatten()(max_pooling_3)

    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    dense2 = Dense(units=32, activation='relu')(dense1)

    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model