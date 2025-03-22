import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dense, Flatten, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    path1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(path1)
    path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
    path1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(path1)

    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    concat_layer = Concatenate()([path1, path2])

    flatten_layer = Flatten()(concat_layer)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model