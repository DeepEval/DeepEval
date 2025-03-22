import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch1)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch1)
    branch1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = GlobalAveragePooling2D()(branch1)

    branch2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch2)
    branch2 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = GlobalAveragePooling2D()(branch2)

    concat = keras.layers.Add()([branch1, branch2])
    flatten_layer = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model