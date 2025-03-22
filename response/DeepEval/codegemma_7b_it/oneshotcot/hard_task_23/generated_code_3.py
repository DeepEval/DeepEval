import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, TransposedConv2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv_init = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_init)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_init)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_init)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = TransposedConv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch3)

    concat = Concatenate()([branch1, branch2, branch3])
    conv_refined = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    flatten_layer = Flatten()(conv_refined)
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model