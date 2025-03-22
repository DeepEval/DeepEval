import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense, BatchNormalization

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_concat = Concatenate()([conv1, input_layer])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_concat)
    conv2_concat = Concatenate()([conv2, conv1_concat])
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_concat)

    flatten_layer = Flatten()(conv3)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model