import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Flatten, Dense, Reshape, multiply, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid')(input_layer)
    batch_norm = BatchNormalization()(conv)
    activation = Activation('relu')(batch_norm)

    gap = GlobalAveragePooling2D()(activation)
    dense1 = Dense(units=64)(gap)
    dense2 = Dense(units=64)(dense1)

    reshaped = Reshape((1, 1, 64))(dense2)
    multiply_output = multiply([reshaped, activation])

    concat = Add()([input_layer, multiply_output])
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_1x1)

    flatten = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model