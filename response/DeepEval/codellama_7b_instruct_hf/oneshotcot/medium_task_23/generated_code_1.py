import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool)
    conv5 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv4)
    conv6 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv5)
    output_tensor = Concatenate()([conv1, conv2, conv3, conv4, conv5, conv6])
    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model