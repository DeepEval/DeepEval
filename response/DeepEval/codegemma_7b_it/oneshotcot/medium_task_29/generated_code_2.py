import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)

    path1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(conv)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    path3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(conv)

    output_tensor = Concatenate()([path1, path2, path3])
    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model