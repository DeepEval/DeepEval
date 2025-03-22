import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)

    path2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, padding='valid')(path2)

    path3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, padding='valid')(path3)

    path4 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path4)

    output_tensor = Concatenate()([path1, path2, path3, path4])
    output_tensor = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)
    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model