import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Dense, BatchNormalization

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Path 1
    path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
    path1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(path1)

    # Path 2
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(path2)

    # Concatenation and Feature Extraction
    concat_path = Concatenate()([path1, path2])

    # Batch Normalization and Flatten
    bath_norm = BatchNormalization()(concat_path)
    flatten_layer = Flatten()(bath_norm)

    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model