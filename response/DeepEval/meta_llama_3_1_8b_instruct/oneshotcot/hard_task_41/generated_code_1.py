import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path1 = Flatten()(path1)
    path1 = Dense(units=64, activation='relu')(path1)
    path1 = Dropout(0.2)(path1)

    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path2 = Flatten()(path2)
    path2 = Dense(units=64, activation='relu')(path2)
    path2 = Dropout(0.2)(path2)

    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    path3 = Flatten()(path3)
    path3 = Dense(units=64, activation='relu')(path3)
    path3 = Dropout(0.2)(path3)

    block1_output = Concatenate()([path1, path2, path3])

    # Transformation layer for Block 2
    flat_layer = Flatten()(block1_output)
    dense_layer = Dense(units=256, activation='relu')(flat_layer)
    reshape_layer = keras.layers.Reshape((4, 64))(dense_layer)

    # Block 2
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path7 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path8 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    path9 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(reshape_layer)
    path9 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path9)

    block2_output = Concatenate()([path4, path5, path6, path7, path8, path9])

    # Flatten and output layer
    flat_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flat_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model