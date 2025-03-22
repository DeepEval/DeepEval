import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1
    path1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2
    path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    path2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    # Path 3
    path3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=16, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=16, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3)

    # Path 4
    path4 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
    path4 = Conv2D(filters=16, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4)
    path4 = Conv2D(filters=16, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4)

    # Concatenate paths
    output = Concatenate()([path1, path2, path3, path4])

    # Batch normalization
    output = BatchNormalization()(output)

    # Flatten
    output = Flatten()(output)

    # Dense layers
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=64, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model