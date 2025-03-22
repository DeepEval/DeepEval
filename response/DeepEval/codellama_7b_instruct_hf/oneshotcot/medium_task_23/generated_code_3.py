import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define path 1
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define path 2
    path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    # Define path 3
    path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Concatenate()([path3, path3])

    # Define path 4
    path4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

    # Define concatenated output
    output = Concatenate()([path1, path2, path3, path4])

    # Define batch normalization layer
    output = BatchNormalization()(output)

    # Define flatten layer
    output = Flatten()(output)

    # Define fully connected layers
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=64, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model