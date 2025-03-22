import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Path 1
    path1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    # Path 2
    path2 = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1), padding='same')(input_layer)
    path2 = Conv2D(filters=16, kernel_size=(7, 1), strides=(1, 1), padding='same')(path2)

    # Path 3
    path3 = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1), padding='same')(input_layer)
    path3 = Conv2D(filters=16, kernel_size=(7, 1), strides=(1, 1), padding='same')(path3)
    path3 = Concatenate()([path3, path2])

    # Path 4
    path4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    path4 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(path4)

    # Fusion
    fusion = Concatenate()([path1, path2, path3, path4])

    # Batch normalization
    batch_norm = BatchNormalization()(fusion)

    # Flatten
    flatten = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model