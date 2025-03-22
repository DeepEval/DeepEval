import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define four parallel branches
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
    path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path3)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

    # Concatenate outputs of parallel branches
    output_tensor = Concatenate()([path1, path2, path3, path4])

    # Apply batch normalization and flatten output
    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)

    # Add dense layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model