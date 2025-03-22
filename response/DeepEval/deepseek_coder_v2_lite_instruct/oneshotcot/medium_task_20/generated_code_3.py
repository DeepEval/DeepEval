import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First path: 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second path: 1x1 -> 3x3 -> 3x3 convolution
    path2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2_1)
    path2_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2_2)

    # Third path: 1x1 -> 3x3 convolution
    path3_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3_1)

    # Fourth path: MaxPooling -> 1x1 convolution
    path4_1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(input_layer)
    path4_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4_1)

    # Concatenate the outputs of these paths
    concatenated = Concatenate()([path1, path2_3, path3_2, path4_2])

    # Batch normalization
    batch_norm = BatchNormalization()(concatenated)

    # Flatten the result
    flattened = Flatten()(batch_norm)

    # Dense layer
    dense1 = Dense(units=128, activation='relu')(flattened)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model