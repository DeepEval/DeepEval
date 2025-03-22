import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first parallel path
    path1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the second parallel path
    path2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)

    # Define the third parallel path
    path3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the fourth parallel path
    path4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    path4 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

    # Concatenate the outputs from the parallel paths
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Apply batch normalization
    batch_norm = BatchNormalization()(concatenated)

    # Flatten the output
    flattened = Flatten()(batch_norm)

    # Add a dense layer with 128 units
    dense = Dense(units=128, activation='relu')(flattened)

    # Add a dense layer with 10 units for the final output
    output_layer = Dense(units=10, activation='softmax')(dense)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model