import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the four parallel branches
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(input_layer)
    path4 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)

    # Concatenate the outputs of the four branches
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Apply batch normalization and flatten the result
    batch_norm = BatchNormalization()(concatenated)
    flattened = Flatten()(batch_norm)

    # Add a fully connected layer for classification
    output = Dense(units=10, activation='softmax')(flattened)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model