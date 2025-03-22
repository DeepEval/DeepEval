import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        return x

    # First pathway
    path1 = block(input_layer)
    path1 = block(path1)
    path1 = block(path1)

    # Second pathway
    path2 = block(input_layer)
    path2 = block(path2)
    path2 = block(path2)

    # Concatenate the outputs of both pathways
    concatenated = Concatenate(axis=-1)([path1, path2])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model