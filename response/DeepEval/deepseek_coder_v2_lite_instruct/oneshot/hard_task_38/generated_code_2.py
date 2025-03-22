import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        return x

    # First pathway
    x1 = block(input_layer)
    x1 = block(x1)
    x1 = block(x1)

    # Second pathway
    x2 = block(input_layer)
    x2 = block(x2)
    x2 = block(x2)

    # Concatenate outputs from both pathways along the channel dimension
    combined = Concatenate(axis=-1)([x1, x2])

    # Flatten the result
    flattened = Flatten()(combined)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model