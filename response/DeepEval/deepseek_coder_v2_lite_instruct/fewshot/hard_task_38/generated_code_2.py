import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    # Pathway 1
    pathway1 = block(input_layer)
    pathway1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(pathway1)
    pathway1 = BatchNormalization()(pathway1)
    pathway1 = ReLU()(pathway1)

    # Pathway 2
    pathway2 = block(input_layer)
    pathway2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(pathway2)
    pathway2 = BatchNormalization()(pathway2)
    pathway2 = ReLU()(pathway2)

    # Merge pathways
    merged = Concatenate(axis=-1)([pathway1, pathway2])

    # Flatten and classify
    flattened = Flatten()(merged)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model