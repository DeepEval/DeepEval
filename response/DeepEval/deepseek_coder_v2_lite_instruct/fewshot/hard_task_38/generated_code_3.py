import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # First apply batch normalization and ReLU activation
        x = BatchNormalization()(input_tensor)
        x = ReLU()(x)
        # Then apply a 3x3 convolutional layer
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        return x

    # Pathway 1
    pathway1 = block(input_layer)
    pathway1 = block(pathway1)
    pathway1 = block(pathway1)

    # Pathway 2
    pathway2 = block(input_layer)
    pathway2 = block(pathway2)
    pathway2 = block(pathway2)

    # Concatenate the outputs from both pathways along the channel dimension
    merged = Concatenate(axis=-1)([pathway1, pathway2])

    # Flatten the merged output
    flattened = Flatten()(merged)

    # Pass the flattened output through two fully connected layers
    output_layer = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model