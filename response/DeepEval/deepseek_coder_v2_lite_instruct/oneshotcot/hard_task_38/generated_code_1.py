import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        return batch_norm

    # Pathway 1
    pathway1 = block(input_layer)
    pathway1 = block(pathway1)
    pathway1 = block(pathway1)

    # Pathway 2
    pathway2 = block(input_layer)
    pathway2 = block(pathway2)
    pathway2 = block(pathway2)

    # Concatenate the outputs of both pathways along the channel dimension
    merged_features = Concatenate(axis=-1)([pathway1, pathway2])

    # Flatten the concatenated features
    flattened = Flatten()(merged_features)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model