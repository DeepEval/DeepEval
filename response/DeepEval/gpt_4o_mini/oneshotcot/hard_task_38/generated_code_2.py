import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def block(input_tensor):
    # Apply batch normalization
    norm = BatchNormalization()(input_tensor)
    # ReLU activation
    activated = ReLU()(norm)
    # Convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(activated)
    return conv

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First pathway
    pathway1 = input_layer
    for _ in range(3):
        pathway1 = block(pathway1)
    pathway1 = Concatenate()([pathway1, input_layer])  # Concatenate original input

    # Second pathway
    pathway2 = input_layer
    for _ in range(3):
        pathway2 = block(pathway2)
    pathway2 = Concatenate()([pathway2, input_layer])  # Concatenate original input

    # Merge the outputs of both pathways
    merged_output = Concatenate()([pathway1, pathway2])
    flattened_output = Flatten()(merged_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model