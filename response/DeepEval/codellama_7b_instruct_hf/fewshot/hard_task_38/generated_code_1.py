import keras
from keras.layers import Input, BatchNormalization, ReLU, Conv2D, Concatenate, Dense, Flatten

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Define the first processing pathway
    pathway1 = input_layer
    for i in range(3):
        pathway1 = BatchNormalization()(pathway1)
        pathway1 = ReLU()(pathway1)
        pathway1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(pathway1)

    # Define the second processing pathway
    pathway2 = input_layer
    for i in range(3):
        pathway2 = BatchNormalization()(pathway2)
        pathway2 = ReLU()(pathway2)
        pathway2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(pathway2)

    # Merge the outputs from both pathways
    merged_output = Concatenate()([pathway1, pathway2])

    # Apply fully connected layers for classification
    flattened_output = Flatten()(merged_output)
    output_layer = Dense(units=10, activation='softmax')(flattened_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model