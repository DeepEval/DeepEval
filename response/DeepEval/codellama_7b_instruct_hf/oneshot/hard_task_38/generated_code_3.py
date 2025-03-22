import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    # Define the input layer with a shape of (28, 28, 1)
    input_layer = Input(shape=(28, 28, 1))

    # Define the first processing pathway
    pathway1 = input_layer
    for i in range(3):
        # Apply batch normalization and ReLU activation
        pathway1 = BatchNormalization()(pathway1)
        pathway1 = ReLU()(pathway1)
        # Apply a 3x3 convolutional layer
        pathway1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(pathway1)

    # Define the second processing pathway
    pathway2 = input_layer
    for i in range(3):
        # Apply batch normalization and ReLU activation
        pathway2 = BatchNormalization()(pathway2)
        pathway2 = ReLU()(pathway2)
        # Apply a 3x3 convolutional layer
        pathway2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(pathway2)

    # Concatenate the outputs from both pathways
    merged_pathways = Concatenate()([pathway1, pathway2])

    # Flatten the concatenated output
    flattened_output = Flatten()(merged_pathways)

    # Apply two fully connected layers
    output_layer = Dense(units=10, activation='softmax')(flattened_output)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model