import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Concatenate, Dense, Flatten

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the first pathway
    pathway1 = Input(shape=input_shape)
    pathway1 = BatchNormalization()(pathway1)
    pathway1 = ReLU()(pathway1)
    pathway1 = Conv2D(32, (3, 3), padding='same')(pathway1)
    pathway1 = MaxPooling2D((2, 2))(pathway1)
    pathway1 = Concatenate()([pathway1, pathway1])

    # Define the second pathway
    pathway2 = Input(shape=input_shape)
    pathway2 = BatchNormalization()(pathway2)
    pathway2 = ReLU()(pathway2)
    pathway2 = Conv2D(32, (3, 3), padding='same')(pathway2)
    pathway2 = MaxPooling2D((2, 2))(pathway2)
    pathway2 = Concatenate()([pathway2, pathway2])

    # Merge the outputs from both pathways
    merged_output = Concatenate()([pathway1, pathway2])

    # Add two fully connected layers for classification
    merged_output = Flatten()(merged_output)
    merged_output = Dense(64, activation='relu')(merged_output)
    merged_output = Dense(10, activation='softmax')(merged_output)

    # Define the model
    model = keras.Model(inputs=[pathway1, pathway2], outputs=merged_output)

    return model