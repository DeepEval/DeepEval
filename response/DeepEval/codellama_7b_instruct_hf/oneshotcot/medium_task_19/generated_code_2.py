import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First branch: dimensionality reduction
    branch1 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation='relu')(input_layer)

    # Second branch: extract features
    branch2 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(branch2)

    # Third branch: capture larger spatial information
    branch3 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu')(branch3)

    # Fourth branch: downsampling
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    branch4 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation='relu')(branch4)

    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten features
    flattened = Flatten()(concatenated)

    # Add fully connected layers
    dense1 = Dense(128, activation='relu')(flattened)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model