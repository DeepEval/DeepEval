import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Define the inputs
    inputs = Input(shape=(32, 32, 3))

    # First path: 1x1 convolution
    path1 = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Second path: two 3x3 convolutions stacked after a 1x1 convolution
    path2 = Conv2D(32, (1, 1), activation='relu')(inputs)
    path2 = Conv2D(32, (3, 3), activation='relu', padding='same')(path2)

    # Third path: single 3x3 convolution following a 1x1 convolution
    path3 = Conv2D(32, (1, 1), activation='relu')(inputs)
    path3 = Conv2D(32, (3, 3), activation='relu', padding='same')(path3)

    # Fourth path: max pooling followed by a 1x1 convolution
    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    path4 = Conv2D(32, (1, 1), activation='relu')(path4)

    # Concatenate the outputs of the four paths
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Pass through a dense layer with 128 units
    dense = Dense(128, activation='relu')(flattened)

    # Final output layer with softmax activation
    outputs = Dense(10, activation='softmax')(dense)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model