import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First path: 1x1 Convolution
    path1 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)

    # Second path: 1x1 Convolution followed by two 3x3 Convolutions
    path2 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    path2 = Conv2D(32, (3, 3), activation='relu', padding='same')(path2)
    path2 = Conv2D(32, (3, 3), activation='relu', padding='same')(path2)

    # Third path: 1x1 Convolution followed by 3x3 Convolution
    path3 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    path3 = Conv2D(32, (3, 3), activation='relu', padding='same')(path3)

    # Fourth path: Max Pooling followed by 1x1 Convolution
    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    path4 = Conv2D(32, (1, 1), activation='relu', padding='same')(path4)

    # Concatenate the outputs from the four paths
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Flatten the concatenated output
    flat = Flatten()(concatenated)

    # Dense layer with 128 units
    dense = Dense(128, activation='relu')(flat)

    # Output layer with softmax activation for 10 classes
    output_layer = Dense(10, activation='softmax')(dense)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()