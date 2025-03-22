import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First feature extraction path: 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second feature extraction path: sequence of convolutions: 1x1, followed by 1x7, and then 7x1
    conv2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Concatenate the outputs from the two paths
    concatenated = Concatenate()([conv1, conv2])

    # Add a 1x1 convolution to align the output dimensions with the input image's channel
    aligned = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Add a branch that merges the output of the main path and the branch through addition
    branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    merged = Add()([aligned, branch])

    # Add batch normalization and flatten layer
    batch_norm = BatchNormalization()(merged)
    flatten = Flatten()(batch_norm)

    # Add two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer
    output = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model