import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First block

    # 2x2 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # 3x3 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv1)

    # Average pooling layer
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Add the output of the first block to the main path
    add = Concatenate()([pool1, input_layer])

    # Second block

    # Global average pooling layer
    gavg = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(add)

    # Fully connected layer with the same number of channels as the output of the first block
    dense1 = Dense(units=32, activation='relu')(gavg)

    # Fully connected layer with the same number of channels as the output of the first block
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Reshape the output of the second block
    reshape = Flatten()(dense2)

    # Multiply the output of the second block with the input
    mul = keras.layers.multiply([reshape, input_layer])

    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(mul)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model