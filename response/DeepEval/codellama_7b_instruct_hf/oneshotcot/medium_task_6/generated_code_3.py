import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolution
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # First parallel block
    block1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    block1 = BatchNormalization()(block1)
    block1 = ReLU()(block1)

    # Second parallel block
    block2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    block2 = BatchNormalization()(block2)
    block2 = ReLU()(block2)

    # Third parallel block
    block3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    block3 = BatchNormalization()(block3)
    block3 = ReLU()(block3)

    # Add the outputs of the parallel blocks to the initial convolution
    output = Concatenate()([block1, block2, block3])

    # Flatten the output
    flatten = Flatten()(output)

    # Fully connected layers
    dense1 = Dense(units=64, activation='relu')(flatten)
    dense2 = Dense(units=32, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model