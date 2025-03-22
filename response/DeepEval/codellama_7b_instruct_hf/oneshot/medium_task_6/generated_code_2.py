import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the first parallel block
    block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    block1 = BatchNormalization()(block1)
    block1 = ReLU()(block1)

    # Define the second parallel block
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    block2 = BatchNormalization()(block2)
    block2 = ReLU()(block2)

    # Define the third parallel block
    block3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    block3 = BatchNormalization()(block3)
    block3 = ReLU()(block3)

    # Add the output of the parallel blocks
    added_output = Concatenate()([block1, block2, block3])

    # Flatten the output and pass through fully connected layers
    flattened_output = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model