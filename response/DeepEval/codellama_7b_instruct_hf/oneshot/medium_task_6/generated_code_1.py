import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the initial convolutional layer
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    # Define the first parallel block
    block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # Define the second parallel block
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    # Define the third parallel block
    block3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    # Add the outputs of the parallel blocks
    added_layer = Add()([block1, block2, block3])

    # Flatten the output
    flatten_layer = Flatten()(added_layer)

    # Pass the flattened output through fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model