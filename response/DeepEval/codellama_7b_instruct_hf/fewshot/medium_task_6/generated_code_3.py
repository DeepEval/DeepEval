import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolution
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Parallel blocks
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    block1_bn = BatchNormalization()(block1)
    block1_relu = Activation('relu')(block1_bn)

    block2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_relu)
    block2_bn = BatchNormalization()(block2)
    block2_relu = Activation('relu')(block2_bn)

    block3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_relu)
    block3_bn = BatchNormalization()(block3)
    block3_relu = Activation('relu')(block3_bn)

    # Add the outputs of the parallel blocks
    added_blocks = keras.layers.add([block1_relu, block2_relu, block3_relu])

    # Flatten the output and pass through fully connected layers
    flattened = Flatten()(added_blocks)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model