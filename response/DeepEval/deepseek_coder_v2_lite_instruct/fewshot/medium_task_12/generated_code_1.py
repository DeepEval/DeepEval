import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters, kernel_size):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation=None)(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    # Block 1
    block1_output = block(input_tensor=input_layer, filters=32, kernel_size=(3, 3))
    # Block 2
    block2_output = block(input_tensor=block1_output, filters=64, kernel_size=(3, 3))
    # Block 3
    block3_output = block(input_tensor=block2_output, filters=128, kernel_size=(3, 3))

    # Concatenate outputs from all blocks
    concatenated_output = Concatenate(axis=-1)([block1_output, block2_output, block3_output])

    # Flatten the concatenated output
    flattened_output = Flatten()(concatenated_output)

    # Pass through two fully connected layers
    dense1 = Dense(units=256, activation='relu')(flattened_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model