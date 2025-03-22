import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters):
        conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = ReLU()(batch_norm)
        return relu

    # First block
    block1_output = block(input_layer, filters=32)

    # Second block, concatenate with the output of the first block
    block2_input = Concatenate(axis=-1)([input_layer, block1_output])
    block2_output = block(block2_input, filters=64)

    # Third block, concatenate with the output of the second block
    block3_input = Concatenate(axis=-1)([block2_input, block2_output])
    block3_output = block(block3_input, filters=128)

    # Flatten and pass through fully connected layers
    flatten = Flatten()(block3_output)
    dense1 = Dense(units=256, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model