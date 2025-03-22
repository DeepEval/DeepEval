import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters):
        conv = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        return batch_norm

    # First block
    block1_output = block(input_layer, filters=32)

    # Second block
    block2_output = block(block1_output, filters=64)
    block2_output = Concatenate(axis=-1)([block2_output, block1_output])

    # Third block
    block3_output = block(block2_output, filters=128)
    block3_output = Concatenate(axis=-1)([block3_output, block2_output])

    # Flatten the output
    flatten_layer = Flatten()(block3_output)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model