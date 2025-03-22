import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters):
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=None)(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    # First block
    block1_output = block(input_layer, filters=32)

    # Concatenate with input_layer
    concat1 = Concatenate(axis=-1)([input_layer, block1_output])

    # Second block
    block2_output = block(concat1, filters=64)

    # Concatenate with concat1
    concat2 = Concatenate(axis=-1)([concat1, block2_output])

    # Third block
    block3_output = block(concat2, filters=128)

    # Concatenate with concat2
    concat3 = Concatenate(axis=-1)([concat2, block3_output])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(concat3)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model