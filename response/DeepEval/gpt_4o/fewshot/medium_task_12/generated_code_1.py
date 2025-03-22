import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters):
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    # First block
    block1_output = block(input_layer, filters=32)

    # Second block, concatenating the output of the first block
    concatenated_1_2 = Concatenate(axis=-1)([input_layer, block1_output])
    block2_output = block(concatenated_1_2, filters=64)

    # Third block, concatenating the output of the second block
    concatenated_2_3 = Concatenate(axis=-1)([concatenated_1_2, block2_output])
    block3_output = block(concatenated_2_3, filters=128)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(block3_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model