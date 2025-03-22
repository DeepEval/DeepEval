import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def conv_block(input_tensor, filters):
        conv = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input_tensor)
        norm = BatchNormalization()(conv)
        relu = ReLU()(norm)
        return relu

    # First block
    block1_output = conv_block(input_layer, filters=32)

    # Second block
    block2_output = conv_block(block1_output, filters=64)
    combined_output_1 = Concatenate()([block1_output, block2_output])

    # Third block
    block3_output = conv_block(combined_output_1, filters=128)
    combined_output_2 = Concatenate()([combined_output_1, block3_output])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(combined_output_2)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model