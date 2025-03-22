import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def conv_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        norm1 = BatchNormalization()(conv1)
        relu1 = ReLU()(norm1)

        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(relu1)
        norm2 = BatchNormalization()(conv2)
        relu2 = ReLU()(norm2)

        return relu2

    # First block
    block1_output = conv_block(input_layer)

    # Second block
    block2_output = conv_block(block1_output)
    concatenated_output_1 = Concatenate()([block1_output, block2_output])

    # Third block
    block3_output = conv_block(concatenated_output_1)
    concatenated_output_2 = Concatenate()([concatenated_output_1, block3_output])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(concatenated_output_2)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model