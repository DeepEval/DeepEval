import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters):
        conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        return batch_norm

    # First block
    conv1 = block(input_layer, 32)
    conv2 = block(conv1, 32)
    conv3 = block(conv2, 32)
    first_block_output = Concatenate(axis=-1)([input_layer, conv1, conv2, conv3])

    # Second block
    conv4 = block(first_block_output, 64)
    conv5 = block(conv4, 64)
    conv6 = block(conv5, 64)
    second_block_output = Concatenate(axis=-1)([first_block_output, conv4, conv5, conv6])

    # Third block
    conv7 = block(second_block_output, 128)
    conv8 = block(conv7, 128)
    conv9 = block(conv8, 128)
    third_block_output = Concatenate(axis=-1)([second_block_output, conv7, conv8, conv9])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(third_block_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model