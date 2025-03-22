import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters):
        conv1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        batch_norm1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
        batch_norm2 = BatchNormalization()(conv2)
        output_tensor = Concatenate(axis=-1)([input_tensor, batch_norm2])
        return output_tensor

    # Block 1
    block1_output = block(input_layer, filters=32)
    block1_output = block(block1_output, filters=32)

    # Block 2
    block2_output = block(block1_output, filters=64)
    block2_output = block(block2_output, filters=64)

    # Block 3
    block3_output = block(block2_output, filters=128)
    block3_output = block(block3_output, filters=128)

    # Flatten the output
    flatten_layer = Flatten()(block3_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model