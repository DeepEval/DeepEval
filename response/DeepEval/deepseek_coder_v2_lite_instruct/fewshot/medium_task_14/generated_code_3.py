import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters):
        x = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    # First block
    block1_output = block(input_tensor=input_layer, filters=32)
    # Second block
    block2_output = block(input_tensor=block1_output, filters=64)
    # Third block
    block3_output = block(input_tensor=block2_output, filters=128)

    # Direct processing branch
    direct_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    direct_output = BatchNormalization()(direct_output)
    direct_output = ReLU()(direct_output)

    # Adding outputs from all paths
    added_output = Add()([block1_output, block2_output, block3_output, direct_output])

    # Flatten and pass through fully connected layers
    flattened_output = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model