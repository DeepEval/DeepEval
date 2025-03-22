import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense, MaxPooling2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters):
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    # First block
    block1_output = block(input_layer, filters=32)
    # Second block
    block2_output = block(block1_output, filters=64)
    # Third block
    block3_output = block(block2_output, filters=128)

    # Direct convolutional path
    direct_path_output = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(input_layer)
    direct_path_output = BatchNormalization()(direct_path_output)
    direct_path_output = ReLU()(direct_path_output)

    # Add all paths
    added_output = Add()([block1_output, block2_output, block3_output, direct_path_output])

    # Flatten and add fully connected layers
    flattened_output = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model